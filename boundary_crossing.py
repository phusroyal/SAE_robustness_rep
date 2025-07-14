"""
Boundary Crossing Analysis for SAE Robustness
============================================

This module implements decision boundary tracing to measure margins and boundary 
crossings in Sparse Autoencoders (SAEs), following the experimental design for 
studying SAE robustness on Gemma2-2B.

Key Functions:
- Decision Boundary Tracing: Find minimal perturbations that cause code changes
- Multi-output DeepFool variant for SAEs
- Targeted boundary search between concept pairs
- Margin analysis and boundary characterization
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundaryResult:
    """Results from boundary tracing experiment"""
    delta_min: float  # Minimal perturbation norm
    perturbation: torch.Tensor  # The actual perturbation vector
    original_code: torch.Tensor  # Original sparse code
    perturbed_code: torch.Tensor  # Code after perturbation
    flipped_features: List[Tuple[int, str]]  # (feature_idx, flip_type: 'activate'|'deactivate')
    num_steps: int  # Number of optimization steps
    boundary_type: str  # Type of boundary crossed
    target_reached: bool  # Whether target was reached (for targeted search)


class SAEBoundaryTracer:
    """
    Implements decision boundary tracing for Sparse Autoencoders.
    
    This class provides methods to find minimal perturbations that cause
    changes in SAE sparse codes, implementing a multi-output variant of DeepFool.
    """
    
    def __init__(self, sae_model, tokenizer, base_model=None, device='cpu'):
        """
        Initialize boundary tracer.
        
        Args:
            sae_model: Trained SAE model
            tokenizer: Tokenizer for the base model
            base_model: Base language model (optional, for getting hidden states)
            device: Device to run computations on
        """
        self.sae = sae_model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        
        # Move models to device
        self.sae = self.sae.to(device)
        if self.base_model is not None:
            self.base_model = self.base_model.to(device)
    
    def get_hidden_representation(self, text: str, layer_idx: int) -> torch.Tensor:
        """Extract hidden representation from base model at specified layer."""
        if self.base_model is None:
            raise ValueError("Base model required for extracting hidden representations")
        
        # Tokenize input
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens['input_ids'].to(self.device)
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.base_model(input_ids, output_hidden_states=True)
            # Take the last token's representation
            hidden_state = outputs.hidden_states[layer_idx + 1][0, -1, :]
        
        return hidden_state.detach()
    
    def get_sae_code(self, hidden_state: torch.Tensor, return_reconstruction: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get sparse code from SAE encoder."""
        hidden_state = hidden_state.to(self.device)
        
        with torch.no_grad():
            if hasattr(self.sae, 'encode'):
                sparse_code = self.sae.encode(hidden_state)
            else:
                # Assume standard SAE structure
                centered = hidden_state - self.sae.b_dec
                sparse_code = F.relu(centered @ self.sae.W_enc.T + self.sae.b_enc)
        
        if return_reconstruction:
            reconstruction = sparse_code @ self.sae.W_dec + self.sae.b_dec
            return sparse_code, reconstruction
        
        return sparse_code
    
    def compute_sae_jacobian(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of SAE encoder with respect to input.
        
        Returns:
            jacobian: (n_features, d_model) tensor representing dz/dh
        """
        hidden_state = hidden_state.to(self.device).requires_grad_(True)
        
        # Forward pass through SAE encoder
        if hasattr(self.sae, 'encode'):
            sparse_code = self.sae.encode(hidden_state)
        else:
            centered = hidden_state - self.sae.b_dec
            pre_activation = centered @ self.sae.W_enc.T + self.sae.b_enc
            sparse_code = F.relu(pre_activation)
        
        # Compute Jacobian
        jacobian = torch.zeros(sparse_code.shape[0], hidden_state.shape[0], device=self.device)
        
        for i in range(sparse_code.shape[0]):
            grad_outputs = torch.zeros_like(sparse_code)
            grad_outputs[i] = 1.0
            
            grad = torch.autograd.grad(
                outputs=sparse_code,
                inputs=hidden_state,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]
            
            jacobian[i] = grad
        
        return jacobian
    
    def find_minimal_boundary_perturbation(
        self, 
        hidden_state: torch.Tensor,
        max_iterations: int = 100,
        step_size: float = 0.01,
        tolerance: float = 1e-6,
        verbose: bool = True
    ) -> BoundaryResult:
        """
        Find minimal L2 perturbation that changes SAE sparse code.
        
        Implements a multi-output variant of DeepFool for SAEs.
        
        Args:
            hidden_state: Input hidden representation
            max_iterations: Maximum optimization steps
            step_size: Step size for perturbation updates
            tolerance: Convergence tolerance
            verbose: Whether to print progress
            
        Returns:
            BoundaryResult containing perturbation details
        """
        if verbose:
            logger.info("Starting minimal boundary perturbation search...")
        
        # Get original sparse code
        original_code = self.get_sae_code(hidden_state)
        original_active = (original_code > 0).float()
        
        # Initialize perturbation
        delta = torch.zeros_like(hidden_state, requires_grad=False)
        current_h = hidden_state.clone()
        
        for step in range(max_iterations):
            # Compute current code
            current_code = self.get_sae_code(current_h)
            current_active = (current_code > 0).float()
            
            # Check if code has changed
            if not torch.equal(original_active, current_active):
                # Found boundary crossing!
                flipped_features = self._identify_flipped_features(original_active, current_active)
                
                if verbose:
                    logger.info(f"Boundary crossed at step {step}, ||δ|| = {delta.norm().item():.6f}")
                    logger.info(f"Flipped features: {flipped_features}")
                
                return BoundaryResult(
                    delta_min=delta.norm().item(),
                    perturbation=delta.clone(),
                    original_code=original_code,
                    perturbed_code=current_code,
                    flipped_features=flipped_features,
                    num_steps=step,
                    boundary_type="activation_change",
                    target_reached=True
                )
            
            # Compute Jacobian
            jacobian = self.compute_sae_jacobian(current_h)
            
            # Find the closest decision boundary
            # For each inactive feature, compute distance to activation
            # For each active feature, compute distance to deactivation
            
            min_distance = float('inf')
            best_direction = None
            
            for i, (original_active_i, current_val) in enumerate(zip(original_active, current_code)):
                grad_i = jacobian[i]
                
                if original_active_i == 0:  # Originally inactive feature
                    # Distance to activate: need current_val + grad_i^T * δ > 0
                    # Minimal δ in direction of grad_i: δ = -current_val * grad_i / ||grad_i||^2
                    if grad_i.norm() > tolerance:
                        distance = abs(current_val) / grad_i.norm()
                        if distance < min_distance:
                            min_distance = distance
                            best_direction = -current_val * grad_i / (grad_i.norm() ** 2)
                
                else:  # Originally active feature
                    # Distance to deactivate: need current_val + grad_i^T * δ <= 0
                    if grad_i.norm() > tolerance:
                        distance = current_val / grad_i.norm()
                        if distance < min_distance:
                            min_distance = distance
                            best_direction = -current_val * grad_i / (grad_i.norm() ** 2)
            
            if best_direction is None:
                logger.warning("No valid direction found, stopping")
                break
            
            # Take a step in the best direction
            step_delta = step_size * best_direction / best_direction.norm()
            delta += step_delta
            current_h = hidden_state + delta
            
            if verbose and step % 10 == 0:
                logger.info(f"Step {step}: ||δ|| = {delta.norm().item():.6f}")
        
        # If we reach here, no boundary was found
        logger.warning(f"No boundary found within {max_iterations} steps")
        return BoundaryResult(
            delta_min=delta.norm().item(),
            perturbation=delta,
            original_code=original_code,
            perturbed_code=self.get_sae_code(current_h),
            flipped_features=[],
            num_steps=max_iterations,
            boundary_type="none",
            target_reached=False
        )
    
    def targeted_boundary_search(
        self,
        source_hidden: torch.Tensor,
        target_code: torch.Tensor,
        max_iterations: int = 200,
        step_size: float = 0.01,
        tolerance: float = 1e-6,
        verbose: bool = True
    ) -> BoundaryResult:
        """
        Find perturbation to make source hidden state produce target sparse code.
        
        Args:
            source_hidden: Starting hidden representation
            target_code: Desired sparse code
            max_iterations: Maximum optimization steps
            step_size: Step size for gradient updates
            tolerance: Convergence tolerance
            verbose: Whether to print progress
            
        Returns:
            BoundaryResult containing perturbation details
        """
        if verbose:
            logger.info("Starting targeted boundary search...")
        
        source_code = self.get_sae_code(source_hidden)
        target_active = (target_code > 0).float()
        
        # Initialize perturbation
        delta = torch.zeros_like(source_hidden, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=step_size)
        
        best_delta = None
        best_loss = float('inf')
        
        for step in range(max_iterations):
            optimizer.zero_grad()
            
            # Get current code
            current_h = source_hidden + delta
            current_code = self.get_sae_code(current_h)
            current_active = (current_code > 0).float()
            
            # Check if target reached
            if torch.equal(current_active, target_active):
                if verbose:
                    logger.info(f"Target reached at step {step}, ||δ|| = {delta.norm().item():.6f}")
                
                return BoundaryResult(
                    delta_min=delta.norm().item(),
                    perturbation=delta.detach().clone(),
                    original_code=source_code,
                    perturbed_code=current_code.detach(),
                    flipped_features=self._identify_flipped_features(
                        (source_code > 0).float(), current_active
                    ),
                    num_steps=step,
                    boundary_type="targeted",
                    target_reached=True
                )
            
            # Loss: L2 penalty + activation pattern matching
            l2_loss = delta.norm()
            activation_loss = F.mse_loss(current_active, target_active)
            code_loss = F.mse_loss(current_code, target_code)
            
            total_loss = l2_loss + 10 * activation_loss + code_loss
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_delta = delta.detach().clone()
            
            total_loss.backward()
            optimizer.step()
            
            if verbose and step % 20 == 0:
                logger.info(f"Step {step}: ||δ|| = {delta.norm().item():.6f}, loss = {total_loss.item():.6f}")
        
        # Return best result found
        final_code = self.get_sae_code(source_hidden + best_delta)
        return BoundaryResult(
            delta_min=best_delta.norm().item(),
            perturbation=best_delta,
            original_code=source_code,
            perturbed_code=final_code,
            flipped_features=self._identify_flipped_features(
                (source_code > 0).float(), (final_code > 0).float()
            ),
            num_steps=max_iterations,
            boundary_type="targeted",
            target_reached=False
        )
    
    def _identify_flipped_features(self, original_active: torch.Tensor, current_active: torch.Tensor) -> List[Tuple[int, str]]:
        """Identify which features changed activation status."""
        flipped = []
        
        for i in range(len(original_active)):
            if original_active[i] == 0 and current_active[i] == 1:
                flipped.append((i, 'activate'))
            elif original_active[i] == 1 and current_active[i] == 0:
                flipped.append((i, 'deactivate'))
        
        return flipped


# Example usage and test cases
if __name__ == "__main__":
    logger.info("Boundary crossing analysis module loaded successfully!")
    logger.info("Example usage:")
    logger.info("1. Initialize tracer: tracer = SAEBoundaryTracer(sae_model, tokenizer, base_model)")
    logger.info("2. Analyze margins: results = tracer.analyze_margins_batch(texts, layer_idx=20)")
    logger.info("3. Concept distances: distances = tracer.concept_distance_analysis(concept_pairs, layer_idx=20)")

    def analyze_margins_batch(
        self,
        texts: List[str],
        layer_idx: int,
        max_iterations: int = 100,
        verbose: bool = True
    ) -> Dict[str, BoundaryResult]:
        """
        Analyze decision boundaries for a batch of text inputs.
        
        Args:
            texts: List of input texts
            layer_idx: Layer index to extract hidden states from
            max_iterations: Maximum iterations per boundary search
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping text to BoundaryResult
        """
        results = {}
        
        if verbose:
            texts_iter = tqdm(texts, desc="Analyzing boundaries")
        else:
            texts_iter = texts
        
        for i, text in enumerate(texts_iter):
            try:
                # Extract hidden representation
                hidden_state = self.get_hidden_representation(text, layer_idx)
                
                # Find minimal boundary perturbation
                result = self.find_minimal_boundary_perturbation(
                    hidden_state, 
                    max_iterations=max_iterations,
                    verbose=False
                )
                
                results[f"text_{i}"] = result
                
                if verbose and i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                continue
        
        return results
    
    def concept_distance_analysis(
        self,
        concept_pairs: List[Tuple[str, str]],
        layer_idx: int,
        max_iterations: int = 200
    ) -> Dict[str, Dict[str, float]]:
        """
        Measure distances between concept pairs in terms of boundary crossings.
        
        Args:
            concept_pairs: List of (concept_A_text, concept_B_text) pairs
            layer_idx: Layer index to extract hidden states from
            max_iterations: Maximum iterations for targeted search
            
        Returns:
            Dictionary with distance metrics for each concept pair
        """
        results = {}
        
        for i, (text_a, text_b) in enumerate(concept_pairs):
            logger.info(f"Analyzing concept pair {i+1}/{len(concept_pairs)}")
            
            # Get hidden representations
            hidden_a = self.get_hidden_representation(text_a, layer_idx)
            hidden_b = self.get_hidden_representation(text_b, layer_idx)
            
            # Get sparse codes
            code_a = self.get_sae_code(hidden_a)
            code_b = self.get_sae_code(hidden_b)
            
            # Bidirectional targeted search
            result_a_to_b = self.targeted_boundary_search(
                hidden_a, code_b, max_iterations=max_iterations, verbose=False
            )
            result_b_to_a = self.targeted_boundary_search(
                hidden_b, code_a, max_iterations=max_iterations, verbose=False
            )
            
            # Single boundary distances
            boundary_a = self.find_minimal_boundary_perturbation(hidden_a, verbose=False)
            boundary_b = self.find_minimal_boundary_perturbation(hidden_b, verbose=False)
            
            pair_key = f"pair_{i}"
            results[pair_key] = {
                'a_to_b_distance': result_a_to_b.delta_min,
                'b_to_a_distance': result_b_to_a.delta_min,
                'a_boundary_distance': boundary_a.delta_min,
                'b_boundary_distance': boundary_b.delta_min,
                'distance_ratio_a': result_a_to_b.delta_min / boundary_a.delta_min if boundary_a.delta_min > 0 else float('inf'),
                'distance_ratio_b': result_b_to_a.delta_min / boundary_b.delta_min if boundary_b.delta_min > 0 else float('inf'),
                'concept_a': text_a,
                'concept_b': text_b,
                'a_to_b_reached': result_a_to_b.target_reached,
                'b_to_a_reached': result_b_to_a.target_reached
            }
        
        return results


def visualize_boundary_analysis(results: Dict[str, BoundaryResult], save_path: Optional[str] = None):
    """
    Visualize results from boundary analysis.
    
    Args:
        results: Results from analyze_margins_batch
        save_path: Optional path to save the plot
    """
    # Extract data for plotting
    delta_mins = [r.delta_min for r in results.values() if r.target_reached]
    num_flips = [len(r.flipped_features) for r in results.values() if r.target_reached]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot distribution of minimal perturbations
    ax1.hist(delta_mins, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Minimal Perturbation Norm ||δ_min||')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Minimal Boundary Distances')
    ax1.grid(True, alpha=0.3)
    
    # Plot number of feature flips
    if num_flips:
        ax2.hist(num_flips, bins=max(num_flips)+1, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Feature Flips')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Feature Flips at Boundaries')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def visualize_concept_distances(concept_results: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
    """
    Visualize concept distance analysis results.
    
    Args:
        concept_results: Results from concept_distance_analysis
        save_path: Optional path to save the plot
    """
    # Extract data
    a_to_b_distances = [data['a_to_b_distance'] for data in concept_results.values()]
    b_to_a_distances = [data['b_to_a_distance'] for data in concept_results.values()]
    distance_ratios_a = [data['distance_ratio_a'] for data in concept_results.values() if data['distance_ratio_a'] != float('inf')]
    distance_ratios_b = [data['distance_ratio_b'] for data in concept_results.values() if data['distance_ratio_b'] != float('inf')]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Bidirectional distances
    if a_to_b_distances and b_to_a_distances:
        ax1.scatter(a_to_b_distances, b_to_a_distances, alpha=0.7)
        max_dist = max(max(a_to_b_distances), max(b_to_a_distances))
        ax1.plot([0, max_dist], [0, max_dist], 'r--', alpha=0.5)
    ax1.set_xlabel('A → B Distance')
    ax1.set_ylabel('B → A Distance')
    ax1.set_title('Bidirectional Concept Distances')
    ax1.grid(True, alpha=0.3)
    
    # Distance ratios
    if distance_ratios_a or distance_ratios_b:
        ax2.hist(distance_ratios_a + distance_ratios_b, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Distance Ratio (Targeted / Single Boundary)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Concept Distance vs Single Boundary Ratios')
    ax2.grid(True, alpha=0.3)
    
    # Success rates
    success_rates_a = [data['a_to_b_reached'] for data in concept_results.values()]
    success_rates_b = [data['b_to_a_reached'] for data in concept_results.values()]
    
    categories = ['A → B', 'B → A']
    success_counts = [sum(success_rates_a), sum(success_rates_b)]
    total_counts = [len(success_rates_a), len(success_rates_b)]
    
    if total_counts[0] > 0 and total_counts[1] > 0:
        ax3.bar(categories, [s/t for s, t in zip(success_counts, total_counts)], alpha=0.7)
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Targeted Boundary Search Success Rates')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Distance distributions
    all_distances = a_to_b_distances + b_to_a_distances
    if all_distances:
        ax4.hist(all_distances, bins=20, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Concept Distance')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Concept Distances')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()
