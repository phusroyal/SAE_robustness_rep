"""
Boundary Crossing Analysis for SAE Robustness - REFINED VERSION
================================================================

This module implements decision boundary tracing in hidden activation space
to identify exact points where SAE's active feature set changes.

REFINED: Enhanced stability, precision, and diagnostic capabilities.
"""

# %%

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

from utils import *

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
    num_boundaries_crossed: int  # Number of intermediate boundaries crossed
    boundary_type: str  # Type of boundary crossed
    target_reached: bool  # Whether target was reached
    trajectory: List[torch.Tensor]  # Trajectory of perturbations
    # New diagnostic fields
    cosine_similarity: float  # Cosine similarity between source and target
    relu_flips: int  # Number of ReLU pattern changes
    sae_flips: int  # Number of SAE feature changes
    trajectory_hidden: List[torch.Tensor]  # Trajectory of hidden states


class SAEBoundaryTracer:
    """
    Refined implementation of decision boundary tracing for Sparse Autoencoders.
    Enhanced with stability improvements and better diagnostics.
    """
    
    def __init__(self, sae_model, tokenizer, base_model=None, device='cpu', 
                 normalize_hidden=False, epsilon=1e-8, max_step_size=1.0):
        """Initialize boundary tracer with refined parameters."""
        self.sae = sae_model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.normalize_hidden = normalize_hidden  # Whether to normalize hidden states
        self.epsilon = epsilon  # Numerical stability parameter
        self.max_step_size = max_step_size  # Maximum step size for stability
    
    def get_hidden_representation(self, text: str, layer_idx: int) -> torch.Tensor:
        """Extract hidden representation from base model at specified layer."""
        if self.base_model is None:
            raise ValueError("Base model required for extracting hidden representations")
        
        post_text = "\nThe previous text is about"
        tokens = self.tokenizer(text + post_text, return_tensors="pt")
        input_ids = tokens['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(input_ids, output_hidden_states=True)
            hidden_state = outputs.hidden_states[layer_idx + 1][0, -1, :]
        
        # REFINED: Apply normalization if enabled
        if self.normalize_hidden:
            hidden_state = F.layer_norm(hidden_state, hidden_state.shape[-1:])
        
        return hidden_state.detach()
    
    def get_sae_code(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Get sparse code from SAE encoder with optional normalization."""
        # REFINED: Ensure consistent normalization
        if self.normalize_hidden:
            hidden_state = F.layer_norm(hidden_state, hidden_state.shape[-1:])
        
        sparse_code = jump_relu(hidden_state @ sae['W_enc'] + sae['b_enc'], sae['threshold'])
        return sparse_code
    
    def get_top_k_active_set(self, sparse_code: torch.Tensor, k: int = None) -> torch.Tensor:
        """Get indices of top-k active features with corrected logic."""
        if k is None:
            # REFINED: Use original indices for active features (not sorted)
            return (sparse_code > 0).nonzero(as_tuple=True)[0]
        else:
            # Get top-k by magnitude (sorted)
            sorted_acts, indices = torch.sort(sparse_code, dim=-1, descending=True)
            return indices[:k]
    
    def find_nearest_boundary_deepfool(
        self,
        hidden_state: torch.Tensor,
        max_iter: int = 50,
        overshoot: float = 0.02,
        top_k: int = None,
        verbose: bool = True
    ) -> BoundaryResult:
        """
        Find minimal perturbation to cross nearest SAE decision boundary.
        REFINED: Improved numerical stability and step size control.
        """
        if verbose:
            logger.info("Starting refined DeepFool-style boundary search...")
        
        original_code = self.get_sae_code(hidden_state)
        original_active_set = self.get_top_k_active_set(original_code, top_k)
        original_relu_pattern = (F.relu(hidden_state) > 0).float()
        
        # Initialize
        pert_hidden = copy.deepcopy(hidden_state)
        r_tot = torch.zeros_like(hidden_state)
        loop_i = 0
        trajectory = [r_tot.clone()]
        trajectory_hidden = [pert_hidden.clone()]
        boundary_crossings = 0
        
        current_active_set = original_active_set
        previous_relu_pattern = original_relu_pattern.clone()
        
        while torch.equal(current_active_set, original_active_set) and loop_i < max_iter:
            
            # Get current code and gradients
            current_code = self.get_sae_code(pert_hidden)
            current_relu_pattern = (F.relu(pert_hidden) > 0).float()
            
            # Count ReLU boundary crossings
            if not torch.equal(current_relu_pattern, previous_relu_pattern):
                boundary_crossings += 1
                if verbose:
                    flipped_units = (previous_relu_pattern != current_relu_pattern).nonzero(as_tuple=True)[0]
                    logger.info(f"Step {loop_i}: ReLU boundary crossing #{boundary_crossings}")
                    logger.info(f"  → Flipped ReLU units: {flipped_units[:10].tolist()}...")
            
            previous_relu_pattern = current_relu_pattern.clone()
            
            # Compute gradients for decision boundaries
            pert_hidden_var = pert_hidden.clone().requires_grad_(True)
            current_code_var = self.get_sae_code(pert_hidden_var)
            
            # Find minimal perturbation using refined DeepFool logic
            min_pert = float('inf')
            best_direction = None
            
            # For each feature, compute distance to flip its activation status
            for i, feature_val in enumerate(current_code):
                if i >= len(current_code_var):
                    continue
                    
                # Compute gradient of this feature
                if current_code_var[i].requires_grad:
                    grad_i = torch.autograd.grad(
                        outputs=current_code_var[i],
                        inputs=pert_hidden_var,
                        retain_graph=True,
                        create_graph=False
                    )[0]
                else:
                    continue
                
                # Distance to cross threshold (activate/deactivate)
                if feature_val <= 0:  # Inactive -> Active
                    f_k = -feature_val.item()
                else:  # Active -> Inactive  
                    f_k = feature_val.item()
                
                w_k = grad_i
                
                # REFINED: Improved numerical stability
                w_k_norm_sq = w_k.norm().item() ** 2
                if w_k_norm_sq > self.epsilon:
                    # Use projected gradient formulation
                    pert_k = abs(f_k) / (w_k_norm_sq + self.epsilon)
                    
                    if pert_k < min_pert:
                        min_pert = pert_k
                        best_direction = f_k * w_k / (w_k_norm_sq + self.epsilon)
            
            if best_direction is None:
                logger.warning("No valid direction found")
                break
            
            # REFINED: Apply step size clamping for stability
            r_i = (min_pert + 1e-4) * best_direction / (best_direction.norm() + self.epsilon)
            r_i = torch.clamp(r_i, -self.max_step_size, self.max_step_size)
            r_tot = r_tot + r_i
            
            # Update perturbed hidden state
            pert_hidden = hidden_state + (1 + overshoot) * r_tot
            
            # Check if active set changed
            new_code = self.get_sae_code(pert_hidden)
            current_active_set = self.get_top_k_active_set(new_code, top_k)
            
            trajectory.append(r_tot.clone())
            trajectory_hidden.append(pert_hidden.clone())
            loop_i += 1
            
            if verbose and loop_i % 10 == 0:
                logger.info(f"Step {loop_i}: ||δ|| = {r_tot.norm().item():.6f}")
        
        # Final perturbation with safety check
        final_perturbation = (1 + overshoot) * r_tot
        final_hidden = hidden_state + final_perturbation
        final_code = self.get_sae_code(final_hidden)
        final_relu_pattern = (F.relu(final_hidden) > 0).float()
        
        # Enhanced diagnostics
        flipped_features = self._identify_flipped_features(
            (original_code > 0).float(), 
            (final_code > 0).float()
        )
        
        sae_flips = torch.sum((original_code > 0) != (final_code > 0)).item()
        relu_flips = torch.sum(original_relu_pattern != final_relu_pattern).item()
        cosine_sim = F.cosine_similarity(hidden_state, final_hidden, dim=0).item()
        
        success = not torch.equal(
            self.get_top_k_active_set(original_code, top_k),
            self.get_top_k_active_set(final_code, top_k)
        )
        
        if verbose:
            logger.info(f"Boundary search completed: ||δ|| = {final_perturbation.norm().item():.6f}")
            logger.info(f"  SAE flips: {sae_flips}, ReLU flips: {relu_flips}")
            logger.info(f"  Cosine similarity: {cosine_sim:.4f}, Success: {success}")
        
        return BoundaryResult(
            delta_min=final_perturbation.norm().item(),
            perturbation=final_perturbation,
            original_code=original_code,
            perturbed_code=final_code,
            flipped_features=flipped_features,
            num_steps=loop_i,
            num_boundaries_crossed=boundary_crossings,
            boundary_type="nearest_refined",
            target_reached=success,
            trajectory=trajectory,
            cosine_similarity=cosine_sim,
            relu_flips=relu_flips,
            sae_flips=sae_flips,
            trajectory_hidden=trajectory_hidden
        )
    
    def targeted_concept_transition(
        self,
        source_text: str,
        target_text: str, 
        layer_idx: int,
        max_iter: int = 200,
        step_size: float = 0.01,
        use_deepfool: bool = True,
        verbose: bool = True

    ) -> Tuple[BoundaryResult, BoundaryResult]:
        """
        Find perturbations for concept transition in both hidden space and SAE space.
        REFINED: Enhanced with better diagnostics and stability.
        """
        if verbose:
            logger.info(f"Analyzing concept transition: '{source_text[:50]}...' -> '{target_text[:50]}...'")
        
        # Get representations
        source_hidden = self.get_hidden_representation(source_text, layer_idx)
        target_hidden = self.get_hidden_representation(target_text, layer_idx)
        source_code = self.get_sae_code(source_hidden)
        target_code = self.get_sae_code(target_hidden)
        
        # Enhanced diagnostics
        cos_sim = F.cosine_similarity(source_hidden, target_hidden, dim=0).item()
        l2_distance = F.mse_loss(source_hidden, target_hidden).item()
        relative_distance = l2_distance / (source_hidden.norm().item() ** 2 + self.epsilon)
        
        if verbose:
            logger.info(f"Source-Target Analysis:")
            logger.info(f"  Cosine similarity: {cos_sim:.4f}")
            logger.info(f"  L2 distance: {l2_distance:.6f}")
            logger.info(f"  Relative distance: {relative_distance:.6f}")
        
        if use_deepfool:
            # DeepFool-style transitions
            hidden_result = self._targeted_transition_hidden_deepfool(
                source_hidden, target_hidden, max_iter, verbose
            )
            sae_result = self._targeted_transition_sae_deepfool(
                source_hidden, target_code, max_iter, verbose
            )
        else:
            # Gradient descent transitions
            hidden_result = self._targeted_transition_hidden(
                source_hidden, target_hidden, max_iter, step_size, verbose
            )
            sae_result = self._targeted_transition_sae(
                source_hidden, target_code, max_iter, step_size, verbose
            )
        
        return hidden_result, sae_result
    
    def _targeted_transition_hidden(
        self,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        max_iter: int,
        step_size: float,
        verbose: bool
    ) -> BoundaryResult:
        """
        REFINED: Targeted transition in hidden space with enhanced polytope boundary tracking.
        """
        
        delta = torch.zeros_like(source_hidden, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=step_size)
        
        # Enhanced tracking
        original_relu_pattern = (F.relu(source_hidden) > 0).float()
        previous_relu_pattern = original_relu_pattern.clone()
        boundary_count = 0
        trajectory = []
        trajectory_hidden = []
        
        best_delta = None
        best_loss = float('inf')
        
        # Get original SAE code for comparison
        original_code = self.get_sae_code(source_hidden)
        
        if verbose:
            logger.info(f"REFINED METHOD: Enhanced ReLU polytope boundary tracking")
            logger.info(f"Original ReLU pattern: {original_relu_pattern.sum().item():.0f}/{len(original_relu_pattern)} active")
        
        for step in range(max_iter):
            optimizer.zero_grad()
            
            current_hidden = source_hidden + delta
            current_relu_pattern = (F.relu(current_hidden) > 0).float()
            
            # Enhanced boundary crossing detection
            if not torch.equal(current_relu_pattern, previous_relu_pattern):
                boundary_count += 1
                
                # Detailed logging of boundary crossings
                differences = current_relu_pattern - previous_relu_pattern
                activated_dims = (differences > 0).nonzero(as_tuple=True)[0]
                deactivated_dims = (differences < 0).nonzero(as_tuple=True)[0]
                
                if verbose and step % 20 == 0:
                    logger.info(f"Step {step}: Hidden polytope boundary crossing #{boundary_count}")
                    if len(activated_dims) > 0:
                        logger.info(f"  → Activated dimensions: {activated_dims[:5].tolist()}...")
                    if len(deactivated_dims) > 0:
                        logger.info(f"  → Deactivated dimensions: {deactivated_dims[:5].tolist()}...")
            
            previous_relu_pattern = current_relu_pattern.clone()
            
            # Loss with L2 regularization
            loss = F.mse_loss(current_hidden, target_hidden) + 0.001 * delta.norm()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.detach().clone()
            
            # Check convergence
            if loss.item() < 1e-6:
                if verbose:
                    logger.info(f"Hidden space target reached at step {step}")
                break
                
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
            
            optimizer.step()
            trajectory.append(delta.detach().clone())
            trajectory_hidden.append(current_hidden.detach().clone())
            
            if verbose and step % 50 == 0:
                logger.info(f"Hidden transition step {step}: loss = {loss.item():.6f}, ||δ|| = {delta.norm().item():.6f}")
        
        # REFINED: Always use best_delta for final results
        final_delta = best_delta if best_delta is not None else delta.detach()
        final_hidden = source_hidden + final_delta
        final_code = self.get_sae_code(final_hidden)
        final_relu_pattern = (F.relu(final_hidden) > 0).float()
        
        # Enhanced diagnostics
        sae_flips = torch.sum((original_code > 0) != (final_code > 0)).item()
        relu_flips = torch.sum(original_relu_pattern != final_relu_pattern).item()
        cosine_sim = F.cosine_similarity(source_hidden, final_hidden, dim=0).item()
        
        if verbose:
            logger.info(f"REFINED RESULT: {boundary_count} ReLU polytope boundaries crossed")
            logger.info(f"Final ReLU pattern: {final_relu_pattern.sum().item():.0f}/{len(final_relu_pattern)} active")
            logger.info(f"SAE changes: {sae_flips}, ReLU changes: {relu_flips}")
            logger.info(f"Final cosine similarity: {cosine_sim:.4f}")
        
        return BoundaryResult(
            delta_min=final_delta.norm().item(),
            perturbation=final_delta,
            original_code=original_code,
            perturbed_code=final_code,
            flipped_features=self._identify_flipped_features(
                (original_code > 0).float(), (final_code > 0).float()
            ),
            num_steps=step,
            num_boundaries_crossed=boundary_count,
            boundary_type="hidden_polytope_refined",
            target_reached=best_loss < 1e-3,
            trajectory=trajectory,
            cosine_similarity=cosine_sim,
            relu_flips=relu_flips,
            sae_flips=sae_flips,
            trajectory_hidden=trajectory_hidden
        )
    
    def _targeted_transition_sae(
        self,
        source_hidden: torch.Tensor,
        target_code: torch.Tensor,
        max_iter: int,
        step_size: float,
        verbose: bool
    ) -> BoundaryResult:
        """REFINED: Targeted transition in SAE feature space with masked loss."""
        
        delta = torch.zeros_like(source_hidden, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=step_size)
        
        original_code = self.get_sae_code(source_hidden)
        target_active = (target_code > 0).float()
        
        # REFINED: Create mask for focusing on important features
        importance_mask = (target_code > 0).float()
        
        trajectory = []
        trajectory_hidden = []
        best_delta = None
        best_loss = float('inf')
        
        for step in range(max_iter):
            optimizer.zero_grad()
            
            current_hidden = source_hidden + delta
            current_code = self.get_sae_code(current_hidden)
            current_active = (current_code > 0).float()
            
            # Check if exact target reached
            if torch.equal(current_active, target_active):
                if verbose:
                    logger.info(f"SAE target reached at step {step}")
                
                final_delta = delta.detach().clone()
                cosine_sim = F.cosine_similarity(source_hidden, current_hidden, dim=0).item()
                sae_flips = torch.sum((original_code > 0) != (current_code > 0)).item()
                
                return BoundaryResult(
                    delta_min=final_delta.norm().item(),
                    perturbation=final_delta,
                    original_code=original_code,
                    perturbed_code=current_code.detach(),
                    flipped_features=self._identify_flipped_features(
                        (original_code > 0).float(), current_active
                    ),
                    num_steps=step,
                    num_boundaries_crossed=1,
                    boundary_type="sae_targeted_refined", 
                    target_reached=True,
                    trajectory=trajectory,
                    cosine_similarity=cosine_sim,
                    relu_flips=0,  # Not tracked in SAE mode
                    sae_flips=sae_flips,
                    trajectory_hidden=trajectory_hidden
                )
            
            # REFINED: Multi-objective loss with masked focus
            l2_loss = delta.norm()
            activation_loss = F.mse_loss(current_active, target_active)
            
            # Masked feature loss (focus on important features)
            masked_feature_loss = F.mse_loss(current_code * importance_mask, target_code * importance_mask)
            
            total_loss = l2_loss + 10 * activation_loss + masked_feature_loss
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_delta = delta.detach().clone()
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
            
            optimizer.step()
            trajectory.append(delta.detach().clone())
            trajectory_hidden.append(current_hidden.detach().clone())
            
            if verbose and step % 50 == 0:
                logger.info(f"SAE transition step {step}: ||δ|| = {delta.norm().item():.6f}, loss = {total_loss.item():.6f}")
        
        # Use best result
        final_delta = best_delta if best_delta is not None else delta.detach()
        final_hidden = source_hidden + final_delta
        final_code = self.get_sae_code(final_hidden)
        
        # Enhanced diagnostics
        cosine_sim = F.cosine_similarity(source_hidden, final_hidden, dim=0).item()
        sae_flips = torch.sum((original_code > 0) != (final_code > 0)).item()
        
        return BoundaryResult(
            delta_min=final_delta.norm().item(),
            perturbation=final_delta,
            original_code=original_code,
            perturbed_code=final_code,
            flipped_features=self._identify_flipped_features(
                (original_code > 0).float(), (final_code > 0).float()
            ),
            num_steps=max_iter,
            num_boundaries_crossed=1,
            boundary_type="sae_targeted_refined",
            target_reached=False,
            trajectory=trajectory,
            cosine_similarity=cosine_sim,
            relu_flips=0,
            sae_flips=sae_flips,
            trajectory_hidden=trajectory_hidden
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
    
    def _targeted_transition_hidden_deepfool(
        self,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        max_iter: int,
        verbose: bool
    ) -> BoundaryResult:
        """REFINED: DeepFool-style targeted transition with enhanced stability."""
        if verbose:
            logger.info("Using refined DeepFool-style approach for hidden space transition")
        
        pert_hidden = copy.deepcopy(source_hidden)
        r_tot = torch.zeros_like(source_hidden)
        loop_i = 0
        trajectory = []
        trajectory_hidden = []
        boundary_count = 0
        
        # Track ReLU polytope boundaries
        original_relu_pattern = (F.relu(source_hidden) > 0).float()
        previous_relu_pattern = original_relu_pattern.clone()
        
        target_distance = F.mse_loss(source_hidden, target_hidden).item()
        current_distance = target_distance
        
        while current_distance > 1e-6 and loop_i < max_iter:
            # Compute gradient toward target
            pert_hidden_var = pert_hidden.clone().requires_grad_(True)
            loss = F.mse_loss(pert_hidden_var, target_hidden)
            
            grad = torch.autograd.grad(outputs=loss, inputs=pert_hidden_var)[0]
            
            # REFINED: Improved step size control
            direction = -grad / (grad.norm() + self.epsilon)
            
            # Adaptive step size with clamping
            step_size = min(0.1, current_distance * 0.1)
            r_i = step_size * direction
            r_i = torch.clamp(r_i, -self.max_step_size, self.max_step_size)
            r_tot = r_tot + r_i
            
            # Update perturbed hidden state
            pert_hidden = source_hidden + r_tot
            
            # Track polytope boundary crossings with detailed logging
            current_relu_pattern = (F.relu(pert_hidden) > 0).float()
            if not torch.equal(current_relu_pattern, previous_relu_pattern):
                boundary_count += 1
                if verbose and loop_i % 10 == 0:
                    flipped_units = (previous_relu_pattern != current_relu_pattern).nonzero(as_tuple=True)[0]
                    logger.info(f"DeepFool step {loop_i}: Polytope boundary #{boundary_count}")
                    logger.info(f"  → Flipped units: {flipped_units[:5].tolist()}...")
            
            previous_relu_pattern = current_relu_pattern.clone()
            current_distance = F.mse_loss(pert_hidden, target_hidden).item();
            
            trajectory.append(r_tot.clone())
            trajectory_hidden.append(pert_hidden.clone())
            loop_i += 1
            
            if verbose and loop_i % 20 == 0:
                logger.info(f"DeepFool step {loop_i}: distance = {current_distance:.6f}, ||δ|| = {r_tot.norm().item():.6f}")
        
        original_code = self.get_sae_code(source_hidden)
        final_code = self.get_sae_code(pert_hidden)
        
        # Enhanced diagnostics
        cosine_sim = F.cosine_similarity(source_hidden, pert_hidden, dim=0).item()
        sae_flips = torch.sum((original_code > 0) != (final_code > 0)).item()
        final_relu_pattern = (F.relu(pert_hidden) > 0).float()
        relu_flips = torch.sum(original_relu_pattern != final_relu_pattern).item()
        
        return BoundaryResult(
            delta_min=r_tot.norm().item(),
            perturbation=r_tot,
            original_code=original_code,
            perturbed_code=final_code,
            flipped_features=self._identify_flipped_features(
                (original_code > 0).float(), (final_code > 0).float()
            ),
            num_steps=loop_i,
            num_boundaries_crossed=boundary_count,
            boundary_type="hidden_deepfool_refined",
            target_reached=current_distance < 1e-3,
            trajectory=trajectory,
            cosine_similarity=cosine_sim,
            relu_flips=relu_flips,
            sae_flips=sae_flips,
            trajectory_hidden=trajectory_hidden
        )

    def _targeted_transition_sae_deepfool(
        self,
        source_hidden: torch.Tensor,
        target_code: torch.Tensor,
        max_iter: int,
        verbose: bool
    ) -> BoundaryResult:
        """REFINED: DeepFool-style SAE transition with better convergence."""
        if verbose:
            logger.info("Using refined DeepFool-style approach for SAE space transition")
        
        pert_hidden = copy.deepcopy(source_hidden)
        r_tot = torch.zeros_like(source_hidden)
        loop_i = 0
        trajectory = []
        trajectory_hidden = []
        
        original_code = self.get_sae_code(source_hidden)
        target_active = (target_code > 0).float()
        current_active = (original_code > 0).float()
        
        # Create importance mask
        importance_mask = (target_code > 0).float()
        
        while not torch.equal(current_active, target_active) and loop_i < max_iter:
            pert_hidden_var = pert_hidden.clone().requires_grad_(True)
            current_code_var = self.get_sae_code(pert_hidden_var)
            current_active_var = (current_code_var > 0).float()
            
            # REFINED: Enhanced loss with masked focus
            activation_loss = F.mse_loss(current_active_var, target_active)
            masked_feature_loss = F.mse_loss(current_code_var * importance_mask, target_code * importance_mask)
            total_loss = activation_loss + 0.1 * masked_feature_loss
            
            # Compute gradient with stability check
            grad = torch.autograd.grad(outputs=total_loss, inputs=pert_hidden_var)[0]
            
            # REFINED: Improved step control
            direction = -grad / (grad.norm() + self.epsilon)
            step_size = 0.01
            
            r_i = step_size * direction
            r_i = torch.clamp(r_i, -self.max_step_size, self.max_step_size)
            r_tot = r_tot + r_i
            
            # Update
            pert_hidden = source_hidden + r_tot
            current_code = self.get_sae_code(pert_hidden)
            current_active = (current_code > 0).float()
            
            trajectory.append(r_tot.clone())
            trajectory_hidden.append(pert_hidden.clone())
            loop_i += 1
            
            if verbose and loop_i % 20 == 0:
                logger.info(f"SAE DeepFool step {loop_i}: loss = {total_loss.item():.6f}, ||δ|| = {r_tot.norm().item():.6f}")
        
        final_code = self.get_sae_code(pert_hidden)
        target_reached = torch.equal((final_code > 0).float(), target_active)
        
        # Enhanced diagnostics
        cosine_sim = F.cosine_similarity(source_hidden, pert_hidden, dim=0).item()
        sae_flips = torch.sum((original_code > 0) != (final_code > 0)).item()
        
        return BoundaryResult(
            delta_min=r_tot.norm().item(),
            perturbation=r_tot,
            original_code=original_code,
            perturbed_code=final_code,
            flipped_features=self._identify_flipped_features(
                (original_code > 0).float(), (final_code > 0).float()
            ),
            num_steps=loop_i,
            num_boundaries_crossed=1,
            boundary_type="sae_deepfool_refined",
            target_reached=target_reached,
            trajectory=trajectory,
            cosine_similarity=cosine_sim,
            relu_flips=0,
            sae_flips=sae_flips,
            trajectory_hidden=trajectory_hidden
        )

    def visualize_trajectory(self, result: BoundaryResult, save_path: str = None):
        """Visualize the trajectory of boundary crossings and feature flips."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Perturbation magnitude over time
        perturbation_norms = [delta.norm().item() for delta in result.trajectory]
        axes[0, 0].plot(perturbation_norms)
        axes[0, 0].set_title('Perturbation Magnitude')
        axes[0, 0].set_xlabel('Optimization Step')
        axes[0, 0].set_ylabel('||δ||')
        
        # Plot 2: Feature flip count over time (if trajectory_hidden available)
        if hasattr(result, 'trajectory_hidden') and result.trajectory_hidden:
            original_pattern = (F.relu(result.trajectory_hidden[0]) > 0).float()
            flip_counts = []
            for hidden_state in result.trajectory_hidden:
                current_pattern = (F.relu(hidden_state) > 0).float()
                flips = torch.sum(original_pattern != current_pattern).item()
                flip_counts.append(flips)
            
            axes[0, 1].plot(flip_counts)
            axes[0, 1].set_title('ReLU Feature Flips')
            axes[0, 1].set_xlabel('Optimization Step')
            axes[0, 1].set_ylabel('# of ReLU Flips')
        
        # Plot 3: Cosine similarity (if available)
        if hasattr(result, 'cosine_similarity'):
            axes[1, 0].text(0.5, 0.5, f'Final Cosine Similarity: {result.cosine_similarity:.4f}', 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Final Metrics')
            axes[1, 0].axis('off')
        
        # Plot 4: Summary statistics
        summary_text = f"""
        Boundary Type: {result.boundary_type}
        Target Reached: {result.target_reached}
        Boundaries Crossed: {result.num_boundaries_crossed}
        SAE Flips: {getattr(result, 'sae_flips', 'N/A')}
        ReLU Flips: {getattr(result, 'relu_flips', 'N/A')}
        Final ||δ||: {result.delta_min:.6f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, ha='left', va='center', 
                        transform=axes[1, 1].transAxes, fontsize=10, family='monospace')
        axes[1, 1].set_title('Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()

# %%
def analysis(hidden_result, sae_result):
    print(f"\n{'='*10}")
    print("REFINED RESULTS COMPARISON")
    print(f"{'='*10}")
    
    print(f"\n1. HIDDEN SPACE TRANSITION (source_hidden → target_hidden):")
    print(f"   - Method: Enhanced ReLU polytope boundary tracking")
    print(f"   - Perturbation magnitude: {hidden_result.delta_min:.6f}")
    print(f"   - ReLU polytope boundaries crossed: {hidden_result.num_boundaries_crossed}")
    print(f"   - Target reached: {hidden_result.target_reached}")
    print(f"   - Boundary type: {hidden_result.boundary_type}")
    print(f"   - Cosine similarity: {getattr(hidden_result, 'cosine_similarity', 'N/A')}")
    print(f"   - ReLU flips: {getattr(hidden_result, 'relu_flips', 'N/A')}")
    print(f"   - SAE flips: {getattr(hidden_result, 'sae_flips', 'N/A')}")
    
    print(f"\n2. SAE SPACE TRANSITION (source_hidden → target_SAE_code):")
    print(f"   - Method: Enhanced SAE feature activation tracking")
    print(f"   - Perturbation magnitude: {sae_result.delta_min:.6f}")
    print(f"   - SAE boundaries crossed: {sae_result.num_boundaries_crossed}")
    print(f"   - Target reached: {sae_result.target_reached}")
    print(f"   - Boundary type: {sae_result.boundary_type}")
    print(f"   - Cosine similarity: {getattr(sae_result, 'cosine_similarity', 'N/A')}")
    print(f"   - SAE flips: {getattr(sae_result, 'sae_flips', 'N/A')}")
    
    # Enhanced insights
    if hidden_result.num_boundaries_crossed > 0 and sae_result.num_boundaries_crossed > 0:
        boundary_ratio = hidden_result.num_boundaries_crossed / sae_result.num_boundaries_crossed
        if sae_result.delta_min > 0:
            perturbation_ratio = hidden_result.delta_min / sae_result.delta_min
        else:
            perturbation_ratio = float('inf')
        
        print(f"\n{'='*10}")
        print("ENHANCED INSIGHTS")
        print(f"{'='*10}")
        print(f"Boundary crossing ratio (Hidden polytope / SAE): {boundary_ratio:.2f}")
        print(f"Perturbation ratio (Hidden / SAE): {perturbation_ratio:.2f}")
        
        # Compare cosine similarities
        if hasattr(hidden_result, 'cosine_similarity') and hasattr(sae_result, 'cosine_similarity'):
            print(f"Cosine similarity ratio (Hidden / SAE): {hidden_result.cosine_similarity / sae_result.cosine_similarity:.2f}")
        
        # Compare feature changes
        if hasattr(hidden_result, 'relu_flips') and hasattr(sae_result, 'sae_flips'):
            feature_ratio = hidden_result.relu_flips / max(sae_result.sae_flips, 1)
            print(f"Feature change ratio (ReLU flips / SAE flips): {feature_ratio:.2f}")
        
        if boundary_ratio > 2:
            print("→ Hidden space transition crosses MANY more polytope boundaries")
            print("  This suggests complex intermediate polytope structure")
        elif boundary_ratio > 1.2:
            print("→ Hidden space has moderately more polytope complexity")
        else:
            print("→ Hidden and SAE space have similar boundary complexity")

# Simple single-case experiment function (no batch analysis)
def run_single_concept_experiment(
    tracer,
    source_text: str,
    target_text: str,
    layer_idx: int,
    max_iter: int = 100,
    use_deepfool: bool = True,
):
    """Run a single concept transition experiment."""
    
    print("="*80)
    print("CORRECTED SINGLE CONCEPT TRANSITION EXPERIMENT")
    print("="*80)
    print(f"Source: '{source_text[:60]}...'")
    print(f"Target: '{target_text[:60]}...'")
    print()
    
    # Run the corrected analysis
    hidden_result, sae_result = tracer.targeted_concept_transition(
        source_text=source_text,
        target_text=target_text,
        layer_idx=layer_idx,
        max_iter=max_iter,
        use_deepfool=use_deepfool,
        verbose=True
    )

    analysis(hidden_result, sae_result)
    
    return hidden_result, sae_result

# %%

# Add experiments directly to the file
import os 
os.environ["https_proxy"] = "http://xen03.iitd.ac.in:3128"
os.environ["http_proxy"] = "http://xen03.iitd.ac.in:3128"

import sys
from sae import Sae
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F

# if __name__ == "__main__":
# Load model
model_type = 'gemma2-2b'
layer_num = 20
device = 'cpu'

print("Loading model and SAE...")
model, tokenizer, sae = load_model_and_sae(model_type, layer_num, device)

# %%
# Test cases
education_text = 'The film explores love and trauma through non-linear storytelling, blending magical realism with emotionally raw performances'
technology_text = 'The film explores love and trauma through non-linear storytelling, blending magical realism with emotionally raw performancesHacker Encryption implementations'

# Initialize corrected tracer
tracer = SAEBoundaryTracer(sae, tokenizer, model, device=device, normalize_hidden=False)

# %%
# Run corrected single experiment
hidden_result, sae_result = run_single_concept_experiment(
    tracer=tracer,
    source_text=education_text,
    target_text=technology_text,
    layer_idx=layer_num,
    max_iter=10000,
    use_deepfool=False
)

# %%
# Comparison experiment: Gradient Descent vs DeepFool-style for targeted transitions

def compare_transition_methods(tracer, source_text, target_text, layer_idx):
    """Compare gradient descent vs DeepFool-style approaches for targeted transitions."""
    
    print("\n" + "="*80)
    print("COMPARISON: GRADIENT DESCENT vs DEEPFOOL-STYLE TARGETED TRANSITIONS")
    print("="*80)
    
    # Method 1: Gradient Descent (original)
    print("\n1. GRADIENT DESCENT APPROACH:")
    print("   - Optimizes directly toward target using Adam optimizer")
    print("   - Continuous optimization with step size")
    
    gd_hidden, gd_sae = tracer.targeted_concept_transition(
        source_text, target_text, layer_idx, max_iter=10000, use_deepfool=False, verbose=False
    )

    analysis(gd_hidden, gd_sae)
    
    print(f"   Results: ||δ|| = {gd_hidden.delta_min:.6f}, boundaries = {gd_hidden.num_boundaries_crossed}")
    
    # Method 2: DeepFool-style (iterative minimal steps)  
    print("\n2. DEEPFOOL-STYLE APPROACH:")
    print("   - Iterative minimal steps toward target")
    print("   - Adaptive step sizing based on distance")
    
    df_hidden, df_sae = tracer.targeted_concept_transition(
        source_text, target_text, layer_idx, max_iter=10000, use_deepfool=True, verbose=False
    )
    
    print(f"   Results: ||δ|| = {df_hidden.delta_min:.6f}, boundaries = {df_hidden.num_boundaries_crossed}")

    analysis(df_hidden, df_sae)
    
    # Comparison
    print(f"\n{'='*60}")
    print("METHOD COMPARISON")
    print(f"{'='*60}")
    print(f"Perturbation ratio (DeepFool/GradDescent): {df_hidden.delta_min/gd_hidden.delta_min:.2f}")
    print(f"Boundary ratio (DeepFool/GradDescent): {df_hidden.num_boundaries_crossed/max(gd_hidden.num_boundaries_crossed,1):.2f}")
    
    if df_hidden.delta_min < gd_hidden.delta_min:
        print("→ DeepFool finds smaller perturbations")
    else:
        print("→ Gradient descent finds smaller perturbations")
    
    return gd_hidden, gd_sae, df_hidden, df_sae

# Run the comparison
print("\nRunning method comparison...")
gd_h, gd_s, df_h, df_s = compare_transition_methods(
    tracer, education_text, technology_text, layer_num
)

# %%
# STEP-BY-STEP DEBUGGING FOR TARGET REACHABILITY

def debug_target_reachability(tracer, source_text, target_text, layer_idx):
    """Debug why the transition is failing to reach targets."""
    
    print("="*80)
    print("DEBUGGING TARGET REACHABILITY")
    print("="*80)
    
    # Step 1: Get representations
    source_hidden = tracer.get_hidden_representation(source_text, layer_idx)
    target_hidden = tracer.get_hidden_representation(target_text, layer_idx)
    
    source_code = tracer.get_sae_code(source_hidden)
    target_code = tracer.get_sae_code(target_hidden)
    
    # Step 2: Analyze the distance between source and target
    hidden_distance = F.mse_loss(source_hidden, target_hidden).item()
    code_distance = F.mse_loss(source_code, target_code).item()
    
    print(f"📊 DISTANCE ANALYSIS:")
    print(f"   Hidden space L2 distance: {hidden_distance:.6f}")
    print(f"   SAE code L2 distance: {code_distance:.6f}")
    print(f"   Hidden norm: {source_hidden.norm().item():.2f} -> {target_hidden.norm().item():.2f}")
    print(f"   Code norm: {source_code.norm().item():.2f} -> {target_code.norm().item():.2f}")
    
    # Step 3: Check how different the concepts actually are
    source_active = (source_code > 0).sum().item()
    target_active = (target_code > 0).sum().item()
    
    source_pattern = (source_code > 0).float()
    target_pattern = (target_code > 0).float()
    overlap = (source_pattern * target_pattern).sum().item()
    
    print(f"\n🎯 CONCEPT DIFFERENCE ANALYSIS:")
    print(f"   Source active features: {source_active}")
    print(f"   Target active features: {target_active}")
    print(f"   Overlapping features: {overlap}")
    print(f"   Features to activate: {target_active - overlap}")
    print(f"   Features to deactivate: {source_active - overlap}")
    
    # Step 4: Check if the distance is reasonable
    relative_distance = hidden_distance / (source_hidden.norm().item() ** 2)
    
    print(f"\n⚠️  FEASIBILITY CHECK:")
    print(f"   Relative distance: {relative_distance:.6f}")
    
    if relative_distance > 0.5:
        print("   🚨 VERY LARGE relative distance - target may be unreachable")
        print("   💡 SOLUTION: Try closer concepts or more iterations")
    elif relative_distance > 0.1:
        print("   ⚠️  LARGE relative distance - may need more iterations")
        print("   💡 SOLUTION: Increase max_iter to 500+")
    else:
        print("   ✅ REASONABLE distance - should be reachable")
    
    if abs(source_active - target_active) > 1000:
        print("   🚨 VERY DIFFERENT activation patterns - SAE transition may be hard")
        print("   💡 SOLUTION: Try more similar concepts")
    elif abs(source_active - target_active) > 100:
        print("   ⚠️  DIFFERENT activation patterns - challenging transition")
        print("   💡 SOLUTION: Adjust loss weights or step size")
    else:
        print("   ✅ SIMILAR activation patterns - SAE transition feasible")
    
    return {
        'hidden_distance': hidden_distance,
        'code_distance': code_distance,
        'relative_distance': relative_distance,
        'source_active': source_active,
        'target_active': target_active,
        'overlap': overlap,
        'diagnosis': 'large_distance' if relative_distance > 0.1 else 'reasonable'
    }

def test_simple_transition(tracer, layer_idx):
    """Test with a simpler, more reachable transition."""
    
    print("\n" + "="*80)
    print("TESTING SIMPLE TRANSITION")
    print("="*80)
    
    # Use very similar texts that should be easier to transition between
    simple_source = "The cat sat on the mat"
    simple_target = "The dog sat on the mat"  # Only one word difference
    
    print(f"Simple source: '{simple_source}'")
    print(f"Simple target: '{simple_target}'")
    
    # Debug this simpler case
    debug_info = debug_target_reachability(tracer, simple_source, simple_target, layer_idx)
    
    if debug_info['diagnosis'] == 'reasonable':
        print("\n🧪 RUNNING TRANSITION ON SIMPLE CASE...")
        
        # Try the transition with more iterations
        hidden_result, sae_result = tracer.targeted_concept_transition(
            simple_source, simple_target, layer_idx, 
            max_iter=500, step_size=0.001, verbose=True
        )
        
        print(f"\n✅ SIMPLE CASE RESULTS:")
        print(f"   Hidden target reached: {hidden_result.target_reached}")
        print(f"   SAE target reached: {sae_result.target_reached}")
        print(f"   Hidden perturbation: {hidden_result.delta_min:.6f}")
        print(f"   SAE perturbation: {sae_result.delta_min:.6f}")
        
        return hidden_result, sae_result
    else:
        print("\n❌ Even simple case is problematic - check implementation")
        return None, None

def improved_targeted_transition(tracer, source_text, target_text, layer_idx):
    """Improved version with better convergence."""
    
    print("\n" + "="*80)
    print("IMPROVED TARGETED TRANSITION")
    print("="*80)
    
    # First debug the target
    debug_info = debug_target_reachability(tracer, source_text, target_text, layer_idx)
    
    if debug_info['relative_distance'] > 0.5:
        print("\n❌ Target too far - using intermediate stepping")
        # Could implement intermediate stepping here
        return None, None
    
    # Adjust parameters based on distance
    if debug_info['relative_distance'] > 0.1:
        max_iter = 1000
        step_size = 0.0001  # Smaller steps
        print(f"\n🔧 USING CONSERVATIVE PARAMETERS: iter={max_iter}, step={step_size}")
    else:
        max_iter = 200
        step_size = 0.01
        print(f"\n🔧 USING STANDARD PARAMETERS: iter={max_iter}, step={step_size}")
    
    # Run with improved parameters
    hidden_result, sae_result = tracer.targeted_concept_transition(
        source_text, target_text, layer_idx,
        max_iter=max_iter, step_size=step_size, verbose=True
    )
    
    return hidden_result, sae_result

# Run debugging on your original texts
print("="*80)
print("STEP 1: DEBUG ORIGINAL CONCEPT PAIR")
print("="*80)

debug_info = debug_target_reachability(tracer, education_text, technology_text, layer_num)

print("\n" + "="*80)
print("STEP 2: TEST SIMPLE CASE")
print("="*80)

simple_results = test_simple_transition(tracer, layer_num)

print("\n" + "="*80)
print("STEP 3: TRY IMPROVED METHOD ON ORIGINAL")
print("="*80)

if debug_info['diagnosis'] != 'large_distance':
    improved_results = improved_targeted_transition(tracer, education_text, technology_text, layer_num)
else:
    print("❌ Original concepts too far apart - need different approach")


# %%
# CORRECTED: Hidden Space Polytope Boundary Analysis
# This fixes the fundamental error in _targeted_transition_hidden

print("=== CORRECTED HIDDEN SPACE POLYTOPE ANALYSIS ===")
print("ISSUE FOUND: Previous implementation tracked SAE boundary crossings")
print("CORRECTION: Now tracking ReLU polytopes boundaries in HIDDEN space")
print()

# def analyze_hidden_polytope_transition(
#     source_hidden: torch.Tensor,
#     target_hidden: torch.Tensor,
#     max_iter: int = 200,
#     step_size: float = 0.01,
#     verbose: bool = True
# ) -> dict:
#     """
#     Analyze transitions between ReLU polytopes in hidden space.
#     A polytope is defined by which dimensions are > 0.

#     Tracks:
#     - Number of distinct polytopes visited (ReLU sign pattern changes)
#     - Hamming distance between ReLU supports
#     - ||delta|| and MSE
#     """
#     delta = torch.zeros_like(source_hidden, requires_grad=True)
#     optimizer = torch.optim.Adam([delta], lr=step_size)

#     target_support = (target_hidden > 0).float()
#     previous_support = (source_hidden > 0).float()

#     visited_polytopes = set()
#     visited_polytopes.add(tuple(previous_support.cpu().numpy().astype(int).tolist()))
#     trajectory = []

#     best_delta = None
#     best_loss = float('inf')

#     for step in range(max_iter):
#         optimizer.zero_grad()

#         current_hidden = source_hidden + delta
#         current_support = (current_hidden > 0).float()
#         support_tuple = tuple(current_support.cpu().numpy().astype(int).tolist())

#         if support_tuple not in visited_polytopes:
#             visited_polytopes.add(support_tuple)

#             if verbose and step % 20 == 0:
#                 differences = current_support - previous_support
#                 activated = (differences > 0).nonzero(as_tuple=True)[0]
#                 deactivated = (differences < 0).nonzero(as_tuple=True)[0]
#                 print(f"Step {step}: New hidden polytope entered (#{len(visited_polytopes)})")
#                 if len(activated) > 0:
#                     print(f"  Activated dims: {activated[:5].tolist()}...")
#                 if len(deactivated) > 0:
#                     print(f"  Deactivated dims: {deactivated[:5].tolist()}...")

#         previous_support = current_support.clone()

#         loss = F.mse_loss(current_hidden, target_hidden)
#         hamming = (current_support != target_support).sum().item()

#         if verbose and step % 50 == 0:
#             print(f"Step {step}: loss = {loss.item():.6f}, ||δ|| = {delta.norm().item():.6f}, "
#                   f"visited polytopes = {len(visited_polytopes)}, hamming = {hamming}")

#         if loss.item() < best_loss:
#             best_loss = loss.item()
#             best_delta = delta.detach().clone()

#         if loss.item() < 1e-6 and hamming == 0:
#             if verbose:
#                 print(f"✅ Target hidden reached at step {step}")
#                 print(f"Hamming distance to target ReLU support: {hamming}")
#             break

#         loss.backward()
#         optimizer.step()
#         trajectory.append(delta.detach().clone())

#     return {
#         'delta_min': best_delta.norm().item(),
#         'perturbation': best_delta,
#         'num_steps': step,
#         'hidden_polytope_boundaries_crossed': len(visited_polytopes) - 1,  # exclude starting polytope
#         'target_reached': best_loss < 1e-3,
#         'final_loss': best_loss,
#         'trajectory': trajectory,
#         'visited_polytope_count': len(visited_polytopes),
#     }

def analyze_hidden_polytope_transition(
    source_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    max_iter: int = 1000,
    support_weight: float = 1.0,
    step_size: float = 0.01,
    verbose: bool = True
) -> dict:
    """
    Analyze transitions between ReLU polytopes in hidden space,
    constraining each optimization step to cross at most 1 boundary.

    Each ReLU polytope is defined by the sign pattern (h > 0).

    Tracks:
    - Number of unique polytopes visited
    - Per-step boundary control (1 flip per step)
    - Hamming to target ReLU support
    """
    delta = torch.zeros_like(source_hidden, requires_grad=True)
    target_support = (target_hidden > 0).float()
    previous_support = (source_hidden > 0).float()

    visited_polytopes = set()
    visited_polytopes.add(tuple(previous_support.cpu().numpy().astype(int).tolist()))
    trajectory = []

    best_delta = None
    best_loss = float('inf')

    step = 0
    # for step in range(max_iter):
    while True:
        step += 1
        current_hidden = source_hidden + delta
        current_support = (current_hidden > 0).float()
        support_tuple = tuple(current_support.cpu().numpy().astype(int).tolist())

        # Log new polytope entry
        if support_tuple not in visited_polytopes:
            visited_polytopes.add(support_tuple)

            if verbose and step % 20 == 0:
                differences = current_support - previous_support
                activated = (differences > 0).nonzero(as_tuple=True)[0]
                deactivated = (differences < 0).nonzero(as_tuple=True)[0]
                print(f"Step {step}: New hidden polytope entered (#{len(visited_polytopes)})")
                if len(activated) > 0:
                    print(f"  Activated dims: {activated[:5].tolist()}...")
                if len(deactivated) > 0:
                    print(f"  Deactivated dims: {deactivated[:5].tolist()}...")

        previous_support = current_support.clone()

        # Compute loss and gradients
        loss = F.mse_loss(current_hidden, target_hidden)
        if support_weight > 0:
            support_loss = F.l1_loss(current_support, target_support)
            loss += support_weight * support_loss
        hamming = (current_support != target_support).sum().item()
        loss.backward()

        with torch.no_grad():
            grad = delta.grad
            proposed_step = -step_size * grad

            # Estimate new support if we take full step
            next_code = source_hidden + delta + proposed_step
            next_support = (next_code > 0).float()
            flip_mask = (next_support != current_support).float()

            # If more than one bit would flip, scale down
            flip_count = flip_mask.sum().item()
            if flip_count > 1:
                # Shrink the step to only allow the most sensitive unit to flip
                signs = current_hidden.sign()
                distances = current_hidden.abs() / (proposed_step.abs() + 1e-8)
                distances[proposed_step == 0] = float('inf')
                distances[signs != -proposed_step.sign()] = float('inf')  # only count approaching boundaries
                best_idx = distances.argmin()
                scale = (current_hidden[best_idx].abs() / proposed_step[best_idx].abs()).item()
                proposed_step = proposed_step * (scale + 1e-4)  # small nudge over boundary

            delta.add_(proposed_step)
            delta.grad.zero_()

        trajectory.append(delta.detach().clone())

        if verbose and step % 50 == 0:
            print(f"Step {step}: loss = {loss.item():.6f}, ||δ|| = {delta.norm().item():.6f}, "
                  f"visited polytopes = {len(visited_polytopes)}, hamming = {hamming}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta.detach().clone()

        if loss.item() < 1e-6 and hamming == 0:
            if verbose:
                print(f"✅ Target hidden reached at step {step}")
                print(f"Hamming distance to target ReLU support: {hamming}")
            break

    return {
        'delta_min': best_delta.norm().item(),
        'perturbation': best_delta,
        'num_steps': step,
        'hidden_polytope_boundaries_crossed': len(visited_polytopes) - 1,
        'target_reached': best_loss < 1e-3,
        'final_loss': best_loss,
        'trajectory': trajectory,
        'visited_polytope_count': len(visited_polytopes),
    }



# Test with single concept pair
source_text = education_text
target_text = technology_text

print(f"Source: '{source_text[:-50]}...'")
print(f"Target: '{target_text[:-50]}...'")
print()

# Get hidden representations
source_hidden = tracer.get_hidden_representation(source_text, layer_num)
target_hidden = tracer.get_hidden_representation(target_text, layer_num)

print(f"Source hidden shape: {source_hidden.shape}")
print(f"Target hidden shape: {target_hidden.shape}")

# Original polytope analysis
source_polytope = (source_hidden > 0).float()
target_polytope = (target_hidden > 0).float()
polytope_hamming = (source_polytope != target_polytope).sum().item()

print(f"Source polytope active dims: {source_polytope.sum().item()}/{len(source_polytope)}")
print(f"Target polytope active dims: {target_polytope.sum().item()}/{len(target_polytope)}")
print(f"Polytope Hamming distance: {polytope_hamming}")
print()

# Run corrected analysis
print("Running CORRECTED hidden polytope transition analysis...")
result = analyze_hidden_polytope_transition(
    source_hidden, target_hidden, 
    max_iter=10000, 
    verbose=True
)

print(f"\n=== CORRECTED RESULTS ===")
print(f"Minimal perturbation: {result['delta_min']:.6f}")
print(f"Hidden polytope boundaries crossed: {result['hidden_polytope_boundaries_crossed']}")
print(f"Target reached: {result['target_reached']}")
print(f"Final loss: {result['final_loss']:.6f}")
print(f"Optimization steps: {result['num_steps']}")

# # Compare with SAE boundary counting for perspective
# original_sae_code = tracer.get_sae_code(source_hidden)
# target_sae_code = tracer.get_sae_code(target_hidden)
# final_sae_code = tracer.get_sae_code(source_hidden + result['perturbation'])
# sae_changes = ((original_sae_code > 0).float() != (final_sae_code > 0).float()).sum().item()

# #print unique values of original_sae_code 



# print(f"\nFor comparison:")
# print(f"SAE feature changes during this transition: {sae_changes}")
# print(f"Hidden polytope boundaries crossed: {result['hidden_polytope_boundaries_crossed']}")
# print(f"Ratio (Hidden boundaries / SAE changes): {result['hidden_polytope_boundaries_crossed'] / max(sae_changes, 1):.2f}")

# print("\n" + "="*60)
# print("KEY INSIGHT:")
# print("Now we're properly measuring transitions between")
# print("ReLU polytopes in the HIDDEN space (h_i > 0),")
# print("not SAE feature boundaries!")
# print("="*60)

# # %%
# def analyze_sae_code_polytope_transition(
#     sparse_code_start: torch.Tensor,
#     sparse_code_target: torch.Tensor,
#     max_iter: int = 200,
#     step_size: float = 0.01,
#     verbose: bool = True
# ) -> dict:
#     """
#     Track transitions between polytope regions in SAE sparse code space.
#     Each polytope is defined by the support set: (z > 0)

#     Inputs:
#         - sparse_code_start: initial SAE code (after jump_relu)
#         - sparse_code_target: target SAE code
#     """
#     # Initialize delta in SAE space
#     delta = torch.zeros_like(sparse_code_start, requires_grad=True)
#     optimizer = torch.optim.Adam([delta], lr=step_size)

#     original_support = (sparse_code_start > 0).float()
#     previous_support = original_support.clone()

#     boundary_count = 0
#     trajectory = []
    
#     best_delta = None
#     best_loss = float('inf')

#     for step in range(max_iter):
#         optimizer.zero_grad()

#         current_code = sparse_code_start + delta
#         current_support = (current_code > 0).float()

#         # Check polytope transition (i.e., support set changed)
#         if not torch.equal(current_support, previous_support):
#             boundary_count += 1
#             flip = current_support - previous_support
#             activated = (flip > 0).nonzero(as_tuple=True)[0]
#             deactivated = (flip < 0).nonzero(as_tuple=True)[0]

#             if verbose and step % 20 == 0:
#                 print(f"Step {step}: SAE code polytope boundary #{boundary_count}")
#                 if len(activated) > 0:
#                     print(f"  Activated features: {activated[:5].tolist()}...")
#                 if len(deactivated) > 0:
#                     print(f"  Deactivated features: {deactivated[:5].tolist()}...")

#         previous_support = current_support.clone()

#         # Objective: get close to target code
#         loss = F.mse_loss(current_code, sparse_code_target)

#         if loss.item() < best_loss:
#             best_loss = loss.item()
#             best_delta = delta.detach().clone()

#         if loss.item() < 1e-6:
#             if verbose:
#                 print(f"Target code reached at step {step}")
#             break

#         loss.backward()
#         optimizer.step()
#         trajectory.append(delta.detach().clone())

#         if verbose and step % 50 == 0:
#             print(f"Step {step}: loss = {loss.item():.6f}, ||δ|| = {delta.norm().item():.6f}, boundaries = {boundary_count}")

#     return {
#         'delta_min': best_delta.norm().item(),
#         'perturbation': best_delta,
#         'num_steps': step,
#         'sae_code_polytope_boundaries_crossed': boundary_count,
#         'target_reached': best_loss < 1e-3,
#         'final_loss': best_loss,
#         'trajectory': trajectory
#     }

# # Test SAE polytope transition
# print("\n" + "="*80)
# print("ANALYZING SAE CODE POLYTOPE TRANSITION")
# print("="*80)   

# # Get initial and target SAE codes
# source_sae_code = tracer.get_sae_code(source_hidden)
# target_sae_code = tracer.get_sae_code(target_hidden)   
# print(f"Source SAE code shape: {source_sae_code.shape}")
# print(f"Target SAE code shape: {target_sae_code.shape}")
# # Original polytope analysis
# source_sae_polytope = (source_sae_code > 0).float()
# target_sae_polytope = (target_sae_code > 0).float()
# sae_polytope_hamming = (source_sae_polytope != target_sae_polytope).sum().item()
# print(f"Source SAE polytope active dims: {source_sae_polytope.sum().item()}/{len(source_sae_polytope)}")
# print(f"Target SAE polytope active dims: {target_sae_polytope.sum().item()}/{len(target_sae_polytope)}")
# print(f"SAE polytope Hamming distance: {sae_polytope_hamming}")
# # Run SAE polytope transition analysis
# print("Running SAE code polytope transition analysis...")  
# result_sae = analyze_sae_code_polytope_transition(
#     sparse_code_start=source_sae_code, 
#     sparse_code_target=target_sae_code, 
#     max_iter=10000, 
#     verbose=True
# )
# print(f"\n=== SAE CODE POLYTOPE TRANSITION RESULTS ===")
# print(f"Minimal perturbation: {result_sae['delta_min']:.6f}")
# print(f"SAE code polytope boundaries crossed: {result_sae['sae_code_polytope_boundaries_crossed']}")   
# print(f"Target reached: {result_sae['target_reached']}")
# print(f"Final loss: {result_sae['final_loss']:.6f}")
# print(f"Optimization steps: {result_sae['num_steps']}")

# %%
# new version 
def support_jaccard(code1, code2):
    s1 = (code1 > 0).float()
    s2 = (code2 > 0).float()
    intersection = (s1 * s2).sum()
    union = (s1 + s2).clamp(max=1).sum()
    return (intersection / union).item() if union.item() > 0 else 1.0

def analyze_sae_code_polytope_transition(
    sparse_code_start: torch.Tensor,
    sparse_code_target: torch.Tensor,
    max_iter: int = 1000,
    step_size: float = 0.01,
    support_weight: float = 0.0,
    verbose: bool = True
) -> dict:
    """
    Analyze transitions between polytopes in SAE sparse code space.

    Adds hybrid loss and metrics:
    - MSE loss + optional support alignment loss
    - Hamming distance and Jaccard score
    """
    delta = torch.zeros_like(sparse_code_start, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=step_size)

    original_support = (sparse_code_start > 0).float()
    target_support = (sparse_code_target > 0).float()
    previous_support = original_support.clone()

    boundary_count = 0
    feature_flip_count = 0
    flip_events = []
    trajectory = []
    metrics = []

    visited_supports = set()
    visited_supports.add(tuple(original_support.cpu().numpy().astype(int).tolist()))

    best_delta = None
    best_loss = float('inf')

    for step in range(max_iter):
        optimizer.zero_grad()
        current_code = sparse_code_start + delta
        current_support = (current_code > 0).float()

        # Polytope support tracking
        current_support_tuple = tuple(current_support.cpu().numpy().astype(int).tolist())
        if current_support_tuple not in visited_supports:
            visited_supports.add(current_support_tuple)
            boundary_count += 1

        # Count flips
        flip = (current_support != previous_support).float()
        flipped_indices = (flip == 1).nonzero(as_tuple=True)[0].tolist()
        flip_events.extend(flipped_indices)
        feature_flip_count += len(flipped_indices)

        # Alignment metrics
        mse = F.mse_loss(current_code, sparse_code_target).item()
        hamming = (current_support != target_support).sum().item()
        jaccard = support_jaccard(current_code, sparse_code_target)

        metrics.append({
            'step': step,
            'mse': mse,
            'hamming': hamming,
            'jaccard': jaccard,
            'delta_norm': delta.norm().item()
        })

        if verbose and step % 100 == 0:
            print(f"Step {step}: MSE = {mse:.6f}, ||δ|| = {delta.norm().item():.2f}, "
                  f"Hamming = {hamming}, Jaccard = {jaccard:.4f}, flips = {feature_flip_count}, "
                  f"boundaries = {boundary_count}")

        previous_support = current_support.clone()

        # Hybrid loss: MSE + λ * support alignment loss (optional)
        loss = F.mse_loss(current_code, sparse_code_target)
        if support_weight > 0:
            support_loss = F.l1_loss(current_support, target_support)
            loss += support_weight * support_loss

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta.detach().clone()

        if mse < 1e-6 and hamming == 0:
            if verbose:
                print(f"✅ Target SAE code reached at step {step}")
                print(f"Hamming distance to target support: {hamming}")
            break

        loss.backward()
        optimizer.step()
        trajectory.append(delta.detach().clone())

    if verbose:
        print("\n=== SAE CODE POLYTOPE TRANSITION RESULTS ===")
        print(f"Minimal perturbation: {best_delta.norm().item():.6f}")
        print(f"SAE polytope boundaries crossed: {boundary_count}")
        print(f"Total individual feature flips: {feature_flip_count}")
        print(f"Unique polytopes visited: {len(visited_supports)}")
        print(f"Target reached: {best_loss < 1e-3}")
        print(f"Final loss: {best_loss:.6f}")
        print(f"Optimization steps: {step}")

    return {
        'delta_min': best_delta.norm().item(),
        'perturbation': best_delta,
        'num_steps': step,
        'sae_code_polytope_boundaries_crossed': boundary_count,
        'total_feature_flips': feature_flip_count,
        'unique_support_patterns': len(visited_supports),
        'target_reached': best_loss < 1e-3,
        'final_loss': best_loss,
        'trajectory': trajectory,
        'flip_events': flip_events,
        'metrics': metrics
    }



# Test SAE polytope transition
print("\n" + "="*80)
print("ANALYZING SAE CODE POLYTOPE TRANSITION")
print("="*80)   

# Get initial and target SAE codes
source_sae_code = tracer.get_sae_code(source_hidden)
target_sae_code = tracer.get_sae_code(target_hidden)   
print(f"Source SAE code shape: {source_sae_code.shape}")
print(f"Target SAE code shape: {target_sae_code.shape}")
# Original polytope analysis
source_sae_polytope = (source_sae_code > 0).float()
target_sae_polytope = (target_sae_code > 0).float()
sae_polytope_hamming = (source_sae_polytope != target_sae_polytope).sum().item()
print(f"Source SAE polytope active dims: {source_sae_polytope.sum().item()}/{len(source_sae_polytope)}")
print(f"Target SAE polytope active dims: {target_sae_polytope.sum().item()}/{len(target_sae_polytope)}")
print(f"SAE polytope Hamming distance: {sae_polytope_hamming}")
# Run SAE polytope transition analysis

results = analyze_sae_code_polytope_transition(
    sparse_code_start=source_sae_code,
    sparse_code_target=target_sae_code,
    support_weight=10,
    max_iter=20000,
    verbose=True
)


print(f"\n=== SAE CODE POLYTOPE TRANSITION RESULTS ===")
print(f"Minimal perturbation: {result_sae['delta_min']:.6f}")
print(f"SAE code polytope boundaries crossed: {result_sae['sae_code_polytope_boundaries_crossed']}")   
print(f"Target reached: {result_sae['target_reached']}")
print(f"Final loss: {result_sae['final_loss']:.6f}")
print(f"Optimization steps: {result_sae['num_steps']}")

# %%
import torch
import torch.nn.functional as F

def greedy_support_path(source_code, target_code, max_steps=1000):
    """
    Greedy bit-flip minimization of support difference.
    At each step, flip the unit whose activation is closest to 0 and reduces Hamming to target.
    
    Returns:
        path_supports: list of support vectors (bitmasks)
        flip_sequence: list of indices flipped
    """
    source = source_code.clone().detach()
    target = target_code.clone().detach()
    current = (source > 0).float()
    target_support = (target > 0).float()
    
    path_supports = [current.clone()]
    flip_sequence = []
    visited = set()
    visited.add(tuple(current.cpu().int().tolist()))

    for step in range(max_steps):
        diff = (current != target_support).float()
        if diff.sum().item() == 0:
            break

        candidates = torch.nonzero(diff).squeeze(1)
        if len(candidates) == 0:
            break

        # Among differing bits, choose one closest to 0 (easiest to flip)
        activations = source[candidates]
        distances = activations.abs()
        best_idx = candidates[distances.argmin()]

        # Flip the support
        current[best_idx] = 1.0 - current[best_idx]
        flip_sequence.append(best_idx.item())
        
        current_tuple = tuple(current.cpu().int().tolist())
        if current_tuple in visited:
            break
        visited.add(current_tuple)
        path_supports.append(current.clone())

    return path_supports, flip_sequence, len(path_supports) - 1

## apply on 

# %%
# Test with single concept pair
source_text = education_text
target_text = technology_text

print(f"Source: '{source_text[:-50]}...'")
print(f"Target: '{target_text[:-50]}...'")
print()

# Get hidden representations
source_hidden = tracer.get_hidden_representation(source_text, layer_num)
target_hidden = tracer.get_hidden_representation(target_text, layer_num)

# greedy support path on source and target hidden representations
path, flips, num_transitions = greedy_support_path(source_hidden, target_hidden, max_steps=10000)
print(f"Greedy support path found {num_transitions} transitions with {len(flips)} flips")

# apply on sae code
source_sae_code = tracer.get_sae_code(source_hidden)
target_sae_code = tracer.get_sae_code(target_hidden)
sae_path, sae_flips, sae_num_transitions = greedy_support_path(source_sae_code, target_sae_code, max_steps=10000)
print(f"Greedy SAE support path found {sae_num_transitions} transitions with {len(sae_flips)} flips")

# %%
