import sys
# sys.path.append('../sae')
from sae import Sae
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import time

def run_population_suffix_attack(args):
    # Load the data
    df = pd.read_csv(args.data_file)
    # Load the model and SAE
    model, tokenizer, sae = load_model_and_sae(args.model_type, args.layer_num)
    
    # Set up logging if required
    original_stdout = None
    if args.log:
        log_file_path = f"./results/{args.model_type}/layer-{args.layer_num}/{args.data_file}/{args.targeted}-population-suffix-{args.sample_idx}.txt"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        original_stdout = sys.stdout
        sys.stdout = open(log_file_path, "w")

    # Process the first sample
    x1_raw_text = df.iloc[args.sample_idx]['x1'][:-1]
    print(f"x1: {x1_raw_text}")
    # Tokenize and prepare the input
    ## x1_raw is the raw input, x1_raw_processed is the processed input with additional context
    x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE) 
    ## x1_raw_processed is the processed input with additional context 
    x1_raw_processed = tokenizer(x1_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
    ## h1_raw is the hidden state of the raw input
    h1_raw = model(x1_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1].detach()

    # Generate an initial input sequence
    ## x1_init is the generated input sequence
    x1_init = model.generate(x1_raw, max_length=x1_raw.shape[-1] + args.suffix_len, do_sample=False, num_return_sequences=1)
    ## x1_init_text is the decoded text of the generated input sequence
    x1_init_text = tokenizer.decode(x1_init[0], skip_special_tokens=True)
    print(f"x1 init: {x1_init_text}")
    x1_init_processed = tokenizer(x1_init_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
    # Get the hidden state of the initial input sequence
    h1_init = model(x1_init_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1].detach()

    # Extract SAE features from the hidden states
    if args.targeted:
        print("\nTargeted attack!!!\n")
        # Process the second sample
        x2_text = df.iloc[args.sample_idx]['x2'][:-1]
        print(f"x2: {x2_text}")
        # Tokenize and prepare the input
        x2 = tokenizer(x2_text, return_tensors="pt")['input_ids'].to(DEVICE)
        x2_processed = tokenizer(x2_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
        # Get the hidden state of the second input sequence
        h2 = model(x2_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1].detach()
        # Extract SAE features for the second input sequence
        ## z2, s2, s2_acts are the SAE features for the second input sequence
        z2, s2, s2_acts = extract_sae_features(h2, sae, args.model_type, k=None)
        ## k is the number of neurons in the second input sequence
        k = len(s2)
        # Extract SAE features for the raw and initial input sequences
        ## z1_raw, s1_raw, s1_acts_raw are the SAE features
        z1_raw, s1_raw, s1_acts_raw = extract_sae_features(h1_raw, sae, args.model_type, k)
        ## z1_init, s1_init, s1_acts_init are the SAE features for the initial input sequence
        z1_init, s1_init, s1_acts_init = extract_sae_features(h1_init, sae, args.model_type, k)
        # Calculate the initial overlap between the raw and initial input sequences
        initial_overlap = get_overlap(s1_raw, s2).item()
        print(f"Initial s1_raw s2 overlap = {initial_overlap}")
        print(f"Initial s1_init s2 overlap = {get_overlap(s1_init, s2)}")
    else:
        z1_raw, s1_raw, s1_acts_raw = extract_sae_features(h1_raw, sae, args.model_type, k=None)
        k = len(s1_raw)
        z1_init, s1_init, s1_acts_init = extract_sae_features(h1_init, sae, args.model_type, k)
        initial_overlap = 1.0
        print(f"Initial s1_raw s1_init overlap = {get_overlap(s1_raw, s1_init)}")

    best_overlap = 0.0 if args.targeted else 1.0
    x1 = x1_init_processed.clone()
    losses = []
    overlaps = []
    # number of iterations for the attack
    for i in range(args.num_iters):
        # Get the hidden states for the current input sequence
        with torch.no_grad():
            embeddings = model.get_input_embeddings()(x1) 
        # Detach and clone the embeddings, then require gradients for backpropagation
        embeddings = embeddings.detach().clone().requires_grad_(True)
        # Get the hidden states from the model
        ## h1 is the hidden state of the current input sequence
        ## z1, s1, s1_acts are the SAE features for the current input sequence
        h1 = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
        z1, s1, s1_acts = extract_sae_features(h1, sae, args.model_type, k)
        # Calculate the loss based on the SAE features
        if args.targeted:
            # Calculate the cosine similarity loss for the targeted attack
            # why? to maximize the similarity between the current and target features
            loss = F.cosine_similarity(z1, z2, dim=0)
        else:
            # Calculate the cosine similarity loss for the untargeted attack
            # why? to minimize the similarity between the current and raw features
            loss = -F.cosine_similarity(z1, z1_raw, dim=0)
        # Backpropagate to get the gradients
        gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=False)[0]
        # Free up memory
        del h1, z1, s1, s1_acts
        torch.cuda.empty_cache()
        # Calculate the dot product of the gradients with the model's input embeddings
        # This helps in identifying the most influential tokens for the attack
        # We set the eos token's dot product to -inf to avoid selecting it
        # as a candidate for the attack
        # This is because the eos token is not a valid token for the attack
        # and we want to avoid selecting it as a candidate for the attack
        dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)
        dot_prod[:, tokenizer.eos_token_id] = -float('inf')
        # Select the top m tokens based on the dot product
        # These tokens are the most influential for the attack
        # We will use these tokens to perturb the input sequence
        # top_m_adv contains the indices of the top m tokens
        # We only consider the last suffix_len tokens of the input sequence
        # This is because we are only perturbing the last suffix_len tokens
        # of the input sequence
        top_m_adv = (torch.topk(dot_prod, args.m).indices)[x1_init.shape[-1] - args.suffix_len:x1_init.shape[-1]]

        # Create a batch of inputs by repeating the current input sequence
        # This is done to parallelize the attack and speed up the process
        # We will perturb each input in the batch independently
        # x1_batch is the batch of inputs
        # We repeat the current input sequence args.batch_size times
        # and clone it to avoid modifying the original input sequence
        # rand_token_idx is a random index for each input in the batch
        # rand_top_m_idx is a random index for each input in the batch
        # batch_indices is a tensor containing the indices of the batch
        # We will use these indices to select the tokens to perturb
        # We will perturb the last suffix_len tokens of the input sequence
        # by replacing them with the top m tokens selected earlier
        # This allows us to explore different perturbations in parallel
        # x1_batch is the batch of inputs with the perturbations applied
        # We replace the last suffix_len tokens of the input sequence
        # with the top m tokens selected earlier
        # This allows us to explore different perturbations in parallel
        # and speed up the attack process
        x1_batch = x1.repeat(args.batch_size, 1).clone()
        rand_token_idx = torch.randint(0, args.suffix_len, (args.batch_size,))
        rand_top_m_idx = torch.randint(0, args.m, (args.batch_size,))
        batch_indices = torch.arange(args.batch_size)
        x1_batch[:, x1_init.shape[-1] - args.suffix_len:x1_init.shape[-1]][batch_indices, rand_token_idx] = top_m_adv[0, rand_top_m_idx]
        
        # Get the hidden states for the batch of inputs
        # This is done to extract the SAE features for the perturbed inputs
        # We will use these features to calculate the overlap with the target or raw features
        # h1_batch contains the hidden states for the batch of inputs
        # z1_batch, s1_batch, s1_acts_batch are the SAE features
        # for the batch of inputs
        # We will use these features to calculate the overlap with the target or raw features
        # depending on whether the attack is targeted or untargeted
        with torch.no_grad():
            h1_batch = model(x1_batch, output_hidden_states=True).hidden_states[args.layer_num + 1]
            z1_batch, s1_batch, s1_acts_batch = extract_sae_features(h1_batch[:, -1, :], sae, args.model_type, k)
        
        # Calculate the overlap between the batch of inputs and the target or raw features
        # depending on whether the attack is targeted or untargeted
        # If the attack is targeted, we want to maximize the overlap with the target features
        # If the attack is untargeted, we want to minimize the overlap with the raw features
        # overlap_batch contains the overlap ratios for each input in the batch
        # We use the get_overlap function to calculate the overlap ratios
        if args.targeted:
            overlap_batch = get_overlap(s1_batch, s2)
            best_idx = torch.argmax(overlap_batch)
        else:
            overlap_batch = get_overlap(s1_batch, s1_raw)
            best_idx = torch.argmin(overlap_batch)
        # Get the best input from the batch based on the overlap ratio
        # This is the input that has the highest overlap with the target or raw features
        # depending on whether the attack is targeted or untargeted
        # x1 is the best input from the batch
        # We will use this input for the next iteration of the attack
        # We also update the best overlap ratio if the current input has a better overlap ratio
        # than the previous best input
        # best_overlap is the best overlap ratio so far
        # best_x1 is the best input so far
        # We will use this input for the next iteration of the attack
        # x1 is the best input from the batch
        # We will use this input for the next iteration of the attack
        # We also update the best overlap ratio if the current input has a better overlap ratio
        # than the previous best input
        # best_overlap is the best overlap ratio so far
        # best_x1 is the best input so far
        # We will use this input for the next iteration of the attack
        # x1 is the best input from the batch
        x1 = x1_batch[best_idx].unsqueeze(0)
        if (args.targeted and overlap_batch[best_idx] > best_overlap) or (not args.targeted and overlap_batch[best_idx] < best_overlap):
            best_overlap = overlap_batch[best_idx].item()
            best_x1 = x1[0][:x1_init.shape[-1]]
        if args.targeted:
            current_loss = F.cosine_similarity(z1_batch[best_idx], z2, dim=0).item()
        else:
            current_loss = F.cosine_similarity(z1_batch[best_idx], z1_raw, dim=0).item()

        # Decode the best input to text
        # This is the input that has the highest overlap with the target or raw features
        # depending on whether the attack is targeted or untargeted
        # We will use this text to evaluate the success of the attack
        # x1_text is the decoded text of the best input
        x1_text = tokenizer.decode(best_x1, skip_special_tokens=True)
        losses.append(current_loss)
        overlaps.append(best_overlap)
        
        print(f"Iteration {i+1} loss = {current_loss}")
        print(f"Iteration {i+1} best overlap ratio = {best_overlap}")
        print(f"Iteration {i+1} input text: {x1_text}")
        print("--------------------")

        del h1_batch, z1_batch, s1_batch, s1_acts_batch
        torch.cuda.empty_cache()

    print(f"Final loss = {current_loss}")
    print(f"Final overlap = {best_overlap}")
    print(f"Final x1: {x1_text}")
    
    # Restore original stdout if logging was enabled
    if args.log and original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout

    return (best_overlap - initial_overlap) / initial_overlap