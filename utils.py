import sys
# sys.path.append('../sae')
from sae import Sae
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import argparse
from torch import nn, Tensor
from huggingface_hub import hf_hub_download, login
import transformers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/home/anwoy/phuhoang/sae_robustness/rep/sae_models/" # for load sae model
CACHE_DIR = "/home/anwoy/phuhoang/sae_robustness/rep/models/" # for load model
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def count_common(x, y):
    return sum(1 for elem in x if elem in y)

def get_overlap(s_batch, s_ref):
    """
    Returns fraction of elements in s_ref that are also in s_batch.

    - s_batch: (k,) or (B, k)
    - s_ref: (k,)
    """
    if s_batch.dim() == 1:
        # Set intersection for single vector
        return (torch.isin(s_ref, s_batch).sum().float() / s_ref.numel())

    elif s_batch.dim() == 2:
        # Batched case
        # For each row in s_batch, check overlap with s_ref
        ref_set = set(s_ref.tolist())
        overlaps = []
        for row in s_batch:
            overlap = len(set(row.tolist()) & ref_set) / len(ref_set)
            overlaps.append(overlap)
        return torch.tensor(overlaps, device=s_batch.device)

    else:
        raise ValueError("s_batch must be 1D or 2D tensor")

def jump_relu(x, theta):
    return x * (x > theta).float()

def load_model_and_sae(model_type, layer_num, device=DEVICE):
    """ Load the specified model and SAE.
    Args:
        model_type (str): Type of the model, either "llama3-8b
                        or "gemma2-9b".
        layer_num (int): Layer number to load the SAE from.
        device (str): Device to load the model onto, default is "cuda:0".
    Returns:
        model (AutoModelForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        sae (Sae): The loaded SAE.
    """
    if model_type == "llama3-8b":
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=CACHE_DIR)
        # model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=CACHE_DIR).to(device)
        # sae = Sae.load_from_disk(BASE_DIR + f"layers.{layer_num}").to(DEVICE)
        raise NotImplementedError("Llama3-8b is not implemented yet.") # use code bove
    elif model_type == "gemma2-9b":
        # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", cache_dir=CACHE_DIR)
        # model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", cache_dir=CACHE_DIR).to(DEVICE)
        # assert layer_num == 30
        # path_to_params = hf_hub_download(
        #     repo_id="google/gemma-scope-9b-pt-res",
        #     filename="layer_30/width_131k/average_l0_170/params.npz",
        #     force_download=False,
        #     cache_dir = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/gemma2-9b-sae"
        # )
        # params = np.load(path_to_params)
        # sae = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
        raise NotImplementedError("Gemma2-9b is not implemented yet.") # use code bove
    elif model_type == "gemma2-2b":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", cache_dir=CACHE_DIR).to(DEVICE)
        assert layer_num == 20
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename="layer_20/width_65k/average_l0_221/params.npz",
            force_download=False,
            cache_dir = os.path.join(BASE_DIR, "gemma2-2b-sae", "layer_20", "width_65k", "average_l0_221")
        )
        params = np.load(path_to_params)
        sae = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    return model, tokenizer, sae

def extract_sae_features(h, sae, model_type, k=None):
    """ Extracts SAE features from the hidden states.
    Args:
        h (Tensor): Hidden states from the model, shape (B, d).
        sae (Sae): The SAE model.
        model_type (str): Type of the model, either "llama3-8b
                        or "gemma2-9b".
        k (int, optional): Number of top activations to return. If None,
                        returns all activations above the threshold.
    Returns:
        z (Tensor): Pre-activation values, shape (B, d).
        s (Tensor): Indices of the top activations, shape (B, k)
        s_acts (Tensor): Values of the top activations, shape (B, k)
    """
    is_batch = (len(h.shape) == 2)
    if model_type == "llama3-8b":
        z = sae.pre_acts(h)
        s = sae.encode(h).top_indices
        s_acts = sae.encode(h).top_acts
    elif model_type == "gemma2-9b" or model_type == "gemma2-2b":
        z = jump_relu(h @ sae['W_enc'] + sae['b_enc'], sae['threshold'])
        sorted_acts, indices = torch.sort(z, dim=-1, descending=True)
        if is_batch:
            s = indices[:, :k]
            s_acts = sorted_acts[:, :k]
        else:
            if k is None:
                mask = sorted_acts > 0
                s = indices[mask]
                s_acts = sorted_acts[mask]
            else:
                s = indices[:k]
                s_acts = sorted_acts[:k]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return z, s, s_acts