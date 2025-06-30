import argparse
from suffix_attack import *
# from replacement_attack import *

import os 
os.environ["https_proxy"] = "http://xen03.iitd.ac.in:3128"
os.environ["http_proxy"] = "http://xen03.iitd.ac.in:3128"

def parse_args():
    parser = argparse.ArgumentParser(description="Run adversarial SAE attack")
    parser.add_argument("--targeted", type=bool, default= True, help="Enable targeted attack")
    parser.add_argument("--level", choices=["population"], default="population", help="Level of analysis")
    parser.add_argument("--mode", choices=["suffix"], default="suffix", help="Perturbation mode")
    # parser.add_argument("--activate", action="store_true", help="Only relevant if level=individual. Whether to activate (True) or deactivate (False) neurons")
    parser.add_argument("--sample_idx", type=int, default=20)
    parser.add_argument("--layer_num", type=int, default=20) # 30 for gemma2-9b, 20 for gemma2-2b
    parser.add_argument("--num_latents", type=int, default=10) # only used in individual
    parser.add_argument("--suffix_len", type=int, default=3) # population level = 3, individual level = 1
    parser.add_argument("--batch_size", type=int, default=400) # 100 for individual level or replacement mode; (2/3) * (m * suffix_len) for population level
    parser.add_argument("--num_iters", type=int, default=50) # 50 for targeted population; 20 for untargeted population; 10 for all individual level
    parser.add_argument("--m", type=int, default=200) # 200 for gemma population suffix; 300 otherwise
    parser.add_argument("--k", type=int, default=172) # 192 for llama; 170 for gemma; not important
    parser.add_argument("--data_file", type=str, default="./two_class_generated.csv")
    # parser.add_argument("--base_dir", type=str, default="/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/")
    parser.add_argument("--model_type", type=str, choices=['gemma2-2b'], default="gemma2-2b", help="Model architecture")
    parser.add_argument("--log", type=bool, default= True, help="Log results")
    
    return parser.parse_args()

def launch():
    args = parse_args()

    d_overlap = run_population_suffix_attack(args)
    print(f"Population attack overlap change = {d_overlap:.4f}")


        
if __name__ == "__main__":
    launch()
