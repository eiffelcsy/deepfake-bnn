#!/usr/bin/env python
import argparse
import torch
from collections import OrderedDict

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="The path to the model checkpoint.",
        required=True,
    )
    args = parser.parse_args()
    return args

def analyze_checkpoint(checkpoint_path):
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Extract state dict
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Analyze structure
    print("\nCheckpoint Structure:")
    print("-" * 50)
    
    # Group keys by prefix
    key_groups = {}
    for key in state_dict.keys():
        prefix = key.split('.')[0]
        if prefix not in key_groups:
            key_groups[prefix] = []
        key_groups[prefix].append(key)
    
    # Print structure by group
    for prefix, keys in key_groups.items():
        print(f"\n{prefix}:")
        for key in sorted(keys):
            tensor = state_dict[key]
            print(f"  {key}: Shape {tensor.shape}, Type {tensor.dtype}")
    
    # Find all fusion_network keys
    fusion_keys = [k for k in state_dict.keys() if 'fusion_network' in k]
    
    print("\nDetailed fusion_network structure:")
    print("-" * 50)
    for key in sorted(fusion_keys):
        tensor = state_dict[key]
        print(f"{key}: Shape {tensor.shape}, Type {tensor.dtype}")
    
    # Check for normalization layers
    norm_keys = [k for k in state_dict.keys() if 'norm' in k]
    
    print("\nNormalization layers:")
    print("-" * 50)
    for key in sorted(norm_keys):
        tensor = state_dict[key]
        print(f"{key}: Shape {tensor.shape}, Type {tensor.dtype}")

if __name__ == "__main__":
    args = args_func()
    analyze_checkpoint(args.checkpoint) 