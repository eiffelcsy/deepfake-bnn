#!/usr/bin/env python
import argparse
import os
import gc
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
import lightning as L

from lib.util import load_config
from enhanced_model import DeepfakeVideoClassifier, DeepfakeVideoDataset

# Import our custom EvalDataset
from eval_dataset import EvalDataset

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="The path to the config.",
        default="./configs/eval_enhanced_model.cfg",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="The path to the model checkpoint.",
        required=True,
    )
    parser.add_argument(
        "--tfrecord",
        type=str,
        help="Path to the TFRecord file with pre-processed features.",
        default="./eval_features.tfrecord",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    args = args_func()

    # Load configs
    cfg = load_config(args.cfg)
    print("Loaded configuration:")
    for section in cfg:
        print(f"[{section}]")
        for key, value in cfg[section].items():
            print(f"  {key}: {value}")

    # Set random seeds for reproducibility
    torch.manual_seed(cfg["test"]["seed"])
    random.seed(cfg["test"]["seed"])
    np.random.seed(cfg["test"]["seed"])
    
    # Check if the tfrecord path is specified in the config
    tfrecord_path = cfg.get("test", {}).get("tfrecord_path", args.tfrecord)
    
    # Load evaluation dataset
    print(f"Loading evaluation dataset from {cfg['dataset']['eval_path']}")
    test_frame_dataset = EvalDataset(
        dataset_path=cfg["dataset"]["eval_path"],
        resolution=cfg["test"]["resolution"],
    )

    # Create the enhanced test dataset that combines frame data with pre-processed features
    print(f"Loading pre-processed features from {tfrecord_path}")
    test_dataset = DeepfakeVideoDataset(
        frame_dataset=test_frame_dataset,
        tfrecord_path=tfrecord_path
    )

    # Create test dataloader
    num_workers = cfg.get("test", {}).get("num_workers", 4)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["test"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load the trained model from checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = DeepfakeVideoClassifier.load_from_checkpoint(args.checkpoint, strict=False)
    print("Model loaded with strict=False to handle architecture differences")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Configure the trainer
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if cfg["test"]["mixed_precision"] else 32,
        limit_test_batches=cfg["test"]["limit_test_batches"],
    )
    
    # Run the test
    print(f"Testing model on {len(test_dataset)} samples...")
    trainer.test(model=model, dataloaders=test_loader)
    
    print("Testing completed!") 