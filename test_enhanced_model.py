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

# Import dataset classes
from cifake_dataset import CIFAKEDataset
from coco_fake_dataset import COCOFakeDataset
from dffd_dataset import DFFDDataset
from pdd_dataset import PDDDataset  # Import the new PDD dataset

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="The path to the config.",
        default="./configs/enhanced_model.cfg",
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
        default="./pdd_features.tfrecord",
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
    
    # Load the appropriate test dataset based on config
    if cfg["dataset"]["name"] == "coco_fake":
        print(f"Loading COCO-Fake test dataset from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}")
        test_frame_dataset = COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="test",
            mode="single",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "dffd":
        print(f"Loading DFFD test dataset from {cfg['dataset']['dffd_path']}")
        test_frame_dataset = DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "cifake":
        print(f"Loading CIFAKE test dataset from {cfg['dataset']['cifake_path']}")
        test_frame_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "pdd":
        print(f"Loading PDD test dataset from {cfg['dataset']['pdd_path']}")
        test_frame_dataset = PDDDataset(
            dataset_path=cfg["dataset"]["pdd_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    else:
        raise ValueError(f"Unsupported dataset name: {cfg['dataset']['name']}")

    # Create the enhanced test dataset that combines frame data with pre-processed features
    print(f"Loading pre-processed features from {args.tfrecord}")
    test_dataset = DeepfakeVideoDataset(
        frame_dataset=test_frame_dataset,
        tfrecord_path=args.tfrecord
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
    model = DeepfakeVideoClassifier.load_from_checkpoint(args.checkpoint)
    
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