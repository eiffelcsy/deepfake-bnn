#!/usr/bin/env python
import argparse
import os
import random
from datetime import datetime
import numpy as np
import gc

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from lib.util import load_config
from model_architecture import BNext4DFR
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
        "--tfrecord",
        type=str,
        help="Path to the TFRecord file with pre-processed features.",
        default="./pdd_features.tfrecord",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
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
    torch.manual_seed(cfg["train"]["seed"])
    random.seed(cfg["train"]["seed"])
    np.random.seed(cfg["train"]["seed"])
    torch.set_float32_matmul_precision("medium")

    # Load the appropriate frame dataset based on config
    if cfg["dataset"]["name"] == "coco_fake":
        print(f"Loading COCO-Fake datasets from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}")
        train_frame_dataset = COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="train",
            mode="single",
            resolution=cfg["train"]["resolution"],
        )
        val_frame_dataset = COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="val",
            mode="single",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "dffd":
        print(f"Loading DFFD dataset from {cfg['dataset']['dffd_path']}")
        train_frame_dataset = DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_frame_dataset = DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="val",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "cifake":
        print(f"Loading CIFAKE dataset from {cfg['dataset']['cifake_path']}")
        train_frame_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_frame_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="test",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "pdd":
        print(f"Loading PDD dataset from {cfg['dataset']['pdd_path']}")
        train_frame_dataset = PDDDataset(
            dataset_path=cfg["dataset"]["pdd_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_frame_dataset = PDDDataset(
            dataset_path=cfg["dataset"]["pdd_path"],
            split="val",
            resolution=cfg["train"]["resolution"],
        )
    else:
        raise ValueError(f"Unsupported dataset name: {cfg['dataset']['name']}")

    # Create the enhanced datasets that combine frame data with pre-processed features
    print(f"Loading pre-processed features from {args.tfrecord}")
    train_dataset = DeepfakeVideoDataset(
        frame_dataset=train_frame_dataset,
        tfrecord_path=args.tfrecord
    )
    
    val_dataset = DeepfakeVideoDataset(
        frame_dataset=val_frame_dataset,
        tfrecord_path=args.tfrecord
    )

    # Create dataloaders
    num_workers = cfg.get("train", {}).get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize the enhanced model
    positive_samples = sum([1 if item["is_real"].sum() > 0 else 0 for item in train_dataset])
    negative_samples = len(train_dataset) - positive_samples
    pos_weight = negative_samples / max(positive_samples, 1)  # Avoid division by zero
    
    print(f"Class balance - Positive: {positive_samples}, Negative: {negative_samples}, Weight: {pos_weight}")
    
    model = DeepfakeVideoClassifier(
        num_classes=cfg["dataset"]["labels"],
        backbone=cfg["model"]["backbone"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        add_magnitude_channel=cfg["model"]["add_magnitude_channel"],
        add_fft_channel=cfg["model"]["add_fft_channel"],
        add_lbp_channel=cfg["model"]["add_lbp_channel"],
        pos_weight=pos_weight,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Configure training with PyTorch Lightning
    date = datetime.now().strftime("%Y%m%d_%H%M")
    project = "Enhanced_Deepfake_Detection"
    run_label = os.path.basename(args.cfg).split(".")[0]
    run = f"{cfg['dataset']['name']}_{date}_{run_label}"
    
    # Initialize WandB logger (if available, otherwise it will fall back to a basic logger)
    try:
        logger = WandbLogger(project=project, name=run, id=run, log_model=False)
    except ImportError:
        print("WandB not available, using TensorBoard logger")
        from lightning.pytorch.loggers import TensorBoardLogger
        logger = TensorBoardLogger("lightning_logs", name=run)
    
    # Create the PyTorch Lightning trainer
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if cfg["train"]["mixed_precision"] else 32,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg["train"]["accumulation_batches"],
        limit_train_batches=cfg["train"]["limit_train_batches"],
        limit_val_batches=cfg["train"]["limit_val_batches"],
        max_epochs=cfg["train"]["epoch_num"],
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_acc",
                save_top_k=1,
                mode="max",
                filename=f"{cfg['dataset']['name']}_{cfg['model']['backbone']}_{{epoch}}-{{train_acc:.2f}}-{{val_acc:.2f}}",
            )
        ],
        logger=logger,
    )
    
    # Start training
    print(f"Starting training with {model.__class__.__name__} on {device}")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print(f"Training completed. Model saved to {trainer.checkpoint_callback.best_model_path}") 