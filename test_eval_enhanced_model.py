#!/usr/bin/env python
import argparse
import os
import gc
import numpy as np
import random
from collections import defaultdict

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
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the video-level predictions.",
        default="./video_predictions.csv",
    )
    parser.add_argument(
        "--save_detailed_results",
        action="store_true",
        help="Whether to save detailed frame-level predictions in addition to video-level results",
    )
    args = parser.parse_args()
    return args


def evaluate_frames(model, test_loader, device):
    """
    Evaluate all frames and collect predictions by video
    
    Args:
        model: The DeepfakeVideoClassifier model
        test_loader: DataLoader with test samples
        device: Device to run evaluation on
        
    Returns:
        Dictionary mapping video filenames to lists of frame predictions
    """
    model.eval()
    video_frame_preds = defaultdict(list)
    video_frame_labels = defaultdict(list)
    
    with torch.no_grad():
        for batch in test_loader:
            frames = batch["image"].to(device)
            processed_features = batch["processed_features"].to(device)
            labels = batch["is_real"].to(device)
            filenames = batch["filename"]
            
            # Forward pass
            outputs = model(frames, processed_features)
            logits = outputs["logits"]
            
            # Convert logits to predictions (binary classification)
            if logits.shape[1] == 1:  # Binary classification
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).float()
            
            # Store predictions by video
            for i, filename in enumerate(filenames):
                video_frame_preds[filename].append(preds[i].cpu().item())
                video_frame_labels[filename].append(labels[i][0].cpu().item())
    
    return video_frame_preds, video_frame_labels


def majority_vote(predictions):
    """
    Apply majority voting to a list of predictions
    
    Args:
        predictions: List of binary predictions (1 for real, 0 for fake)
        
    Returns:
        Final prediction (1 for real, 0 for fake)
    """
    fake_count = sum(1 for p in predictions if p < 0.5)
    real_count = len(predictions) - fake_count
    
    # If equal, default to fake (more conservative approach)
    return 1 if real_count > fake_count else 0


def calculate_metrics(video_preds, video_labels):
    """
    Calculate evaluation metrics for video-level predictions
    
    Args:
        video_preds: Dictionary mapping video filenames to predictions
        video_labels: Dictionary mapping video filenames to ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for video_id in video_preds:
        pred = video_preds[video_id]
        label = video_labels[video_id]
        
        if pred == 1 and label == 1:
            true_positives += 1
        elif pred == 1 and label == 0:
            false_positives += 1
        elif pred == 0 and label == 0:
            true_negatives += 1
        elif pred == 0 and label == 1:
            false_negatives += 1
    
    # Calculate metrics
    accuracy = (true_positives + true_negatives) / len(video_preds) if len(video_preds) > 0 else 0
    
    # Avoid division by zero
    epsilon = 1e-7
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }


def save_predictions(video_preds, video_labels, video_frame_counts, video_frame_preds, output_file):
    """
    Save video-level predictions to a CSV file
    
    Args:
        video_preds: Dictionary mapping video filenames to predictions
        video_labels: Dictionary mapping video filenames to ground truth labels
        video_frame_counts: Dictionary mapping video filenames to frame counts
        video_frame_preds: Dictionary mapping video filenames to lists of frame predictions
        output_file: Path to save the predictions
    """
    with open(output_file, "w") as f:
        f.write("filename,expected_label,prediction,confidence,frame_count\n")
        for video_id in sorted(video_preds.keys()):
            pred = video_preds[video_id]
            label = video_labels[video_id]
            frame_count = video_frame_counts[video_id]
            
            # Convert numerical labels to text for clarity
            expected_label = "real" if label == 1 else "fake"
            prediction = "real" if pred == 1 else "fake"
            
            # Calculate confidence as the percentage of frames that agree with the final prediction
            agreement_count = sum(1 for p in video_frame_preds[video_id] if (p >= 0.5) == (pred == 1))
            confidence = agreement_count / frame_count if frame_count > 0 else 0
            
            f.write(f"{video_id},{expected_label},{prediction},{confidence:.4f},{frame_count}\n")


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

    # Evaluate all frames and collect predictions by video
    print(f"Evaluating {len(test_dataset)} frames...")
    video_frame_preds, video_frame_labels = evaluate_frames(model, test_loader, device)
    
    # Count frames per video
    video_frame_counts = {video_id: len(preds) for video_id, preds in video_frame_preds.items()}
    
    # Apply majority voting to get video-level predictions
    print("Applying majority voting for video-level predictions...")
    video_preds = {video_id: majority_vote(preds) for video_id, preds in video_frame_preds.items()}
    
    # Get ground truth for each video (most common label)
    video_labels = {}
    for video_id, labels in video_frame_labels.items():
        # Since all frames of a video should have the same label, we take the most frequent one
        # This handles potential inconsistencies in the dataset
        video_labels[video_id] = round(sum(labels) / len(labels))
    
    # Calculate and print metrics
    metrics = calculate_metrics(video_preds, video_labels)
    print("\nVideo-level evaluation metrics:")
    print(f"Total videos evaluated: {len(video_preds)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    # Save predictions to file
    save_predictions(video_preds, video_labels, video_frame_counts, video_frame_preds, args.output_file)
    print(f"Video-level predictions saved to {args.output_file}")
    
    # Also save detailed frame-level predictions if requested
    if args.save_detailed_results:
        detailed_output_file = args.output_file.replace('.csv', '_detailed.csv')
        with open(detailed_output_file, "w") as f:
            f.write("filename,frame_idx,prediction,confidence\n")
            for video_id in sorted(video_frame_preds.keys()):
                for i, pred in enumerate(video_frame_preds[video_id]):
                    confidence = pred if pred >= 0.5 else 1 - pred  # Convert to 0.5-1.0 range
                    prediction = "real" if pred >= 0.5 else "fake"
                    f.write(f"{video_id},{i},{prediction},{confidence:.4f}\n")
        print(f"Detailed frame-level predictions saved to {detailed_output_file}")
    
    print("Testing completed!") 