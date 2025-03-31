#!/usr/bin/env python
import argparse
import os
import gc
import numpy as np
import torch
import cv2
import tensorflow as tf
from tqdm import tqdm

from enhanced_model import DeepfakeVideoClassifier, TFRecordDatasetLoader
from preprocess_dfd import (extract_video_info, extract_10_frames, detect_flicker, 
                           get_lip_movement, detect_blinks, extract_head_pose, 
                           detect_pulse, compute_ssim_psnr)
from lib.util import load_config

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="The path to the config.",
        default="./configs/pdd_enhanced_model.cfg",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="The path to the model checkpoint.",
        required=True,
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to the video file for inference.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save temporary frames.",
        default="./temp_frames",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Resolution to resize frames to (square).",
        default=224,
    )
    args = parser.parse_args()
    return args

def extract_features_from_video(video_path, output_dir):
    """
    Extract required features from a video for model inference
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save temporary frames
        
    Returns:
        dict: Dictionary containing extracted features
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Extracting features from {video_path}")
    
    # Get video info
    total_frames, fps, duration = extract_video_info(video_path)
    segment_timestamps = [(i * (duration / 5), (i + 1) * (duration / 5)) for i in range(5)]
    
    # Initialize feature lists
    flicker_vals = []
    lip_var = []
    blink_feature = []
    head_movement = []
    pulse_vals = []
    psnr_vals = []
    ssim_vals = []
    all_frames = []
    
    # Process each segment
    for i, (start_time, end_time) in enumerate(segment_timestamps):
        print(f"Processing segment {i+1}/5...")
        
        # Extract frames from this segment
        frames = extract_10_frames(
            video_path, fps, start_time, end_time, 
            save_dir=output_dir, segment_idx=i
        )
        
        if not frames:
            print(f"Warning: No frames extracted for segment {i+1}")
            continue
            
        # Add to all frames list
        all_frames.extend(frames)
        
        # Extract features from frames
        flicker_vals += detect_flicker(frames)
        lip_var += get_lip_movement(frames)
        blink_feature += detect_blinks(frames)
        head_movement += extract_head_pose(frames)
        pulse_vals += detect_pulse(frames)
        psnr_val, ssim_val = compute_ssim_psnr(frames)
        psnr_vals += psnr_val
        ssim_vals += ssim_val
    
    return {
        "frames": all_frames,
        "flicker": flicker_vals,
        "lip_movement_variance": lip_var,
        "blink": blink_feature,
        "head_movement": head_movement,
        "pulse": pulse_vals,
        "psnr": psnr_vals,
        "ssim": ssim_vals
    }

def prepare_frame_for_model(frame, resolution):
    """Prepare a frame for input to the model"""
    # Resize to the required resolution
    frame = cv2.resize(frame, (resolution, resolution))
    
    # Convert from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PyTorch tensor format (C, H, W) and normalize
    frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
    
    return frame

def average_features(feature_list):
    """Average a list of features to get a single value"""
    if not feature_list:
        return 0.0
    return np.mean(feature_list)

def main():
    # Parse arguments
    args = args_func()
    
    # Load config
    cfg = load_config(args.cfg)
    print("Loaded configuration:")
    for section in cfg:
        print(f"[{section}]")
        for key, value in cfg[section].items():
            print(f"  {key}: {value}")
    
    # Extract features from video
    features = extract_features_from_video(args.video, args.output_dir)
    
    if not features["frames"]:
        print("Error: No frames could be extracted from the video.")
        return
    
    # Load the model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = DeepfakeVideoClassifier.load_from_checkpoint(args.checkpoint)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Prepare processed features
    processed_features = torch.tensor([
        average_features(features["flicker"]),
        average_features(features["lip_movement_variance"]),
        average_features(features["blink"]),
        average_features(features["head_movement"]),
        average_features(features["pulse"]),
        average_features(features["psnr"]),
        average_features(features["ssim"])
    ], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Process individual frames and make predictions
    predictions = []
    batch_size = 1
    
    with torch.no_grad():
        for i, frame in enumerate(tqdm(features["frames"], desc="Processing frames")):
            # Prepare frame
            processed_frame = prepare_frame_for_model(frame, args.resolution)
            processed_frame = processed_frame.unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(processed_frame, processed_features)
            logits = outputs["logits"]
            
            # Get probability
            if logits.shape[1] == 1:  # Binary classification
                prob = torch.sigmoid(logits)[0][0].item()
            else:  # Multi-class
                prob = torch.softmax(logits, dim=1)[0][1].item()  # Probability of fake class
                
            predictions.append(prob)
    
    # Calculate final prediction
    avg_prediction = np.mean(predictions)
    
    # Output result
    print("\n" + "="*50)
    print(f"Video: {os.path.basename(args.video)}")
    print(f"Probability of being fake: {avg_prediction:.4f} ({avg_prediction*100:.2f}%)")
    print(f"Classification: {'FAKE' if avg_prediction > 0.5 else 'REAL'}")
    print("="*50)
    
    # Clean up temporary files if needed
    print(f"Temporary frames saved in: {args.output_dir}")
    print("You can delete them manually when no longer needed.")

if __name__ == "__main__":
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    main() 