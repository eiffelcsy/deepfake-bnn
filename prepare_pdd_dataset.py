#!/usr/bin/env python
"""
Utility script to prepare the PDD dataset for training with the enhanced model.
This script helps organize pre-extracted frames into the expected directory structure.
"""

import os
import argparse
import shutil
from pathlib import Path
import random
import tensorflow as tf
import numpy as np

def setup_dataset_structure(output_dir):
    """
    Creates the required directory structure for the PDD dataset
    
    Args:
        output_dir: Base directory for the dataset
    """
    # Create main dataset directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train/val splits
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    print(f"Created dataset structure in {output_dir}")


def copy_frames(frames_dir, output_dir, video_id, split='train', num_frames=10):
    """
    Copies existing frames to the appropriate directory in the dataset structure
    
    Args:
        frames_dir: Path to the directory containing the pre-extracted frames
        output_dir: Base directory for the dataset
        video_id: Identifier string for the video (should match tfrecord entry)
        split: 'train', 'val', or 'test'
        num_frames: Maximum number of frames to copy (will select evenly distributed frames)
    """
    # Source directory for this video's frames
    source_frames_dir = os.path.join(frames_dir, video_id)
    
    # Target directory for this video's frames
    target_frames_dir = os.path.join(output_dir, split, video_id)
    os.makedirs(target_frames_dir, exist_ok=True)
    
    # Get list of all frame files in the source directory
    frame_files = []
    if os.path.exists(source_frames_dir):
        for file in os.listdir(source_frames_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                frame_files.append(file)
    else:
        print(f"Warning: Frame directory {source_frames_dir} does not exist, skipping.")
        return
    
    # Sort frame files (assuming they are named with some numerical order)
    frame_files.sort()
    
    if len(frame_files) == 0:
        print(f"Warning: No frames found in {source_frames_dir}")
        return
    
    # Select evenly distributed frames
    if len(frame_files) <= num_frames:
        selected_frames = frame_files  # Use all frames if fewer than requested
    else:
        # Choose evenly distributed frames
        indices = [int(i * len(frame_files) / num_frames) for i in range(num_frames)]
        selected_frames = [frame_files[i] for i in indices]
    
    # Copy selected frames
    for i, frame_file in enumerate(selected_frames):
        source_path = os.path.join(source_frames_dir, frame_file)
        target_path = os.path.join(target_frames_dir, f"frame_{i:02d}{os.path.splitext(frame_file)[1]}")
        shutil.copy2(source_path, target_path)
    
    print(f"Copied {len(selected_frames)} frames from {source_frames_dir} to {target_frames_dir}")


def read_tfrecord_map(tfrecord_path):
    """
    Reads the TFRecord file to get a mapping of filenames to labels
    
    Args:
        tfrecord_path: Path to the TFRecord file
        
    Returns:
        Dictionary mapping filenames to fake labels (0=real, 1=fake)
    """
    # Define feature description for parsing
    feature_description = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'fake': tf.io.FixedLenFeature([1], tf.int64),
    }
    
    filename_to_fake = {}
    
    # Parse the TFRecord file
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    for raw_record in raw_dataset:
        example = tf.io.parse_single_example(raw_record, feature_description)
        filename = example['filename'].numpy().decode('utf-8')
        print(filename)
        fake = int(example['fake'].numpy()[0])  # 0=real, 1=fake
        filename_to_fake[filename] = fake
    
    print(f"Read {len(filename_to_fake)} entries from {tfrecord_path}")
    return filename_to_fake


def process_dataset(frames_dir, output_dir, tfrecord_path, train_ratio=0.8, val_ratio=0.2, num_frames=10):
    """
    Processes a directory of pre-extracted frames and organizes them into the PDD dataset structure
    
    Args:
        frames_dir: Directory containing pre-extracted frames (./data/[filename]/)
        output_dir: Base directory for the dataset
        tfrecord_path: Path to the TFRecord file with labels
        train_ratio: Proportion of videos to use for training
        val_ratio: Proportion of videos to use for validation
        num_frames: Number of frames to use from each video
    """
    # Create the dataset structure
    setup_dataset_structure(output_dir)
    
    # Read the label mapping from TFRecord
    filename_to_fake = read_tfrecord_map(tfrecord_path)
    
    # Get list of video directories (each containing frames)
    video_ids = []
    for item in os.listdir(frames_dir):
        item_path = os.path.join(frames_dir, item)
        if os.path.isdir(item_path):
            video_ids.append(item)
    
    # Shuffle the videos for random split
    random.shuffle(video_ids)
    
    # Split the videos into train/val
    train_end_idx = int(len(video_ids) * train_ratio)
    
    train_videos = video_ids[:train_end_idx]
    val_videos = video_ids[train_end_idx:]
    
    print(f"Split {len(video_ids)} videos into Train: {len(train_videos)}, Val: {len(val_videos)}")
    
    # Process each set of videos
    for videos, split in [(train_videos, 'train'), (val_videos, 'val')]:
        for video_id in videos:
            # Verify the video exists in TFRecord
            if video_id not in filename_to_fake:
                print(f"Warning: Video {video_id} not found in TFRecord, skipping.")
                continue
            
            # Copy frames
            copy_frames(frames_dir, output_dir, video_id, split, num_frames)


def main():
    parser = argparse.ArgumentParser(description="Prepare PDD dataset for training using pre-extracted frames")
    
    parser.add_argument('--frames_dir', type=str, default='./data/',
                        help='Path to directory containing pre-extracted frames (./data/[filename]/)')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for the prepared dataset')
    
    parser.add_argument('--tfrecord', type=str, default='./pdd_features.tfrecord',
                        help='Path to TFRecord file with pre-processed features')
    
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Proportion of videos to use for training')
    
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Proportion of videos to use for validation')
    
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to use from each video')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
        print("Warning: Train and val ratios should sum to 1.0")
    
    # Process dataset
    process_dataset(
        args.frames_dir,
        args.output_dir,
        args.tfrecord,
        args.train_ratio,
        args.val_ratio,
        args.num_frames
    )
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main() 