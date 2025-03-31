#!/usr/bin/env python
"""
Utility script to prepare the PDD dataset for training with the enhanced model.
This script helps organize frames from videos into the expected directory structure.
"""

import os
import argparse
import shutil
from pathlib import Path
import random
import cv2
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
    
    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    print(f"Created dataset structure in {output_dir}")


def extract_frames(video_path, output_dir, video_id, split='train', num_frames=10):
    """
    Extracts frames from a video and saves them to the appropriate directory
    
    Args:
        video_path: Path to the video file
        output_dir: Base directory for the dataset
        video_id: Identifier string for the video (should match tfrecord entry)
        split: 'train', 'val', or 'test'
        num_frames: Number of frames to extract
    """
    # Create directory for this video's frames
    video_frames_dir = os.path.join(output_dir, split, video_id)
    os.makedirs(video_frames_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Warning: Video {video_path} has 0 frames")
        return
    
    # Get frames at regular intervals
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    for i, frame_index in enumerate(frame_indices):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.read()
        
        if ret:
            output_path = os.path.join(video_frames_dir, f"frame_{i:02d}.jpg")
            cv2.imwrite(output_path, frame)
        else:
            print(f"Warning: Could not read frame {frame_index} from {video_path}")
    
    video.release()
    print(f"Extracted {num_frames} frames from {video_path} to {video_frames_dir}")


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
        fake = int(example['fake'].numpy()[0])  # 0=real, 1=fake
        filename_to_fake[filename] = fake
    
    print(f"Read {len(filename_to_fake)} entries from {tfrecord_path}")
    return filename_to_fake


def process_videos(videos_dir, output_dir, tfrecord_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_frames=10):
    """
    Processes a directory of videos and organizes them into the PDD dataset structure
    
    Args:
        videos_dir: Directory containing video files
        output_dir: Base directory for the dataset
        tfrecord_path: Path to the TFRecord file with labels
        train_ratio: Proportion of videos to use for training
        val_ratio: Proportion of videos to use for validation
        test_ratio: Proportion of videos to use for testing
        num_frames: Number of frames to extract from each video
    """
    # Create the dataset structure
    setup_dataset_structure(output_dir)
    
    # Read the label mapping from TFRecord
    filename_to_fake = read_tfrecord_map(tfrecord_path)
    
    # Get list of video files
    video_files = []
    for root, _, files in os.walk(videos_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    
    # Shuffle the videos for random split
    random.shuffle(video_files)
    
    # Split the videos into train/val/test
    train_end_idx = int(len(video_files) * train_ratio)
    val_end_idx = train_end_idx + int(len(video_files) * val_ratio)
    
    train_videos = video_files[:train_end_idx]
    val_videos = video_files[train_end_idx:val_end_idx]
    test_videos = video_files[val_end_idx:]
    
    print(f"Split {len(video_files)} videos into Train: {len(train_videos)}, Val: {len(val_videos)}, Test: {len(test_videos)}")
    
    # Process each set of videos
    for videos, split in [(train_videos, 'train'), (val_videos, 'val'), (test_videos, 'test')]:
        for video_path in videos:
            # Extract video ID from filename - adjust this based on your naming convention
            video_id = Path(video_path).stem
            
            # Verify the video exists in TFRecord
            if video_id not in filename_to_fake:
                print(f"Warning: Video {video_id} not found in TFRecord, skipping.")
                continue
            
            # Extract frames
            extract_frames(video_path, output_dir, video_id, split, num_frames)


def main():
    parser = argparse.ArgumentParser(description="Prepare PDD dataset for training")
    
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Path to directory containing video files')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for the prepared dataset')
    
    parser.add_argument('--tfrecord', type=str, default='./pdd_features.tfrecord',
                        help='Path to TFRecord file with pre-processed features')
    
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Proportion of videos to use for training')
    
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Proportion of videos to use for validation')
    
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Proportion of videos to use for testing')
    
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to extract from each video')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("Warning: Train, val, and test ratios should sum to 1.0")
    
    # Process videos
    process_videos(
        args.videos_dir,
        args.output_dir,
        args.tfrecord,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.num_frames
    )
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main() 