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
import re

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


def copy_frames(frames_dir, output_dir, video_id, split='train', frames_per_segment=10, num_segments=5):
    """
    Copies existing frames to the appropriate directory in the dataset structure,
    preserving segment structure with multiple frames per segment
    
    Args:
        frames_dir: Path to the directory containing the pre-extracted frames
        output_dir: Base directory for the dataset
        video_id: Identifier string for the video (should match tfrecord entry)
        split: 'train', 'val', or 'test'
        frames_per_segment: Number of frames in each segment
        num_segments: Number of segments per video
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
    # Try to extract numbers from filenames for proper numerical sorting
    def get_frame_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    frame_files.sort(key=get_frame_number)
    
    if len(frame_files) == 0:
        print(f"Warning: No frames found in {source_frames_dir}")
        return
    
    # Copy all frames, preserving segment structure
    total_frames = frames_per_segment * num_segments
    
    # If we have fewer frames than expected, use all available frames
    if len(frame_files) <= total_frames:
        selected_frames = frame_files
    else:
        # Choose evenly distributed frames across all segments
        indices = [int(i * len(frame_files) / total_frames) for i in range(total_frames)]
        selected_frames = [frame_files[i] for i in indices]
    
    # Copy selected frames, maintaining segment structure
    for i, frame_file in enumerate(selected_frames):
        source_path = os.path.join(source_frames_dir, frame_file)
        
        # Calculate segment and frame within segment
        segment_idx = i // frames_per_segment
        frame_in_segment = i % frames_per_segment
        
        # Create target filename preserving segment information
        target_path = os.path.join(
            target_frames_dir, 
            f"segment_{segment_idx:01d}_frame_{frame_in_segment:02d}{os.path.splitext(frame_file)[1]}"
        )
        
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
        clean_filename = os.path.splitext(filename)[0]
        fake = int(example['fake'].numpy()[0])  # 0=real, 1=fake
        filename_to_fake[clean_filename] = fake
        filename_to_fake[filename] = fake
    
    print(f"Read {len(filename_to_fake) // 2} entries from {tfrecord_path}")
    return filename_to_fake


def process_dataset(frames_dir, output_dir, tfrecord_path, train_ratio=0.8, val_ratio=0.2, 
                   frames_per_segment=10, num_segments=5):
    """
    Processes a directory of pre-extracted frames and organizes them into the PDD dataset structure
    
    Args:
        frames_dir: Directory containing pre-extracted frames (./data/[filename]/)
        output_dir: Base directory for the dataset
        tfrecord_path: Path to the TFRecord file with labels
        train_ratio: Proportion of videos to use for training
        val_ratio: Proportion of videos to use for validation
        frames_per_segment: Number of frames in each segment
        num_segments: Number of segments per video
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
    
    # Count of videos processed and skipped
    processed_count = 0
    skipped_count = 0
    
    # Process each set of videos
    for videos, split in [(train_videos, 'train'), (val_videos, 'val')]:
        for video_id in videos:
            # Try different variations of video_id to match TFRecord entries
            if video_id in filename_to_fake:
                # Direct match
                match_found = True
            elif video_id + ".mp4" in filename_to_fake:
                # Try with .mp4 extension
                match_found = True
                video_id_for_label = video_id + ".mp4"
            elif os.path.splitext(video_id)[0] in filename_to_fake:
                # Try with extension stripped
                match_found = True
                video_id = os.path.splitext(video_id)[0]
            else:
                match_found = False
            
            if not match_found:
                print(f"Warning: Video {video_id} not found in TFRecord, skipping.")
                skipped_count += 1
                continue
            
            # Copy frames with segment structure preserved
            copy_frames(frames_dir, output_dir, video_id, split, frames_per_segment, num_segments)
            processed_count += 1
    
    print(f"Processed {processed_count} videos, skipped {skipped_count} videos")


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
    
    parser.add_argument('--frames_per_segment', type=int, default=10,
                        help='Number of frames per segment')
    
    parser.add_argument('--num_segments', type=int, default=5,
                        help='Number of segments per video')
    
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
        args.frames_per_segment,
        args.num_segments
    )
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main() 