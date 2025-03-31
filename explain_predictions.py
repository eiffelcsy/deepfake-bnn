import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from pdd_dataset import PDDDataset
from enhanced_model import DeepfakeVideoDataset
from shap_explain import ModelExplainer

def args_func():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the PDD dataset.",
        required=True,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to analyze",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Image resolution for the model",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to analyze",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_func()
    
    # Load the PDD dataset
    print(f"Loading PDD {args.split} dataset from {args.dataset_path}")
    frame_dataset = PDDDataset(
        dataset_path=args.dataset_path,
        split=args.split,
        resolution=args.resolution,
    )
    
    # Create the enhanced dataset that combines frame data with pre-processed features
    print(f"Loading pre-processed features from {args.tfrecord}")
    dataset = DeepfakeVideoDataset(
        frame_dataset=frame_dataset,
        tfrecord_path=args.tfrecord
    )
    
    # Create a DataLoader for the dataset
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize the model explainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    explainer = ModelExplainer(args.checkpoint, device=device)
    
    # Get a batch of data to analyze
    real_samples = []
    fake_samples = []
    real_frames = []
    fake_frames = []
    
    # Collect samples
    print("Collecting samples for analysis...")
    for i, batch in enumerate(loader):
        for j in range(len(batch['is_real'])):
            if batch['is_real'][j].item() == 1 and len(real_samples) < args.num_samples // 2:
                real_samples.append(batch['processed_features'][j])
                real_frames.append(batch['image'][j])
            elif batch['is_real'][j].item() == 0 and len(fake_samples) < args.num_samples // 2:
                fake_samples.append(batch['processed_features'][j])
                fake_frames.append(batch['image'][j])
                
        if len(real_samples) >= args.num_samples // 2 and len(fake_samples) >= args.num_samples // 2:
            break
    
    # Convert to tensors
    real_samples = torch.stack(real_samples)
    fake_samples = torch.stack(fake_samples)
    real_frames = torch.stack(real_frames)
    fake_frames = torch.stack(fake_frames)
    
    print(f"Collected {len(real_samples)} real samples and {len(fake_samples)} fake samples")
    
    # Analyze global feature importance
    feature_names = ["Flicker", "Lip Movement", "Blink", "Head Movement", "Pulse", "PSNR", "SSIM"]
    
    # Create a feature explainer
    print("Creating feature explainer...")
    feature_explainer, feature_names = explainer.create_feature_explainer(dataset)
    
    # Analyze real samples
    print("Analyzing real samples...")
    explainer.visualize_feature_importance(feature_explainer, real_samples, feature_names)
    
    # Analyze fake samples
    print("Analyzing fake samples...")
    explainer.visualize_feature_importance(feature_explainer, fake_samples, feature_names)
    
    # Visualize frame activations for a few samples
    print("Visualizing frame activations...")
    for i in range(min(5, len(real_frames))):
        frame = real_frames[i:i+1]
        features = real_samples[i:i+1]
        explainer.visualize_frame_activations(frame, features)
        plt.savefig(f'real_frame_{i}_activation.png')
        
    for i in range(min(5, len(fake_frames))):
        frame = fake_frames[i:i+1]
        features = fake_samples[i:i+1]
        explainer.visualize_frame_activations(frame, features)
        plt.savefig(f'fake_frame_{i}_activation.png')
    
    print("Analysis complete! Visualizations saved to the current directory.") 