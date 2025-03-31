import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
import numpy as np

from pdd_dataset import PDDDataset
from enhanced_model import DeepfakeVideoClassifier, DeepfakeVideoDataset
from shap_explain import ModelExplainer
from gradcam import apply_gradcam

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze a trained DeepfakeVideoClassifier')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to PDD dataset')
    parser.add_argument('--tfrecord', type=str, required=True, help='Path to TFRecord file')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to analyze')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to analyze')
    return parser.parse_args()

def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DeepfakeVideoClassifier.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()
    
    # Load dataset
    frame_dataset = PDDDataset(
        dataset_path=args.dataset_path,
        split=args.split,
        resolution=224
    )
    
    dataset = DeepfakeVideoDataset(
        frame_dataset=frame_dataset,
        tfrecord_path=args.tfrecord
    )
    
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    
    # Get a batch of samples
    batch = next(iter(loader))
    frames = batch['image'].to(device)
    features = batch['processed_features'].to(device)
    labels = batch['is_real'].to(device)
    filenames = batch['filename']
    
    # 1. Use SHAP to analyze feature importance
    explainer = ModelExplainer(args.checkpoint, device)
    feature_explainer, feature_names = explainer.create_feature_explainer(dataset)
    
    # Global feature importance
    print("Analyzing global feature importance...")
    shap_values = explainer.visualize_feature_importance(feature_explainer, features, feature_names)
    
    # 2. Activation analysis for frames
    print("Analyzing frame activations...")
    for i in range(min(5, len(frames))):
        explainer.visualize_frame_activations(frames[i:i+1], features[i:i+1])
        is_real = "real" if labels[i].item() == 1 else "fake"
        plt.savefig(f'{is_real}_sample_{i}_activation.png')
    
    # 3. Grad-CAM visualization
    print("Applying Grad-CAM visualization...")
    gradcam_results = apply_gradcam(model, frames, features)
    
    # 4. Create a comprehensive visualization
    print("Creating comprehensive visualization...")
    for i in range(min(5, len(frames))):
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 2, 1)
        img = frames[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        is_real = "Real" if labels[i].item() == 1 else "Fake"
        plt.title(f'{is_real} Sample: {filenames[i]}')
        plt.axis('off')
        
        # Activation map
        plt.subplot(2, 2, 2)
        activations = explainer.analyze_frame_features(frames[i:i+1], features[i:i+1])
        attention_map = torch.mean(activations, dim=1).cpu().numpy()
        plt.imshow(attention_map[0], cmap='viridis')
        plt.title('Feature Activation Map')
        plt.axis('off')
        
        # Grad-CAM
        plt.subplot(2, 2, 3)
        plt.imshow(gradcam_results[i])
        plt.title('Grad-CAM Visualization')
        plt.axis('off')
        
        # SHAP values for processed features
        plt.subplot(2, 2, 4)
        sample_features = features[i:i+1].cpu().numpy()
        sample_shap = explainer.explain_features(feature_explainer, sample_features, feature_names)
        shap_data = np.column_stack([feature_names, sample_features[0], sample_shap[0]])
        y_pos = np.arange(len(feature_names))
        plt.barh(y_pos, sample_shap[0], align='center')
        plt.yticks(y_pos, feature_names)
        plt.title('Feature Importance (SHAP)')
        
        plt.tight_layout()
        plt.savefig(f'comprehensive_analysis_sample_{i}.png')
        plt.close()
    
    print("Analysis completed! Check the output files.")

if __name__ == '__main__':
    args = parse_args()
    main(args) 