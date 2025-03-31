import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from enhanced_model import DeepfakeVideoClassifier

class ModelExplainer:
    def __init__(self, model_checkpoint, device='cuda'):
        """
        Initialize the explainer with a trained DeepfakeVideoClassifier
        
        Args:
            model_checkpoint: Path to the trained model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        # Load the trained model
        self.model = DeepfakeVideoClassifier.load_from_checkpoint(model_checkpoint)
        self.model.to(device)
        self.model.eval()
        
    def create_feature_explainer(self, background_dataset, num_samples=100):
        """
        Create a SHAP explainer for the processed features
        
        Args:
            background_dataset: Dataset to use for background distribution
            num_samples: Number of samples to use for background
        
        Returns:
            SHAP explainer for the processed features
        """
        # Create a wrapper for the model that only explains the processed features
        def model_wrapper(processed_features):
            # Use a fixed frame for all explanations
            if isinstance(processed_features, np.ndarray):
                proc_features = torch.tensor(processed_features, dtype=torch.float32).to(self.device)
                batch_size = proc_features.shape[0]
                # Use the first image from the background as a fixed frame
                fixed_frame = torch.stack([background_dataset[0]['image']] * batch_size).to(self.device)
            else:
                proc_features = processed_features
                batch_size = proc_features.shape[0]
                fixed_frame = torch.stack([background_dataset[0]['image']] * batch_size).to(self.device)
            
            with torch.no_grad():
                out = self.model(fixed_frame, proc_features)
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(out["logits"])
                return probs.cpu().numpy()
        
        # Get background samples for the explainer
        background_samples = []
        feature_names = ["Flicker", "Lip Movement", "Blink", "Head Movement", "Pulse", "PSNR", "SSIM"]
        
        for i in range(min(num_samples, len(background_dataset))):
            background_samples.append(background_dataset[i]['processed_features'].numpy())
        
        background = np.array(background_samples)
        
        # Create the SHAP explainer
        explainer = shap.KernelExplainer(model_wrapper, background)
        return explainer, feature_names
    
    def explain_features(self, explainer, samples, feature_names):
        """
        Generate SHAP values for the processed features
        
        Args:
            explainer: SHAP explainer for the processed features
            samples: Samples to explain
            feature_names: Names of the features
            
        Returns:
            SHAP values for the samples
        """
        # Convert samples to numpy if they're torch tensors
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(samples)
        
        return shap_values
    
    def visualize_feature_importance(self, explainer, samples, feature_names):
        """
        Visualize the importance of processed features
        
        Args:
            explainer: SHAP explainer for the processed features
            samples: Samples to explain
            feature_names: Names of the features
        """
        # Convert samples to numpy if they're torch tensors
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        
        # Calculate and plot SHAP values
        shap_values = explainer.shap_values(samples)
        
        # Summary plot showing the importance of each feature
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, samples, feature_names=feature_names)
        plt.tight_layout()
        plt.savefig('feature_importance_summary.png')
        
        # Bar plot showing average feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, samples, feature_names=feature_names, plot_type="bar")
        plt.tight_layout()
        plt.savefig('feature_importance_bar.png')
        
        # Force plot for detailed inspection of individual samples
        for i in range(min(5, len(samples))):
            plt.figure(figsize=(20, 3))
            force_plot = shap.force_plot(
                explainer.expected_value, 
                shap_values[i, :], 
                samples[i, :], 
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.savefig(f'sample_{i}_force_plot.png', bbox_inches='tight')
            plt.close()
        
        return shap_values

    def analyze_frame_features(self, frame, processed_features):
        """
        Analyze the activation patterns in the frame feature extractor
        
        Args:
            frame: Input frame
            processed_features: Processed temporal features
        
        Returns:
            Activation maps for the frame
        """
        # Ensure the inputs are on the right device
        frame = frame.to(self.device)
        processed_features = processed_features.to(self.device)
        
        # Hook for getting the activations
        activations = {}
        def hook_fn(module, input, output):
            activations['features'] = output.detach()
            
        # Register the hook
        hook = self.model.frame_feature_extractor.base_model.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            self.model(frame, processed_features)
            
        # Remove the hook
        hook.remove()
        
        # Return the activations
        return activations['features']

    def visualize_frame_activations(self, frame, processed_features):
        """
        Visualize the activation patterns in the frame feature extractor
        
        Args:
            frame: Input frame
            processed_features: Processed temporal features
        
        Returns:
            Activation visualization
        """
        # Get the activations
        activations = self.analyze_frame_features(frame, processed_features)
        
        # Average across channels to get attention map
        attention_map = torch.mean(activations, dim=1).cpu().numpy()
        
        # Plot the original image and activation map
        plt.figure(figsize=(12, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        img = frame[0].permute(1, 2, 0).cpu().numpy()
        # Normalize for visualization
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Activation map
        plt.subplot(1, 2, 2)
        plt.imshow(attention_map[0], cmap='viridis')
        plt.title('Feature Activation Map')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('frame_activation_map.png')
        
        return attention_map 