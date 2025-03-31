import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from enhanced_model import DeepfakeVideoClassifier

class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.gradient = None
        self.activation = None
        
        # Register hooks
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self._forward_hook)
                module.register_backward_hook(self._backward_hook)
                
    def _forward_hook(self, module, input, output):
        self.activation = output
        
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]
        
    def __call__(self, frame, processed_features, class_idx=None):
        # Forward pass
        self.model.eval()
        self.model.zero_grad()
        
        # Get model output
        output = self.model(frame, processed_features)
        logits = output["logits"]
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)
            
        # Target for backprop
        one_hot = torch.zeros_like(logits)
        one_hot[:, class_idx] = 1
        
        # Backward pass
        logits.backward(gradient=one_hot)
        
        # Get weights
        gradients = self.gradient
        activations = self.activation
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))
        
        # Weight the activations by the gradients
        cam = torch.zeros(activations.shape[0], activations.shape[2], activations.shape[3], device=activations.device)
        
        for i, w in enumerate(weights):
            cam += torch.sum(w[:, None, None] * activations[i], dim=0)
            
        # Apply ReLU and normalize
        cam = torch.nn.functional.relu(cam)
        cam = cam / (torch.max(cam) + 1e-10)
        
        return cam

def visualize_cam(image, cam):
    # Convert to numpy
    cam = cam.cpu().numpy()
    img = image.permute(1, 2, 0).cpu().numpy()
    
    # Normalize image for display
    img = (img - img.min()) / (img.max() - img.min())
    
    # Resize CAM to match image size
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0
    
    # Combine image and heatmap
    result = heatmap * 0.4 + img * 0.6
    result = result / result.max()
    
    return result

# Example usage:
def apply_gradcam(model, frame, processed_features):
    # Initialize Grad-CAM with the last convolutional layer
    cam = GradCAM(model, 'frame_feature_extractor.base_model.stages.3')
    
    # Apply Grad-CAM
    activation_map = cam(frame, processed_features)
    
    # Visualize for each image in the batch
    results = []
    for i in range(frame.shape[0]):
        result = visualize_cam(frame[i], activation_map[i])
        results.append(result)
        
        # Save the visualization
        plt.figure(figsize=(10, 5))
        plt.imshow(result)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'gradcam_result_{i}.png')
        
    return results 