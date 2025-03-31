# DeepFake Detection Model Architecture

This repository contains the model architecture for a deepfake detection system based on the BNext backbone with additional feature extraction for improved detection.

## Model Overview

The BNext4DFR model is designed for binary classification of real vs fake images. It consists of:

1. **BNext Backbone**: Pre-trained backbone network (comes in various sizes: tiny, small, middle, large)
2. **Feature Enrichment**: Adds additional image processing channels to enhance detection:
   - Magnitude channel (edge detection using Sobel filters)
   - FFT channel (frequency domain analysis)
   - LBP channel (local binary patterns for texture analysis)
3. **Channel Adapter**: Converts the multi-channel input back to RGB using a convolutional layer
4. **Classification Head**: A fully connected layer for binary classification

## Requirements

```
matplotlib
pandas
pyarrow
wandb
opencv-python
scipy
scikit-learn
scikit-image
einops
timm
torch
torchvision
torchinfo
torchmetrics
lightning
fvcore
```

## How to Use

1. First, you'll need to implement or obtain the BNext backbone model
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Import the model in your project:
   ```python
   from model_architecture import BNext4DFR
   ```
4. Create and use the model:
   ```python
   # For binary classification (real vs fake)
   model = BNext4DFR(num_classes=2)
   
   # Make predictions
   with torch.no_grad():
       outputs = model(images)
       predictions = torch.sigmoid(outputs["logits"])
   ```

## Customization Options

- `backbone`: Choose from 'BNext-T', 'BNext-S', 'BNext-M', 'BNext-L'
- `freeze_backbone`: Whether to freeze the backbone weights during training
- `add_magnitude_channel`, `add_fft_channel`, `add_lbp_channel`: Toggle additional feature channels
- `learning_rate`: Control the learning rate for training
- `pos_weight`: Adjust the positive class weight for imbalanced datasets

## Important Notes

1. The BNext backbone implementation is required and should be placed in the proper directory structure
2. Pre-trained weights for the backbone should be available in the 'pretrained/' directory
3. The model is implemented using PyTorch Lightning for easy training and evaluation 