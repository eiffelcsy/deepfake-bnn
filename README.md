# Enhanced Deepfake Detection Model

This implementation extends the BNext4DFR deepfake detection model to incorporate pre-processed video features for improved accuracy.

## Overview

The enhanced model builds upon the BNext4DFR architecture by:

1. Using BNext4DFR to extract frame-level features
2. Combining extracted frame features with pre-processed video features:
   - Flicker
   - Lip movement variance
   - Blink detection
   - Head movement
   - Pulse detection
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
3. Passing the combined features through a fusion network for final classification

## File Structure

- `enhanced_model.py` - Contains the implementation of the enhanced model
- `train_enhanced_model.py` - Script for training the model
- `test_enhanced_model.py` - Script for evaluating the model
- `pdd_dataset.py` - Dataset class for the PDD dataset
- `prepare_pdd_dataset.py` - Utility script to prepare the PDD dataset
- `configs/enhanced_model.cfg` - General configuration file for the model
- `configs/pdd_enhanced_model.cfg` - Configuration file for the PDD dataset

## Requirements

This implementation requires all dependencies of the original BNext4DFR model, plus:

```
tensorflow>=2.0.0
opencv-python>=4.5.0
```

## Using with PDD Dataset

### Preparing the PDD Dataset

The PDD dataset consists of video frames organized in a specific directory structure, along with pre-processed features in a TFRecord file.

To prepare your own PDD dataset from videos:

1. Organize your videos in a directory
2. Run the preparation script:

```bash
python prepare_pdd_dataset.py --videos_dir /path/to/videos --output_dir ../../datasets/pdd --tfrecord ./pdd_features.tfrecord
```

This will:
- Extract 10 frames from each video
- Organize them into train/val/test splits
- Create the expected directory structure:
  ```
  pdd/
  ├── train/
  │   ├── video1/
  │   │   ├── frame_00.jpg
  │   │   ├── frame_01.jpg
  │   │   └── ...
  │   ├── video2/
  │   │   └── ...
  │   └── ...
  ├── val/
  │   ├── video3/
  │   │   └── ...
  │   └── ...
  └── test/
      ├── video4/
      │   └── ...
      └── ...
  ```

### Training with the PDD Dataset

To train the enhanced model with the PDD dataset:

```bash
python train_enhanced_model.py --cfg configs/pdd_enhanced_model.cfg --tfrecord ./pdd_features.tfrecord
```

### Testing with the PDD Dataset

To test the trained model:

```bash
python test_enhanced_model.py --cfg configs/pdd_enhanced_model.cfg --checkpoint path/to/checkpoint.ckpt --tfrecord ./pdd_features.tfrecord
```

## Data Format

### Frame Dataset

The model expects frames from videos to be provided in a directory structure where each video has its own subdirectory containing the extracted frames.

### Pre-processed Features

Pre-processed features should be stored in a TFRecord file with the following structure:

- `filename`: String identifying the video
- `fake`: Binary label (0=real, 1=fake)
- `flicker`: 45-dimensional flicker feature array
- `lip_movement_variance`: 5-dimensional lip movement variance feature array
- `blink`: 5-dimensional blink detection feature array
- `head_movement`: 50-dimensional head movement feature array
- `pulse`: 50-dimensional pulse detection feature array
- `psnr`: 45-dimensional PSNR feature array
- `ssim`: 45-dimensional SSIM feature array

## Model Architecture

The model consists of three main components:

1. **Frame Feature Extractor**: Uses the BNext4DFR model to extract features from video frames
2. **Feature Fusion**: Concatenates frame features with pre-processed video features
3. **Classification Head**: A fully connected network that processes the combined features for binary classification

## Customization

You can customize the model behavior through the configuration file:

- `backbone`: Choose from 'BNext-T', 'BNext-S', 'BNext-M', 'BNext-L'
- `freeze_backbone`: Whether to freeze the backbone weights during training
- `add_magnitude_channel`, `add_fft_channel`, `add_lbp_channel`: Toggle additional feature channels for frame processing 