import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional.classification import accuracy, auroc
import tensorflow as tf
import numpy as np
import os

from model_architecture import BNext4DFR

class DeepfakeVideoClassifier(L.LightningModule):
    def __init__(self, num_classes=2, backbone='BNext-T', 
                 freeze_backbone=True, add_magnitude_channel=True, 
                 add_fft_channel=True, add_lbp_channel=True,
                 learning_rate=1e-5, pos_weight=1.):
        """
        Enhanced deepfake detector that combines frame features with pre-processed video features
        
        Args:
            num_classes: Number of output classes (usually 2 for real/fake classification)
            backbone: BNext backbone type ('BNext-T', 'BNext-S', 'BNext-M', 'BNext-L')
            freeze_backbone: Whether to freeze the backbone weights
            add_magnitude_channel: Whether to add edge magnitude channel to input
            add_fft_channel: Whether to add FFT channel to input
            add_lbp_channel: Whether to add Local Binary Pattern channel to input
            learning_rate: Learning rate for optimizer
            pos_weight: Positive class weight for loss function
        """
        super(DeepfakeVideoClassifier, self).__init__()
        
        # Initialize the frame feature extractor (BNext4DFR)
        self.frame_feature_extractor = BNext4DFR(
            num_classes=num_classes,
            backbone=backbone,
            freeze_backbone=freeze_backbone,
            add_magnitude_channel=add_magnitude_channel,
            add_fft_channel=add_fft_channel,
            add_lbp_channel=add_lbp_channel,
            learning_rate=learning_rate,
            pos_weight=pos_weight
        )
        
        # Get the feature dimension from the backbone
        self.frame_feature_dim = self.frame_feature_extractor.inplanes
        
        # Dimension for the pre-processed features
        # [Flicker, lip_movement_variance, blink, head_movement, pulse, psnr, ssim]
        self.processed_feature_dim = 7
        
        # Create a fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(self.frame_feature_dim + self.processed_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),  # Increased dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(256, num_classes if num_classes >= 3 else 1)
        )
        
        # Initialize fusion network weights to prevent exploding gradients 
        for m in self.fusion_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)  # Lower gain for better stability
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Feature normalizers for stabilizing training
        self.frame_feature_norm = nn.LayerNorm(self.frame_feature_dim)
        self.processed_feature_norm = nn.LayerNorm(self.processed_feature_dim)
        
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight
        self.num_classes = num_classes
        self.epoch_outs = []
        
        # Add hooks for feature extraction
        self.intermediate_features = {}
        
        self.save_hyperparameters()
    
    def extract_frame_features(self, frame):
        """
        Extract frame features from the backbone
        
        Args:
            frame: Video frame tensor (B, C, H, W)
            
        Returns:
            Frame features tensor
        """
        # Preprocess the frame if needed
        if (self.frame_feature_extractor.add_magnitude_channel or 
            self.frame_feature_extractor.add_fft_channel or 
            self.frame_feature_extractor.add_lbp_channel):
            frame = self.frame_feature_extractor.add_new_channels(frame)
        
        # Apply the adapter
        frame = self.frame_feature_extractor.adapter(frame)
        
        # Extract features using the backbone
        frame_features = self.frame_feature_extractor.base_model(frame)
        
        return frame_features
    
    def forward(self, frame, processed_features):
        """
        Forward pass through the model
        
        Args:
            frame: Video frame tensor (B, C, H, W)
            processed_features: Pre-processed features tensor (B, 7)
            
        Returns:
            Dictionary containing logits
        """
        outs = {}
        
        # Extract frame features
        frame_features = self.extract_frame_features(frame)
        
        # Apply normalization to stabilize features
        frame_features = self.frame_feature_norm(frame_features)
        processed_features = self.processed_feature_norm(processed_features)
        
        # Add noise during training for better generalization
        if self.training:
            # Add small Gaussian noise to features
            frame_features = frame_features + torch.randn_like(frame_features) * 0.05
            processed_features = processed_features + torch.randn_like(processed_features) * 0.05
            
            # Randomly zero out some features (feature dropout)
            if torch.rand(1).item() < 0.3:  # 30% chance to apply feature masking
                # Create random mask for frame features (keep 80-95% of features)
                frame_mask = torch.bernoulli(torch.ones_like(frame_features) * 0.9).to(self.device)
                frame_features = frame_features * frame_mask
                
                # Create random mask for processed features (keep 80-95% of features)
                proc_mask = torch.bernoulli(torch.ones_like(processed_features) * 0.9).to(self.device)
                processed_features = processed_features * proc_mask
        
        # Store intermediate features for analysis
        self.intermediate_features['frame_features'] = frame_features
        self.intermediate_features['processed_features'] = processed_features
        
        # Concatenate frame features with processed features
        combined_features = torch.cat([frame_features, processed_features], dim=1)
        self.intermediate_features['combined_features'] = combined_features
        
        # Pass through fusion network
        outs["logits"] = self.fusion_network(combined_features)
        
        return outs
    
    def configure_optimizers(self):
        # Create optimizer for all trainable parameters
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-4  # Increased weight decay for stronger regularization
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss"
            }
        }
    
    def _step(self, batch, i_batch, phase=None):
        # Extract data from batch
        frames = batch["image"].to(self.device)
        processed_features = batch["processed_features"].to(self.device)
        
        outs = {
            "phase": phase,
            "labels": batch["is_real"][:, 0].float().to(self.device),
        }
        
        # Forward pass
        model_outputs = self(frames, processed_features)
        outs.update(model_outputs)
        
        # Calculate loss with a dynamic pos_weight based on training progress
        if self.num_classes == 2:
            # Scale down pos_weight during initial training to prevent loss explosion
            if phase == "train":
                current_epoch = self.current_epoch + 1  # epochs start at 0
                warmup_epochs = 5
                dynamic_pos_weight = self.pos_weight * min(current_epoch / warmup_epochs, 1.0)
                
                # Apply label smoothing during training for better generalization
                # Instead of using hard 0/1 labels, use 0.1/0.9 to prevent overconfidence
                smoothed_labels = outs["labels"].clone()
                smoothed_labels = smoothed_labels * 0.9 + 0.05  # Smooth labels between 0.05 and 0.95
                
                loss = F.binary_cross_entropy_with_logits(
                    input=outs["logits"][:, 0], 
                    target=smoothed_labels, 
                    pos_weight=torch.as_tensor(dynamic_pos_weight, device=self.device)
                )
            else:
                # Use normal labels for validation
                dynamic_pos_weight = self.pos_weight
                loss = F.binary_cross_entropy_with_logits(
                    input=outs["logits"][:, 0], 
                    target=outs["labels"], 
                    pos_weight=torch.as_tensor(dynamic_pos_weight, device=self.device)
                )
        else:
            raise NotImplementedError("Only binary classification is implemented!")
        
        # Clip gradients to prevent explosion
        if phase == "train":
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Transfer tensors to CPU
        for k in outs:
            if isinstance(outs[k], torch.Tensor):
                outs[k] = outs[k].detach().cpu()
        
        # Log metrics
        if phase in {"train", "val"}:
            self.log_dict({
                f"{phase}_{k}": v for k, v in [
                    ("loss", loss.detach().cpu()), 
                    ("learning_rate", self.optimizers().param_groups[0]["lr"])
                ]}, 
                prog_bar=False, logger=True
            )
        else:
            self.log_dict({
                f"{phase}_{k}": v for k, v in [
                    ("loss", loss.detach().cpu())
                ]}, 
                prog_bar=False, logger=True
            )
        
        # Save outputs
        self.epoch_outs.append(outs)
        return loss
    
    def training_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="train")
    
    def validation_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="val")
    
    def test_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="test")
    
    def on_train_epoch_start(self):
        self._on_epoch_start()
        
    def on_test_epoch_start(self):
        self._on_epoch_start()
        
    def on_train_epoch_end(self):
        self._on_epoch_end()
        
    def on_test_epoch_end(self):
        self._on_epoch_end()
        
    def _on_epoch_start(self):
        self._clear_memory()
        self.epoch_outs = []
    
    def _on_epoch_end(self):
        self._clear_memory()
        with torch.no_grad():
            labels = torch.cat([batch["labels"] for batch in self.epoch_outs], dim=0)
            logits = torch.cat([batch["logits"] for batch in self.epoch_outs], dim=0)[:, 0]
            phases = [phase for batch in self.epoch_outs for phase in [batch["phase"]] * len(batch["labels"])]
            
            assert len(labels) == len(logits), f"{len(labels)} != {len(logits)}"
            assert len(phases) == len(labels), f"{len(phases)} != {len(labels)}"
            
            for phase in ["train", "val", "test"]:
                indices_phase = [i for i in range(len(phases)) if phases[i] == phase]
                if len(indices_phase) == 0:
                    continue                
                
                # Get predictions and targets for this phase
                phase_logits = logits[indices_phase]
                phase_labels = labels[indices_phase]
                
                # Calculate predictions (binary classification)
                preds = (torch.sigmoid(phase_logits) > 0.5).float()
                
                # Calculate metrics
                metrics = {
                    "acc": accuracy(preds=phase_logits, target=phase_labels, task="binary", average="micro"),
                    "auc": auroc(preds=phase_logits, target=phase_labels.long(), task="binary", average="micro"),
                }
                
                # Calculate TP, FP, TN, FN for detailed metrics
                true_positives = torch.sum((preds == 1) & (phase_labels == 1)).item()
                false_positives = torch.sum((preds == 1) & (phase_labels == 0)).item()
                true_negatives = torch.sum((preds == 0) & (phase_labels == 0)).item()
                false_negatives = torch.sum((preds == 0) & (phase_labels == 1)).item()
                
                # Avoid division by zero
                epsilon = 1e-7
                
                # Calculate precision, recall, and F1 score
                precision = true_positives / (true_positives + false_positives + epsilon)
                recall = true_positives / (true_positives + false_negatives + epsilon)
                f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
                
                # Add these metrics to the dictionary
                metrics.update({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1_score,
                })
                
                # Log all metrics
                self.log_dict({
                    f"{phase}_{k}": v for k, v in metrics.items() if isinstance(v, (torch.Tensor, int, float))
                }, prog_bar=True, logger=True)
    
    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()


class TFRecordDatasetLoader:
    """Helper class for loading TFRecord datasets with pre-processed features"""
    
    @staticmethod
    def parse_tfrecord(example):
        """Parse a TFRecord example containing pre-processed features"""
        feature_description = {
            'filename': tf.io.FixedLenFeature([], tf.string),
            'fake': tf.io.FixedLenFeature([1], tf.int64),  # Changed from 'is_real' to 'fake'
            'flicker': tf.io.FixedLenFeature([45], tf.float32),
            'lip_movement_variance': tf.io.FixedLenFeature([5], tf.float32),
            'blink': tf.io.FixedLenFeature([5], tf.float32),
            'head_movement': tf.io.FixedLenFeature([50], tf.float32),
            'pulse': tf.io.FixedLenFeature([50], tf.float32),
            'psnr': tf.io.FixedLenFeature([45], tf.float32),
            'ssim': tf.io.FixedLenFeature([45], tf.float32),
        }
        
        return tf.io.parse_single_example(example, feature_description)
    
    @staticmethod
    def extract_features(parsed_example):
        """
        Extract features from parsed TFRecord example
        
        Returns:
            dict containing:
            - filename: video filename
            - processed_features: tensor of concatenated features
            - is_real: binary label (0=real, 1=fake)
        """
        # For each feature, take the mean to get a single value
        flicker = tf.reduce_mean(parsed_example['flicker'])
        lip_movement = tf.reduce_mean(parsed_example['lip_movement_variance'])
        blink = tf.reduce_mean(parsed_example['blink'])
        head_movement = tf.reduce_mean(parsed_example['head_movement'])
        pulse = tf.reduce_mean(parsed_example['pulse'])
        psnr = tf.reduce_mean(parsed_example['psnr'])
        ssim = tf.reduce_mean(parsed_example['ssim'])
        
        # Clip extreme values to prevent numerical instability
        # Use tf.clip_by_value to limit the range of each feature
        features_raw = [flicker, lip_movement, blink, head_movement, pulse, psnr, ssim]
        features_clipped = [tf.clip_by_value(f, -10.0, 10.0) for f in features_raw]
        
        # Min-max normalize the features to [0, 1] range after clipping
        # This helps stabilize training by keeping all features in a consistent range
        processed_features = tf.stack(features_clipped)
        
        # Convert fake to is_real (0=real, 1=fake) -> (1=real, 0=fake)
        is_real = 1 - parsed_example['fake']
        
        return {
            'filename': parsed_example['filename'],
            'processed_features': processed_features,
            'is_real': is_real
        }
    
    @staticmethod
    def load_dataset(tfrecord_path):
        """
        Load a TFRecord dataset with pre-processed features
        
        Args:
            tfrecord_path: Path to TFRecord file
            
        Returns:
            TensorFlow dataset containing processed features
        """
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        return raw_dataset.map(TFRecordDatasetLoader.parse_tfrecord).map(TFRecordDatasetLoader.extract_features)


# Custom PyTorch dataset for deepfake detection with pre-processed features
class DeepfakeVideoDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dataset, tfrecord_path, transform=None):
        """
        Dataset for deepfake detection that combines frame images with pre-processed features
        
        Args:
            frame_dataset: Dataset containing frame images
            tfrecord_path: Path to TFRecord file with pre-processed features
            transform: Optional transforms to apply to images
        """
        self.frame_dataset = frame_dataset
        self.transform = transform
        
        # Load TFRecord dataset
        tf_dataset = TFRecordDatasetLoader.load_dataset(tfrecord_path)
        
        # Convert TF dataset to list for easier indexing
        self.processed_features = []
        for item in tf_dataset:
            filename = item['filename'].numpy().decode('utf-8')
            is_real_value = item['is_real'].numpy()[0]
            # The prefix is already added in EvalDataset, so just use the original filename here
            
            self.processed_features.append({
                'filename': filename,
                'processed_features': item['processed_features'].numpy(),
                'is_real': item['is_real'].numpy()
            })
        
        # Create a mapping from filename to processed features
        self.filename_to_features = {}
        for item in self.processed_features:
            # Original filename (no need to add category prefix)
            filename = item['filename']
            # Filename without extension
            basename = os.path.splitext(filename)[0]
            
            # Determine category from is_real value
            category = 'real' if item['is_real'][0] == 1 else 'fake'
            
            feature_data = {
                'processed_features': item['processed_features'],
                'is_real': item['is_real']
            }
            
            # Store both with and without extension (original filename)
            self.filename_to_features[filename] = feature_data
            self.filename_to_features[basename] = feature_data
            
            # Also store with category prefix to match EvalDataset format
            self.filename_to_features[f"{category}_{filename}"] = feature_data
            self.filename_to_features[f"{category}_{basename}"] = feature_data
        
        print(f"Loaded {len(self.processed_features)} items from TFRecord, created {len(self.filename_to_features)} mappings")
        
        # Debug information about the first few processed features
        if len(self.processed_features) > 0:
            print("First 3 TFRecord entries:")
            for i in range(min(3, len(self.processed_features))):
                filename = self.processed_features[i]['filename']
                category = 'real' if self.processed_features[i]['is_real'][0] == 1 else 'fake'
                print(f"  Original filename: {filename} - is_real: {self.processed_features[i]['is_real'][0]}")
                print(f"  Mapping includes: {filename}, {os.path.splitext(filename)[0]}, {category}_{filename}, {category}_{os.path.splitext(filename)[0]}")
            
            # Show a sample of mapping keys
            all_keys = list(self.filename_to_features.keys())
            print(f"Sample of mapping keys (from {len(all_keys)} total):")
            for key in all_keys[:10]:
                print(f"  - {key}")
                
            # Print distribution of real/fake in mapping
            real_count = sum(1 for k, v in self.filename_to_features.items() if v['is_real'][0] == 1 and not ('_' in k))
            fake_count = sum(1 for k, v in self.filename_to_features.items() if v['is_real'][0] == 0 and not ('_' in k))
            print(f"Mapping distribution (original filenames only): {real_count} real, {fake_count} fake")
        
        # Update the frame dataset labels based on the TFRecord data
        updated_items = []
        matched_count = 0
        for idx in range(len(frame_dataset)):
            item = frame_dataset.items[idx]
            filename = item['filename']  # This should already have the category prefix
            
            # Check if filename starts with "real_" or "fake_" and strip the prefix
            # to get the original filename used in TFRecord
            if filename.startswith("real_"):
                original_filename = filename[5:]  # Remove "real_" prefix
                category = "real"
            elif filename.startswith("fake_"):
                original_filename = filename[5:]  # Remove "fake_" prefix
                category = "fake"
            else:
                # No prefix, use as is
                original_filename = filename
                # Determine category from the is_real value
                category = "real" if item['is_real'][0] == 1 else "fake"
            
            # Try different filename variations for matching, with category prefix added
            filename_variations = [
                f"{category}_{original_filename}",  # With category prefix
                f"{category}_{os.path.splitext(original_filename)[0]}",  # Without extension, with prefix
                f"{category}_{original_filename}.mp4",  # With .mp4 extension, with prefix
            ]
            
            matched = False
            for fname in filename_variations:
                if fname in self.filename_to_features:
                    # Update the is_real label from the TFRecord
                    item['is_real'] = torch.tensor(self.filename_to_features[fname]['is_real'], dtype=torch.long)
                    matched = True
                    matched_count += 1
                    break
            
            if not matched:
                print(f"Warning: Could not match {filename} to any TFRecord entry")
            
            updated_items.append(item)
        
        print(f"Updated {matched_count} out of {len(frame_dataset)} frame dataset items with TFRecord labels")
        
        # Replace the frame dataset items with the updated ones
        self.frame_dataset.items = updated_items
    
    def __len__(self):
        return len(self.frame_dataset)
    
    def __getitem__(self, idx):
        # Get frame image from the frame dataset
        frame_item = self.frame_dataset[idx]
        
        # Get filename from frame item (already has category prefix)
        filename = frame_item['filename']
        
        # Check if filename starts with "real_" or "fake_" and strip the prefix
        # to get the original filename used in TFRecord
        if filename.startswith("real_"):
            original_filename = filename[5:]  # Remove "real_" prefix
            category = "real"
        elif filename.startswith("fake_"):
            original_filename = filename[5:]  # Remove "fake_" prefix
            category = "fake"
        else:
            # No prefix, use as is
            original_filename = filename
            # Determine category from the is_real value
            category = "real" if frame_item['is_real'][0] == 1 else "fake"
        
        # Try different filename variations for matching, with category prefix added
        filename_variations = [
            f"{category}_{original_filename}",  # With category prefix
            f"{category}_{os.path.splitext(original_filename)[0]}",  # Without extension, with prefix
            f"{category}_{original_filename}.mp4",  # With .mp4 extension, with prefix
        ]
        
        # Find matching features
        matched = False
        for fname in filename_variations:
            if fname in self.filename_to_features:
                processed_features = self.filename_to_features[fname]['processed_features']
                is_real = torch.tensor(self.filename_to_features[fname]['is_real'], dtype=torch.long)
                matched = True
                break
        
        if not matched:
            # If not found, use zeros as placeholder and the label from the frame dataset
            processed_features = np.zeros(7, dtype=np.float32)
            is_real = frame_item['is_real']
            print(f"Warning: Video {filename} not found in TFRecord. Using frame dataset label.")
        
        # Apply transform if provided
        image = frame_item['image']
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'processed_features': torch.tensor(processed_features, dtype=torch.float32),
            'is_real': is_real,
            'filename': filename
        } 