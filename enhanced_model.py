import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional.classification import accuracy, auroc
import tensorflow as tf
import numpy as np

from model_architecture import BNext4DFR

class DeepfakeVideoClassifier(L.LightningModule):
    def __init__(self, num_classes=2, backbone='BNext-T', 
                 freeze_backbone=True, add_magnitude_channel=True, 
                 add_fft_channel=True, add_lbp_channel=True,
                 learning_rate=1e-4, pos_weight=1.):
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
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes if num_classes >= 3 else 1)
        )
        
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight
        self.num_classes = num_classes
        self.epoch_outs = []
        
        self.save_hyperparameters()
    
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
        
        # Extract frame features using BNext4DFR
        frame_features = self.frame_feature_extractor.base_model(
            self.frame_feature_extractor.adapter(
                self.frame_feature_extractor.add_new_channels(frame) 
                if (self.frame_feature_extractor.add_magnitude_channel or 
                    self.frame_feature_extractor.add_fft_channel or 
                    self.frame_feature_extractor.add_lbp_channel) 
                else frame
            )
        )
        
        # Concatenate frame features with processed features
        combined_features = torch.cat([frame_features, processed_features], dim=1)
        
        # Pass through fusion network
        outs["logits"] = self.fusion_network(combined_features)
        
        return outs
    
    def configure_optimizers(self):
        # Create optimizer for all trainable parameters
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1., end_factor=0.1, total_iters=5
        )
        return [optimizer], [scheduler]
    
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
        
        # Calculate loss
        if self.num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(
                input=outs["logits"][:, 0], 
                target=outs["labels"], 
                pos_weight=torch.as_tensor(self.pos_weight, device=self.device)
            )
        else:
            raise NotImplementedError("Only binary classification is implemented!")
        
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
                metrics = {
                    "acc": accuracy(preds=logits[indices_phase], target=labels[indices_phase], task="binary", average="micro"),
                    "auc": auroc(preds=logits[indices_phase], target=labels[indices_phase].long(), task="binary", average="micro"),
                }
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
        
        # Combine all features into a single tensor
        processed_features = tf.stack([
            flicker, lip_movement, blink, head_movement, 
            pulse, psnr, ssim
        ])
        
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
            self.processed_features.append({
                'filename': item['filename'].numpy().decode('utf-8'),
                'processed_features': item['processed_features'].numpy(),
                'is_real': item['is_real'].numpy()
            })
        
        # Create a mapping from filename to processed features
        self.filename_to_features = {
            item['filename']: {
                'processed_features': item['processed_features'],
                'is_real': item['is_real']
            } for item in self.processed_features
        }
        
        # Update the frame dataset labels based on the TFRecord data
        updated_items = []
        for idx in range(len(frame_dataset)):
            item = frame_dataset.items[idx]
            filename = item['filename']
            
            if filename in self.filename_to_features:
                # Update the is_real label from the TFRecord
                item['is_real'] = torch.tensor(self.filename_to_features[filename]['is_real'], dtype=torch.long)
            
            updated_items.append(item)
        
        # Replace the frame dataset items with the updated ones
        self.frame_dataset.items = updated_items
    
    def __len__(self):
        return len(self.frame_dataset)
    
    def __getitem__(self, idx):
        # Get frame image from the frame dataset
        frame_item = self.frame_dataset[idx]
        
        # Get filename from frame item
        filename = frame_item['filename']
        
        # Get corresponding processed features
        if filename in self.filename_to_features:
            processed_features = self.filename_to_features[filename]['processed_features']
            is_real = torch.tensor(self.filename_to_features[filename]['is_real'], dtype=torch.long)
        else:
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