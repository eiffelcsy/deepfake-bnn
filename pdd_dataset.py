import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class PDDDataset(Dataset):
    """Dataset class for PDD (Presidential Deepfake Detection) dataset"""
    
    def __init__(self, dataset_path, split="train", resolution=224, transform=None):
        """
        Args:
            dataset_path: Path to the PDD dataset root directory
            split: 'train', 'val', or 'test' split
            resolution: Image resolution to resize to
            transform: Optional transform to be applied on a sample
        """
        super(PDDDataset, self).__init__()
        
        self.dataset_path = dataset_path
        self.split = split
        self.resolution = resolution
        
        # Create default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load the dataset structure
        self.items = self._load_dataset()
        
    def _load_dataset(self):
        """
        Loads dataset structure, creating a list of items with paths and labels
        
        Returns:
            list of dictionaries containing:
                - image_path: path to the frame image
                - is_real: binary label (1=real, 0=fake)
                - filename: video identifier string (to match with tfrecord)
        """
        items = []
        
        # Construct the path to the split directory
        split_dir = os.path.join(self.dataset_path, self.split)
        
        # Check if directory exists
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Each video has its own directory containing frames
        for video_id in os.listdir(split_dir):
            video_dir = os.path.join(split_dir, video_id)
            
            # Skip non-directory items
            if not os.path.isdir(video_dir):
                continue
                
            # Find all frame images in the video directory
            for frame_file in os.listdir(video_dir):
                if frame_file.endswith(('.jpg', '.png', '.jpeg')):
                    items.append({
                        'image_path': os.path.join(video_dir, frame_file),
                        'is_real': torch.tensor([1], dtype=torch.long),  # Placeholder value, will be updated from TFRecord
                        'filename': video_id
                    })
        
        print(f"Loaded {len(items)} frame images for {self.split} split.")
        
        return items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load and transform the image
        try:
            image = Image.open(item['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            # Return a blank image in case of errors
            image = torch.zeros(3, self.resolution, self.resolution)
        
        return {
            'image': image,
            'is_real': item['is_real'],
            'filename': item['filename']
        } 