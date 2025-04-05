import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class EvalDataset(Dataset):
    """Dataset class for evaluation dataset with real/fake videos"""
    
    def __init__(self, dataset_path, resolution=224, transform=None):
        """
        Args:
            dataset_path: Path to the evaluation dataset root directory
            resolution: Image resolution to resize to
            transform: Optional transform to be applied on a sample
        """
        super(EvalDataset, self).__init__()
        
        self.dataset_path = dataset_path
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
        
        # Load real frames
        real_frames_dir = os.path.join(self.dataset_path, 'real')
        if os.path.exists(real_frames_dir):
            for video_id in os.listdir(real_frames_dir):
                video_dir = os.path.join(real_frames_dir, video_id)
                
                # Skip non-directory items
                if not os.path.isdir(video_dir):
                    continue
                    
                # Find all frame images in the video directory
                for frame_file in os.listdir(video_dir):
                    if frame_file.endswith(('.jpg', '.png', '.jpeg')):
                        items.append({
                            'image_path': os.path.join(video_dir, frame_file),
                            'is_real': torch.tensor([1], dtype=torch.long),  # Real = 1
                            'filename': f"real_{video_id}"  # Add "real_" prefix
                        })
        
        # Load fake frames
        fake_frames_dir = os.path.join(self.dataset_path, 'fake')
        if os.path.exists(fake_frames_dir):
            for video_id in os.listdir(fake_frames_dir):
                video_dir = os.path.join(fake_frames_dir, video_id)
                
                # Skip non-directory items
                if not os.path.isdir(video_dir):
                    continue
                    
                # Find all frame images in the video directory
                for frame_file in os.listdir(video_dir):
                    if frame_file.endswith(('.jpg', '.png', '.jpeg')):
                        items.append({
                            'image_path': os.path.join(video_dir, frame_file),
                            'is_real': torch.tensor([0], dtype=torch.long),  # Fake = 0
                            'filename': f"fake_{video_id}"  # Add "fake_" prefix
                        })
        
        print(f"Loaded {len(items)} frame images for the evaluation dataset")
        print(f"Real samples: {sum(1 for item in items if item['is_real'][0] == 1)}")
        print(f"Fake samples: {sum(1 for item in items if item['is_real'][0] == 0)}")
        
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