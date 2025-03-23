import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional, Union
import cv2
import albumentations as A

from crackdetect.data.preprocessing import ImagePreprocessor

class CrackDataset(Dataset):
    """Dataset for crack detection and segmentation."""
    
    def __init__(self, 
                 image_dir: Union[str, Path], 
                 mask_dir: Optional[Union[str, Path]] = None,
                 image_size: Tuple[int, int] = (512, 512),
                 transform: Optional[A.Compose] = None,
                 preprocessing: bool = True):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing mask annotations (if None, inference mode)
            image_size: Target image size
            transform: Albumentations transformations to apply
            preprocessing: Whether to apply standard preprocessing
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.preprocessing = preprocessing
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(target_size=image_size)
        
        # Get image paths
        self.image_paths = sorted(
            [p for p in self.image_dir.glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        )
        
        # Training mode (with masks) or inference mode
        self.train_mode = self.mask_dir is not None
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image path
        image_path = self.image_paths[idx]
        
        # Read and preprocess image
        image = self.preprocessor.read_image(image_path)
        
        if self.preprocessing:
            image = self.preprocessor.preprocess(image)
        
        # For training mode, get corresponding mask
        if self.train_mode:
            mask_path = self.mask_dir / f"{image_path.stem}.png"
            
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found for {image_path.name}")
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.preprocessor.target_size[1], self.preprocessor.target_size[0]))
            mask = (mask > 0).astype(np.float32)
            
            # Apply transformations
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            # Convert to tensor
            if len(image.shape) == 3:  # Color image
                image = image.transpose(2, 0, 1)
            else:  # Grayscale
                image = np.expand_dims(image, 0)
            
            mask = np.expand_dims(mask, 0)
            
            return {
                'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask),
                'filename': image_path.name
            }
        
        # For inference mode, return only the image
        else:
            # Apply transformations
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            # Convert to tensor
            if len(image.shape) == 3:  # Color image
                image = image.transpose(2, 0, 1)
            else:  # Grayscale
                image = np.expand_dims(image, 0)
            
            return {
                'image': torch.from_numpy(image),
                'filename': image_path.name
            }
