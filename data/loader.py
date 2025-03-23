import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class DataLoader:
    """Class for loading and preparing crack detection datasets."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"
        
        # Create directories if they don't exist
        for directory in [self.train_dir, self.val_dir, self.test_dir]:
            (directory / "images").mkdir(exist_ok=True, parents=True)
            (directory / "masks").mkdir(exist_ok=True, parents=True)
    
    def split_dataset(self, 
                     input_images: Union[str, Path], 
                     input_masks: Union[str, Path],
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     random_state: int = 42) -> None:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            input_images: Path to input images directory
            input_masks: Path to input masks directory
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_state: Random seed for reproducibility
        """
        if train_ratio + val_ratio + test_ratio != 1.0:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        input_images = Path(input_images)
        input_masks = Path(input_masks)
        
        # Get all image files
        image_files = sorted([f for f in input_images.glob("*") 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        # Ensure corresponding masks exist
        valid_images = []
        for img_path in image_files:
            mask_path = input_masks / f"{img_path.stem}.png"
            if mask_path.exists():
                valid_images.append(img_path)
        
        # Split into train, validation, and test sets
        train_val_images, test_images = train_test_split(
            valid_images, test_size=test_ratio, random_state=random_state
        )
        
        train_images, val_images = train_test_split(
            train_val_images, 
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=random_state
        )
        
        # Copy files to respective directories
        for img_path in tqdm(train_images, desc="Copying training data"):
            shutil.copy(img_path, self.train_dir / "images" / img_path.name)
            shutil.copy(input_masks / f"{img_path.stem}.png", 
                      self.train_dir / "masks" / f"{img_path.stem}.png")
        
        for img_path in tqdm(val_images, desc="Copying validation data"):
            shutil.copy(img_path, self.val_dir / "images" / img_path.name)
            shutil.copy(input_masks / f"{img_path.stem}.png", 
                      self.val_dir / "masks" / f"{img_path.stem}.png")
        
        for img_path in tqdm(test_images, desc="Copying test data"):
            shutil.copy(img_path, self.test_dir / "images" / img_path.name)
            shutil.copy(input_masks / f"{img_path.stem}.png", 
                      self.test_dir / "masks" / f"{img_path.stem}.png")
        
        print(f"Dataset split complete: {len(train_images)} training, "
              f"{len(val_images)} validation, {len(test_images)} test images")
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as a numpy array (RGB)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image from {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """
        Load a mask from disk.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            Mask as a binary numpy array
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask from {mask_path}")
        return (mask > 0).astype(np.float32)
    
    def get_dataset_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        for split in ["train", "val", "test"]:
            split_dir = getattr(self, f"{split}_dir")
            
            images_dir = split_dir / "images"
            masks_dir = split_dir / "masks"
            
            image_count = len(list(images_dir.glob("*")))
            mask_count = len(list(masks_dir.glob("*")))
            
            stats[split] = {
                "images": image_count,
                "masks": mask_count
            }
        
        return stats