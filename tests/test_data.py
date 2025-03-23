import pytest
import numpy as np
import cv2
from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crackdetect.data.preprocessing import ImagePreprocessor
from crackdetect.data.dataset import CrackDataset

class TestImagePreprocessor:
    def setup_method(self):
        self.preprocessor = ImagePreprocessor(target_size=(256, 256))
        
        # Create a test image (black square with a white crack-like line)
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.line(self.test_image, (50, 150), (250, 150), (255, 255, 255), 5)
    
    def test_resize(self):
        resized = self.preprocessor.resize(self.test_image)
        assert resized.shape[:2] == (256, 256)
    
    def test_normalize(self):
        normalized = self.preprocessor.normalize(self.test_image)
        assert normalized.dtype == np.float32
        assert 0 <= normalized.min() <= normalized.max() <= 1.0
    
    def test_enhance_contrast(self):
        enhanced = self.preprocessor.enhance_contrast(self.test_image)
        assert enhanced.shape == self.test_image.shape
        
        # Should maintain the same dtype
        assert enhanced.dtype == self.test_image.dtype
    
    def test_denoise(self):
        denoised = self.preprocessor.denoise(self.test_image)
        assert denoised.shape == self.test_image.shape
        
        # Should maintain the same dtype
        assert denoised.dtype == self.test_image.dtype
    
    def test_extract_edges(self):
        edges = self.preprocessor.extract_edges(self.test_image)
        assert edges.shape[:2] == self.test_image.shape[:2]
        assert edges.dtype == np.uint8
        
        # Should contain some edges (the white line)
        assert np.sum(edges) > 0
    
    def test_preprocess(self):
        processed = self.preprocessor.preprocess(self.test_image)
        
        # Check shape and dtype
        assert processed.shape[:2] == (256, 256)
        assert processed.dtype == np.float32
        assert 0 <= processed.min() <= processed.max() <= 1.0

class TestCrackDataset:
    def setup_method(self):
        # Create temporary directory structure for testing
        self.test_dir = Path("test_data")
        self.images_dir = self.test_dir / "images"
        self.masks_dir = self.test_dir / "masks"
        
        self.images_dir.mkdir(exist_ok=True, parents=True)
        self.masks_dir.mkdir(exist_ok=True, parents=True)
        
        # Create test images and masks
        for i in range(3):
            # Create a test image
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.line(img, (20, 50), (80, 50), (255, 255, 255), 2)
            cv2.imwrite(str(self.images_dir / f"image_{i}.jpg"), img)
            
            # Create a test mask
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.line(mask, (20, 50), (80, 50), 255, 2)
            cv2.imwrite(str(self.masks_dir / f"image_{i}.png"), mask)
    
    def teardown_method(self):
        # Remove test directories
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_dataset_length(self):
        dataset = CrackDataset(
            image_dir=self.images_dir,
            mask_dir=self.masks_dir,
            image_size=(64, 64),
            transform=None,
            preprocessing=True
        )
        
        assert len(dataset) == 3
    
    def test_dataset_getitem(self):
        dataset = CrackDataset(
            image_dir=self.images_dir,
            mask_dir=self.masks_dir,
            image_size=(64, 64),
            transform=None,
            preprocessing=True
        )
        
        # Get an item
        item = dataset[0]
        
        # Check keys
        assert 'image' in item
        assert 'mask' in item
        assert 'filename' in item
        
        # Check shapes and types
        assert item['image'].shape[0] == 3  # RGB channels
        assert item['image'].shape[1:] == (64, 64)  # Image size
        assert item['mask'].shape[0] == 1  # Single channel
        assert item['mask'].shape[1:] == (64, 64)  # Image size
        
        assert item['image'].dtype == torch.float32
        assert item['mask'].dtype == torch.float32
        
        # Filename should be a string
        assert isinstance(item['filename'], str)
    
    def test_inference_mode(self):
        dataset = CrackDataset(
            image_dir=self.images_dir,
            mask_dir=None,  # No mask directory for inference mode
            image_size=(64, 64),
            transform=None,
            preprocessing=True
        )
        
        # Get an item
        item = dataset[0]
        
        # Check keys
        assert 'image' in item
        assert 'mask' not in item
        assert 'filename' in item
        
        # Check shapes and types
        assert item['image'].shape[0] == 3  # RGB channels
        assert item['image'].shape[1:] == (64, 64)  # Image size
        
        assert item['image'].dtype == torch.float32
        
        # Filename should be a string
        assert isinstance(item['filename'], str)