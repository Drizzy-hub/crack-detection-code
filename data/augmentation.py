import albumentations as A
from typing import Dict, Tuple, Optional, List

def get_training_augmentation(height: int = 512, width: int = 512) -> A.Compose:
    """
    Get augmentation pipeline for training.
    
    Args:
        height: Target height
        width: Target width
        
    Returns:
        Augmentation pipeline
    """
    return A.Compose([
        A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1.0, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ], p=0.6),
        A.OneOf([
            A.GaussianBlur(blur_limit=7, p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
            A.MedianBlur(blur_limit=7, p=0.5),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.5),
        ], p=0.2),
    ])

def get_validation_augmentation(height: int = 512, width: int = 512) -> A.Compose:
    """
    Get augmentation pipeline for validation.
    
    Args:
        height: Target height
        width: Target width
        
    Returns:
        Augmentation pipeline
    """
    return A.Compose([
        A.Resize(height=height, width=width, p=1.0),
    ])

def get_preprocessing(mean: Optional[List[float]] = None, 
                    std: Optional[List[float]] = None) -> A.Compose:
    """
    Get preprocessing transforms.
    
    Args:
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Preprocessing pipeline
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
        
    preprocessing = [
        A.Normalize(mean=mean, std=std),
    ]
    
    return A.Compose(preprocessing)