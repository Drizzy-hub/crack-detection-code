import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union, List, Optional

class ImagePreprocessor:
    """Class for preprocessing construction images for crack detection."""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
    
    def read_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Read an image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as a numpy array
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image from {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                          interpolation=cv2.INTER_AREA)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1].
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            return enhanced
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to the image.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def extract_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Extract edges from the image using Canny edge detector.
        
        Args:
            image: Input image
            
        Returns:
            Edge map
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges
    
    def preprocess(self, image: Union[np.ndarray, str, Path], 
                  enhance: bool = True,
                  denoise_img: bool = True) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image: Input image or path to image
            enhance: Whether to apply contrast enhancement
            denoise_img: Whether to apply denoising
            
        Returns:
            Preprocessed image
        """
        if isinstance(image, (str, Path)):
            image = self.read_image(image)
            
        if denoise_img:
            image = self.denoise(image)
            
        if enhance:
            image = self.enhance_contrast(image)
            
        image = self.resize(image)
        image = self.normalize(image)
        
        return image