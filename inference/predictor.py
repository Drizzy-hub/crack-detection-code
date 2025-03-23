import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import cv2
import time

from crackdetect.models.segmentation import UNet
from crackdetect.data.preprocessing import ImagePreprocessor
from crackdetect.utils.crack_analysis import CrackAnalyzer, CrackProperties

class Predictor:
    """Class for making predictions with a trained crack detection model."""
    
    def __init__(self, 
                model_path: Union[str, Path], 
                device: Optional[torch.device] = None,
                confidence_threshold: float = 0.5,
                pixel_mm_ratio: float = 1.0,
                min_crack_area: int = 100):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model weights
            device: Device to run inference on
            confidence_threshold: Confidence threshold for binary prediction
            pixel_mm_ratio: Ratio of pixels to millimeters
            min_crack_area: Minimum crack area to consider (in pixels)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # Initialize crack analyzer
        self.analyzer = CrackAnalyzer(pixel_mm_ratio=pixel_mm_ratio)
        self.min_crack_area = min_crack_area
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self) -> torch.nn.Module:
        """
        Load model from disk.
        
        Returns:
            Loaded PyTorch model
        """
        model = UNet(in_channels=3, out_channels=1)
        
        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image: Union[np.ndarray, str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image or path to image
            
        Returns:
            Tuple of (preprocessed image, tensor for model input)
        """
        # Load image if needed
        if isinstance(image, (str, Path)):
            image = self.preprocessor.read_image(image)
        
        # Preprocess image
        processed_image = self.preprocessor.preprocess(image.copy())
        
        # Prepare tensor for model
        tensor = torch.from_numpy(processed_image.transpose(2, 0, 1)).unsqueeze(0).float()
        
        return processed_image, tensor
    
    def predict_mask(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Predict crack mask for an image.
        
        Args:
            image: Input image or path to image
            
        Returns:
            Binary prediction mask
        """
        # Preprocess image
        processed_image, tensor = self.preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
            
        # Convert prediction to mask
        pred_mask = output.squeeze().cpu().numpy() > self.confidence_threshold
        
        return pred_mask
    
    def analyze_image(self, image: Union[np.ndarray, str, Path]) -> Dict:
        """
        Analyze cracks in an image.
        
        Args:
            image: Input image or path to image
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        # Load image if needed
        if isinstance(image, (str, Path)):
            original_image = self.preprocessor.read_image(image)
            image_path = Path(image).name
        else:
            original_image = image.copy()
            image_path = "input_image"
        
        # Preprocess image and predict mask
        processed_image, tensor = self.preprocess_image(original_image)
        
        # Time the prediction
        pred_start = time.time()
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
        pred_time = time.time() - pred_start
        
        # Convert prediction to mask
        pred_mask = output.squeeze().cpu().numpy() > self.confidence_threshold
        
        # Analyze cracks
        analysis_start = time.time()
        crack_properties = self.analyzer.analyze_mask(pred_mask, min_area=self.min_crack_area)
        analysis_time = time.time() - analysis_start
        
        # Visualize results
        result_image = self.analyzer.visualize_analysis(processed_image, pred_mask, crack_properties)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Prepare results
        results = {
            "image_path": image_path,
            "original_image": original_image,
            "processed_image": processed_image,
            "prediction_mask": pred_mask,
            "crack_properties": crack_properties,
            "result_image": result_image,
            "timing": {
                "prediction": pred_time,
                "analysis": analysis_time,
                "total": total_time
            },
            "metadata": {
                "num_cracks": len(crack_properties),
                "model_path": str(self.model_path),
                "confidence_threshold": self.confidence_threshold,
                "pixel_mm_ratio": self.analyzer.pixel_mm_ratio
            }
        }
        
        return results