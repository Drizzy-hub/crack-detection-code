import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CrackProperties:
    """Class to store crack properties."""
    area: float
    length: float
    width_avg: float
    width_max: float
    orientation: float
    severity: str

class CrackAnalyzer:
    """Class for analyzing detected cracks in images."""
    
    def __init__(self, pixel_mm_ratio: float = 1.0):
        """
        Initialize the crack analyzer.
        
        Args:
            pixel_mm_ratio: Ratio of pixels to millimeters
        """
        self.pixel_mm_ratio = pixel_mm_ratio
    
    def extract_contours(self, mask: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
        """
        Extract contours from a binary mask.
        
        Args:
            mask: Binary mask of cracks
            min_area: Minimum contour area to consider
            
        Returns:
            List of contours
        """
        # Ensure mask is binary
        if mask.max() > 1:
            mask = mask / 255
        
        # Convert to uint8 for OpenCV functions
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return filtered_contours
    
    def calculate_crack_width(self, mask: np.ndarray, contour: np.ndarray) -> Tuple[float, float]:
        """
        Calculate average and maximum width of a crack.
        
        Args:
            mask: Binary mask of cracks
            contour: Contour of the crack
            
        Returns:
            Tuple of (average_width, maximum_width)
        """
        # Create a skeletonized version of the crack
        skeleton = self._skeletonize(mask)
        
        # Create a distance transform
        dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        
        # Calculate widths at skeleton points
        widths = []
        y, x = np.where(skeleton > 0)
        
        for i, j in zip(y, x):
            width = dist_transform[i, j] * 2  # Multiply by 2 because distance is to edge
            widths.append(width)
        
        if not widths:
            return 0, 0
        
        avg_width = np.mean(widths) * self.pixel_mm_ratio
        max_width = np.max(widths) * self.pixel_mm_ratio
        
        return avg_width, max_width
    
    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """
        Create a skeleton of the mask using morphological operations.
        
        Args:
            mask: Binary mask
            
        Returns:
            Skeletonized mask
        """
        mask_uint8 = mask.astype(np.uint8)
        skeleton = np.zeros_like(mask_uint8)
        
        # Create a copy of the mask
        img = mask_uint8.copy()
        
        # Get a kernel for erosion and dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        # Skeletonize
        while True:
            # Erode the image
            eroded = cv2.erode(img, kernel)
            
            # Dilate the eroded image
            dilated = cv2.dilate(eroded, kernel)
            
            # Get the difference between the original and the dilated image
            diff = cv2.subtract(img, dilated)
            
            # Add the difference to the skeleton
            skeleton = cv2.bitwise_or(skeleton, diff)
            
            # Set the eroded image as the new image
            img = eroded.copy()
            
            # If the image is all zeros, we're done
            if cv2.countNonZero(img) == 0:
                break
        
        return skeleton
    
    def calculate_crack_length(self, skeleton: np.ndarray) -> float:
        """
        Calculate the length of a crack from its skeleton.
        
        Args:
            skeleton: Skeletonized mask of the crack
            
        Returns:
            Length of the crack in millimeters
        """
        # Count non-zero pixels in the skeleton
        length_pixels = np.sum(skeleton > 0)
        
        # Convert to millimeters
        length_mm = length_pixels * self.pixel_mm_ratio
        
        return length_mm
    
    def calculate_orientation(self, contour: np.ndarray) -> float:
        """
        Calculate the principal orientation of a crack.
        
        Args:
            contour: Contour of the crack
            
        Returns:
            Orientation in degrees (0-180)
        """
        if len(contour) < 5:  # Need at least 5 points for ellipse fitting
            return 0
        
        try:
            _, (_, _), angle = cv2.fitEllipse(contour)
            return angle
        except:
            return 0
    
    def classify_severity(self, width_mm: float) -> str:
        """
        Classify crack severity based on its width.
        
        Args:
            width_mm: Width of the crack in millimeters
            
        Returns:
            Severity classification as string
        """
        if width_mm < 0.1:
            return "Hairline"
        elif width_mm < 0.3:
            return "Fine"
        elif width_mm < 1.0:
            return "Medium"
        elif width_mm < 2.0:
            return "Wide"
        else:
            return "Severe"
    
    def analyze_crack(self, mask: np.ndarray, contour: np.ndarray) -> CrackProperties:
        """
        Analyze properties of a single crack.
        
        Args:
            mask: Binary mask of the crack
            contour: Contour of the crack
            
        Returns:
            CrackProperties object with analysis results
        """
        # Calculate area
        area_pixels = cv2.contourArea(contour)
        area_mm2 = area_pixels * (self.pixel_mm_ratio ** 2)
        
        # Create an individual mask for this contour
        individual_mask = np.zeros_like(mask)
        cv2.drawContours(individual_mask, [contour], 0, 1, -1)
        
        # Skeletonize for length calculation
        skeleton = self._skeletonize(individual_mask)
        
        # Calculate length
        length_mm = self.calculate_crack_length(skeleton)
        
        # Calculate width
        avg_width, max_width = self.calculate_crack_width(individual_mask, contour)
        
        # Calculate orientation
        orientation = self.calculate_orientation(contour)
        
        # Classify severity
        severity = self.classify_severity(avg_width)
        
        return CrackProperties(
            area=area_mm2,
            length=length_mm,
            width_avg=avg_width,
            width_max=max_width,
            orientation=orientation,
            severity=severity
        )
    
    def analyze_mask(self, mask: np.ndarray, min_area: int = 100) -> List[CrackProperties]:
        """
        Analyze all cracks in a mask.
        
        Args:
            mask: Binary mask of cracks
            min_area: Minimum contour area to consider
            
        Returns:
            List of CrackProperties for each crack
        """
        contours = self.extract_contours(mask, min_area)
        results = []
        
        for contour in contours:
            crack_props = self.analyze_crack(mask, contour)
            results.append(crack_props)
        
        return results
    
    def visualize_analysis(self, image: np.ndarray, mask: np.ndarray, 
                         properties_list: List[CrackProperties]) -> np.ndarray:
        """
        Visualize the crack analysis results on the image.
        
        Args:
            image: Original image
            mask: Binary mask of cracks
            properties_list: List of CrackProperties for each crack
            
        Returns:
            Annotated image with crack analysis
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Convert to BGR if in RGB format
        if vis_image.shape[2] == 3 and vis_image.dtype == np.float32:
            vis_image = (vis_image * 255).astype(np.uint8)
        
        # Extract contours
        contours = self.extract_contours(mask)
        
        # Draw contours
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
        
        # Add text annotations for each crack
        for i, (contour, props) in enumerate(zip(contours, properties_list)):
            # Get centroid of contour
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Add text for crack properties
            text = f"#{i+1}: {props.severity} crack"
            cv2.putText(vis_image, text, (cx, cy - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            text = f"W: {props.width_avg:.2f}mm L: {props.length:.1f}mm"
            cv2.putText(vis_image, text, (cx, cy), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return vis_image
