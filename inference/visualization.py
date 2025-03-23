import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import cv2

from crackdetect.utils.crack_analysis import CrackProperties

def create_result_figure(results: Dict, 
                        save_path: Optional[Union[str, Path]] = None,
                        show: bool = True) -> None:
    """
    Create a visualization of crack detection and analysis results.
    
    Args:
        results: Results dictionary from Predictor.analyze_image
        save_path: Path to save the figure (if None, no saving)
        show: Whether to show the figure
    """
    # Extract data from results
    original_image = results['original_image']
    processed_image = results['processed_image']
    prediction_mask = results['prediction_mask']
    result_image = results['result_image']
    crack_properties = results['crack_properties']
    
    # Create figure
    plt.figure(figsize=(18, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')
    
    # Processed image
    plt.subplot(2, 2, 2)
    plt.title("Processed Image")
    plt.imshow(processed_image)
    plt.axis('off')
    
    # Prediction mask
    plt.subplot(2, 2, 3)
    plt.title("Crack Detection Mask")
    plt.imshow(prediction_mask, cmap='gray')
    plt.axis('off')
    
    # Analysis result
    plt.subplot(2, 2, 4)
    plt.title("Crack Analysis")
    plt.imshow(result_image)
    plt.axis('off')
    
    # Add text summary
    num_cracks = len(crack_properties)
    summary_text = f"Detected {num_cracks} cracks\n"
    
    if num_cracks > 0:
        # Calculate statistics
        widths = [prop.width_avg for prop in crack_properties]
        lengths = [prop.length for prop in crack_properties]
        severities = [prop.severity for prop in crack_properties]
        
        # Count by severity
        severity_counts = {}
        for severity in severities:
            if severity in severity_counts:
                severity_counts[severity] += 1
            else:
                severity_counts[severity] = 1
        
        # Add to summary
        summary_text += f"Average width: {np.mean(widths):.2f} mm\n"
        summary_text += f"Max width: {np.max(widths) if widths else 0:.2f} mm\n"
        summary_text += f"Total length: {np.sum(lengths):.2f} mm\n"
        summary_text += "Severity breakdown:\n"
        
        for severity, count in severity_counts.items():
            summary_text += f"  - {severity}: {count}\n"
    
    plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=12, 
               bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()

def create_overlay_image(image: np.ndarray, 
                        mask: np.ndarray, 
                        color: Tuple[int, int, int] = (0, 255, 0),
                        alpha: float = 0.5) -> np.ndarray:
    """
    Create an overlay of the mask on the image.
    
    Args:
        image: Input image
        mask: Binary mask
        color: RGB color for the overlay
        alpha: Transparency of the overlay
        
    Returns:
        Image with overlay
    """
    # Ensure the image is in the correct format
    if image.dtype == np.float32 and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create a colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Create the overlay
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    return overlay

def create_severity_colormap(crack_properties: List[CrackProperties], 
                           image: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
    """
    Create a colormap of crack severity.
    
    Args:
        crack_properties: List of CrackProperties
        image: Input image
        mask: Binary mask
        
    Returns:
        Image with severity colormap
    """
    # Ensure the image is in the correct format
    if image.dtype == np.float32 and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create a colored mask for severity
    severity_mask = np.zeros_like(image)
    
    # Define colors for severity levels (from green to red)
    severity_colors = {
        "Hairline": (0, 255, 0),    # Green
        "Fine": (0, 255, 255),      # Yellow
        "Medium": (0, 165, 255),    # Orange
        "Wide": (0, 0, 255),        # Red
        "Severe": (0, 0, 128)       # Dark Red
    }
    
    # Create individual masks for each crack
    for i, props in enumerate(crack_properties):
        # Create a mask for this specific crack
        contours = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)[0]
        
        # Skip if no contours are found
        if not contours:
            continue
        
        # Sort contours by area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get the color for this crack's severity
        color = severity_colors.get(props.severity, (255, 255, 255))
        
        # Draw the contour on the severity mask
        cv2.drawContours(severity_mask, [contours[i % len(contours)]], 0, color, -1)
    
    # Create the overlay
    severity_overlay = cv2.addWeighted(image, 1, severity_mask, 0.7, 0)
    
    # Add a legend
    legend_start_y = 30
    for i, (severity, color) in enumerate(severity_colors.items()):
        y_pos = legend_start_y + i * 25
        cv2.putText(severity_overlay, severity, (20, y_pos), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return severity_overlay