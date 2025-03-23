import pytest
import numpy as np
import cv2
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crackdetect.utils.crack_analysis import CrackAnalyzer, CrackProperties

class TestCrackAnalyzer:
    def setup_method(self):
        self.analyzer = CrackAnalyzer(pixel_mm_ratio=1.0)
        
        # Create a test mask with a simple crack
        self.test_mask = np.zeros((100, 100), dtype=np.float32)
        cv2.line(self.test_mask, (20, 50), (80, 50), 1.0, 5)
    
    def test_extract_contours(self):
        contours = self.analyzer.extract_contours(self.test_mask)
        
        # Should have one contour
        assert len(contours) == 1
        
        # Contour should have significant area
        assert cv2.contourArea(contours[0]) > 100
    
    def test_calculate_crack_width(self):
        contours = self.analyzer.extract_contours(self.test_mask)
        avg_width, max_width = self.analyzer.calculate_crack_width(self.test_mask, contours[0])
        
        # Width should be approximately 5 pixels (the line width)
        assert 4.5 <= avg_width <= 5.5
        assert 4.5 <= max_width <= 5.5
    
    def test_calculate_crack_length(self):
        # Create a skeleton of the mask
        skeleton = self.analyzer._skeletonize(self.test_mask)
        length = self.analyzer.calculate_crack_length(skeleton)
        
        # Length should be approximately 60 pixels (80 - 20)
        assert 55 <= length <= 65
    
    def test_calculate_orientation(self):
        contours = self.analyzer.extract_contours(self.test_mask)
        orientation = self.analyzer.calculate_orientation(contours[0])
        
        # Orientation should be approximately 0 or 180 degrees (horizontal line)
        assert (0 <= orientation <= 10) or (170 <= orientation <= 180)
    
    def test_classify_severity(self):
        # Test different widths
        assert self.analyzer.classify_severity(0.05) == "Hairline"
        assert self.analyzer.classify_severity(0.2) == "Fine"
        assert self.analyzer.classify_severity(0.5) == "Medium"
        assert self.analyzer.classify_severity(1.5) == "Wide"
        assert self.analyzer.classify_severity(2.5) == "Severe"
    
    def test_analyze_crack(self):
        contours = self.analyzer.extract_contours(self.test_mask)
        props = self.analyzer.analyze_crack(self.test_mask, contours[0])
        
        # Check that properties are calculated
        assert isinstance(props, CrackProperties)
        assert props.area > 0
        assert props.length > 0
        assert props.width_avg > 0
        assert props.width_max > 0
        assert 0 <= props.orientation <= 180
        assert props.severity in ["Hairline", "Fine", "Medium", "Wide", "Severe"]
    
    def test_analyze_mask(self):
        props_list = self.analyzer.analyze_mask(self.test_mask)
        
        # Should have one crack
        assert len(props_list) == 1
        
        # Check properties
        props = props_list[0]
        assert props.area > 0
        assert props.length > 0
        assert props.width_avg > 0
        assert props.width_max > 0
    
    def test_pixel_mm_ratio(self):
        # Test with different pixel_mm_ratio values
        self.analyzer.pixel_mm_ratio = 0.1
        props_list_01 = self.analyzer.analyze_mask(self.test_mask)
        
        self.analyzer.pixel_mm_ratio = 0.5
        props_list_05 = self.analyzer.analyze_mask(self.test_mask)
        
        # Area should scale quadratically with pixel_mm_ratio
        assert abs(props_list_05[0].area / props_list_01[0].area - 25) < 1
        
        # Length should scale linearly with pixel_mm_ratio
        assert abs(props_list_05[0].length / props_list_01[0].length - 5) < 1
        
        # Width should scale linearly with pixel_mm_ratio
        assert abs(props_list_05[0].width_avg / props_list_01[0].width_avg - 5) < 1
