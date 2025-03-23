import pytest
import torch
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crackdetect.models.segmentation import UNet
from models.deeplab import DeepLabV3Plus

class TestUNet:
    def setup_method(self):
        self.model = UNet(in_channels=3, out_channels=1)
        self.input_size = (1, 3, 256, 256)  # (batch_size, channels, height, width)
    
    def test_forward(self):
        # Create a random input tensor
        x = torch.randn(self.input_size)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        assert output.shape == (1, 1, 256, 256)
          # Check output range (should be between 0 and 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_encoder_blocks(self):
        # Test that all encoder blocks work independently
        x = torch.randn(self.input_size)
        
        # First encoder block
        enc1 = self.model.enc1(x)
        assert enc1.shape == (1, 64, 256, 256)
        
        # Second encoder block (after pooling)
        x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(enc1)
        enc2 = self.model.enc2(x2)
        assert enc2.shape == (1, 128, 128, 128)
        
        # Third encoder block (after pooling)
        x3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(enc2)
        enc3 = self.model.enc3(x3)
        assert enc3.shape == (1, 256, 64, 64)
        
        # Fourth encoder block (after pooling)
        x4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(enc3)
        enc4 = self.model.enc4(x4)
        assert enc4.shape == (1, 512, 32, 32)
    
    def test_bridge(self):
        # Test the bridge
        x = torch.randn(1, 512, 32, 32)
        bridge = self.model.bridge(x)
        assert bridge.shape == (1, 1024, 32, 32)
    
    def test_decoder_blocks(self):
        # Create sample tensors for testing decoder blocks
        bridge = torch.randn(1, 1024, 32, 32)
        enc4 = torch.randn(1, 512, 32, 32)
        enc3 = torch.randn(1, 256, 64, 64)
        enc2 = torch.randn(1, 128, 128, 128)
        enc1 = torch.randn(1, 64, 256, 256)
        
        # Test decoder blocks
        dec4 = self.model.dec4(bridge, enc4)
        assert dec4.shape == (1, 512, 64, 64)
        
        dec3 = self.model.dec3(dec4, enc3)
        assert dec3.shape == (1, 256, 128, 128)
        
        dec2 = self.model.dec2(dec3, enc2)
        assert dec2.shape == (1, 128, 256, 256)
        
        dec1 = self.model.dec1(dec2, enc1)
        assert dec1.shape == (1, 64, 256, 256)
    
    def test_final_layer(self):
        # Test the final layer
        x = torch.randn(1, 64, 256, 256)
        final = self.model.final(x)
        assert final.shape == (1, 1, 256, 256)

class TestDeepLab:
    def setup_method(self):
        try:
            # Skip test if torchvision is not available
            import torchvision
            self.model = DeepLabV3Plus(in_channels=3, out_channels=1, backbone="resnet50", pretrained=False)
            self.input_size = (1, 3, 256, 256)  # (batch_size, channels, height, width)
        except ImportError:
            pytest.skip("torchvision not available")
    
    def test_forward(self):
        # Create a random input tensor
        x = torch.randn(self.input_size)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        assert output.shape == (1, 1, 256, 256)
        
        # Check output range (should be between 0 and 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_aspp_module(self):
        # Test ASPP module
        x = torch.randn(1, 2048, 8, 8)  # Feature size after ResNet50
        aspp_output = self.model.aspp(x)
        assert aspp_output.shape == (1, 256, 8, 8)
    
    def test_low_level_conv(self):
        # Create a sample low-level feature tensor
        x = torch.randn(1, 256, 32, 32)  # Low-level feature size
        
        # Test low-level feature processing
        low_level_output = self.model.low_level_conv(x)
        assert low_level_output.shape == (1, 48, 32, 32)
    
    def test_decoder(self):
        # Create sample tensors for testing decoder
        x = torch.randn(1, 304, 32, 32)  # 256 (ASPP) + 48 (low level)
        
        # Test decoder
        output = self.model.decoder(x)
        assert output.shape == (1, 1, 32, 32)
