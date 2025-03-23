import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class ASPPConv(nn.Sequential):
    """Atrous Spatial Pyramid Pooling convolution module."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    """Atrous Spatial Pyramid Pooling pooling module."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()[2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: List[int] = [6, 12, 18]):
        super().__init__()
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Project to output channels
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ model for semantic segmentation."""
    
    def __init__(self, 
                in_channels: int = 3, 
                out_channels: int = 1,
                backbone: Optional[str] = "resnet50",
                pretrained: bool = True):
        """
        Initialize DeepLabV3+ model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            backbone: Backbone architecture
            pretrained: Whether to use pretrained backbone
        """
        super().__init__()
        
        # Import torchvision for ResNet backbone
        import torchvision
        
        # Get backbone
        if backbone == "resnet50":
            self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            self.backbone = torchvision.models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove last layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Get output channels from backbone
        if backbone.startswith("resnet"):
            backbone_out_channels = 2048
            low_level_channels = 256  # After first residual block
        
        # ASPP module
        self.aspp = ASPP(backbone_out_channels, 256)
        
        # Low-level features processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 256 (ASPP) + 48 (low level)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.size()[2:]
        
        # Save low-level features
        low_level_features = None
        
        # Forward through backbone layers
        for i, module in enumerate(self.backbone):
            x = module(x)
            if i == 4:  # After first residual block
                low_level_features = x
        
        # ASPP
        x = self.aspp(x)
        
        # Upsample ASPP features
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=False)
        
        # Process low-level features
        low_level_features = self.low_level_conv(low_level_features)
        
        # Concatenate features
        x = torch.cat([x, low_level_features], dim=1)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return torch.sigmoid(x)
