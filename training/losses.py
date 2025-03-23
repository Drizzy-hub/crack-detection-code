import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def dice_loss(pred: torch.Tensor, 
             target: torch.Tensor, 
             smooth: float = 1.0) -> torch.Tensor:
    """
    Dice loss function for segmentation.
    
    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Target mask (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss
    """
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_loss(pred: torch.Tensor, 
            target: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss function for segmentation.
    
    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Target mask (B, 1, H, W)
        
    Returns:
        BCE loss
    """
    return F.binary_cross_entropy(pred, target)

def focal_loss(pred: torch.Tensor, 
              target: torch.Tensor, 
              alpha: float = 0.25, 
              gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss function for handling class imbalance.
    
    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Target mask (B, 1, H, W)
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss
    """
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal_loss = alpha * (1 - pt) ** gamma * bce
    return focal_loss.mean()

def combo_loss(pred: torch.Tensor, 
              target: torch.Tensor, 
              alpha: float = 0.5) -> torch.Tensor:
    """
    Combination of Dice and BCE loss.
    
    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Target mask (B, 1, H, W)
        alpha: Weighting factor
        
    Returns:
        Combined loss
    """
    return alpha * dice_loss(pred, target) + (1 - alpha) * bce_loss(pred, target)

# ==================== crackdetect/training/metrics.py ====================
import torch
from typing import Optional

def iou_score(pred: torch.Tensor, 
             target: torch.Tensor, 
             smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate IoU (Intersection over Union) score.
    
    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Target mask (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score
    """
    # Flatten predictions and targets
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.mean()

def dice_coefficient(pred: torch.Tensor, 
                    target: torch.Tensor, 
                    smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice coefficient.
    
    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Target mask (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    # Flatten predictions and targets
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    # Calculate intersection and sum
    intersection = (pred * target).sum(dim=1)
    sum_pred_target = pred.sum(dim=1) + target.sum(dim=1)
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (sum_pred_target + smooth)
    
    return dice.mean()

def precision(pred: torch.Tensor, 
             target: torch.Tensor, 
             smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate precision.
    
    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Target mask (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Precision score
    """
    # Flatten predictions and targets
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    # Calculate true positives and false positives
    tp = (pred * target).sum(dim=1)
    fp = pred.sum(dim=1) - tp
    
    # Calculate precision
    prec = (tp + smooth) / (tp + fp + smooth)
    
    return prec.mean()

def recall(pred: torch.Tensor, 
          target: torch.Tensor, 
          smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate recall.
    
    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Target mask (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Recall score
    """
    # Flatten predictions and targets
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    # Calculate true positives and false negatives
    tp = (pred * target).sum(dim=1)
    fn = target.sum(dim=1) - tp
    
    # Calculate recall
    rec = (tp + smooth) / (tp + fn + smooth)
    
    return rec.mean()
