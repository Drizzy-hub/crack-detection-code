import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import time
import datetime

from crackdetect.training.losses import dice_loss, combo_loss
from crackdetect.training.metrics import iou_score, dice_coefficient

class Trainer:
    """Class for training crack detection models."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 criterion: Callable = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 num_epochs: int = 100,
                 save_dir: Union[str, Path] = "saved_models",
                 model_name: str = "crack_segmentation"):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            num_epochs: Number of epochs to train for
            save_dir: Directory to save models
            model_name: Name for saved model
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion if criterion else combo_loss
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        
        # Create save directory
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer
        
        # Initialize learning rate scheduler
        self.lr_scheduler = lr_scheduler
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / f"{model_name}_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_dices = []
        self.best_val_metric = 0.0
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Get data
            images = batch['image'].float().to(self.device)
            masks = batch['mask'].float().to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            batch_loss = loss.item()
            epoch_loss += batch_loss
            progress_bar.set_postfix({"Batch Loss": f"{batch_loss:.4f}"})
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> Tuple[float, float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average validation loss, IoU score, Dice coefficient)
        """
        if self.val_loader is None:
            return 0.0, 0.0, 0.0
        
        self.model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch in progress_bar:
                # Get data
                images = batch['image'].float().to(self.device)
                masks = batch['mask'].float().to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                preds = (outputs > 0.5).float()
                batch_iou = iou_score(preds, masks).item()
                batch_dice = dice_coefficient(preds, masks).item()
                
                val_iou += batch_iou
                val_dice += batch_dice
                
                # Update progress bar
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "IoU": f"{batch_iou:.4f}",
                    "Dice": f"{batch_dice:.4f}"
                })
        
        # Calculate averages
        avg_val_loss = val_loss / num_batches
        avg_val_iou = val_iou / num_batches
        avg_val_dice = val_dice / num_batches
        
        self.val_losses.append(avg_val_loss)
        self.val_ious.append(avg_val_iou)
        self.val_dices.append(avg_val_dice)
        
        return avg_val_loss, avg_val_iou, avg_val_dice
    
    def save_model(self, epoch: int, metric: float) -> None:
        """
        Save the model.
        
        Args:
            epoch: Current epoch
            metric: Metric value (for filename)
        """
        filename = f"{self.model_name}_epoch{epoch}_metric{metric:.4f}.pth"
        best_filename = f"{self.model_name}_best.pth"
        
        torch.save(self.model.state_dict(), self.save_dir / filename)
        if metric > self.best_val_metric:
            torch.save(self.model.state_dict(), self.save_dir / best_filename)
            self.best_val_metric = metric
    
    def train(self) -> Dict:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary with training history
        """
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_iou, val_dice = self.validate()
            
            # Learning rate scheduler step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Save model
            self.save_model(epoch, val_dice)
            
            # Log progress
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch}/{self.num_epochs} - "
                f"Time: {datetime.timedelta(seconds=int(epoch_time))} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val IoU: {val_iou:.4f} - "
                f"Val Dice: {val_dice:.4f}"
            )
        
        # Log total training time
        total_time = time.time() - start_time
        self.logger.info(
            f"Training completed in {datetime.timedelta(seconds=int(total_time))}"
        )
        
        # Return training history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'val_dices': self.val_dices
        }
