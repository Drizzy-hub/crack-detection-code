import argparse
import torch
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader

from crackdetect.models.segmentation import UNet
from crackdetect.data.dataset import CrackDataset
from crackdetect.training.trainer import Trainer
from crackdetect.training.losses import combo_loss
from config.config import Config

def main():
    parser = argparse.ArgumentParser(description="Train a crack detection model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--model-name", type=str, default=None, help="Model name")
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.model_name:
        config.model_name = args.model_name
    else:
        config.model_name = "crack_segmentation"
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data augmentation
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])
    
    val_transform = A.Compose([])
    
    # Create datasets
    train_dataset = CrackDataset(
        image_dir=config.data_dir / "train" / "images",
        mask_dir=config.data_dir / "train" / "masks",
        image_size=config.image_size,
        transform=train_transform,
        preprocessing=True
    )
    
    val_dataset = CrackDataset(
        image_dir=config.data_dir / "val" / "images",
        mask_dir=config.data_dir / "val" / "masks",
        image_size=config.image_size,
        transform=val_transform,
        preprocessing=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=combo_loss,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        num_epochs=config.num_epochs,
        save_dir=config.models_dir,
        model_name=config.model_name
    )
    
    # Train model
    history = trainer.train()
    
    # Plot training history
    epochs = range(1, config.num_epochs + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_ious'], 'g-')
    plt.title('Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_dices'], 'm-')
    plt.title('Validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    
    plt.tight_layout()
    plt.savefig(config.models_dir / f"{config.model_name}_training_history.png")

if __name__ == "__main__":
    main()
