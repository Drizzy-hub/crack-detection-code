import argparse
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import random

from crackdetect.data.loader import DataLoader
from config.config import Config

def main():
    parser = argparse.ArgumentParser(description="Prepare data for crack detection")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to input data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to output data directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of training data")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio of validation data")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Ratio of test data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.data_dir
    
    # Set up input paths
    input_dir = Path(args.input_dir)
    input_images = input_dir / "images"
    input_masks = input_dir / "masks"
    
    # Check if input directories exist
    if not input_images.exists() or not input_masks.exists():
        print(f"Error: Input directories not found: {input_images} or {input_masks}")
        return
    
    # Create data loader
    data_loader = DataLoader(output_dir)
    
    # Split dataset
    data_loader.split_dataset(
        input_images=input_images,
        input_masks=input_masks,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    
    # Print dataset statistics
    stats = data_loader.get_dataset_stats()
    print("\nDataset Statistics:")
    for split, counts in stats.items():
        print(f"  {split.capitalize()}: {counts['images']} images, {counts['masks']} masks")

if __name__ == "__main__":
    main()