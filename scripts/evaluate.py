import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader as TorchDataLoader
import pandas as pd
from tqdm import tqdm
import cv2

from crackdetect.models.segmentation import UNet
from crackdetect.data.dataset import CrackDataset
from crackdetect.training.metrics import iou_score, dice_coefficient, precision, recall
from crackdetect.utils.crack_analysis import CrackAnalyzer
from crackdetect.inference.visualization import create_result_figure
from config.config import Config

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained crack detection model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to test data directory")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Path to output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    parser.add_argument("--save-images", action="store_true", help="Save prediction images")
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Set data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = config.data_dir / "test"
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.save_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    test_dataset = CrackDataset(
        image_dir=data_dir / "images",
        mask_dir=data_dir / "masks",
        image_size=config.image_size,
        transform=None,
        preprocessing=True
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Load model
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Initialize metrics
    metrics = {
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "filename": []
    }
    
    # Evaluate model
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get data
            images = batch['image'].float().to(device)
            masks = batch['mask'].float().to(device)
            filenames = batch['filename']
            
            # Forward pass
            outputs = model(images)
            
            # Convert to binary predictions
            preds = (outputs > args.threshold).float()
            
            # Calculate metrics batch-wise
            batch_iou = iou_score(preds, masks).cpu().numpy()
            batch_dice = dice_coefficient(preds, masks).cpu().numpy()
            batch_precision = precision(preds, masks).cpu().numpy()
            batch_recall = recall(preds, masks).cpu().numpy()
            
            # Process each image in the batch
            for i in range(images.size(0)):
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                mask = masks[i, 0].cpu().numpy()
                pred = preds[i, 0].cpu().numpy()
                filename = filenames[i]
                
                # Record metrics
                metrics["iou"].append(batch_iou)
                metrics["dice"].append(batch_dice)
                metrics["precision"].append(batch_precision)
                metrics["recall"].append(batch_recall)
                metrics["filename"].append(filename)
                
                # Save prediction images if requested
                if args.save_images:
                        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                        
                        ax[0].imshow(image)
                        ax[0].set_title("Original Image")
                        ax[0].axis("off")
                        
                        ax[1].imshow(mask, cmap="gray")
                        ax[1].set_title("Ground Truth")
                        ax[1].axis("off")
                        
                        ax[2].imshow(pred, cmap="gray")
                        ax[2].set_title(f"Prediction (IoU: {batch_iou:.4f})")
                        ax[2].axis("off")
                        
                        plt.tight_layout()
                        plt.savefig(images_dir / f"{filename}_pred.png", dpi=200)
                        plt.close()
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(value) for key, value in metrics.items() if key != "filename"}
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "Filename": metrics["filename"],
        "IoU": metrics["iou"],
        "Dice": metrics["dice"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"]
    })
    
    # Save results to CSV
    results_df.to_csv(output_dir / "metrics_per_image.csv", index=False)
    
    # Calculate confusion matrix
    confusion_matrix = np.zeros((2, 2))
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calculating confusion matrix"):
            # Get data
            images = batch['image'].float().to(device)
            masks = batch['mask'].float().to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Convert to binary predictions
            preds = (outputs > args.threshold).float()
            
            # Calculate true positives, false positives, false negatives, true negatives
            tp = torch.sum((preds == 1) & (masks == 1)).item()
            fp = torch.sum((preds == 1) & (masks == 0)).item()
            fn = torch.sum((preds == 0) & (masks == 1)).item()
            tn = torch.sum((preds == 0) & (masks == 0)).item()
            
            confusion_matrix[0, 0] += tn  # True negative
            confusion_matrix[0, 1] += fp  # False positive
            confusion_matrix[1, 0] += fn  # False negative
            confusion_matrix[1, 1] += tp  # True positive
    
    # Create summary table
    summary = pd.DataFrame({
        "Metric": ["IoU", "Dice", "Precision", "Recall", "Accuracy", "F1-Score"],
        "Value": [
            avg_metrics["iou"],
            avg_metrics["dice"],
            avg_metrics["precision"],
            avg_metrics["recall"],
            (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix),
            2 * avg_metrics["precision"] * avg_metrics["recall"] / (avg_metrics["precision"] + avg_metrics["recall"])
        ]
    })
    
    # Save summary to CSV
    summary.to_csv(output_dir / "summary.csv", index=False)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(confusion_matrix, cmap="Blues")
    
    # Add text to cells
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{int(confusion_matrix[i, j])}", 
                   ha="center", va="center", 
                   color="white" if confusion_matrix[i, j] > np.sum(confusion_matrix)/4 else "black")
    
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.colorbar(label="Count")
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()

                    