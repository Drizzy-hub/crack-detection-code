import os
import torch
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from crackdetect.models.segmentation import UNet
from crackdetect.data.preprocessing import ImagePreprocessor
from crackdetect.utils.crack_analysis import CrackAnalyzer
from config.config import Config

def main():
    parser = argparse.ArgumentParser(description="Detect and analyze cracks in construction images")
    parser.add_argument("--image", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    parser.add_argument("--output", type=str, default="results", help="Path to output directory")
    parser.add_argument("--pixel-mm-ratio", type=float, default=1.0, help="Pixels to millimeters ratio")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum crack area to consider (in pixels)")
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=config.image_size)
    
    # Initialize model
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1)
    
    # Load model weights
    if args.model:
        model_path = args.model
    else:
        model_path = config.models_dir / "crack_segmentation.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
    
    model = model.to(device)
    model.eval()
    
    # Initialize analyzer
    analyzer = CrackAnalyzer(pixel_mm_ratio=args.pixel_mm_ratio)
    
    # Process input image or directory
    input_path = Path(args.image)
    if input_path.is_dir():
        image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    else:
        image_paths = [input_path]
    
    for img_path in image_paths:
        # Read and preprocess image
        image = preprocessor.read_image(img_path)
        processed_image = preprocessor.preprocess(image, enhance=True, denoise_img=True)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(processed_image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert prediction to mask
        pred_mask = output.squeeze().cpu().numpy() > 0.5
        
        # Analyze cracks
        crack_properties = analyzer.analyze_mask(pred_mask, min_area=args.min_area)
        
        # Visualize results
        result_image = analyzer.visualize_analysis(processed_image, pred_mask, crack_properties)
        
        # Save results
        output_path = output_dir / f"{img_path.stem}_result.png"
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(processed_image)
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title("Crack Detection")
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Analysis Result")
        plt.imshow(result_image)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        # Print analysis summary
        print(f"\nAnalysis Results for {img_path.name}:")
        print(f"Number of cracks detected: {len(crack_properties)}")
        
        if crack_properties:
            for i, props in enumerate(crack_properties):
                print(f"\nCrack #{i+1}:")
                print(f"  Severity: {props.severity}")
                print(f"  Average Width: {props.width_avg:.2f} mm")
                print(f"  Maximum Width: {props.width_max:.2f} mm")
                print(f"  Length: {props.length:.2f} mm")
                print(f"  Area: {props.area:.2f} mm²")
                print(f"  Orientation: {props.orientation:.1f}°")

if __name__ == "__main__":
    main()