import argparse
import torch
import os
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np

from crackdetect.models.segmentation import UNet
from config.config import Config

def export_to_onnx(model, save_path, input_shape=(1, 3, 512, 512)):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        save_path: Path to save ONNX model
        input_shape: Input shape for the model
    """
    # Create dummy input
    dummy_input = torch.randn(input_shape, requires_grad=True)
    
    # Export model
    torch.onnx.export(
        model,                                 # model being run
        dummy_input,                           # model input (or a tuple for multiple inputs)
        save_path,                             # where to save the model
        export_params=True,                    # store the trained parameter weights inside the model file
        opset_version=12,                      # the ONNX version to export the model to
        do_constant_folding=True,              # whether to execute constant folding for optimization
        input_names=['input'],                 # the model's input names
        output_names=['output'],               # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},        # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {save_path}")

def verify_onnx_model(onnx_path, input_shape=(1, 3, 512, 512)):
    """
    Verify ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Input shape for the model
    """
    # Load ONNX model
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model checked successfully")
    
    # Create ONNX runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create a random input
    x = np.random.randn(*input_shape).astype(np.float32)
    
    # Run the model
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Check output
    print(f"ONNX model output shape: {ort_outs[0].shape}")
    print(f"ONNX model output range: [{ort_outs[0].min()}, {ort_outs[0].max()}]")
    
    print("ONNX model verified successfully")

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model")
    parser.add_argument("--output", type=str, default=None, help="Path to save ONNX model")
    parser.add_argument("--input-shape", type=str, default="1,3,512,512", help="Input shape (comma-separated)")
    parser.add_argument("--verify", action="store_true", help="Verify exported ONNX model")
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = Path(args.model).stem
        output_path = config.models_dir / f"{model_name}.onnx"
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Load PyTorch model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Export model to ONNX
    export_to_onnx(model, output_path, input_shape)
    
    # Verify ONNX model if requested
    if args.verify:
        verify_onnx_model(output_path, input_shape)

if __name__ == "__main__":
    main()
