import argparse
import yaml
import torch
from thop import profile

# Import the baseline models we just defined
from src.models.baseline_models import Baseline_ResNet44, Baseline_P4CNN_MNIST

def main(args):
    """
    This script calculates the FLOPs for a dense baseline model
    based on a given configuration file.
    """
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"--- Calculating FLOPs for baseline model in: {args.config} ---")

    # Determine the model and input shape from the config
    model_name = config['model']
    in_channels = config['in_channels']
    
    if 'MNIST' in config['dataset'] or 'FashionMNIST' in config['dataset']:
        input_shape = (in_channels, 28, 28)
    elif 'CIFAR' in config['dataset'] or 'GTSRB' in config['dataset']:
        input_shape = (in_channels, 32, 32)
    else:
        raise ValueError(f"Dataset {config['dataset']} not recognized for input shape.")

    # Instantiate the correct baseline model
    if 'P4CNN_MNIST' in model_name:
        model = Baseline_P4CNN_MNIST(
            num_classes=config['num_classes'],
            in_channels=in_channels
        )
    elif 'ResNet44' in model_name:
        model = Baseline_ResNet44(
            n=config.get('resnet_n', 7),
            num_classes=config['num_classes'],
            in_channels=in_channels,
            group=config['group'],
            widths=config['widths']
        )
    else:
        raise ValueError(f"Model '{model_name}' not recognized for baseline FLOPs calculation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, *input_shape).to(device)

    # Calculate FLOPs using thop
    total_ops, total_params = profile(model, inputs=(dummy_input,), verbose=False)
    
    gflops = total_ops / 1e9
    mflops = total_ops / 1e6
    params_m = total_params / 1e6

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Parameters: {params_m:.2f}M")
    print(f"GFLOPs: {gflops:.6f}G")
    print(f"MFLOPs: {mflops:.2f}M")
    print("-" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate FLOPs for baseline G-CNN models.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file for the experiment.")
    args = parser.parse_args()
    
    # Set PYTHONPATH to include the project root
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    
    main(args)
