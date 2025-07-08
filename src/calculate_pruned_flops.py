import argparse
import yaml
import torch
import os
import sys

# Add the project root to the Python path to allow imports from 'src'
sys.path.insert(0, os.getcwd())

from src.models.pose_gcnn import PoseSelectiveSparse_ResNet44
from src.models.mnist_architectures import PoseSelective_P4CNN_MNIST
from src.utils import calculate_gflops

def main(args):
    """
    This script loads a trained sparse model checkpoint and calculates its
    final pruned FLOPs.
    """
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"--- Calculating Pruned FLOPs for checkpoint: {args.checkpoint} ---")
    print(f"--- Using configuration from: {args.config} ---")

    # Determine the model and input shape from the config
    model_name = config['model']
    in_channels = config['in_channels']
    
    if 'MNIST' in config['dataset'] or 'FashionMNIST' in config['dataset']:
        input_shape = (in_channels, 28, 28)
    elif 'CIFAR' in config['dataset'] or 'GTSRB' in config['dataset']:
        input_shape = (in_channels, 32, 32)
    else:
        raise ValueError(f"Dataset {config['dataset']} not recognized for input shape.")

    # Instantiate the correct model architecture
    if 'P4CNN_MNIST' in model_name:
        model = PoseSelective_P4CNN_MNIST(
            num_classes=config['num_classes'],
            in_channels=in_channels
        )
    elif 'ResNet44' in model_name:
        model = PoseSelectiveSparse_ResNet44(
            n=config.get('resnet_n', 7),
            num_classes=config['num_classes'],
            in_channels=in_channels,
            group=config['group'],
            widths=config['widths']
        )
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

    # Load the learned weights and gate logits from the checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file not found at {args.checkpoint}")
        return

    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Calculate the pruned GFLOPs using the corrected utility
    gflops = calculate_gflops(model, input_shape, device)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Pruned GFLOPs: {gflops:.6f}G")
    print(f"Pruned MFLOPs: {gflops * 1000:.2f}M")
    print("-" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Pruned FLOPs for a trained model checkpoint.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file for the experiment.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model's .pth checkpoint file.")
    args = parser.parse_args()
    main(args)
