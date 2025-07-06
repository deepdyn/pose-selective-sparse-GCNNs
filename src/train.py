import argparse
import yaml
import torch
import torch.nn as nn
import os
import time
import logging
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

# Import the models and the specific gate for type checking
from src.models.pose_gcnn import PoseSelectiveSparse_ResNet44, DifferentiableMaskGate
from src.models.mnist_architectures import PoseSelective_P4CNN_MNIST # <-- ADD THIS IMPORT
from src.data_loader import get_dataloaders_with_fixed_splits
from src.utils import (
    set_seed, setup_logging, save_results, get_gate_weights, 
    count_parameters, compute_ece, calculate_gflops
)

# ... (train_epoch and evaluate functions remain unchanged) ...
def train_epoch(model, dataloader, optimizer, criterion, device, lambda_penalty):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        ce_loss = criterion(outputs, labels)
        
        sparsity_loss = 0
        num_gates = 0
        for module in model.modules():
            if isinstance(module, DifferentiableMaskGate):
                sparsity_loss += module.get_mask().sum()
                num_gates += 1
        
        if num_gates > 0:
            sparsity_loss /= num_gates

        total_loss = ce_loss + lambda_penalty * sparsity_loss
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, description="Evaluating"):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=description):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    
    ece = compute_ece(model, dataloader, device)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return epoch_loss, epoch_acc, ece, all_preds, all_labels


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Updated save path to include model name for better organization
    save_path = os.path.join(config['results_dir'], config['dataset'], config['model'], f"seed_{args.seed}")
    os.makedirs(save_path, exist_ok=True)
    setup_logging(os.path.join(save_path, 'train.log'))

    set_seed(args.seed)
    logging.info(f"Starting experiment with config:\n{yaml.dump(config)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders_with_fixed_splits(
        name=config['dataset'],
        batch_size=config['batch_size'],
        path=config['data_path']
    )
    
    if 'MNIST' in config['dataset'] or 'FashionMNIST' in config['dataset']:
        input_shape = (config['in_channels'], 28, 28)
    elif 'CIFAR' in config['dataset'] or 'GTSRB' in config['dataset']:
        input_shape = (config['in_channels'], 32, 32)
    else:
        input_shape = (config.get('in_channels', 3), 32, 32)
        logging.warning(f"Could not derive input shape for {config['dataset']}. Defaulting to {input_shape}.")

    # --- REPLACE THE MODEL CREATION BLOCK WITH THIS ---
    logging.info(f"Creating model: {config['model']}")
    if config['model'] == 'PoseSelective_P4CNN_MNIST':
        model = PoseSelective_P4CNN_MNIST(
            num_classes=config['num_classes'],
            in_channels=config['in_channels']
        ).to(device)
    elif config['model'] == 'PoseSelectiveSparse_ResNet44':
        model = PoseSelectiveSparse_ResNet44(
            n=config.get('resnet_n', 7),
            num_classes=config['num_classes'],
            in_channels=config['in_channels'],
            group=config['group'],
            widths=config['widths']
        ).to(device)
    else:
        raise ValueError(f"Model '{config['model']}' not recognized.")
    # --- END OF REPLACEMENT ---
    
    num_params = count_parameters(model)
    logging.info(f"Model initialized with {num_params:,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])

    best_val_acc = 0.0
    best_model_path = os.path.join(save_path, 'model_best.pth')
    
    sparsity_schedule = config.get('sparsity_schedule', {'lambda_initial': 0.0, 'lambda_final': 0.0, 'anneal_epochs': 1})
    temp_schedule = config.get('temperature_schedule', {'temp_initial': 1.0, 'temp_final': 1.0, 'anneal_epochs': 1})

    for epoch in range(config['epochs']):
        progress = min(epoch / temp_schedule['anneal_epochs'], 1.0) if temp_schedule['anneal_epochs'] > 0 else 1.0
        
        current_temp = temp_schedule['temp_initial'] - (temp_schedule['temp_initial'] - temp_schedule['temp_final']) * progress
        current_lambda = sparsity_schedule['lambda_initial'] + (sparsity_schedule['lambda_final'] - sparsity_schedule['lambda_initial']) * progress
        use_noise_flag = progress < 1.0

        for module in model.modules():
            if isinstance(module, DifferentiableMaskGate):
                module.temp.fill_(current_temp)
                module.use_noise = use_noise_flag

        logging.info(f"Epoch {epoch+1}/{config['epochs']} | Temp: {current_temp:.4f} | Lambda: {current_lambda:.6f} | Noise: {use_noise_flag}")

        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, current_lambda)
        val_loss, val_acc, val_ece, _, _ = evaluate(model, val_loader, criterion, device, "Validating")
        scheduler.step()
        
        logging.info(f"  -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | Val ECE: {val_ece:.4f} | Time: {time.time() - start_time:.2f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  -> New best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        gate_weights = get_gate_weights(model)
        logging.info(f"  -> Gate Weights: {gate_weights}")

    logging.info("Training finished. Evaluating best model on the test set.")
    model.load_state_dict(torch.load(best_model_path))
    
    logging.info("Setting final gate temperature to a near-zero value for hard-gate evaluation.")
    for module in model.modules():
        if isinstance(module, DifferentiableMaskGate):
            module.temp.fill_(1e-5)

    logging.info("Calculating final model GFLOPs...")
    gflops = calculate_gflops(model, input_shape, device)
    if gflops != -1:
        logging.info(f"Final Pruned Model GFLOPs: {gflops:.4f}G")
    
    test_loss, test_acc, test_ece, _, _ = evaluate(model, test_loader, criterion, device, "Testing")
    logging.info(f"Final Test Results | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test ECE: {test_ece:.4f}")

    final_results = {
        'best_validation_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'test_ece': test_ece,
        'model_params': num_params,
        'gflops': gflops
    }
    save_results(final_results, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Pose-Selective Sparse G-CNN Experiments")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for the experiment.")
    args = parser.parse_args()
    main(args)
