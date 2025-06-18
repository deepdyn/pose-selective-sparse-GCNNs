import argparse
import yaml
import torch
import torch.nn as nn
import os
import time
import logging
from tqdm import tqdm

from src.models.partial_gcnn import P4mW_ResNet44
from src.data_loader import get_dataloaders_with_fixed_splits
from src.utils import set_seed, setup_logging, save_results, get_alpha_weights

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    return epoch_loss, epoch_acc

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_path = os.path.join(config['results_dir'], config['dataset'], config['model'], f"seed_{args.seed}")
    os.makedirs(save_path, exist_ok=True)
    setup_logging(os.path.join(save_path, 'train.log'))

    set_seed(args.seed)
    config['seed'] = args.seed
    logging.info(f"Starting experiment with config: {config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders_with_fixed_splits(
        name=config['dataset'],
        batch_size=config['batch_size'],
        path=config['data_path']
    )

    model = P4mW_ResNet44(
        n=config['resnet_n'],
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
        group=config['group']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])

    best_test_acc = 0.0
    for epoch in range(config['epochs']):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
        scheduler.step()
        
        logging.info(f"Epoch {epoch+1}/{config['epochs']} | "
                     f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                     f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
                     f"Time: {time.time() - start_time:.2f}s")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'model_best.pth'))
            logging.info(f"New best model saved with accuracy: {best_test_acc:.2f}%")
        
        # Log alpha weights
        alpha_weights = get_alpha_weights(model)
        logging.info(f"Epoch {epoch+1} Alpha Weights: {alpha_weights}")

    final_results = {'best_test_accuracy': best_test_acc}
    save_results(final_results, save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 'model_final.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Partial G-CNN Experiments")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for the experiment.")
    args = parser.parse_args()
    main(args)