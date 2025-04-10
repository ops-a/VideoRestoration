import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np

from models.restoration_model import VideoRestorationModel
from data.video_processor import MediaDataset
from utils.metrics import ImageQualityMetrics

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def parse_args():
    parser = argparse.ArgumentParser(description='Train Video Restoration Model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                      help='Path to configuration file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(model, train_loader, criterion, optimizer, device, epoch, writer, scaler):
    model.train()
    total_loss = 0
    total_metrics = {
        'psnr': 0.0,
        'ssim': 0.0,
        'l1_loss': 0.0,
        'mse_loss': 0.0
    }
    
    with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate metrics
            batch_metrics = ImageQualityMetrics.evaluate_batch(model, data, target, device)
            
            # Update totals
            total_loss += loss.item()
            for metric in total_metrics:
                total_metrics[metric] += batch_metrics[metric]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'psnr': total_metrics['psnr'] / (batch_idx + 1)
            })
            
            # Log metrics
            if batch_idx % 100 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Training/Loss', loss.item(), step)
                for metric, value in batch_metrics.items():
                    writer.add_scalar(f'Training/{metric.upper()}', value, step)
                
    # Calculate averages
    num_batches = len(train_loader)
    avg_metrics = {metric: value / num_batches for metric, value in total_metrics.items()}
    avg_loss = total_loss / num_batches
    
    return avg_loss, avg_metrics

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_metrics = {
        'psnr': 0.0,
        'ssim': 0.0,
        'l1_loss': 0.0,
        'mse_loss': 0.0
    }
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            # Calculate metrics
            batch_metrics = ImageQualityMetrics.evaluate_batch(model, data, target, device)
            
            # Update totals
            total_loss += loss.item()
            for metric in total_metrics:
                total_metrics[metric] += batch_metrics[metric]
    
    # Calculate averages
    num_batches = len(val_loader)
    avg_metrics = {metric: value / num_batches for metric, value in total_metrics.items()}
    avg_loss = total_loss / num_batches
    
    return avg_loss, avg_metrics

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_metrics, val_metrics, path):
    """Save model checkpoint with all necessary information"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }, path)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = VideoRestorationModel(
        in_channels=config['model']['in_channels'],
        num_blocks=config['model']['num_blocks']
    ).to(device)
    
    # Create datasets and dataloaders
    train_dataset = MediaDataset(
        config['data']['train_data'],
        frame_size=tuple(config['data']['frame_size']),
        is_video=config['data'].get('is_video', None)
    )
    val_dataset = MediaDataset(
        config['data']['val_data'],
        frame_size=tuple(config['data']['frame_size']),
        is_video=config['data'].get('is_video', None)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=1e-4  # L2 regularization
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Setup mixed precision training
    scaler = GradScaler()
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=10)
    
    # Setup logging
    writer = SummaryWriter(config['training']['log_dir'])
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        try:
            # Training phase
            train_loss, train_metrics = train(model, train_loader, criterion, optimizer, device, epoch, writer, scaler)
            
            # Validation phase
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            
            # Log metrics
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            for metric, value in val_metrics.items():
                writer.add_scalar(f'Validation/{metric.upper()}', value, epoch)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    train_metrics, val_metrics,
                    Path(config['training']['checkpoint_dir']) / 'best_model.pth'
                )
            
            # Print epoch summary
            print(f'\nEpoch {epoch}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print('Train Metrics:')
            for metric, value in train_metrics.items():
                print(f'  {metric.upper()}: {value:.4f}')
            print('Val Metrics:')
            for metric, value in val_metrics.items():
                print(f'  {metric.upper()}: {value:.4f}')
            
            # Early stopping check
            if early_stopping(val_loss):
                print("Early stopping triggered")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    writer.close()

if __name__ == '__main__':
    main()