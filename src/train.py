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
import logging

from models.restoration_model import VideoRestorationModel
from data.video_processor import MediaDataset
from utils.metrics import ImageQualityMetrics

def setup_logging(log_dir: Path):
    """Set up logging configuration"""
    log_file = log_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

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
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set up logging
    log_dir = Path(config['training']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        # Enable memory efficient algorithms
        torch.backends.cudnn.benchmark = True
        # Clear any existing CUDA cache
        torch.cuda.empty_cache()

    # Initialize model
    model = VideoRestorationModel(
        in_channels=config['model']['in_channels'],
        num_blocks=config['model']['num_blocks']
    )

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    # Create data loaders with reduced batch size if needed
    batch_size = config['training']['batch_size']
    if torch.cuda.is_available():
        # Try to determine optimal batch size
        try:
            # Test with a small batch first
            test_batch = torch.randn(1, 3, *config['data']['frame_size']).to(device)
            model(test_batch)
            # If successful, try with full batch size
            test_batch = torch.randn(batch_size, 3, *config['data']['frame_size']).to(device)
            model(test_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.warning("Reducing batch size due to memory constraints")
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()

    train_dataset = MediaDataset(
        distorted_path=config['data']['train_distorted'],
        clean_path=config['data']['train_clean'],
        frame_size=config['data']['frame_size']
    )
    val_dataset = MediaDataset(
        distorted_path=config['data']['val_distorted'],
        clean_path=config['data']['val_clean'],
        frame_size=config['data']['frame_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True  # Enable pinned memory for faster data transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        try:
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    if batch_idx % 10 == 0:
                        logging.info(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')
                    
                    # Clear memory after each batch
                    del output, loss
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.warning("Out of memory during training batch. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    try:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        val_loss += criterion(output, target).item()
                        
                        # Clear memory after each batch
                        del output
                        torch.cuda.empty_cache()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logging.warning("Out of memory during validation batch. Skipping batch.")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e

            # Calculate average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_dir = Path(config['training']['checkpoint_dir'])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_dir / 'best_model.pth')
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("Out of memory during epoch. Reducing batch size and retrying.")
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                # Recreate data loaders with new batch size
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=config['training']['num_workers'],
                    pin_memory=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=config['training']['num_workers'],
                    pin_memory=True
                )
                continue
            else:
                raise e

if __name__ == '__main__':
    main()