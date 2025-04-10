import torch
import torch.nn as nn
import argparse
import yaml
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from models.restoration_model import VideoRestorationModel
from data.video_processor import MediaDataset, MediaProcessor
from utils.metrics import ImageQualityMetrics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Video Restoration Model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/raw/test',
                      help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save evaluation results')
    parser.add_argument('--save_images', action='store_true',
                      help='Save restored images')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for evaluation')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, config, device):
    """Load model from checkpoint"""
    model = VideoRestorationModel(
        in_channels=config['model']['in_channels'],
        num_blocks=config['model']['num_blocks']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model on test dataset"""
    model.eval()
    total_metrics = {
        'psnr': 0.0,
        'ssim': 0.0,
        'l1_loss': 0.0,
        'mse_loss': 0.0
    }
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate metrics
            batch_metrics = ImageQualityMetrics.evaluate_batch(model, data, target, device)
            
            # Update totals
            for metric in total_metrics:
                total_metrics[metric] += batch_metrics[metric]
    
    # Calculate averages
    num_batches = len(test_loader)
    avg_metrics = {metric: value / num_batches for metric, value in total_metrics.items()}
    
    return avg_metrics

def save_results(metrics, output_dir):
    """Save evaluation results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.txt")
    
    with open(results_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
    
    print(f"Results saved to {results_file}")
    return results_file

def visualize_results(model, test_loader, output_dir, device, num_samples=5):
    """Visualize and save sample results"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Convert tensors to numpy arrays
            input_img = data[0].cpu().numpy().transpose(1, 2, 0)
            target_img = target[0].cpu().numpy().transpose(1, 2, 0)
            output_img = output[0].cpu().numpy().transpose(1, 2, 0)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot images
            axes[0].imshow(input_img)
            axes[0].set_title('Input')
            axes[0].axis('off')
            
            axes[1].imshow(target_img)
            axes[1].set_title('Target')
            axes[1].axis('off')
            
            axes[2].imshow(output_img)
            axes[2].set_title('Restored')
            axes[2].axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{i+1}.png"))
            plt.close()

def process_test_data(model, test_data_path, output_dir, device, is_video=None):
    """Process test data and save restored outputs"""
    # Determine if input is video or images
    if is_video is None:
        if os.path.isfile(test_data_path):
            # Check file extension
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            is_video = Path(test_data_path).suffix.lower() in video_extensions
        else:
            # Assume directory contains images
            is_video = False
    
    # Create processor
    processor = MediaProcessor(
        input_path=test_data_path,
        output_path=os.path.join(output_dir, "restored_output"),
        frame_size=(256, 256),  # Use default size, can be updated from config
        is_video=is_video
    )
    
    # Process data
    processor.process_media(model, device)
    print(f"Processed test data saved to {os.path.join(output_dir, 'restored_output')}")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Create test dataset
    test_dataset = MediaDataset(
        args.test_data,
        frame_size=tuple(config['data']['frame_size']),
        is_video=config['data'].get('is_video', None)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Evaluate model
    print("Evaluating model on test dataset...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Save results
    results_file = save_results(metrics, output_dir)
    
    # Visualize results
    if args.save_images:
        print("Generating visualizations...")
        visualize_results(model, test_loader, output_dir, device)
    
    # Process test data
    print("Processing test data...")
    process_test_data(model, args.test_data, output_dir, device, config['data'].get('is_video', None))
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 