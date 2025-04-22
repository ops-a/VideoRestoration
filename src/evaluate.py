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
import logging
import cv2
from typing import Dict, List, Tuple

from models.restoration_model import VideoRestorationModel
from data.video_processor import MediaDataset
from utils.metrics import ImageQualityMetrics

def setup_logging(output_dir: Path):
    """Set up logging configuration"""
    log_file = output_dir / 'evaluation.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path: str, config: dict, device: torch.device) -> VideoRestorationModel:
    """Load model from checkpoint"""
    model = VideoRestorationModel(
        in_channels=config['model']['in_channels'],
        num_blocks=config['model']['num_blocks']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model

def evaluate_model(model: VideoRestorationModel, 
                  test_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  output_dir: Path) -> Dict[str, float]:
    """Evaluate model on test dataset and save restored images"""
    model.eval()
    total_metrics = {
        'psnr': 0.0,
        'ssim': 0.0,
        'l1_loss': 0.0,
        'mse_loss': 0.0
    }
    num_batches = len(test_loader)

    # Create directory for restored images
    restored_dir = output_dir / 'restored_images'
    restored_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(test_loader, desc="Evaluating")):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate metrics for this batch
            batch_metrics = ImageQualityMetrics.evaluate_batch(model, data, target, device)
            
            # Accumulate metrics
            for metric in total_metrics:
                total_metrics[metric] += batch_metrics[metric]

            # Save restored image
            restored_img = output[0].cpu().numpy().transpose(1, 2, 0)
            restored_img = (restored_img * 255).astype(np.uint8)
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
            
            # Get original filename from the dataset
            original_filename = os.path.basename(test_loader.dataset.pairs[idx][0])
            base_name = os.path.splitext(original_filename)[0]
            restored_filename = f"{base_name}_restored{os.path.splitext(original_filename)[1]}"
            
            cv2.imwrite(str(restored_dir / restored_filename), restored_img)

    # Calculate average metrics
    for metric in total_metrics:
        total_metrics[metric] /= num_batches

    return total_metrics

def save_results(metrics: Dict[str, float], output_dir: Path) -> str:
    """Save evaluation results to file"""
    results_file = output_dir / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")
        f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
    return str(results_file)

def visualize_results(model: VideoRestorationModel,
                     test_loader: torch.utils.data.DataLoader,
                     output_dir: Path,
                     device: torch.device,
                     num_samples: int = 5):
    """Visualize and save sample results"""
    model.eval()
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
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
            axes[0].set_title('Input (Distorted)')
            axes[0].axis('off')
            
            axes[1].imshow(target_img)
            axes[1].set_title('Target (Clean)')
            axes[1].axis('off')
            
            axes[2].imshow(output_img)
            axes[2].set_title('Restored')
            axes[2].axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(vis_dir / f"comparison_{i+1}.png")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate Video Restoration Model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save evaluation results')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='Number of samples to visualize')
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logging.info("Configuration loaded successfully")
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load model
    try:
        model = load_model(args.checkpoint, config, device)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

    # Create test dataset
    try:
        test_dataset = MediaDataset(
            distorted_path=config['data']['test_distorted'],
            clean_path=config['data']['test_clean'],
            frame_size=config['data']['frame_size']
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,  # Process one image at a time for evaluation
            shuffle=False,
            num_workers=config['training']['num_workers']
        )
        logging.info(f"Test dataset created with {len(test_dataset)} samples")
    except Exception as e:
        logging.error(f"Error creating test dataset: {e}")
        raise

    # Evaluate model
    try:
        logging.info("Starting model evaluation...")
        metrics = evaluate_model(model, test_loader, device, output_dir)
        logging.info("Evaluation completed successfully")
        
        # Save results
        results_file = save_results(metrics, output_dir)
        logging.info(f"Results saved to {results_file}")
        
        # Visualize results
        logging.info("Generating visualizations...")
        visualize_results(model, test_loader, output_dir, device, args.num_samples)
        logging.info("Visualizations generated successfully")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

    logging.info(f"\nEvaluation complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 