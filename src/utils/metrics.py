import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, Tuple

class ImageQualityMetrics:
    @staticmethod
    def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        Args:
            img1: First image tensor (B, C, H, W)
            img2: Second image tensor (B, C, H, W)
        Returns:
            PSNR value
        """
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    @staticmethod
    def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
        """
        Calculate Structural Similarity Index (SSIM)
        Args:
            img1: First image tensor (B, C, H, W)
            img2: Second image tensor (B, C, H, W)
            window_size: Size of the Gaussian window
        Returns:
            SSIM value
        """
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2

        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()

    @staticmethod
    def l1_loss(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate L1 Loss
        Args:
            img1: First image tensor (B, C, H, W)
            img2: Second image tensor (B, C, H, W)
        Returns:
            L1 loss value
        """
        return F.l1_loss(img1, img2).item()

    @staticmethod
    def mse_loss(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate Mean Squared Error (MSE)
        Args:
            img1: First image tensor (B, C, H, W)
            img2: Second image tensor (B, C, H, W)
        Returns:
            MSE value
        """
        return F.mse_loss(img1, img2).item()

    @staticmethod
    def evaluate_batch(model: torch.nn.Module, 
                      data: torch.Tensor, 
                      target: torch.Tensor,
                      device: torch.device) -> Dict[str, float]:
        """
        Evaluate a batch of images using multiple metrics
        Args:
            model: The restoration model
            data: Input images (B, C, H, W)
            target: Target images (B, C, H, W)
            device: Device to run evaluation on
        Returns:
            Dictionary of metric names and values
        """
        model.eval()
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            metrics = {
                'psnr': ImageQualityMetrics.psnr(output, target),
                'ssim': ImageQualityMetrics.ssim(output, target),
                'l1_loss': ImageQualityMetrics.l1_loss(output, target),
                'mse_loss': ImageQualityMetrics.mse_loss(output, target)
            }
            
        return metrics

    @staticmethod
    def evaluate_dataset(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device: torch.device) -> Dict[str, float]:
        """
        Evaluate the entire dataset
        Args:
            model: The restoration model
            dataloader: DataLoader containing the dataset
            device: Device to run evaluation on
        Returns:
            Dictionary of average metric values
        """
        model.eval()
        total_metrics = {
            'psnr': 0.0,
            'ssim': 0.0,
            'l1_loss': 0.0,
            'mse_loss': 0.0
        }
        num_batches = len(dataloader)

        with torch.no_grad():
            for data, target in dataloader:
                batch_metrics = ImageQualityMetrics.evaluate_batch(model, data, target, device)
                for metric in total_metrics:
                    total_metrics[metric] += batch_metrics[metric]

        # Calculate averages
        for metric in total_metrics:
            total_metrics[metric] /= num_batches

        return total_metrics 