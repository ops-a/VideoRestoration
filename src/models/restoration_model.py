import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class VideoRestorationModel(nn.Module):
    def __init__(self, in_channels=3, num_blocks=16):
        super(VideoRestorationModel, self).__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks with attention
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(64),
                AttentionBlock(64)
            ) for _ in range(num_blocks)
        ])
        
        # Upsampling and final reconstruction
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, in_channels, 3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 channels, got {x.size(1)}")
            
        # Initial feature extraction
        out = self.conv1(x)
        out = self.relu(out)
        
        # Residual blocks with attention
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Final reconstruction
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        return out

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def resize_input(self, x, target_size=(256, 256)):
        """Resize input to target size while maintaining aspect ratio"""
        if x.size(2) != target_size[0] or x.size(3) != target_size[1]:
            return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x