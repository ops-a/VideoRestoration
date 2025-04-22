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

# Squeeze-and-Excitation Block for RainRemoval architecture
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = x.view(batch, channels, -1).mean(dim=2)  # Global Average Pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y

class RainRemovalModel(nn.Module):
    def __init__(self, in_channels=3):
        super(RainRemovalModel, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck with SEBlock
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output normalized to [0, 1]
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3)
        
        # Decoder with skip connections
        dec3 = self.decoder3(bottleneck + enc3)  # Skip connection
        dec2 = self.decoder2(dec3 + enc2)        # Skip connection
        dec1 = self.decoder1(dec2 + enc1)        # Skip connection
        
        return dec1

class EncoderDecoderModel(nn.Module):
    def __init__(self, in_channels=3):
        super(EncoderDecoderModel, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),  # fc1
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),  # fc2
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 1, 1)),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True)
        )
        
        # Additional bottleneck layers
        self.bottleneck_extra = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),  # bottleneck.4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),  # bottleneck.6
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),  # bottleneck.9
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, in_channels, 3, padding=1)
        )
        
    def forward(self, x):
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 channels, got {x.size(1)}")
            
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        b = self.bottleneck_extra(b)
        
        # Decoder
        d3 = self.decoder3(b)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        
        return d1

    def resize_input(self, x, target_size=(256, 256)):
        """Resize input to target size while maintaining aspect ratio"""
        if x.size(2) != target_size[0] or x.size(3) != target_size[1]:
            return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

# For backward compatibility
VideoRestorationModel = EncoderDecoderModel