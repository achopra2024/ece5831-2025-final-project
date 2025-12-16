"""
Improved CNN model architecture with residual connections and batch normalization.
"""

import torch
import torch.nn as nn


class ImprovedAudioCNN(nn.Module):
    """
    Enhanced CNN with residual connections and deeper architecture.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels
        dropout (float): Dropout rate
    """
    
    def __init__(self, num_classes=10, input_channels=1, dropout=0.3):
        super(ImprovedAudioCNN, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Residual Block 1
        self.res_block1 = self._make_residual_block(32, 64, dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual Block 2
        self.res_block2 = self._make_residual_block(64, 128, dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual Block 3
        self.res_block3 = self._make_residual_block(128, 256, dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual Block 4
        self.res_block4 = self._make_residual_block(256, 512, dropout)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with attention-like mechanism
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, num_classes)
        )
    
    def _make_residual_block(self, in_channels, out_channels, dropout):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Initial conv
        x = self.initial_conv(x)
        
        # Residual blocks with pooling
        x = self.res_block1(x)
        x = self.pool1(x)
        
        x = self.res_block2(x)
        x = self.pool2(x)
        
        x = self.res_block3(x)
        x = self.pool3(x)
        
        x = self.res_block4(x)
        
        # Global pooling and classification
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        
        return x


class AttentionAudioCNN(nn.Module):
    """
    CNN with attention mechanism for better feature selection.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels
        dropout (float): Dropout rate
    """
    
    def __init__(self, num_classes=10, input_channels=1, dropout=0.3):
        super(AttentionAudioCNN, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.3),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.4),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Convolutional features
        features = self.conv_blocks(x)
        
        # Attention weights
        attention_weights = self.attention(features)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention
        features = features * attention_weights
        
        # Global pooling and classification
        x = self.global_pool(features)
        x = self.classifier(x)
        
        return x
