# models folder: cosar.py

from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict

from utils.utils import calculate_output_size

class CoSAR(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 num_sensors: int,
                 num_filters: int = 3,
                 num_channels: int = 1,
                 feature_size: tuple = (27, 75),
                 use_bias: bool = True,
                 noise_alpha: float = 2.5,
                 ):
        super(CoSAR, self).__init__()

        self.num_classes    = num_classes
        self.num_sensors    = num_sensors
        self.feature_size   = feature_size
        self._is_use_bias   = use_bias
        self.noise_alpha    = noise_alpha
        
        # Create the variables for the model
        self.kernel_sizes   = [(4,4), (6,6)]
        self.strides        = [(1,1), (2,2)]
        self.num_filters    = num_filters
        
        # Calculate the output sizes for each convolutional layer
        H_size, W_size = feature_size
        for kernel, stride in zip(self.kernel_sizes, self.strides):
            H_size, W_size = calculate_output_size(H_size, W_size, kernel, stride)
            
        # Create the unified-convolutional layers
        self.unified_convs = nn.Sequential(
            nn.LayerNorm([self.num_filters, self.feature_size[0], self.feature_size[1]]),
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=self.kernel_sizes[0], stride=self.strides[0], bias=self._is_use_bias),
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=self.kernel_sizes[1], stride=self.strides[1], bias=self._is_use_bias),
        )
        
        # Get the size of sequence and embedding features
        self.embedding_size  = self.num_sensors * self.num_filters * H_size
        self.sequence_size   = W_size
        self.new_feature_size = self.embedding_size
        
        self.new_feature_size = 128
        # # Create feature expansion model
        self.fe = nn.Sequential(
            nn.Linear(self.embedding_size, self.new_feature_size),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # Create multihead self-attention
        self.num_heads = 2
        self.mha = nn.MultiheadAttention(embed_dim=self.new_feature_size, 
                                         num_heads=self.num_heads, 
                                         batch_first=True,
                                         dropout=0.1)
        
        # Create the classification head
        self.fc = nn.Sequential(            
            nn.BatchNorm1d(self.new_feature_size),
            nn.Linear(self.new_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, self.num_classes)
        )
        
    def forward(self, x, train=False, test=False):
        """
        The forward pass of the model.
        Args:
            x: (batch_size, num_sensors, num_channels, H, W) -> the real number is [batch_size, 2, 3, 27, 75]
        """
        x = x.float()
        # Get the size of the input
        b, s, c, h, w = x.size()
        x = x.view(b * s, c, h, w)  # Reshape to (batch_size * num_sensors, num_channels, H, W)
        
        # Pass through the unified-convolutional layers ===============
        x = self.unified_convs(x)
        
        # Reshape back to (batch_size, W', num_sensors * num_filters * H')
        _, f, h, w = x.size()
        x = x.view(b, s * f * h, w)
        x = x.transpose(1, 2)  # Transpose to (batch_size, sequence_size, embedding_size)
        
        # # FE module
        x = self.fe(x) # [batch_size, sequence_size, 128]
        
        # Multihead self-attention ====================================
        if train:
            dims = np.prod(x.shape[1:])
            noise_mag = self.noise_alpha / np.sqrt(dims)
            x = x + torch.zeros_like(x).uniform_(-noise_mag, noise_mag)
        
        x, _ = self.mha(x, x, x) # [batch_size, sequence_size, 128]
        
        # Average pooling over the sequence dimension =================
        x = torch.mean(x, dim=1).squeeze(-1)  # # [bs, hidden_features]
        
        # Classification head =========================================
        x = self.fc(x)
        return x