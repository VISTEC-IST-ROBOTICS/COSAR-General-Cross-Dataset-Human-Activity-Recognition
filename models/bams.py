# models folder
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict

from utils.utils import calculate_output_size

class BAMS(nn.Module):
    def __init__(self, 
                 num_classes, 
                 num_sensors,
                 num_channels, 
                 use_bias=True,
                 noise_alpha=2.5) -> None:
        super(BAMS, self).__init__()

        self.num_classes    = num_classes
        self.num_sensors    = num_sensors
        self.num_channels   = num_channels

        self._use_bias   = use_bias
        self.noise_alpha = noise_alpha
        self.kernel_size = [(9,5), (9,5)]
        self.num_filters = self.num_channels * 3
        self.stride     = [(1,1), (1,1)]
        print('BAMS Model Initialized')
        
        self.conv = nn.ModuleList([
            nn.Conv2d(self.num_filters,
                    self.num_filters,     
                    kernel_size=self.kernel_size[0],
                    stride=self.stride[0])
            for _ in range(self.num_sensors)  # 2 sensors → 2 conv layers
        ])
        self.conv_relu = nn.ModuleList([nn.ReLU() for _ in range(self.num_sensors)])
        
        # calculate the output size after conv layers
        H_in, W_in  = 27, 75
        H_in, W_in = calculate_output_size(H_in, W_in, (9,5), (1,1))
        
        # Get the size of sequence and embedding features
        self.embed_feautres     = H_in * W_in
        self.sequence_length    = self.num_sensors * (self.num_channels * 3)
        
        # define the values for multi-head attention
        num_heads   = 1
        drop_out    = 0.0
        self.attention_heads = nn.MultiheadAttention(self.embed_feautres, 
                                                     num_heads=num_heads, 
                                                     batch_first=True, 
                                                     dropout=drop_out)
        # define the values for layer normalization, and MLPs
        self.norm = nn.LayerNorm(self.embed_feautres)
        self.fc = nn.Sequential(
            nn.Linear(self.embed_feautres, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, self.num_classes)
        )

    def forward(self, input_tensor, train=False, test=False, fine_tune=False, visualize=False, labels=None):
        
        input_tensor = input_tensor.float()
        batch, n_sensor, n_channel, H, W = input_tensor.size()

        output_tensor = []
        # input_tensor  = input_tensor.transpose(0, 1) # transpose the dimension-> [n_sensor, bs, n_channel, H, W]
        
        # Sensor-wise Convolution
        for s in range(n_sensor):
            ip = input_tensor[:, s]     # [bs, 1, H, W]
            ip = self.conv[s](ip)       # [bs, 3, H_out, W_out]
            ip = self.conv_relu[s](ip)
            output_tensor.append(ip)

        x = torch.stack(output_tensor, dim=0)           # Stack all output from all sensor together
        x = x.transpose(0, 1)                           # [bs, n_sensor, n_channels, H, W]

        # self-attention  =========================================================
        
        # Keep time (W) as sequence dimension
        x = x.flatten(3)
        x = x.flatten(1,2)                  # [bs, n_sensorxn_ch, H_outxW_out] 
        
        if train:
            dims = np.prod(x.shape[1:])
            noise_mag = self.noise_alpha / np.sqrt(dims)
            x = x + torch.zeros_like(x).uniform_(-noise_mag, noise_mag)
        
        output, attention_weights = self.attention_heads(x, x, x)   
        x = torch.mean(output, dim=1).squeeze(-1)  # # [bs, hidden_features]

        # Classification =========================================================
        x = self.fc(x)
        
        return x