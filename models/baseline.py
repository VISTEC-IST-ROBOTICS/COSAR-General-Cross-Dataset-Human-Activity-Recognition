# models folder

import os
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go

class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
        self.cnn_sequence = nn.Sequential(
            nn.Conv1d(6, 16, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1),
            
            nn.Conv1d(16, 32, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1),
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
            # nn.Softmax(dim=1)
        )
    
    def forward(self, x, train=None, test=None):
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.cnn_sequence(x)
        x = torch.mean(x, dim=-1)
        x = self.fc(x)
        return x