# data/preprocessing/preprocessing.py
import torch.nn.functional as F
from data.preprocessing.base import BasePreprocessor
from data.preprocessing.utils import apply_filter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Preprocessing(BasePreprocessor):
    def __init__(self, dataset_sampling:int, desired_sampling:int=50):
        self.dataset_sampling   = dataset_sampling
        
        # Default parameters for preprocessing
        self.desired_sampling   = 50
        self.low_cutoff         = 20.0
        self.butterworth_order  = 6
        
        if dataset_sampling != self.desired_sampling:
            self.resample_factor    = self.desired_sampling / dataset_sampling
            self.sampling_rate      = self.resample_factor * dataset_sampling
        else:
            self.resample_factor    = 1.0
            self.sampling_rate      = dataset_sampling
        print(f'Initialized Preprocessing with dataset sampling rate {dataset_sampling} Hz and desired sampling rate {desired_sampling} Hz. Resample factor: {self.resample_factor:.2f}')
    
    def transform(self, dataset):
        # desc = f'Preprocessing using resample factor {self.resample_factor} and low-pass filter with cutoff {self.low_cutoff} Hz'
        return self._iterate_dataset(dataset, 'Data Preprocessing', self._process)
    
    def _process(self, x):
        n_times = x.size(-1)
        
        # Convert from m/s^2 to g
        x /= 9.81
                
        # downsample to 50 Hz and check the number of dimension of input
        if x.ndim == 3:
            # no sample dimension, just a sequence
            n_size  = int(n_times * self.resample_factor) 
            x = F.interpolate(x, size=n_size, mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=(1, self.resample_factor), mode='bicubic')
        
        # Filter to remove high-frequency noise
        x = apply_filter(x, self.low_cutoff, self.butterworth_order, filter_type='lowpass')
        
        return x