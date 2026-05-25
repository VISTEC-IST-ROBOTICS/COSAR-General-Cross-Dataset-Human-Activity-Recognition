# data/preprocessing/aug_cwt.py
import torch
from torchvision import transforms

import numpy as np
from tqdm import tqdm
from pywt import cwt
from data.preprocessing.base import BasePreprocessor
from data.preprocessing.utils import apply_filter


class AugmentedCWT(BasePreprocessor):
    def __init__(self, wavelet='mexh', n_scales=24, scale_type='geo', sampling=50):
        self.wavelet     = wavelet
        self.n_scales    = n_scales
        self.scale_type  = scale_type
        self.max_scale   = 64
        self.high_cutoff = 1.0
        self.sampling    = sampling
        self.n_repeat    = 3
        self.time_resize = 75
        
        self.resizer = transforms.Resize(
            size=(self.n_scales + self.n_repeat, self.time_resize),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True  # <--- CRITICAL FOR SIGNALS
        )

    def transform(self, dataset):
        return self._iterate_dataset(dataset, 'Augmented scalogram computation', self._apply_augmented)

    def _apply_augmented(self, x):
        # Compute CWT and concatenate original input
        additional_x = x.unsqueeze(-2)
        x = self.compute_cwt(x)

        for _ in range(self.n_repeat):
            x = torch.cat((additional_x, x), dim=-2)

        # Resize time dimension
        bs, n_s, n_ch, current_scales, t = x.shape
        
        before_x = x.clone()
        
        x = x.view(bs * n_s, n_ch, current_scales, t)
        x = self.resizer(x)
        x = x.view(bs, n_s, n_ch, current_scales, self.time_resize)
    
        
        return x

    def _create_scale_range(self):
        if self.scale_type == 'geo':
            return np.geomspace(1, self.max_scale, num=self.n_scales, dtype=np.int32)
        elif self.scale_type == 'linear':
            return np.linspace(1, self.max_scale, num=self.n_scales, dtype=np.int32)
        else:
            raise ValueError(f"Invalid scale type: {self.scale_type}. Choose 'geo' or 'linear'.")

    def compute_cwt(self, x):
        if len(x.size()) != 4:
            raise ValueError(f"Input tensor must be 4D, got {len(x.size())}D.")

        self.scale_range = self._create_scale_range()

        # Filter DC offset before CWT
        x = apply_filter(x, cutoff=self.high_cutoff, order=6, filter_type='highpass', sampling_rate=self.sampling)

        # Compute CWT
        coeffs, _ = cwt(x.numpy(), self.scale_range, self.wavelet, sampling_period=1/self.sampling)
        coeffs = torch.abs(torch.tensor(coeffs, dtype=torch.float32))  # [scales, bs, n_sensor, n_ch, coeffs]
        coeffs = coeffs.permute(1, 2, 3, 0, 4)

        return coeffs
    
    