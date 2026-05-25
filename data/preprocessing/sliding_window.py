# data/preprocessing/sliding_window.py
from data.preprocessing.base import BasePreprocessor
import torch

class SlidingWindow(BasePreprocessor):
    def __init__(self, window: int, step: int, sampling:int):
        self.window_length = int(window * sampling)
        self.step_length   = int(step* sampling)

    def transform(self, dataset):
        desc = f'Sliding Window: w={self.window_length}, s={self.step_length}'
        return self._iterate_dataset(dataset, desc, self._apply_sliding_window)

    def _apply_sliding_window(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unfold(-1, self.window_length, self.step_length)  # [n_sensor, n_channels, n_windows, window_length]

        if x.ndim == 4:
            # [n_windows, n_sensors, n_channels, n_times]
            x = x.permute(2, 0, 1, 3)
        else:
            # flatten sample and sliding window dimensions
            x = x.permute(0, 3, 1, 2, 4)
            x = x.reshape(-1, x.size(2), x.size(3), x.size(4))

        return x