# data/preprocessing/aug_stft.py
import torch
from torchvision import transforms
from scipy import signal
from data.preprocessing.base import BasePreprocessor
from data.preprocessing.utils import apply_filter, sign_sequence

class AugmentedSTFT(BasePreprocessor):
    def __init__(self, window_size=32, hop=2, sampling=50):
        self.high_cutoff = 1.0
        self.sampling    = sampling
        self.n_repeat    = 3
        self.time_resize = 75
        self.new_coeffs  = 27
        self.window_size     = window_size
        self.hop             = hop  

        window_func      = signal.windows.hamming(window_size)
        self.stft_func   = signal.ShortTimeFFT(window_func, hop=hop, fs=self.sampling, scale_to='magnitude')
        self.t_stft      = self.stft_func.t(3 * self.sampling)
        self.mid_indexs  = self.stft_func.m_num_mid
        
        self.resizer = transforms.Resize(
            size=(self.new_coeffs, self.time_resize),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True  # <--- CRITICAL FOR SIGNALS
        )
        
    def transform(self, dataset):
        return self._iterate_dataset(dataset, f'Augmented spectrogram with w={self.window_size},hop={self.hop}', self._apply_augmented)

    def _apply_augmented(self, x):
        # Compute sign sequence
        binary_sequence = sign_sequence(x, self.t_stft, self.mid_indexs, fs=self.sampling, method='mean')

        # Filter the signals
        x = apply_filter(x, cutoff=self.high_cutoff, order=6, filter_type='highpass', sampling_rate=self.sampling)

        # Compute STFT per sample
        output_tensor = []
        for n_sample in range(x.size(0)):
            sample_tensor = x[n_sample]
            sample_tensor = self.stft_func.spectrogram(sample_tensor.cpu().numpy())
            sample_tensor = torch.tensor(sample_tensor, dtype=torch.float32).unsqueeze(0)
            output_tensor.append(sample_tensor)

        x = torch.cat(output_tensor, dim=0)  # [n_samples, n_sensors, n_channels, n_freqs, n_frames]

        for _ in range(self.n_repeat):
            x = torch.cat((binary_sequence, x), dim=-2)
        before_x = x.clone()
        # Resize time dimension
        bs, n_s, n_ch, coeffs, t = x.shape
        
        x = x.view(bs * n_s, n_ch, coeffs, t)
        x = self.resizer(x)
        x = x.view(bs, n_s, n_ch, self.new_coeffs, self.time_resize)
        return x

        