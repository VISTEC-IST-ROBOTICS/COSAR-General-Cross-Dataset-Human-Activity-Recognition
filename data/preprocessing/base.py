# data/preprocessing/base.py
from abc import ABC, abstractmethod
from tqdm import tqdm
from termcolor import colored

class BasePreprocessor(ABC):
    def __call__(self, dataset):
        return self.transform(dataset)
    
    def status_info(self) -> list[str]:
        """Override in subclass to show info after progress bar."""
        return []
    
    @abstractmethod
    def transform(self, dataset):
        """Each subclass must implement this"""
        raise NotImplementedError
    
    def _iterate_dataset(self, dataset, desc, operation):
        BAR_DESC_WIDTH = 30
        classes    = dataset['classes']
        n_subjects = len(dataset['data'])
        
        if 'scalogram' in desc:         title = 'scalogram'
        elif 'spectrogram' in desc:     title = 'spectrogram'
        
        with tqdm(range(n_subjects), desc=f'{desc}'.ljust(BAR_DESC_WIDTH), unit='subject') as pbar:
            for s in pbar:
                for activity in tqdm(classes, desc=f"    └── Subject {s+1}", leave=False):
                    x = dataset['data'][s][activity]
                    if x is None:
                        continue
                    dataset['data'][s][activity] = operation(x) 

        return dataset
    
    def _plot(self, x, title):
        import os
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # import torch

        def get_unique_filename(filepath):
            if not os.path.exists(filepath):
                return filepath
            base, ext = os.path.splitext(filepath)
            counter = 1
            while os.path.exists(f"{base}_{counter}{ext}"):
                counter += 1
            return f"{base}_{counter}{ext}"

        SAMPLE_IDX  = 10
        CHANNEL_IDX = 2
        fig = go.Figure()
            # cmin = torch.min(x[SAMPLE_IDX, 1, CHANNEL_IDX]).item()
            # cmax = torch.max(x[SAMPLE_IDX, 1, CHANNEL_IDX]).item()
        cmin, cmax = -2.1, 3.2
        print(f"cmin: {cmin}, cmax: {cmax}")
        fig.add_trace(go.Heatmap(z=x[SAMPLE_IDX, 1, CHANNEL_IDX].cpu(),coloraxis="coloraxis"))
        fig.update_layout(
            coloraxis=dict(colorscale='Twilight', cmin=cmin, cmax=cmax, colorbar=dict(thickness=15, len=300,lenmode="pixels"
            )),
            height=100, width=250,
            font=dict(size=15,          # global font size
                      family="Open Sans, sans-serif",),
            margin=dict(t=5, b=0, l=10, r=10),  # Reduce top margin
        )
        fig.write_image(f"images/{title}_resize_{self.focus_activity}.png")