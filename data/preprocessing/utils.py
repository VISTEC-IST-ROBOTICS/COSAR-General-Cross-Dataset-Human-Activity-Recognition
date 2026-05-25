# data/preprocessing/utils.py
from scipy import signal
import torch
import numpy as np
from typing import Literal
from itertools import product

def apply_filter(x:torch.Tensor, sampling_rate=50, cutoff=20, order=6, filter_type:str='lowpass') -> np.ndarray:
    nyquist = 0.5 * sampling_rate
    if filter_type == 'bandpass':
        if len(cutoff) != 2:
            raise ValueError("For bandpass filter, cutoff must be a list of two values [low, high].")
        low     = cutoff[0] / nyquist
        high    = cutoff[1] / nyquist
        cutoff  = [low, high]
    else:
        if not isinstance(cutoff, (int, float)):
            raise ValueError("For lowpass filter, cutoff must be a single value.")
        cutoff = cutoff / nyquist
    
    x   = x.detach().cpu().numpy()
    sos = signal.butter(order, cutoff, btype=filter_type, output='sos')
    x   = signal.sosfiltfilt(sos, x, padtype='even', axis=-1)
    return torch.from_numpy(x.copy())

def sign_sequence(x: torch.Tensor,indexs: list,mid_index: int,fs: int, 
                  method: Literal['mean', 'std', 'combine']) -> torch.Tensor:
    
    n_t    = x.size(-1)
    p_index = [int(tp * fs) for tp in indexs]
    k_slices = [
        (max(0, idx - mid_index), min(idx + mid_index, n_t))
        for idx in p_index
    ]

    # ── Helper: aggregate a 1D slice per window ──────────────────────────────
    def _aggregate(data: torch.Tensor, fn) -> torch.Tensor:
        return torch.tensor([fn(data[s:e]) for s, e in k_slices])

    # ── 1D input ─────────────────────────────────────────────────────────────
    if x.dim() == 1:
        if method == 'mean':
            return _aggregate(x, lambda t: t.mean())
        elif method == 'std':
            return _aggregate(x, lambda t: t.std())
        else:  # combine
            avg = _aggregate(x, lambda t: t.mean())
            std = _aggregate(x, lambda t: t.std())
            return torch.stack([avg, std], dim=-2)

    # ── Multi-dim input ───────────────────────────────────────────────────────
    n_sensor, n_channels, n_samples = x.size(0), x.size(1), x.size(2)
    out_shape      = list(x.shape)
    out_shape[-1]  = len(k_slices)

    def _apply(fn) -> torch.Tensor:
        out = torch.zeros(*out_shape)
        for sensor, ch, sample in product(range(n_sensor), range(n_channels), range(n_samples)):
            data = x[sensor, ch, sample]
            out[sensor, ch, sample] = _aggregate(data, fn)
        return out

    if method == 'mean':
        return _apply(lambda t: t.mean()).unsqueeze(-2)

    elif method == 'std':
        return _apply(lambda t: t.std()).unsqueeze(-2)

    else:  # combine
        avg_tensor = _apply(lambda t: t.mean())
        std_tensor = _apply(lambda t: t.std())
        return torch.cat([avg_tensor.unsqueeze(-2),
                          std_tensor.unsqueeze(-2)], dim=-2)
        
def animate_heatmaps(x, subtitles, title="CWT animation"):
    """
    Animate a sequence of 2D heatmaps using Plotly.
    The input tensor `x` is expected to have the shape [n_samples, n_sensors, n_channels, n_scales, time], 
    and the function will create an animation that iterates over the sample index (first dimension). 
    Each frame of the animation will display heatmaps for the first and last sensor across all channels and scales.
    """
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # x shape assumption: [n_samples, ?, ?, ?]
    
    cmax = x.max().item()
    cmin = x.min().item()
    
    n_frames = x.shape[0]   # loop over sample index
    fig = make_subplots(rows=2, cols=3, subplot_titles=subtitles,
                        horizontal_spacing=0.08, vertical_spacing=0.15)

    # Initial traces (first frame)
    scales = x.shape[3]  # Assuming scales is the 4th dimension
    ts = 1 / (scales * 25)
    fig.add_trace(go.Heatmap(z=x[0, 0, 0].numpy(), showscale=True, coloraxis="coloraxis"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=x[0, 0, 1].numpy(), showscale=False, coloraxis="coloraxis"), row=1, col=2)
    fig.add_trace(go.Heatmap(z=x[0, 0, 2].numpy(), showscale=False, coloraxis="coloraxis"), row=1, col=3)
    fig.add_trace(go.Heatmap(z=x[0, -1, 0].numpy(), showscale=False, coloraxis="coloraxis"), row=2, col=1)
    fig.add_trace(go.Heatmap(z=x[0, -1, 1].numpy(), showscale=False, coloraxis="coloraxis"), row=2, col=2)
    fig.add_trace(go.Heatmap(z=x[0, -1, 2].numpy(), showscale=False, coloraxis="coloraxis"), row=2, col=3)

    # Build frames
    frames = []
    for k in range(n_frames):
        frames.append(go.Frame(
            data=[
                go.Heatmap(z=x[k, 0, 0].numpy(), coloraxis="coloraxis"),
                go.Heatmap(z=x[k, 0, 1].numpy(), coloraxis="coloraxis"),
                go.Heatmap(z=x[k, 0, 2].numpy(), coloraxis="coloraxis"),
                go.Heatmap(z=x[k, -1, 0].numpy(), coloraxis="coloraxis"),
                go.Heatmap(z=x[k, -1, 1].numpy(), coloraxis="coloraxis"),
                go.Heatmap(z=x[k, -1, 2].numpy(), coloraxis="coloraxis"),
            ],
            name=str(k)
        ))

    # Add play/pause buttons and slider
    fig.update(frames=frames)
    fig.update_layout(
        title=title,
        coloraxis={'colorscale': 'Viridis',
                    'cmin':cmin, 'cmax':cmax,
                'colorbar': {'title': 'Magnitude', 'thickness': 20}},
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True}],
                "label": "▶ Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate"}],
                "label": "⏸ Pause", "method": "animate"}
            ],
            "direction": "left", "pad": {"r": 10, "t": 70}, "type": "buttons",
            "x": 0.15, "y": 0.075
        }],
        sliders=[{
            "steps": [
                {"args": [[str(k)], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate"}],
                "label": str(k), "method": "animate"}
                for k in range(n_frames)
            ],
            "x": 0.15, "y": 0, "len": 0.9
        }]
    )

    fig.show(renderer="browser")
    
def animate_multiplot(x, plot_titles:list=None, title="Line animation", ylim=None):
    """
    x: np.array or torch.Tensor with shape [n_frames, n_plots, n_signals, n_times]
    - n_plots = number of subplots
    - n_signals = 3 (x,y,z)
    plot_titles: list of titles for each subplot
    """
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not isinstance(x, np.ndarray):
        x = x.numpy()

    n_frames, n_rows, n_signals, n_times = x.shape
    t = np.arange(n_times)
    if n_frames > 300:
        n_frames = 300

    # Subplots: 2 rows, 1 column
    fig = make_subplots(rows=n_rows, cols=1,
                        subplot_titles=plot_titles,
                        shared_xaxes=True, horizontal_spacing=0.05, vertical_spacing=0.1,)

    # Colors for x,y,z
    colors = ["red", "green", "blue"]
    names = ["x", "y", "z"]

    # Add initial traces (frame 0)
    for r in range(n_rows):
        for s in range(n_signals):
            fig.add_trace(
                go.Scatter(y=x[0, r, s], x=t, mode="lines",
                        name=f"{plot_titles[r]}-{names[s]}",
                        ),
                row=r+1, col=1
            )

    # Build frames
    frames = []
    for k in range(n_frames):
        frame_data = []
        for r in range(n_rows):
            for s in range(n_signals):
                frame_data.append(
                    go.Scatter(y=x[k, r, s], x=t, mode="lines",)
                )
        frames.append(go.Frame(data=frame_data, name=str(k)))

    # Add play/pause and slider
    fig.update(frames=frames)
    fig.update_layout(
        title=title,
        updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "y": -0.15,
                "x": 0,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {"label": "Play", "method": "animate", 
                    "args": [None, {"frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 0}}]},
                    {"label": "Pause", "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]}
                ]
            }],
        sliders=[{
            "steps": [
                {
                    "args": [[str(s)], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                    "label": f"{s}",
                    "method": "animate",
                }
                for s in range(n_frames)
            ],
            "x": 0.15,
            "y": -0.05,
            "xanchor": "left",
            "yanchor": "top"
        }]
    )
    fig.update_yaxes(range=ylim)
    fig.show(renderer="browser")
