import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class SensorDataAdaptation(nn.Module):
    """
    Module 1: Sensor Data Adaptation
    - Instance Normalization (per channel)
    - Channel Independency
    - Sensor Segmentation
    - Conv1D Embedding
    """
    def __init__(self, in_channels=6, segment_len=20, d_model=256):
        super().__init__()
        self.segment_len   = segment_len
        self.instance_norm = nn.InstanceNorm1d(in_channels)
        # Conv1D: each segment → d_model dimensional token
        self.conv1d = nn.Conv1d(
            in_channels  = 1,
            out_channels = d_model,
            kernel_size  = segment_len,
            stride       = segment_len
        )

    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)          # (B, C, T)
        x = self.instance_norm(x)        # normalize each channel independently

        B, C, T = x.shape
        # Channel Independency: treat each channel as separate sequence
        x = x.reshape(B * C, 1, T)      # (B*C, 1, T)
        # Segmentation + Embedding via Conv1D
        x = self.conv1d(x)              # (B*C, d_model, num_segments)
        x = x.permute(0, 2, 1)          # (B*C, num_segments, d_model)
        return x, B, C


class SensorKnowledgeLearning(nn.Module):
    def __init__(self, d_model=128, num_classes=4, num_layers=2):
        super().__init__()

        n_head = 8
        assert d_model % n_head == 0, \
            f"d_model ({d_model}) must be divisible by n_head ({n_head})"

        self.pos_embedding = nn.Parameter(torch.randn(1, 256, d_model))

        config = GPT2Config(
            n_embd      = d_model,
            n_layer     = num_layers,
            n_head      = n_head,
            resid_pdrop = 0.1,
            attn_pdrop  = 0.1,
            embd_pdrop  = 0.0,
        )
        self.gpt2 = GPT2Model(config)


        # Freeze all first
        for param in self.gpt2.parameters():
            param.requires_grad = False

        for name, param in self.gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name or 'c_proj' in name:
                param.requires_grad = True

        self.layer_norm = nn.LayerNorm(d_model)

        self.activity_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, seq_len):
        # x: (B*C, num_segments, d_model)

        # Only add pos_embedding if more than 1 segment
        # With seq_len=1, it has no effect so skip it
        if seq_len > 1:
            x = x + self.pos_embedding[:, :seq_len, :]

        outputs = self.gpt2(inputs_embeds=x)
        x = outputs.last_hidden_state    # (B*C, num_segments, d_model)

        x   = self.layer_norm(x)
        x   = x.mean(dim=1)             # (B*C, d_model)
        out = self.activity_projection(x)
        return out


class LLM4HAR(nn.Module):
    """
    LLM4HAR: Generalizable On-device Human Activity Recognition
    with Pretrained LLMs (KDD 2025)

    Input shape (your pipeline): (B, n_sensors, n_channels, T)
    Output shape:                (B, num_classes)
    """
    def __init__(
        self,
        in_channels = 6,     # n_sensors * n_channels
        T           = 120,   # time steps
        segment_len = 20,    # length of each segment (token)
        d_model     = 256,   # embedding dimension (smaller → ~8M params)
        num_classes = 4,     # number of activity classes
        num_layers  = 4      # number of GPT-2 decoder layers
    ):
        super().__init__()

        assert T % segment_len == 0, \
            f"T ({T}) must be divisible by segment_len ({segment_len})"

        self.num_segments = T // segment_len  # number of tokens per channel

        self.adaptation = SensorDataAdaptation(in_channels, segment_len, d_model)
        self.knowledge  = SensorKnowledgeLearning(d_model, num_classes, num_layers)

    def forward(self, x, **kwargs):
        # Support both input formats:
        if x.ndim == 4:
            # Pipeline format: (B, n_sensors, n_channels, T)
            B, S, C, T = x.shape
            x = x.reshape(B, S * C, T)  # (B, in_channels, T)
            x = x.permute(0, 2, 1)      # (B, T, in_channels)
        elif x.ndim == 3:
            # Direct format: (B, T, in_channels)
            pass

        x, B, C = self.adaptation(x)                      # (B*C, num_segments, d_model)
        x = self.knowledge(x, self.num_segments)           # (B*C, num_classes)
        x = x.reshape(B, C, -1).mean(dim=1)               # (B, num_classes)
        return x


# ─────────────────────────────────────────
# Parameter count utility
# ─────────────────────────────────────────
def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total/1e6:.2f}M")
    print(f"Trainable params: {trainable/1e6:.2f}M")
    print(f"Frozen params:    {(total - trainable)/1e6:.2f}M")


# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Your pipeline format: (B, n_sensors, n_channels, T)
    bs, n_sensors, n_channels, T = 8, 2, 3, 150
    segment_len = 15   # 150 / 15 = 10 segments

    model = LLM4HAR(
        in_channels = n_sensors * n_channels,  # 6
        T           = T,
        segment_len = segment_len,
        d_model     = 256,   # reduced from 768 → closer to 8M params
        num_classes = 7,
        num_layers  = 4
    )

    x   = torch.randn(bs, n_sensors, n_channels, T)
    out = model(x)

    print(f"Input shape:  {x.shape}")    # (8, 2, 3, 150)
    print(f"Output shape: {out.shape}")  # (8, 7)
    print()
    count_parameters(model)