from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionEmbedding3D(nn.Module):
    def __init__(self, hidden_dim: int, temperature: float = 10_000.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        frequencies = max(1, math.ceil(hidden_dim / 6))
        scale = torch.arange(frequencies, dtype=torch.float32) / frequencies
        self.register_buffer("frequency", temperature**scale, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(frequencies * 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, xyz: Tensor) -> Tensor:
        phase = xyz.unsqueeze(-1) / self.frequency.to(dtype=xyz.dtype)
        encoded = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        return self.projection(encoded)

