from __future__ import annotations

import torch
from torch import nn


class ForecastMLP(nn.Module):
    def __init__(self, context: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

