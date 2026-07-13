from __future__ import annotations

import torch
from torch import nn


class TinyConvNet(nn.Module):
    def __init__(self, classes: int = 2, width: int = 24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Linear(width * 2 * 4, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

