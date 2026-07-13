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


class ImageMLP(nn.Module):
    """Batch-one exact-recovery baseline with a raw-input first linear layer."""

    def __init__(self, image_shape: tuple[int, int, int], classes: int = 2, hidden: int = 128):
        super().__init__()
        channels, height, width = image_shape
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * height * width, hidden),
            nn.GELU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SmallLeNet(nn.Module):
    """DLG-style shallow convolutional victim for high-fidelity inversion."""

    def __init__(self, image_shape: tuple[int, int, int], classes: int = 2, width: int = 6):
        super().__init__()
        channels, height, width_pixels = image_shape
        self.network = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(width * height * width_pixels, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
