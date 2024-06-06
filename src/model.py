"""3D CNN models for video anomaly detection.

Two heads are exposed:
- C3D supervised classifier
- 3D conv autoencoder (unsupervised, reconstruction error -> anomaly score)
"""
from __future__ import annotations

import torch
from torch import nn


class C3D(nn.Module):
    """Tran et al. 2015 style C3D, slightly modernized.

    Input shape: (B, 3, T, H, W). Default T=16, H=W=112.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        # placeholder; filled out in next commits
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h.mean(dim=(2, 3, 4))
        return self.fc(h)
