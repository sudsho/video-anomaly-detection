"""3D CNN models for video anomaly detection.

Two heads are exposed:
- C3D supervised classifier (Tran et al. 2015, modernized)
- 3D conv autoencoder (unsupervised, reconstruction error -> anomaly score)
"""
from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_c: int, out_c: int, pool: bool = True, pool_t: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    ]
    if pool:
        kt = 2 if pool_t else 1
        layers.append(nn.MaxPool3d(kernel_size=(kt, 2, 2), stride=(kt, 2, 2)))
    return nn.Sequential(*layers)


class C3D(nn.Module):
    """Tran et al. 2015 C3D, BN added.

    Input shape: (B, 3, T, H, W). Default T=16, H=W=112.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 64, pool=True, pool_t=False),     # T=16, H=56
            _conv_block(64, 128, pool=True, pool_t=True),    # T=8,  H=28
            _conv_block(128, 256, pool=True, pool_t=True),   # T=4,  H=14
            _conv_block(256, 512, pool=True, pool_t=True),   # T=2,  H=7
            _conv_block(512, 512, pool=True, pool_t=True),   # T=1,  H=3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.classifier(h)
