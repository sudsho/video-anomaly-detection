"""Clip-level transforms. Operate on tensors of shape (T, H, W, 3) uint8."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class ClipTransform:
    """Compose: resize -> CHW float -> normalize -> (3, T, H, W)."""

    def __init__(
        self,
        size: int = 112,
        mean: Sequence[float] = (0.43216, 0.394666, 0.37645),
        std: Sequence[float] = (0.22803, 0.22145, 0.216989),
    ) -> None:
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, clip_thwc: np.ndarray) -> torch.Tensor:
        # clip_thwc: (T, H, W, 3) uint8
        try:
            import cv2

            T = clip_thwc.shape[0]
            out = np.zeros((T, self.size, self.size, 3), dtype=np.uint8)
            for i in range(T):
                out[i] = cv2.resize(clip_thwc[i], (self.size, self.size))
            x = torch.from_numpy(out).float() / 255.0
            x = x.permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)
        except ModuleNotFoundError:
            # cv2 is optional; fall back to a pure-torch bilinear resize so the
            # transform works in environments without OpenCV (offline smoke,
            # minimal deploys). Numerically close to cv2.resize for our use.
            x = torch.from_numpy(np.ascontiguousarray(clip_thwc)).float() / 255.0
            x = x.permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)
            x = torch.nn.functional.interpolate(
                x, size=(self.size, self.size), mode="bilinear", align_corners=False
            )
        x = (x - self.mean) / self.std
        return x
