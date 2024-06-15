"""Label utilities for UCSD Ped2 / ShanghaiTech.

UCSD Ped2 ships per-frame anomaly ground truth as a struct array
(`UCSDped2.m`) and as `_gt` mask folders. We only need binary frame labels
for AUC, so this module flattens those into a 1D array per video.
"""
from __future__ import annotations

import glob
import os
from typing import Dict

import numpy as np


def load_ucsdped2_frame_labels(gt_root: str) -> Dict[str, np.ndarray]:
    """Return {video_basename: np.ndarray of {0,1} per frame}.

    The Ped2 distribution lays out gt as ``Test001_gt/<frame>.bmp`` etc.
    A non-zero gt mask is treated as anomaly = 1.
    """
    out: Dict[str, np.ndarray] = {}
    for d in sorted(glob.glob(os.path.join(gt_root, "*_gt"))):
        base = os.path.basename(d).replace("_gt", "")
        masks = sorted(
            glob.glob(os.path.join(d, "*.bmp"))
            + glob.glob(os.path.join(d, "*.png"))
        )
        labels = np.zeros(len(masks), dtype=np.int64)
        for i, m in enumerate(masks):
            try:
                import cv2
                img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
                labels[i] = int((img > 0).any())
            except Exception:
                labels[i] = 0
        out[base] = labels
    return out
