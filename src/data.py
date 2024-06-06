"""Video clip dataset built on top of decord.

UCSD Ped2 / ShanghaiTech are stored as numbered frame folders. We treat each
training clip as `clip_len` consecutive frames sampled with `stride`.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np


def list_videos(root: str) -> List[str]:
    """Return absolute paths of video files (or frame folders) under root."""
    out: List[str] = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) or name.endswith((".avi", ".mp4")):
            out.append(p)
    return out
