"""Video clip dataset built on top of decord.

UCSD Ped2 / ShanghaiTech are stored as numbered frame folders. We treat each
training clip as `clip_len` consecutive frames sampled with `stride`.
Decord is much faster than PyAV/OpenCV for sequential clip reads.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from .transforms import ClipTransform

try:
    import decord  # type: ignore
    decord.bridge.set_bridge("native")
    HAS_DECORD = True
except Exception:  # pragma: no cover
    HAS_DECORD = False


def list_videos(root: str) -> List[str]:
    """Return sorted absolute paths of video files / frame folders under root."""
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) or name.endswith((".avi", ".mp4")):
            out.append(p)
    return out


@dataclass
class ClipIndex:
    video_path: str
    start: int          # start frame index
    label: int          # 0 normal, 1 anomaly (best-effort, may be -1 if unknown)


def _read_frames(path: str) -> Tuple[int, callable]:
    """Return (num_frames, getter(idx_list) -> np.ndarray (T,H,W,3) uint8)."""
    if os.path.isdir(path):
        frame_files = sorted(
            glob.glob(os.path.join(path, "*.tif"))
            + glob.glob(os.path.join(path, "*.jpg"))
            + glob.glob(os.path.join(path, "*.png"))
        )
        n = len(frame_files)

        def get(idxs):
            import cv2
            frames = []
            for i in idxs:
                img = cv2.imread(frame_files[i])
                if img is None:
                    raise IOError(f"could not read {frame_files[i]}")
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return np.stack(frames, axis=0)

        return n, get

    if not HAS_DECORD:
        raise RuntimeError("decord not installed; cannot read mp4/avi")

    vr = decord.VideoReader(path)
    n = len(vr)

    def get(idxs):
        return vr.get_batch(idxs).asnumpy()  # (T, H, W, 3)

    return n, get


class VideoClipDataset(Dataset):
    """Sliding-window clip dataset for 3D CNNs.

    Args:
        root: dataset root containing video files or per-video frame dirs.
        clip_len: temporal length of each clip (frames).
        stride: hop between successive clip starts.
        labels: optional dict {video_basename: int} mapping to 0/1; missing -> -1.
    """

    def __init__(
        self,
        root: str,
        clip_len: int = 16,
        stride: int = 8,
        img_size: int = 112,
        labels: Optional[dict] = None,
    ) -> None:
        self.root = root
        self.clip_len = clip_len
        self.stride = stride
        self.transform = ClipTransform(size=img_size)
        self.videos = list_videos(root)
        if not self.videos:
            raise FileNotFoundError(f"no videos found under {root}")
        self.index: List[ClipIndex] = []
        for vp in self.videos:
            n, _ = _read_frames(vp)
            base = os.path.basename(vp)
            label = (labels or {}).get(base, -1)
            for s in range(0, max(1, n - clip_len + 1), stride):
                self.index.append(ClipIndex(vp, s, label))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ci = self.index[i]
        _, getter = _read_frames(ci.video_path)
        idxs = list(range(ci.start, ci.start + self.clip_len))
        clip = getter(idxs)
        x = self.transform(clip)
        return x, ci.label, ci.video_path, ci.start
