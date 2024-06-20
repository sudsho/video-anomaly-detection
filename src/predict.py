"""Streaming clip inference.

Reads a video file in chunks of clip_len frames and produces a stream of
anomaly scores. Designed for surveillance feeds where end-to-end latency
matters more than batch throughput.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Iterator, Tuple

import numpy as np
import torch
import yaml

from .model import C3D, ConvAE3D, I3DLite
from .transforms import ClipTransform


def _build_model(arch: str, num_classes: int) -> torch.nn.Module:
    arch = arch.lower()
    if arch == "c3d":
        return C3D(num_classes=num_classes)
    if arch == "i3d":
        return I3DLite(num_classes=num_classes)
    return ConvAE3D()


def load_model(ckpt: str, num_classes: int = 2, device: str = "cpu") -> Tuple[torch.nn.Module, str]:
    state = torch.load(ckpt, map_location=device)
    arch = state.get("arch", "c3d")
    m = _build_model(arch, num_classes).to(device)
    m.load_state_dict(state["state_dict"])
    m.eval()
    return m, arch


def stream_clips(video_path: str, clip_len: int = 16, stride: int = 8) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (start_frame_idx, clip[T,H,W,3] uint8). Uses decord.

    Handles short videos by zero-padding the last clip (rather than dropping).
    """
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    n = len(vr)
    if n == 0:
        return
    if n < clip_len:
        idxs = list(range(n)) + [n - 1] * (clip_len - n)
        yield 0, vr.get_batch(idxs).asnumpy()
        return
    for s in range(0, n - clip_len + 1, stride):
        idxs = list(range(s, s + clip_len))
        yield s, vr.get_batch(idxs).asnumpy()


@torch.no_grad()
def score_clip(model: torch.nn.Module, arch: str, clip: np.ndarray, transform: ClipTransform, device: str) -> float:
    x = transform(clip).unsqueeze(0).to(device)
    if arch.lower() in ("autoencoder", "ae", "convae3d"):
        return float(model.reconstruction_error(x).item())
    logits = model(x)
    return float(torch.softmax(logits, dim=1)[0, 1].item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--video", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["infer"]["device"] if torch.cuda.is_available() else "cpu"
    model, arch = load_model(args.ckpt, cfg["model"]["num_classes"], device)
    tx = ClipTransform(size=cfg["data"]["img_size"])

    print("frame_start,score,latency_ms")
    for s, clip in stream_clips(args.video, cfg["data"]["clip_len"], cfg["data"]["stride"]):
        t0 = time.time()
        sc = score_clip(model, arch, clip, tx, device)
        dt_ms = (time.time() - t0) * 1000.0
        print(f"{s},{sc:.4f},{dt_ms:.1f}")


if __name__ == "__main__":
    main()
