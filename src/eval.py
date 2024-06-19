"""Frame-level AUC-ROC evaluation.

Anomaly score per clip is mapped back to its constituent frames; multiple
clip scores covering the same frame are aggregated (max by default).
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import VideoClipDataset
from .labels import load_ucsdped2_frame_labels
from .model import C3D, ConvAE3D, I3DLite


def _build_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "c3d":
        return C3D(num_classes=num_classes)
    if arch == "i3d":
        return I3DLite(num_classes=num_classes)
    return ConvAE3D()


@torch.no_grad()
def score_clips(model, loader, device: str, is_ae: bool) -> List[dict]:
    model.eval()
    out: List[dict] = []
    for x, y, paths, starts in loader:
        x = x.to(device)
        if is_ae:
            score = model.reconstruction_error(x).cpu().numpy()
        else:
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            score = prob.cpu().numpy()
        for s, p, st in zip(score, paths, starts.tolist()):
            out.append({"video": os.path.basename(p), "start": int(st), "score": float(s)})
    return out


def aggregate_to_frames(
    clip_scores: List[dict],
    clip_len: int,
    n_frames_per_video: Dict[str, int],
    agg: str = "max",
) -> Dict[str, np.ndarray]:
    """Map per-clip scores back onto per-frame scores."""
    buckets: Dict[str, List[List[float]]] = {}
    for vid, n in n_frames_per_video.items():
        buckets[vid] = [[] for _ in range(n)]
    for c in clip_scores:
        vid = c["video"]
        if vid not in buckets:
            continue
        for f in range(c["start"], min(c["start"] + clip_len, len(buckets[vid]))):
            buckets[vid][f].append(c["score"])
    out: Dict[str, np.ndarray] = {}
    for vid, frames in buckets.items():
        arr = np.zeros(len(frames), dtype=np.float32)
        for i, vals in enumerate(frames):
            if not vals:
                continue
            if agg == "max":
                arr[i] = max(vals)
            else:
                arr[i] = float(np.mean(vals))
        out[vid] = arr
    return out


def frame_auc(
    frame_scores: Dict[str, np.ndarray],
    frame_labels: Dict[str, np.ndarray],
) -> float:
    y_true: List[int] = []
    y_score: List[float] = []
    for vid, scores in frame_scores.items():
        if vid not in frame_labels:
            continue
        labs = frame_labels[vid]
        n = min(len(scores), len(labs))
        y_true.extend(labs[:n].tolist())
        y_score.extend(scores[:n].tolist())
    if not y_true:
        raise ValueError("no overlap between frame scores and labels")
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gt-root", required=True, help="dir with <vid>_gt mask folders")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds = VideoClipDataset(
        root=cfg["data"]["root"],
        clip_len=cfg["data"]["clip_len"],
        stride=cfg["data"]["stride"],
        img_size=cfg["data"]["img_size"],
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device)
    arch = state.get("arch", cfg["model"]["arch"])
    model = _build_model(arch, cfg["model"]["num_classes"]).to(device)
    model.load_state_dict(state["state_dict"])

    is_ae = arch.lower() in ("autoencoder", "ae", "convae3d")
    clip_scores = score_clips(model, loader, device, is_ae)

    # frame counts come from the dataset's index
    n_frames: Dict[str, int] = defaultdict(int)
    for ci in ds.index:
        n_frames[os.path.basename(ci.video_path)] = max(
            n_frames[os.path.basename(ci.video_path)], ci.start + cfg["data"]["clip_len"]
        )

    frame_scores = aggregate_to_frames(
        clip_scores, cfg["data"]["clip_len"], dict(n_frames), agg=cfg["eval"]["agg"]
    )
    labels = load_ucsdped2_frame_labels(args.gt_root)
    auc = frame_auc(frame_scores, labels)
    print(f"frame AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
