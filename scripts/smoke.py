"""Tiny-CPU offline smoke for video-anomaly-detection.

No downloads, no GPU, no decord/OpenCV/AWS. This exercises the real repo code
paths (the ConvAE3D autoencoder from ``src.model`` and the frame-level scoring
in ``src.eval``) end to end on tiny SYNTHETIC video clips:

  * normal clips  = smooth, low-frequency spatiotemporal patterns
  * anomaly clips = the same, with a bright blob + noise injected into a segment

The 3D conv autoencoder is trained for a few steps on normal clips only, so its
reconstruction error is low on normal footage and rises on the injected
anomalies. We then reuse ``src.eval.aggregate_to_frames`` and
``src.eval.frame_auc`` (the exact functions used for real UCSD Ped2 / ShanghaiTech
evaluation) to turn per-clip scores into a frame-level ROC-AUC.

This is a SMOKE, not a benchmark. The published headline numbers need the real
datasets and a GPU. Here we only prove the pipeline learns and separates normal
from anomalous on synthetic data.

Run:  python scripts/smoke.py     (or: make smoke)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

# Make ``src`` importable when run as a plain script (python scripts/smoke.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import aggregate_to_frames, frame_auc  # noqa: E402
from src.model import ConvAE3D  # noqa: E402

# ---- tiny config -----------------------------------------------------------
SEED = 0
CLIP_LEN = 8         # frames per clip (must be divisible by 4 for ConvAE3D pools)
STRIDE = 4
IMG = 32             # HxW, small so CPU is fast
CH = 3
FRAMES_PER_VIDEO = 24
N_TRAIN_VIDEOS = 6   # normal-only, for training
TRAIN_STEPS = 80
LR = 1e-3


def _smooth_normal_video(rng: np.random.Generator, frames: int) -> np.ndarray:
    """A smooth, slowly drifting low-frequency clip in (T, H, W, C), floats [0,1]."""
    ys, xs = np.meshgrid(
        np.linspace(0, np.pi, IMG), np.linspace(0, np.pi, IMG), indexing="ij"
    )
    phase = rng.uniform(0, 2 * np.pi)
    freq = rng.uniform(0.8, 1.4)
    vid = np.empty((frames, IMG, IMG, CH), dtype=np.float32)
    for t in range(frames):
        drift = 0.35 * t / frames
        base = 0.5 + 0.35 * np.sin(freq * (xs + ys) + phase + 2 * np.pi * drift)
        base = base + 0.02 * rng.standard_normal((IMG, IMG))  # mild sensor noise
        for c in range(CH):
            vid[t, :, :, c] = base * (0.85 + 0.1 * c)
    return np.clip(vid, 0.0, 1.0)


def _inject_anomaly(vid: np.ndarray, rng: np.random.Generator, seg: range) -> np.ndarray:
    """Add a bright moving blob + noise into frames in ``seg`` (in place copy)."""
    out = vid.copy()
    ys, xs = np.meshgrid(np.arange(IMG), np.arange(IMG), indexing="ij")
    for t in seg:
        cy = int(6 + (IMG - 12) * (t - seg.start) / max(1, len(seg)))
        cx = int(IMG // 2 + 6 * np.sin(0.7 * t))
        blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 4.0 ** 2))
        out[t, :, :, :] += 0.9 * blob[..., None]
        out[t, :, :, :] += 0.15 * rng.standard_normal((IMG, IMG, CH))
    return np.clip(out, 0.0, 1.0)


def _clip_tensor(frames: np.ndarray, start: int) -> torch.Tensor:
    """(T,H,W,C) window -> (C,T,H,W) float tensor, no cv2/decord needed."""
    win = frames[start : start + CLIP_LEN]                 # (T,H,W,C)
    x = torch.from_numpy(np.ascontiguousarray(win)).float()
    return x.permute(3, 0, 1, 2).contiguous()              # (C,T,H,W)


def _clips_from_video(frames: np.ndarray):
    """Yield (start, clip_tensor) sliding windows over a video."""
    n = frames.shape[0]
    for s in range(0, n - CLIP_LEN + 1, STRIDE):
        yield s, _clip_tensor(frames, s)


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cpu"  # tiny-CPU smoke: never touch CUDA even if present
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    print("== video-anomaly-detection tiny-CPU smoke ==")
    print(f"device={device} clip={CH}x{CLIP_LEN}x{IMG}x{IMG} "
          f"train_videos={N_TRAIN_VIDEOS} steps={TRAIN_STEPS}")

    # --- build synthetic training clips (normal only) -----------------------
    train_clips = []
    for _ in range(N_TRAIN_VIDEOS):
        vid = _smooth_normal_video(rng, FRAMES_PER_VIDEO)
        for _s, clip in _clips_from_video(vid):
            train_clips.append(clip)
    train_batch = torch.stack(train_clips, dim=0)  # (N, C, T, H, W)
    print(f"train clips: {tuple(train_batch.shape)}")

    # --- tiny model + a few training steps ----------------------------------
    model = ConvAE3D(in_channels=CH).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossfn = torch.nn.MSELoss()
    print(f"model=ConvAE3D params={n_params:,}")

    model.train()
    bs = 8
    first_loss = None
    last_loss = None
    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, train_batch.shape[0], (bs,))
        xb = train_batch[idx].to(device)
        xhat = model(xb)
        loss = lossfn(xhat, xb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(loss)
        last_loss = float(loss)
        if step % 20 == 0 or step == TRAIN_STEPS - 1:
            print(f"  step {step:3d}  recon_loss {float(loss):.5f}")

    # --- build a labelled eval set (mix of normal + anomalous videos) -------
    model.eval()
    clip_scores = []           # list of {"video","start","score"} for src.eval
    frame_labels = {}          # {video: np.ndarray {0,1}}
    n_eval_videos = 8
    for v in range(n_eval_videos):
        vid = _smooth_normal_video(rng, FRAMES_PER_VIDEO)
        labels = np.zeros(FRAMES_PER_VIDEO, dtype=np.int64)
        if v % 2 == 1:  # half the videos contain an anomalous segment
            seg = range(8, 16)
            vid = _inject_anomaly(vid, rng, seg)
            labels[seg.start : seg.stop] = 1
        name = f"vid{v:02d}"
        frame_labels[name] = labels
        with torch.no_grad():
            for s, clip in _clips_from_video(vid):
                err = float(model.reconstruction_error(clip.unsqueeze(0))[0])
                clip_scores.append({"video": name, "start": s, "score": err})

    # separation check on clips whose window is fully normal vs fully anomalous
    norm_scores, anom_scores = [], []
    for c in clip_scores:
        seg_lab = frame_labels[c["video"]][c["start"] : c["start"] + CLIP_LEN]
        if seg_lab.max() == 0:
            norm_scores.append(c["score"])
        elif seg_lab.min() == 1:
            anom_scores.append(c["score"])
    norm_mean = float(np.mean(norm_scores))
    anom_mean = float(np.mean(anom_scores))

    # --- reuse the REAL eval code to get a frame-level ROC-AUC --------------
    n_frames = {name: FRAMES_PER_VIDEO for name in frame_labels}
    frame_scores = aggregate_to_frames(clip_scores, CLIP_LEN, n_frames, agg="max")
    auc = frame_auc(frame_scores, frame_labels)

    print("-- results --")
    print(f"recon_loss: first {first_loss:.5f} -> last {last_loss:.5f}")
    print(f"clip recon-error  normal_mean {norm_mean:.4f}  anomaly_mean {anom_mean:.4f}"
          f"  (ratio {anom_mean / norm_mean:.1f}x)")
    print(f"frame ROC-AUC (synthetic): {auc:.3f}")

    # --- assertions ---------------------------------------------------------
    ok = True
    if not (last_loss < first_loss):
        print("FAIL: reconstruction loss did not decrease"); ok = False
    if not (anom_mean > norm_mean):
        print("FAIL: anomalous clips did not score higher than normal"); ok = False
    if not (auc > 0.75):
        print(f"FAIL: frame AUC {auc:.3f} <= 0.75"); ok = False

    if ok:
        print("SMOKE OK")
        return 0
    print("SMOKE FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
