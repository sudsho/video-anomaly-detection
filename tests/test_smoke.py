"""Fast offline check of the anomaly-detection mechanism.

A compact version of ``scripts/smoke.py``: train the real ConvAE3D on tiny
synthetic normal clips for a handful of steps, then confirm reconstruction
error is higher on an anomalous clip. Also exercises the real frame-level
aggregation / AUC helpers from ``src.eval``. CPU only, no downloads.
"""
import numpy as np
import torch

from src.eval import aggregate_to_frames, frame_auc
from src.model import ConvAE3D


def _normal_clip(rng, t=8, s=16, c=3):
    ys, xs = np.meshgrid(np.linspace(0, np.pi, s), np.linspace(0, np.pi, s), indexing="ij")
    ph = rng.uniform(0, 2 * np.pi)
    vid = np.empty((c, t, s, s), dtype=np.float32)
    for k in range(t):
        base = 0.5 + 0.35 * np.sin(xs + ys + ph + 0.2 * k)
        for ch in range(c):
            vid[ch, k] = base
    return torch.from_numpy(vid)


def _anomalous_clip(base: torch.Tensor, rng) -> torch.Tensor:
    out = base.clone()
    out[:, 2:6, 4:12, 4:12] += 0.9  # bright blob in a temporal segment
    out += 0.1 * torch.from_numpy(rng.standard_normal(out.shape).astype(np.float32))
    return out.clamp(0, 1)


def test_convae3d_trains_and_separates_anomaly():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    clips = torch.stack([_normal_clip(rng) for _ in range(8)], dim=0)

    model = ConvAE3D(in_channels=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = torch.nn.MSELoss()

    model.train()
    first = last = None
    for _ in range(30):
        xhat = model(clips)
        loss = lossfn(xhat, clips)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)

    assert last < first  # reconstruction loss decreased

    model.eval()
    with torch.no_grad():
        normal = _normal_clip(rng).unsqueeze(0)
        anomalous = _anomalous_clip(normal[0], rng).unsqueeze(0)
        e_norm = float(model.reconstruction_error(normal)[0])
        e_anom = float(model.reconstruction_error(anomalous)[0])

    assert e_anom > e_norm  # anomaly reconstructs worse -> higher score


def test_frame_auc_pipeline_perfect_separation():
    # Perfectly separated per-clip scores must give AUC 1.0 through the real
    # aggregation + AUC code used for UCSD Ped2 / ShanghaiTech eval.
    clip_scores = [
        {"video": "v0", "start": 0, "score": 0.1},
        {"video": "v0", "start": 4, "score": 0.1},
        {"video": "v1", "start": 0, "score": 0.1},
        {"video": "v1", "start": 4, "score": 0.9},  # covers anomalous frames
    ]
    n_frames = {"v0": 8, "v1": 8}
    labels = {
        "v0": np.zeros(8, dtype=np.int64),
        "v1": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
    }
    frame_scores = aggregate_to_frames(clip_scores, clip_len=4, n_frames_per_video=n_frames, agg="max")
    auc = frame_auc(frame_scores, labels)
    assert auc == 1.0
