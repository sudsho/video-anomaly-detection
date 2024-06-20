"""Streamlit demo: upload a video, see anomaly score timeline."""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import streamlit as st
import torch
import yaml

from src.predict import load_model, score_clip, stream_clips
from src.transforms import ClipTransform


CONFIG_PATH = os.environ.get("VAD_CONFIG", "configs/default.yaml")
CKPT_PATH = os.environ.get("VAD_CKPT", "checkpoints/model.pt")


@st.cache_resource
def _bootstrap():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = cfg["infer"]["device"] if torch.cuda.is_available() else "cpu"
    if not os.path.exists(CKPT_PATH):
        return cfg, None, "", None, device
    model, arch = load_model(CKPT_PATH, cfg["model"]["num_classes"], device)
    tx = ClipTransform(size=cfg["data"]["img_size"])
    return cfg, model, arch, tx, device


def main() -> None:
    st.set_page_config(page_title="Video Anomaly Detection", layout="wide")
    st.title("Video Anomaly Detection")
    st.caption("3D CNN scoring with sliding-window clips. UCSD Ped2 / ShanghaiTech.")

    cfg, model, arch, tx, device = _bootstrap()
    if model is None:
        st.error(f"checkpoint not found at {CKPT_PATH}. train first or set VAD_CKPT.")
        return

    up = st.file_uploader("upload a video chunk (.mp4 / .avi)", type=["mp4", "avi"])
    if up is None:
        st.info("upload a clip to score")
        return

    suffix = os.path.splitext(up.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(up.read())
        tmp_path = tmp.name

    starts = []
    scores = []
    t0 = time.time()
    bar = st.progress(0.0)
    log = st.empty()
    for i, (s, clip) in enumerate(
        stream_clips(tmp_path, cfg["data"]["clip_len"], cfg["data"]["stride"])
    ):
        sc = score_clip(model, arch, clip, tx, device)
        starts.append(s)
        scores.append(sc)
        log.text(f"clip {i+1}: start={s} score={sc:.3f}")
    elapsed = time.time() - t0

    bar.progress(1.0)
    st.metric("clips scored", len(scores))
    st.metric("max anomaly score", f"{max(scores):.3f}" if scores else "n/a")
    st.metric("elapsed (s)", f"{elapsed:.2f}")

    if scores:
        st.subheader("Score timeline")
        st.line_chart({"score": scores}, x_label="clip index")
        st.caption(f"threshold = {cfg['infer'].get('threshold', 0.6)}")

    try:
        os.unlink(tmp_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
