"""FastAPI service for surveillance-grade video anomaly scoring.

POST /score_clip with a multipart video chunk -> JSON {score, latency_ms, ...}.
Designed for low-latency serving, not batch eval.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

import torch
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field

from ..predict import load_model, score_clip, stream_clips
from ..transforms import ClipTransform


CONFIG_PATH = os.environ.get("VAD_CONFIG", "configs/default.yaml")
CKPT_PATH = os.environ.get("VAD_CKPT", "checkpoints/model.pt")


class ScoreResponse(BaseModel):
    score: float = Field(..., description="anomaly score in [0,1] for clf, MSE for AE")
    arch: str
    n_clips: int
    latency_ms: float
    threshold: float
    is_anomaly: bool


class HealthResponse(BaseModel):
    status: str
    arch: str
    device: str


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


cfg = _load_cfg()
device = cfg["infer"]["device"] if torch.cuda.is_available() else "cpu"
model: Optional[torch.nn.Module] = None
arch_name: str = ""
transform: Optional[ClipTransform] = None


def _ensure_model() -> None:
    global model, arch_name, transform
    if model is None:
        if not os.path.exists(CKPT_PATH):
            raise HTTPException(status_code=503, detail=f"checkpoint not found: {CKPT_PATH}")
        model, arch_name = load_model(CKPT_PATH, cfg["model"]["num_classes"], device)
        transform = ClipTransform(size=cfg["data"]["img_size"])


app = FastAPI(title="video-anomaly-detection", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    _ensure_model()
    return HealthResponse(status="ok", arch=arch_name, device=device)


@app.post("/score_clip", response_model=ScoreResponse)
async def score_clip_endpoint(file: UploadFile = File(...)) -> ScoreResponse:
    _ensure_model()
    threshold = float(cfg["infer"].get("threshold", 0.6))
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        t0 = time.time()
        scores = []
        n = 0
        for _start, clip in stream_clips(
            tmp_path,
            clip_len=cfg["data"]["clip_len"],
            stride=cfg["data"]["stride"],
        ):
            scores.append(score_clip(model, arch_name, clip, transform, device))
            n += 1
        if not scores:
            raise HTTPException(status_code=400, detail="video too short for one clip")
        agg = max(scores)
        dt_ms = (time.time() - t0) * 1000.0
        return ScoreResponse(
            score=float(agg),
            arch=arch_name,
            n_clips=n,
            latency_ms=dt_ms,
            threshold=threshold,
            is_anomaly=bool(agg >= threshold),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
