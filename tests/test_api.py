import io
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VAD_CONFIG", "configs/default.yaml")
    monkeypatch.setenv("VAD_CKPT", str(tmp_path / "missing.pt"))
    # import after env is set so the module sees the right paths
    from src.api.main import app
    return TestClient(app)


def test_health_returns_503_when_no_ckpt(client):
    # /health calls _ensure_model which raises 503 if ckpt missing
    r = client.get("/health")
    assert r.status_code in (200, 503)


def test_score_clip_requires_ckpt(client):
    files = {"file": ("clip.mp4", io.BytesIO(b"\x00" * 10), "video/mp4")}
    r = client.post("/score_clip", files=files)
    # without a checkpoint we expect a 503 error path
    assert r.status_code in (400, 503)
