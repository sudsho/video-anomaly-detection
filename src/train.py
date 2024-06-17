"""Training entrypoint. Single-GPU, AMP enabled by default. MLflow logging."""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .data import VideoClipDataset
from .model import C3D, ConvAE3D, I3DLite

try:
    import mlflow  # type: ignore
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "c3d":
        return C3D(num_classes=num_classes)
    if name == "i3d":
        return I3DLite(num_classes=num_classes)
    if name in ("autoencoder", "ae", "convae3d"):
        return ConvAE3D()
    raise ValueError(f"unknown arch: {name}")


def _maybe_mlflow_start(cfg: Dict[str, Any]) -> bool:
    if not HAS_MLFLOW or "mlflow" not in cfg:
        return False
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    mlflow.start_run()
    mlflow.log_params({
        "arch": cfg["model"]["arch"],
        "lr": cfg["train"]["lr"],
        "batch_size": cfg["train"]["batch_size"],
        "clip_len": cfg["data"]["clip_len"],
    })
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.get("seed", 42))

    train_ds = VideoClipDataset(
        root=cfg["data"]["root"],
        clip_len=cfg["data"]["clip_len"],
        stride=cfg["data"]["stride"],
        img_size=cfg["data"]["img_size"],
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = cfg["model"]["arch"].lower()
    model = build_model(arch, cfg["model"]["num_classes"]).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    is_ae = arch in ("autoencoder", "ae", "convae3d")
    crit_cls = nn.CrossEntropyLoss(ignore_index=-1)
    crit_rec = nn.MSELoss()

    use_mlflow = _maybe_mlflow_start(cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", True) and device == "cuda")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, (x, y, *_rest) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=cfg["train"].get("amp", True) and device == "cuda"):
                if is_ae:
                    x_hat = model(x)
                    loss = crit_rec(x_hat, x)
                else:
                    y = y.to(device, non_blocking=True)
                    logits = model(x)
                    loss = crit_cls(logits, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
            if step % cfg["train"]["log_interval"] == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        avg = running / max(1, len(loader))
        if use_mlflow:
            mlflow.log_metric("train_loss", avg, step=epoch)
        print(f"epoch {epoch} done in {time.time() - t0:.1f}s avg loss {avg:.4f}")

    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    out = os.path.join(cfg["train"]["ckpt_dir"], "model.pt")
    torch.save({"state_dict": model.state_dict(), "arch": arch}, out)
    if use_mlflow:
        mlflow.log_artifact(out)
        mlflow.end_run()


if __name__ == "__main__":
    main()
