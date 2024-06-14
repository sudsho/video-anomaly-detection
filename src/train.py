"""Training entrypoint. Single-GPU for now, AMP enabled by default."""
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
from .model import C3D, I3DLite


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "c3d":
        return C3D(num_classes=num_classes)
    if name == "i3d":
        return I3DLite(num_classes=num_classes)
    raise ValueError(f"unknown arch: {name}")


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
    model = build_model(cfg["model"]["arch"], cfg["model"]["num_classes"]).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    crit = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()
        for step, (x, y, *_rest) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % cfg["train"]["log_interval"] == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        print(f"epoch {epoch} done in {time.time() - t0:.1f}s")

    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["train"]["ckpt_dir"], "model.pt"))


if __name__ == "__main__":
    main()
