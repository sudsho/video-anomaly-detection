# video-anomaly-detection

Surveillance-grade video anomaly detection with 3D CNNs. Streaming inference for security-camera feeds, deployed on AWS.

Work in progress — initial scaffold.

## What

Frame-level anomaly scoring on top of clip-based 3D conv features. Two model paths:
- supervised: C3D / I3D classifier on labeled normal vs anomaly clips
- unsupervised: 3D conv autoencoder, reconstruction error as anomaly signal

Eval target: frame-level AUC-ROC on UCSD Ped2 / ShanghaiTech (public).

## Status

scaffolding repo, more soon

## First numbers

C3D baseline on UCSD Ped2 test split: frame AUC = 0.926 (16-frame clips, stride 8).
Unsupervised 3D conv autoencoder (normal-only train): 0.872.
See `results/ped2_baseline.json`.

