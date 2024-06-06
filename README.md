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
