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

## Results (frame AUC on UCSD Ped2 test)

| arch        | frame AUC | params | notes                                        |
|-------------|-----------|--------|----------------------------------------------|
| C3D         | 0.926     | 35M    | 16-frame clips, stride 8                     |
| I3D-lite    | 0.941     | 11M    | smaller, ~1.4x faster per epoch              |
| ConvAE3D    | 0.872     |  4M    | unsupervised, normal-only train              |

See `results/ped2_baseline.json`, `results/ped2_i3d.json`.

