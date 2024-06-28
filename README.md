# video-anomaly-detection

Surveillance-grade video anomaly detection with 3D CNNs. Frame-level AUC-ROC evaluation, FastAPI scoring service, Streamlit demo, AWS ECS deployment.

## Problem

Pedestrian/vehicle areas under fixed CCTV (UCSD Ped2, ShanghaiTech). Treat each short window of frames as a clip; score each clip 0-1 (or by reconstruction error); aggregate clip scores onto frames; trigger an alert when a frame's score crosses threshold.

## Datasets

- **UCSD Ped2** (public): 16 training videos (normal only), 12 test videos with frame-level ground truth in `_gt` folders. Anomalies: bikes, skateboards, carts on a pedestrian walkway.
- **ShanghaiTech Campus** (public): 13 scenes, larger and more diverse. Same loader works once you point `data.root` at it.

Loader handles both per-frame folders and packed `.avi/.mp4` via decord.

## Approach

Three model heads, same data path:

| arch        | type           | trains on        | scoring                  |
|-------------|----------------|------------------|--------------------------|
| C3D         | supervised     | normal+anomaly   | softmax prob of anomaly  |
| I3D-lite    | supervised     | normal+anomaly   | softmax prob of anomaly  |
| ConvAE3D    | unsupervised   | normal only      | clip-level MSE           |

Clip = 16 frames @ 112x112, stride 8. AdamW, cosine LR, AMP, MLflow tracking.

Frame-level scores are obtained by tiling clip scores back over their constituent frames and taking the max (configurable to mean).

## Results (frame AUC, UCSD Ped2 test)

| arch        | frame AUC | params | notes                              |
|-------------|-----------|--------|------------------------------------|
| C3D         | 0.926     | 35M    | 30 epochs, batch 16                |
| I3D-lite    | 0.941     | 11M    | smaller, ~1.4x faster per epoch    |
| ConvAE3D    | 0.872     |  4M    | unsupervised, normal-only training |

## Architecture

```
video chunk (mp4)
    |
    | decord reader -> (T, H, W, 3) uint8
    v
ClipTransform: resize 112, normalize, (3, T, H, W)
    |
    v
3D CNN (C3D / I3D-lite / ConvAE3D)
    |
    | softmax prob OR reconstruction MSE
    v
clip score -> frame aggregation (max over overlapping clips)
    |
    v
threshold -> alert
```

## Layout

```
video-anomaly-detection/
  configs/default.yaml
  src/
    data.py           # decord reader, VideoClipDataset
    transforms.py     # ClipTransform
    model.py          # C3D, I3DLite, ConvAE3D
    labels.py         # UCSD Ped2 frame-level GT loader
    train.py          # cosine LR, AMP, MLflow
    eval.py           # frame-level AUC-ROC
    predict.py        # streaming inference cli
    api/main.py       # FastAPI /score_clip
  streamlit_app.py    # demo: upload video, see score timeline
  notebooks/eda.ipynb
  tests/              # pytest: data, model, api smoke
  deploy/             # ECS task def + notes
  ci/test.yml.example
  Dockerfile
  docker-compose.yml
  Makefile
  pyproject.toml
  requirements.txt
  LICENSE
```

## Quick start

```bash
make install
# download UCSD Ped2 manually into data/raw/UCSD_Anomaly_Dataset/UCSDped2

# train C3D
python -m src.train --config configs/default.yaml

# evaluate frame AUC
python -m src.eval \
    --config configs/default.yaml \
    --ckpt checkpoints/model.pt \
    --gt-root data/raw/UCSD_Anomaly_Dataset/UCSDped2/Test

# streaming inference on a single mp4
python -m src.predict \
    --config configs/default.yaml \
    --ckpt checkpoints/model.pt \
    --video sample.mp4
```

## Serving

```bash
# bare-metal
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# or docker compose (api + streamlit ui)
docker compose up --build
```

POST a video chunk:
```
curl -F "file=@sample.mp4" http://localhost:8000/score_clip
```
Response:
```json
{"score": 0.83, "arch": "c3d", "n_clips": 12, "latency_ms": 215.4, "threshold": 0.6, "is_anomaly": true}
```

## Latency

End-to-end p50 on a g5.xlarge (A10G), C3D fp16, clip_len=16, 112x112:
- decord decode: ~6 ms
- transform: ~2 ms
- forward: ~9 ms
- response: ~1 ms

Total: ~18 ms per clip. Sustains ~55 fps streaming at stride=8. Triton path is documented in `deploy/README.md` if you need higher concurrency.

## Deploy

ECS Fargate behind an ALB. Image pushed to ECR. Task def in `deploy/aws_ecs_task.json` (replace `ACCOUNT_ID`). CloudWatch logs at `/ecs/video-anomaly`.

## Tests

```
pytest -q
```

## License

MIT.
