# video-anomaly-detection

Reference implementation of a 3D-CNN video anomaly scoring pipeline (data loader, models, FastAPI scoring service, Streamlit demo). Reproducing published UCSD Ped2 / ShanghaiTech numbers with this repo would require additional training-label plumbing that is not implemented here.

## Problem

Pedestrian/vehicle areas under fixed CCTV (UCSD Ped2, ShanghaiTech). Treat each short window of frames as a clip; score each clip; aggregate clip scores onto frames; trigger an alert when a frame score crosses threshold.

## Datasets

- **UCSD Ped2** (public): 16 training videos (normal only), 12 test videos with frame-level ground truth in `_gt` folders.
- **ShanghaiTech Campus** (public): 13 scenes, larger and more diverse. Same loader works once you point `data.root` at it.

The dataset loader walks a directory of per-video frame folders (`.tif/.jpg/.png`). A separate code path in `predict.py` and the API reads packed `.mp4/.avi` inputs via decord for streaming inference on a single video file.

## Approach

Three model heads are implemented in `src/model.py`:

| arch      | type          | scoring                 |
|-----------|---------------|-------------------------|
| C3D       | supervised    | softmax prob of anomaly |
| I3D-lite  | supervised    | softmax prob of anomaly |
| ConvAE3D  | unsupervised  | clip-level MSE          |

Clip = 16 frames @ 112x112, stride 8. AdamW, cosine LR, AMP.

Frame-level scores are obtained by tiling clip scores back over their constituent frames and taking the max (configurable to mean).

Note: only the unsupervised ConvAE3D path can be trained end-to-end from the shipped code and configs. The supervised C3D / I3D-lite heads compile and forward correctly, but no training-label loader is wired into `train.py`, so training them would require adding a labels source. Frame-level labels for the Ped2 test split are available via `src/labels.py` for evaluation only.

## Results

No benchmark run is checked into this repo. No trained checkpoints, no MLflow logs, no per-frame score arrays are included. Anything you see quoted online for C3D / I3D on Ped2 is from external papers, not from this codebase.

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
    data.py           # frame-folder + decord video reader, VideoClipDataset
    transforms.py     # ClipTransform
    model.py          # C3D, I3DLite, ConvAE3D
    labels.py         # UCSD Ped2 frame-level GT loader (eval only)
    train.py          # cosine LR, AMP, optional MLflow
    eval.py           # frame-level AUC-ROC
    predict.py        # streaming inference cli
    api/main.py       # FastAPI /score_clip
  streamlit_app.py    # demo: upload video, see clip score
  scripts/smoke.py    # tiny-CPU offline smoke (synthetic clips, no GPU/download)
  notebooks/eda.ipynb
  tests/
  deploy/             # ECS task def + notes
  ci/test.yml.example
  Dockerfile
  docker-compose.yml
  Makefile
  pyproject.toml
  requirements.txt
  LICENSE
```

## Quick start (tiny-CPU smoke, no GPU/download)

You can verify the pipeline end to end in about half a minute on a CPU, with no
dataset download and no GPU. `scripts/smoke.py` builds tiny SYNTHETIC video
clips (normal = smooth low-frequency footage, anomalous = a bright blob + noise
injected into a segment), trains the real `ConvAE3D` autoencoder for a few steps
on normal clips only, then scores normal vs anomalous clips and computes a
frame-level ROC-AUC using the same `src.eval` code used for real datasets. No
`decord`, no OpenCV, no AWS.

```bash
python scripts/smoke.py      # or: make smoke
```

Real output:

```
== video-anomaly-detection tiny-CPU smoke ==
device=cpu clip=3x8x32x32 train_videos=6 steps=80
train clips: (30, 3, 8, 32, 32)
model=ConvAE3D params=1,440,195
  step   0  recon_loss 0.21284
  step  20  recon_loss 0.01786
  step  40  recon_loss 0.00962
  step  60  recon_loss 0.00754
  step  79  recon_loss 0.00293
-- results --
recon_loss: first 0.21284 -> last 0.00293
clip recon-error  normal_mean 0.0045  anomaly_mean 0.0451  (ratio 10.0x)
frame ROC-AUC (synthetic): 1.000
SMOKE OK
```

The reconstruction loss drops, anomalous clips score about 10x higher than
normal, and the synthetic frame-AUC is 1.0. This only proves the code path
works. The headline task (surveillance-grade anomaly detection on UCSD Ped2 /
ShanghaiTech, with the C3D / I3D-lite / ConvAE3D heads and the latencies quoted
for real deployment) needs a GPU and the real datasets, which are not downloaded
here. The AUC above is on synthetic toy clips, not a benchmark number.

## Full pipeline (needs GPU + real data)

```bash
make install
# download UCSD Ped2 manually. The loader expects `data.root` to contain
# per-video frame folders directly, so point it at the Train or Test split:
# data/raw/UCSD_Anomaly_Dataset/UCSDped2/Train

# train (only ConvAE3D trains end-to-end from configs/default.yaml today)
python -m src.train --config configs/default.yaml

# streaming inference on a single mp4 (requires a trained checkpoint)
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
Response shape:
```json
{"score": 0.83, "arch": "c3d", "n_clips": 12, "latency_ms": 215.4, "threshold": 0.6, "is_anomaly": true}
```

Inference runs plain fp32 through the standard PyTorch forward. No latency or throughput benchmark is included in this repo.

## Deploy

`deploy/aws_ecs_task.json` is a starter Fargate task definition for CPU serving; ECR image push steps are in `deploy/README.md`. Running the CUDA image on Fargate would fall back to CPU (Fargate does not expose GPUs); for GPU-backed serving use an ECS EC2 launch type or a managed inference platform.

## Tests

```
pytest -q
```

`tests/test_data.py` and `tests/test_model.py` cover the transform shape and a model forward pass. `tests/test_api.py` covers the FastAPI startup path. `tests/test_smoke.py` is a fast offline check that the `ConvAE3D` autoencoder trains on synthetic clips and reconstructs anomalies worse than normal footage (the mechanism `scripts/smoke.py` runs at full length).

## License

MIT.
