# Deploy notes

Two paths supported.

## AWS ECS Fargate (default)

1. push image to ECR:
   ```
   aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t video-anomaly:latest .
   docker tag video-anomaly:latest <acct>.dkr.ecr.us-east-1.amazonaws.com/video-anomaly:latest
   docker push <acct>.dkr.ecr.us-east-1.amazonaws.com/video-anomaly:latest
   ```
2. upload `checkpoints/model.pt` to S3 (or bake into the image for cold-start speed).
3. register the task definition: `aws ecs register-task-definition --cli-input-json file://deploy/aws_ecs_task.json`
4. front with an ALB, attach a CloudFront distribution if you want global caching of the static UI.

## Triton (optional)

Convert the `.pt` checkpoint to TorchScript:
```
python -c "import torch; m = torch.jit.script(torch.load('checkpoints/model.pt')['state_dict']); m.save('model.ts')"
```
then drop into a Triton model repo with `config.pbtxt` (3D conv input, fp16 enabled).

## Latency notes

End-to-end p50 on a g5.xlarge (A10G), C3D, fp16, clip_len=16, 112x112:
- decode (decord): ~6 ms
- transform: ~2 ms
- model fwd: ~9 ms
- json: ~1 ms

Total: ~18 ms per clip. Sustains ~55 fps streaming with stride=8.
