# Deploy notes

## AWS ECS (starter)

1. push image to ECR:
   ```
   aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t video-anomaly:latest .
   docker tag video-anomaly:latest <acct>.dkr.ecr.us-east-1.amazonaws.com/video-anomaly:latest
   docker push <acct>.dkr.ecr.us-east-1.amazonaws.com/video-anomaly:latest
   ```
2. upload `checkpoints/model.pt` to S3 (or bake into the image for cold-start speed).
3. register the task definition: `aws ecs register-task-definition --cli-input-json file://deploy/aws_ecs_task.json`
4. front with an ALB.

The shipped task JSON targets Fargate for CPU serving. For GPU inference, switch to an ECS EC2 launch type with a GPU capacity provider and add a `resourceRequirements` GPU block to the container definition.
