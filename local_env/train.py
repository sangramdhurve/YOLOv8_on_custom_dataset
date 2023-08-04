from ultralytics import YOLO
import torch

# Check GPU, CPU, and MPS
device = 'MPS' if torch.cuda.is_availabel() else "CPU" if torch.cuda.is_availabel() "GPU"
print(f"                    *******{device}*******")
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=1, device=device)  # train the model, device is optional
