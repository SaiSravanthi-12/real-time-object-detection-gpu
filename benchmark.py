
import torch
import time
import numpy as np
from ultralytics import YOLO

def benchmark(device, runs=50):
    model = YOLO("yolov8n.pt")
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model(img, device=device)

    start = time.time()
    for _ in range(runs):
        model(img, device=device)
    end = time.time()

    total_time = end - start
    fps = runs / total_time
    return fps

print("Benchmarking YOLOv8...")

cpu_fps = benchmark("cpu")
print(f"CPU FPS: {cpu_fps:.2f}")

if torch.cuda.is_available():
    gpu_fps = benchmark("cuda")
    print(f"GPU FPS: {gpu_fps:.2f}")
    improvement = ((gpu_fps - cpu_fps) / cpu_fps) * 100
    print(f"GPU is {improvement:.2f}% faster than CPU")
else:
    print("CUDA not available. GPU benchmark skipped.")
