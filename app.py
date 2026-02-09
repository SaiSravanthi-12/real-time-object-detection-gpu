
import cv2
import torch
from ultralytics import YOLO
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Webcam
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, device=device)

    # Plot results
    annotated_frame = results[0].plot()

    # FPS calculation
    fps = 1 / (time.time() - start_time)
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
