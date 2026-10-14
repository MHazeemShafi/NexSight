
import cv2
import time
import torch
from collections import Counter
from ultralytics import YOLO

MODEL_NAME = "yolo26s.pt"
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
CAMERA_INDEX = 0

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("Starting NexSight...")
print("Loading model:", MODEL_NAME)
print("Device:", torch.cuda.get_device_name(0)
      if DEVICE == 0 else "CPU")

model = YOLO(MODEL_NAME)

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError("Cannot open webcam.")

previous_time = time.time()
fps = 0.0

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]
        display = result.plot()
        counts = Counter()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = model.names[class_id]
                counts[label] += 1

        current_time = time.time()
        instant_fps = 1.0 / max(
            current_time - previous_time, 0.001
        )
        fps = instant_fps if fps == 0 else (
            0.85 * fps + 0.15 * instant_fps
        )
        previous_time = current_time

        cv2.rectangle(
            display, (10, 10), (430, 92), (12, 20, 32), -1
        )

        cv2.putText(
            display, "NEXSIGHT | AI VISION",
            (22, 38), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 220, 80), 2
        )

        cv2.putText(
            display,
            f"Objects: {sum(counts.values())} | FPS: {fps:.1f}",
            (22, 70), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (230, 245, 255), 1
        )

        y = 120
        for label, count in counts.most_common(6):
            cv2.putText(
                display,
                f"{label}: {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (100, 255, 210), 2
            )
            y += 27

        cv2.imshow("NexSight - Improved Detector", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        # Press + or - to adjust confidence
        elif key in (ord("+"), ord("=")):
            CONFIDENCE = min(0.90, CONFIDENCE + 0.05)
            print("Confidence:", CONFIDENCE)

        elif key in (ord("-"), ord("_")):
            CONFIDENCE = max(0.05, CONFIDENCE - 0.05)
            print("Confidence:", CONFIDENCE)

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("NexSight stopped.")
