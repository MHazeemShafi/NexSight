
import sys
import csv
import json
import time
import urllib.request
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOWorld

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QSlider, QLineEdit, QFileDialog, QListWidget,
    QVBoxLayout, QHBoxLayout, QMessageBox, QInputDialog, QFrame
)

# ============================================================
# NEXSIGHT CONFIGURATION
# ============================================================

APP_NAME = "NexSight"
STANDARD_MODEL = "yolo26s.pt"
WORLD_MODEL = "yolov8s-worldv2.pt"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PROFILE_FILE = DATA_DIR / "face_profiles.json"

YUNET_FILE = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_FILE = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

FACE_MATCH_THRESHOLD = 0.40

DEFAULT_CLASSES = (
    "person, dog, cat, bird, car, bicycle, motorcycle, "
    "bottle, cup, chair, backpack, book, laptop, cell phone, "
    "keyboard, mouse, handbag, umbrella, headphones, remote control"
)

for folder in (OUTPUT_DIR, DATA_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITIES
# ============================================================

def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def ensure_model_file(path, url, minimum_size):
    """Download a model and reject small error pages or Git LFS pointers."""
    if path.exists() and path.stat().st_size >= minimum_size:
        with path.open("rb") as file:
            header = file.read(512)

        if (
            b"git-lfs.github.com/spec" not in header
            and not header.lstrip().startswith(b"<!DOCTYPE")
            and not header.lstrip().lower().startswith(b"<html")
        ):
            return

    temporary = path.with_suffix(path.suffix + ".download")

    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "NexSight/1.0"}
        )

        with urllib.request.urlopen(request, timeout=90) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        with temporary.open("rb") as file:
            header = file.read(512)

        if temporary.stat().st_size < minimum_size:
            raise RuntimeError(
                f"{path.name} download is too small "
                f"({temporary.stat().st_size:,} bytes)."
            )

        if (
            b"git-lfs.github.com/spec" in header
            or header.lstrip().startswith(b"<!DOCTYPE")
            or header.lstrip().lower().startswith(b"<html")
        ):
            raise RuntimeError(
                f"The server did not return a valid model: {path.name}"
            )

        temporary.replace(path)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_profiles():
    if not PROFILE_FILE.exists():
        return {}

    try:
        with PROFILE_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_profiles(profiles):
    temporary = PROFILE_FILE.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(profiles, file, indent=2)
    temporary.replace(PROFILE_FILE)


# ============================================================
# FACE ENGINE
# ============================================================

class FaceEngine:
    def __init__(self):
        ensure_model_file(YUNET_FILE, YUNET_URL, 100_000)
        ensure_model_file(SFACE_FILE, SFACE_URL, 5_000_000)

        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise RuntimeError(
                "YuNet is unavailable in this OpenCV installation. "
                "Install opencv-contrib-python only."
            )

        if not hasattr(cv2, "FaceRecognizerSF_create"):
            raise RuntimeError(
                "SFace is unavailable in this OpenCV installation."
            )

        self.detector = cv2.FaceDetectorYN_create(
            str(YUNET_FILE), "", (320, 320), 0.85, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF_create(
            str(SFACE_FILE), ""
        )

    def get_embeddings(self, frame):
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)

        results = []
        if faces is None:
            return results

        for face in faces:
            try:
                aligned = self.recognizer.alignCrop(frame, face)
                feature = self.recognizer.feature(aligned).flatten()
                feature = feature.astype(np.float32)
                norm = np.linalg.norm(feature)

                if norm > 0:
                    results.append((face, feature / norm))
            except cv2.error:
                continue

        return results

    def identify(self, vector, profiles):
        best_name = None
        best_score = -1.0

        for name, profile in profiles.items():
            for sample in profile.get("embeddings", []):
                try:
                    reference = np.asarray(sample, dtype=np.float32)
                    norm = np.linalg.norm(reference)
                    if norm == 0:
                        continue

                    score = float(np.dot(vector, reference / norm))
                    if score > best_score:
                        best_name, best_score = name, score
                except (ValueError, TypeError):
                    continue

        if best_name is not None and best_score >= FACE_MATCH_THRESHOLD:
            return best_name

        return None

    def annotate(self, frame, profiles):
        output = frame.copy()
        faces = self.get_embeddings(frame)

        for face, vector in faces:
            x, y, w, h = [int(v) for v in face[:4]]
            name = self.identify(vector, profiles)

            if name:
                label = f"{name} // RECOGNIZED"
                color = (0, 255, 100)
            else:
                label = "FACE DETECTED"
                color = (0, 220, 255)

            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                output, label, (x, max(22, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                cv2.LINE_AA
            )

        return output, len(faces)


# ============================================================
# DETECTION WORKER
# ============================================================

class DetectionWorker(QThread):
    frame_ready = Signal(object)
    status_ready = Signal(str)
    log_ready = Signal(str)
    error_occurred = Signal(str)
    finished_ok = Signal()

    def __init__(
        self, source, use_world, confidence,
        target_classes, enable_faces, profiles
    ):
        super().__init__()
        self.source = source
        self.use_world = use_world
        self.confidence = confidence
        self.target_classes = target_classes
        self.enable_faces = enable_faces
        self.profiles = profiles
        self.running = True
        self.face_engine = None

    def stop(self):
        self.running = False

    def run(self):
        capture = None

        try:
            model_name = WORLD_MODEL if self.use_world else STANDARD_MODEL
            self.status_ready.emit(f"LOADING // {model_name}")

            model = (
                YOLOWorld(model_name)
                if self.use_world
                else YOLO(model_name)
            )

            if self.use_world and self.target_classes:
                model.set_classes(self.target_classes)

            if self.enable_faces:
                try:
                    self.status_ready.emit("LOADING FACE MODELS")
                    self.face_engine = FaceEngine()
                    self.status_ready.emit("FACE MODULE ONLINE")
                except Exception as error:
                    self.log_ready.emit(
                        f"FACE MODULE OFFLINE // {error}"
                    )
                    self.status_ready.emit(
                        "OBJECT DETECTION ONLINE // FACE MODULE OFFLINE"
                    )

            if isinstance(self.source, str) and Path(self.source).suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
            }:
                frame = cv2.imread(self.source)
                if frame is None:
                    raise RuntimeError("Could not open the selected image.")
                self.process_frame(model, frame, track=False)
                self.status_ready.emit("IMAGE PROCESSED")
                return

            if isinstance(self.source, int):
                capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not capture.isOpened():
                    capture.release()
                    capture = cv2.VideoCapture(self.source)
            else:
                capture = cv2.VideoCapture(self.source)

            if not capture.isOpened():
                raise RuntimeError(
                    "Could not open camera/video. Close other camera "
                    "applications and check Windows camera permissions."
                )

            frame_number = 0
            started = time.time()

            while self.running:
                success, frame = capture.read()
                if not success:
                    break

                self.process_frame(model, frame, track=True)
                frame_number += 1

                elapsed = max(time.time() - started, 0.001)
                self.status_ready.emit(
                    f"PROCESSING // {frame_number} FRAMES // "
                    f"{frame_number / elapsed:.1f} FPS"
                )

        except Exception as error:
            self.error_occurred.emit(str(error))

        finally:
            if capture is not None:
                capture.release()
            self.finished_ok.emit()

    def process_frame(self, model, frame, track):
        if not self.running:
            return

        if track:
            results = model.track(
                frame,
                persist=True,
                conf=self.confidence,
                device=get_device(),
                verbose=False,
                tracker="bytetrack.yaml"
            )
        else:
            results = model.predict(
                frame,
                conf=self.confidence,
                device=get_device(),
                verbose=False
            )

        result = results[0]
        annotated = result.plot()
        count = len(result.boxes) if result.boxes is not None else 0

        if self.face_engine is not None:
            try:
                annotated, face_count = self.face_engine.annotate(
                    annotated, self.profiles
                )
                if face_count:
                    self.log_ready.emit(
                        f"FACE SCAN // {face_count} FACE(S)"
                    )
            except Exception as error:
                self.log_ready.emit(f"FACE SCAN ERROR // {error}")
                self.face_engine = None

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = result.names.get(class_id, str(class_id))
                confidence = float(box.conf[0].item())

                track_id = ""
                if box.id is not None:
                    track_id = f" // ID {int(box.id[0].item())}"

                self.log_ready.emit(
                    f"{label.upper()} // {confidence:.0%}{track_id}"
                )

        self.status_ready.emit(
            f"OBJECTS {count} // DEVICE {get_device()}"
        )
        self.frame_ready.emit(annotated)


# ============================================================
# ENROLLMENT WORKER
# ============================================================

class EnrollmentWorker(QThread):
    profile_ready = Signal(str, object)
    status_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, person_name):
        super().__init__()
        self.person_name = person_name
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        capture = None

        try:
            engine = FaceEngine()
            capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(0)

            if not capture.isOpened():
                raise RuntimeError(
                    "Cannot open camera for enrollment. Stop other "
                    "camera processes and try again."
                )

            samples = []
            last_sample = 0.0
            started = time.time()

            self.status_ready.emit(
                "ENROLLMENT // LOOK AT CAMERA AND MOVE YOUR HEAD SLIGHTLY"
            )

            while (
                self.running
                and time.time() - started < 30
                and len(samples) < 25
            ):
                success, frame = capture.read()
                if not success:
                    continue

                if time.time() - last_sample >= 0.5:
                    faces = engine.get_embeddings(frame)

                    if faces:
                        _, vector = max(
                            faces,
                            key=lambda item: item[0][2] * item[0][3]
                        )
                        samples.append(vector.tolist())
                        last_sample = time.time()

                        self.status_ready.emit(
                            f"ENROLLMENT // {len(samples)}/25 SAMPLES"
                        )

                time.sleep(0.02)

            if len(samples) < 3:
                raise RuntimeError(
                    "Not enough face samples captured. Improve lighting "
                    "and face the camera, then try again."
                )

            self.profile_ready.emit(self.person_name, samples)

        except Exception as error:
            self.error_occurred.emit(str(error))

        finally:
            if capture is not None:
                capture.release()


# ============================================================
# USER INTERFACE
# ============================================================

STYLE = """
QMainWindow, QWidget {
    background-color: #050907;
    color: #a8ffc0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}
QLabel { color: #8dffad; }
QLabel#brand {
    color: #00ff66;
    font-size: 28px;
    font-weight: bold;
}
QLabel#subtitle { color: #478d5c; font-size: 10px; }
QLabel#screen {
    background-color: #020403;
    border: 1px solid #155b31;
    color: #00ff66;
}
QPushButton {
    background-color: #07140b;
    color: #00ff66;
    border: 1px solid #1a8b45;
    padding: 9px;
}
QPushButton:hover {
    background-color: #0d2b17;
    border: 1px solid #5bff8a;
}
QPushButton:disabled {
    color: #385844;
    border-color: #183323;
}
QComboBox, QLineEdit, QListWidget {
    background-color: #020604;
    color: #a8ffc0;
    border: 1px solid #155b31;
    padding: 6px;
}
QSlider::groove:horizontal {
    background: #14351e;
    height: 4px;
}
QSlider::handle:horizontal {
    background: #00ff66;
    width: 12px;
    margin: -5px 0;
}
QFrame#separator {
    background-color: #124b29;
    max-height: 1px;
}
"""


class NexSightWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NexSight // Visual Intelligence")
        self.resize(1320, 850)

        self.worker = None
        self.enrollment_worker = None
        self.last_frame = None
        self.profiles = load_profiles()

        self.setStyleSheet(STYLE)
        self.build_ui()
        self.refresh_profiles()

        self.log("NEXSIGHT INITIALIZED")
        self.log(f"COMPUTE DEVICE // {get_device()}")
        self.log(f"LOCAL FACE PROFILES // {len(self.profiles)}")

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main = QHBoxLayout(root)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(9)
        left.setAlignment(Qt.AlignmentFlag.AlignTop)

        brand = QLabel("NEXSIGHT")
        brand.setObjectName("brand")
        subtitle = QLabel("REAL-TIME VISUAL INTELLIGENCE")
        subtitle.setObjectName("subtitle")

        left.addWidget(brand)
        left.addWidget(subtitle)
        left.addWidget(QLabel("See the world. Understand what matters."))
        left.addWidget(self.separator())

        left.addWidget(QLabel(">> INPUT SOURCE"))
        self.source_box = QComboBox()
        self.source_box.addItems(["Webcam", "Image file", "Video file"])
        self.source_box.currentIndexChanged.connect(self.source_changed)
        left.addWidget(self.source_box)

        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select image or video...")
        left.addWidget(self.file_path)

        browse = QPushButton("BROWSE FILE")
        browse.clicked.connect(self.browse_file)
        left.addWidget(browse)

        left.addWidget(self.separator())
        left.addWidget(QLabel(">> DETECTION ENGINE"))

        self.model_box = QComboBox()
        self.model_box.addItems(["Standard YOLO", "YOLO World"])
        left.addWidget(self.model_box)

        left.addWidget(QLabel("Confidence threshold"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(10, 95)
        self.confidence_slider.setValue(40)
        left.addWidget(self.confidence_slider)

        self.confidence_value = QLabel("0.40")
        self.confidence_slider.valueChanged.connect(
            lambda value: self.confidence_value.setText(
                f"{value / 100:.2f}"
            )
        )
        left.addWidget(self.confidence_value)

        left.addWidget(QLabel("World model target classes"))
        self.classes_input = QLineEdit(DEFAULT_CLASSES)
        left.addWidget(self.classes_input)

        self.face_button = QPushButton("FACE RECOGNITION: ENABLED")
        self.face_button.setCheckable(True)
        self.face_button.setChecked(True)
        self.face_button.clicked.connect(self.toggle_face_button)
        left.addWidget(self.face_button)

        left.addWidget(self.separator())
        left.addWidget(QLabel(">> REGISTERED PEOPLE"))

        self.profile_list = QListWidget()
        self.profile_list.setMinimumHeight(95)
        left.addWidget(self.profile_list)

        register = QPushButton("+ REGISTER PERSON")
        register.clicked.connect(self.register_person)
        left.addWidget(register)

        remove = QPushButton("- REMOVE SELECTED PERSON")
        remove.clicked.connect(self.remove_person)
        left.addWidget(remove)

        left.addWidget(self.separator())

        controls = QHBoxLayout()
        self.start_button = QPushButton("START CAMERA")
        self.start_button.clicked.connect(self.start_process)

        self.stop_button = QPushButton("STOP PROCESS")
        self.stop_button.clicked.connect(self.stop_process)
        self.stop_button.setEnabled(False)

        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        left.addLayout(controls)

        snapshot = QPushButton("SAVE SNAPSHOT")
        snapshot.clicked.connect(self.save_snapshot)
        left.addWidget(snapshot)

        export = QPushButton("EXPORT CSV")
        export.clicked.connect(self.export_csv)
        left.addWidget(export)

        main.addLayout(left, 0)

        right = QVBoxLayout()
        right.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(QLabel(">> VISUAL FEED"))
        top.addStretch()

        self.status_label = QLabel("SYSTEM READY")
        top.addWidget(self.status_label)
        right.addLayout(top)

        self.screen = QLabel("NEXSIGHT // WAITING FOR INPUT")
        self.screen.setObjectName("screen")
        self.screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screen.setMinimumSize(640, 420)
        right.addWidget(self.screen, 1)

        right.addWidget(QLabel(">> EVENT LOG"))
        self.log_list = QListWidget()
        self.log_list.setMinimumHeight(140)
        right.addWidget(self.log_list)

        footer = QLabel(
            "NEXSIGHT // LOCAL PROCESSING // VISUAL INTELLIGENCE SYSTEM"
        )
        footer.setObjectName("subtitle")
        right.addWidget(footer)

        main.addLayout(right, 1)

    def separator(self):
        line = QFrame()
        line.setObjectName("separator")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def source_changed(self, index):
        self.file_path.setEnabled(index != 0)

    def toggle_face_button(self, enabled):
        self.face_button.setText(
            "FACE RECOGNITION: ENABLED"
            if enabled else "FACE RECOGNITION: DISABLED"
        )

    def log(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_list.insertItem(0, f"[{stamp}] {message}")

        while self.log_list.count() > 250:
            self.log_list.takeItem(self.log_list.count() - 1)

    def set_status(self, message):
        self.status_label.setText(message)

    def refresh_profiles(self):
        self.profile_list.clear()
        for name in sorted(self.profiles):
            self.profile_list.addItem(name)

    def browse_file(self):
        if self.source_box.currentIndex() == 1:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select image", "",
                "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select video", "",
                "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.webm)"
            )

        if path:
            self.file_path.setText(path)

    def start_process(self):
        if self.worker is not None and self.worker.isRunning():
            return

        if (
            self.enrollment_worker is not None
            and self.enrollment_worker.isRunning()
        ):
            QMessageBox.information(
                self, "NexSight",
                "Wait for face enrollment to finish before starting detection."
            )
            return

        if self.source_box.currentIndex() == 0:
            source = 0
        else:
            source = self.file_path.text().strip()
            if not source or not Path(source).is_file():
                QMessageBox.warning(
                    self, "NexSight", "Select a valid image or video file."
                )
                return

        targets = [
            item.strip()
            for item in self.classes_input.text().split(",")
            if item.strip()
        ]

        self.worker = DetectionWorker(
            source,
            self.model_box.currentIndex() == 1,
            self.confidence_slider.value() / 100.0,
            targets,
            self.face_button.isChecked(),
            self.profiles.copy()
        )

        self.worker.frame_ready.connect(self.show_frame)
        self.worker.status_ready.connect(self.set_status)
        self.worker.log_ready.connect(self.log)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.finished_ok.connect(self.process_finished)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log("PROCESS START REQUESTED")
        self.worker.start()

    def stop_process(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.set_status("STOPPING PROCESS...")

    def process_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log("PROCESS STOPPED")

    def show_frame(self, frame):
        self.last_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape

        image = QImage(
            rgb.data, width, height, channels * width,
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image).scaled(
            self.screen.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.screen.setPixmap(pixmap)

    def show_error(self, message):
        self.log("ERROR // " + message)
        QMessageBox.warning(self, "NexSight error", message)

    def save_snapshot(self):
        if self.last_frame is None:
            QMessageBox.information(
                self, "NexSight", "No frame is available to save yet."
            )
            return

        default_path = str(
            OUTPUT_DIR / (
                "snapshot_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".png"
            )
        )

        path, _ = QFileDialog.getSaveFileName(
            self, "Save snapshot", default_path, "PNG image (*.png)"
        )

        if path:
            if cv2.imwrite(path, self.last_frame):
                self.log("SNAPSHOT SAVED // " + path)
            else:
                QMessageBox.warning(self, "NexSight", "Snapshot save failed.")

    def export_csv(self):
        default_path = str(
            OUTPUT_DIR / (
                "detections_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".csv"
            )
        )

        path, _ = QFileDialog.getSaveFileName(
            self, "Export event log", default_path, "CSV file (*.csv)"
        )

        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "event"])

                for index in range(self.log_list.count() - 1, -1, -1):
                    item = self.log_list.item(index).text()
                    timestamp = item[1:9] if len(item) >= 9 else ""
                    event = item[11:] if len(item) >= 11 else item
                    writer.writerow([timestamp, event])

            self.log("CSV EXPORTED // " + path)

        except OSError as error:
            QMessageBox.warning(self, "NexSight", str(error))

    def register_person(self):
        if (
            self.enrollment_worker is not None
            and self.enrollment_worker.isRunning()
        ):
            QMessageBox.information(
                self, "NexSight", "Enrollment is already running."
            )
            return

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self, "NexSight",
                "Stop the current process before registering a person."
            )
            return

        name, accepted = QInputDialog.getText(
            self, "Register person", "Person's name:"
        )
        name = name.strip()

        if not accepted or not name:
            return

        if name in self.profiles:
            answer = QMessageBox.question(
                self, "Update profile",
                f"{name} already exists. Add more samples?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self.enrollment_worker = EnrollmentWorker(name)
        self.enrollment_worker.profile_ready.connect(
            self.enrollment_complete
        )
        self.enrollment_worker.status_ready.connect(self.set_status)
        self.enrollment_worker.error_occurred.connect(self.show_error)

        self.start_button.setEnabled(False)
        self.log("FACE ENROLLMENT STARTED // " + name)
        self.enrollment_worker.finished.connect(
            lambda: self.start_button.setEnabled(True)
        )
        self.enrollment_worker.start()

    def enrollment_complete(self, name, samples):
        previous = self.profiles.get(name, {}).get("embeddings", [])
        combined = (previous + samples)[-100:]
        self.profiles[name] = {"embeddings": combined}

        try:
            save_profiles(self.profiles)
        except OSError as error:
            QMessageBox.warning(self, "NexSight", str(error))
            return

        self.refresh_profiles()
        self.log(f"PERSON REGISTERED // {name} // {len(combined)} SAMPLES")
        self.set_status("PROFILE SAVED // " + name)

        QMessageBox.information(
            self, "NexSight",
            f"Registered {name} successfully.\n"
            f"Face samples saved: {len(combined)}"
        )

    def remove_person(self):
        selected = self.profile_list.currentItem()

        if selected is None:
            QMessageBox.information(
                self, "NexSight", "Select a registered person first."
            )
            return

        name = selected.text()
        answer = QMessageBox.question(
            self, "Remove person",
            f"Delete the local face profile for {name}?"
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        self.profiles.pop(name, None)

        try:
            save_profiles(self.profiles)
        except OSError as error:
            QMessageBox.warning(self, "NexSight", str(error))
            return

        self.refresh_profiles()
        self.log("PROFILE REMOVED // " + name)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)

        if (
            self.enrollment_worker is not None
            and self.enrollment_worker.isRunning()
        ):
            self.enrollment_worker.stop()
            self.enrollment_worker.wait(3000)

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = NexSightWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
