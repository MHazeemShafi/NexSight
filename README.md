# NexSight

<p align="center">
  <strong>REAL-TIME VISUAL INTELLIGENCE</strong>
  <br>
  <em>See the world. Understand what matters.</em>
</p>

<p align="center">
  A desktop computer vision application for real-time object detection, tracking, and face recognition using Python, OpenCV, YOLO, and PySide6.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-00FF66?style=flat-square&logo=python&logoColor=black" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-00FF66?style=flat-square&logo=opencv&logoColor=black" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLO-Object%20Detection-00FF66?style=flat-square" alt="YOLO">
  <img src="https://img.shields.io/badge/PySide6-Desktop%20UI-00FF66?style=flat-square" alt="PySide6">
  <img src="https://img.shields.io/badge/License-MIT-00FF66?style=flat-square" alt="MIT License">
</p>

---

## Overview

**NexSight** is a Python-based visual intelligence desktop application built to explore real-time computer vision in a practical, interactive environment.

It combines YOLO-based object detection with a desktop interface for viewing camera feeds, inspecting detections, tracking objects, and recording results. Optional face-recognition functionality supports locally enrolled profiles.

The project brings together computer vision, machine learning, and desktop application development in one application.

## Features

- **Real-time object detection** — detect objects in live camera feeds using YOLO.
- **Multiple input sources** — work with webcam feeds, images, and video files.
- **Object tracking** — follow detected objects across frames.
- **YOLO-World support** — explore open-vocabulary detection where supported by the configured model.
- **Face recognition** — enroll local face profiles and identify matching enrolled faces when the required models are available.
- **Unknown-face handling** — display a generic face-detected label when a face cannot be matched to an enrolled profile.
- **Adjustable confidence threshold** — control the minimum confidence used for detections.
- **Target-class filtering** — focus on selected object classes.
- **Detection log** — inspect recorded detection events.
- **Snapshot capture** — save useful frames for later review.
- **CSV export** — export detection records for further analysis.
- **Desktop interface** — a PySide6 interface with a dark, terminal-inspired visual design.

*Some features depend on the selected model, input source, and availability of the required model files.*

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| PySide6 | Desktop user interface |
| OpenCV | Image and video processing |
| Ultralytics YOLO | Object detection and supported tracking workflows |
| NumPy | Numerical and image-array operations |
| YuNet / SFace | Face detection and recognition, when configured |

## Getting Started

### Requirements

- Windows, Linux, or another compatible Python environment
- Python compatible with the installed dependencies (Python 3.10+ recommended)
- A webcam for live camera detection
- Internet access for the initial installation and any required model downloads
- A compatible GPU is optional; performance depends on the hardware and configured inference device

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NexSight.git
cd NexSight
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows — PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the virtual environment's Python executable directly instead.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch NexSight

```powershell
python app.py
```

If your local entry-point file has a different name, use that file instead.

## Model Setup

NexSight relies on compatible model files for its detection and optional face-recognition workflows.

- **YOLO models:** the application may download the configured Ultralytics weights on first use.
- **YOLO-World:** additional weights may be required to use open-vocabulary detection.
- **Face recognition:** the configured YuNet and SFace ONNX model files must be available for face functionality.

Model availability, filenames, and download behavior depend on the application configuration. If optional face models are missing, object detection may still be available.

Model weights are intentionally not included in this repository. Follow the relevant model providers' download instructions and licensing terms.

## Privacy

NexSight can use locally enrolled face profiles for recognition. Face data is sensitive.

- Enroll only people who have given appropriate permission.
- Keep local face-profile data private.
- Do not commit personal face profiles, embeddings, or other sensitive data to a public repository.
- Review the applicable laws and obtain consent before using face recognition in real-world settings.

NexSight is a learning and development project, not a guarantee of identity or a substitute for human verification.

## Project Structure

```text
NexSight/
├── app.py
├── detector.py
├── requirements.txt
├── .gitignore
├── models/              # Local model files
├── weights/             # Optional local weights
├── data/                # Local application data
└── README.md
```

The exact contents of `models/`, `weights/`, and `data/` depend on your local setup. Model files and personal data should remain untracked unless there is a specific, safe reason to distribute them.

## Roadmap

Potential future improvements:

- More flexible model selection
- Expanded detection analytics
- Improved export and session management
- Additional configurable vision workflows
- Further UI and accessibility refinements

## Contributing

Suggestions, bug reports, and improvements are welcome. Open an issue to discuss a change before submitting a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>NEXSIGHT</strong>
  <br>
  <sub>Perception through code.</sub>
</p>
