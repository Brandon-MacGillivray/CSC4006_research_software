# Requirements

This document describes the expected environment for the standalone
`mp_handpose_demo` artefact.

## Hardware Requirements

- A 64-bit desktop or laptop capable of running the StereoLabs ZED SDK
- A connected ZED camera
- A USB 3.x port suitable for the camera
- A display that can comfortably present a portrait layout
  - preferred: 1080 x 1920 fullscreen
  - acceptable for development: windowed 1080 x 1920

## Software Requirements

- Operating system: Windows 10 or Windows 11, 64-bit, for this packaged setup
- Python 3.10
- StereoLabs ZED SDK with the Python API (`pyzed`)
- The Python packages listed in `requirements.txt`

## Python Package Notes

- `pyzed` is not installed from `requirements.txt`; it is provided by the ZED
  SDK installation.
- The application depends on PySide6, OpenCV, MediaPipe, Pillow, and NumPy.

## Storage Requirements

- Bundled demo assets under `data/` require roughly 60 MB
- Additional space is required for:
  - the Python virtual environment
  - the ZED SDK installation
  - optional generated screenshots or recordings during demonstration

## Runtime Assumptions

- The application is launched from the repository root so that relative
  `data/` paths resolve correctly.
- The default configuration assumes portrait orientation, mirrored horizontal
  hand control, and fullscreen display.
- No dedicated training GPU is required because this artefact is a
  presentation-layer application rather than a model-training workflow.
