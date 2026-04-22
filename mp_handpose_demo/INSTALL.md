# Installation

This document describes a straightforward local installation for the standalone
hand UI demonstrator.

## 1. Install the ZED SDK

Install the StereoLabs ZED SDK for your operating system and ensure that the
Python API is included. The demo expects the import `from pyzed import sl` to
work in the active Python environment.

After installation:

- connect the ZED camera
- confirm that the camera is visible to the SDK tools
- confirm that the Python binding is available

## 2. Create a Python 3.10 Environment

From the repository root:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are using Conda instead of `venv`, create a Python 3.10 environment and
then install the same package set from `requirements.txt`.

## 3. Confirm the Repository Layout

Before launch, confirm that these paths exist relative to the repository root:

- `hand_18.py`
- `data/training_001.tif`
- `data/training_groundtruth_001.tif`
- `data/rnvu_data/Sarah_050.tif`

If these files are not found, the application will not be able to load the
demo content.

## 4. Run the Demo

From the repository root:

```bash
python hand_18.py
```

## 5. Smoke Test

The installation can be considered successful when the following sequence
works:

1. The application window opens without import errors.
2. The start page displays "Wave your hand to begin."
3. Showing one hand reveals the cursor and hand overlay.
4. Making a fist activates the on-screen selection once per intended click.
5. The menu allows navigation to both the segmentation and tone-mapping pages.

## 6. Development Mode

For local development on a non-kiosk display, edit `hand_18.py` and set:

```python
FULLSCREEN = False
```

This keeps the same application logic but uses a window rather than fullscreen
display.

## Troubleshooting

### "Failed to open ZED."

- confirm that the ZED SDK is installed correctly
- confirm that the camera is connected before launch
- confirm that no other application is holding the camera

### `ModuleNotFoundError` for `pyzed`

- reinstall or repair the ZED SDK Python binding
- activate the intended Python environment before launch

### Assets do not load

- run the application from the repository root
- verify that the `data/` directory is present and complete
