# Hand UI Demos

This repository contains a standalone gesture-driven PySide6 demonstrator for
two precomputed image-processing showcases:

- Segmentation: select a region on microscopy images and reveal either a
  connected component or a precomputed RNvU retina mask.
- Tone mapping: move your hand to reveal the improved rendering inside a
  spotlight.

The application uses a ZED camera plus MediaPipe Hands for interaction. It is
not a training pipeline and does not run segmentation or tone-mapping models
live; it visualises prepared demo assets through hand-controlled interaction.

## Artefact Summary

- Main entry point: `hand_18.py`
- Bundled assets: `data/`
- Interaction model: palm-centre cursor plus fist-to-click selection
- Display mode: portrait fullscreen by default

## Repository Layout

```text
hand_18.py
data/
README.md
REQUIREMENTS.md
requirements.txt
INSTALL.md
REPLICATION_GUIDE.md
ASSET_ATTRIBUTION.md
LICENSE
```

## Quick Start

1. Install the hardware and software prerequisites described in
   `REQUIREMENTS.md`.
2. Follow `INSTALL.md` to create the Python environment and install
   dependencies.
3. From the repository root, run:

```bash
python hand_18.py
```

Expected initial behaviour:

- a portrait window opens
- the start page displays "Wave your hand to begin."
- showing one hand reveals the cursor
- making a fist acts as a click

## Demo Behaviour

### Segmentation

- Example 1 loads `training_001.tif` plus `training_groundtruth_001.tif`
  and highlights the selected connected component.
- Example 2 loads `Sarah_050.tif` plus the RNvU class masks under
  `data/rnvu_data/` and reveals the mask covering the selected pixel.

### Tone Mapping

- The tone-mapping page cycles through fixed pairs of baseline and improved
  images.
- The improved image is only shown inside a spotlight whose size scales with
  the user's palm width.

## Limitations

- A connected ZED camera is required.
- The current interaction logic tracks one hand only.
- The app relies on relative `data/` paths and should be launched from the
  repository root.
- The bundled assets are demonstration inputs and outputs, not live model
  predictions.

## Additional Documentation

- `REQUIREMENTS.md`: hardware, software, and storage assumptions
- `INSTALL.md`: installation and smoke-test steps
- `REPLICATION_GUIDE.md`: how to reproduce the demo workflow
- `ASSET_ATTRIBUTION.md`: asset provenance and redistribution notes
