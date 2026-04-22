# Replication Guide

This guide describes how to reproduce the demonstration behaviour presented by
the standalone hand UI artefact.

## Scope

This artefact is a presentation-layer demonstrator. Replication in this context
means reproducing the interaction flow and visible behaviour of the two demo
modes, not retraining or recomputing the underlying segmentation or tone-mapped
assets.

## Prerequisites

- complete the steps in `INSTALL.md`
- connect a working ZED camera
- launch the application from the repository root

## Launch Procedure

Run:

```bash
python hand_18.py
```

The demo should open to the start page.

## Reproducing the Segmentation Demo

1. Show one hand to the camera and wait for the cursor to appear.
2. Make a fist to select the menu.
3. Choose `Segmentation Demo`.
4. On the first example:
   - move the cursor over the microscopy image
   - make a fist to select a location
   - verify that the connected component at that point is highlighted
5. Press `Next` to move to the RNvU example.
6. Select different pixels on the retina image.
7. Verify that the visible overlay changes according to the class mask covering
   the selected location.

Relevant assets:

- `data/training_001.tif`
- `data/training_groundtruth_001.tif`
- `data/rnvu_data/Sarah_050.tif`
- `data/rnvu_data/*.tif` class masks

## Reproducing the Tone-Mapping Demo

1. Return to the menu.
2. Choose `Tone Mapping Demo`.
3. Move the tracked hand across the image area.
4. Verify that the improved image is revealed only inside the spotlight.
5. Move the hand closer to or further from the camera.
6. Verify that the spotlight radius changes with palm width.
7. Use `Previous` and `Next` to cycle through the hard-coded image pairs.

Relevant assets:

- `data/*_ref*.png`
- `data/*_Khan20.png`
- `data/*_Cao23.png`

## Reset Behaviour

To reproduce the idle-reset behaviour:

1. Enter either demo page.
2. Remove the hand from view.
3. Wait longer than the configured timeout.
4. Verify that the interface returns to the start page.

## Reproducing Report Screenshots

If the report includes screenshots from this artefact:

1. navigate to the relevant page
2. reproduce the desired cursor or spotlight state
3. capture the screen using the operating-system screenshot tool

## Data and Asset Notes

- The bundled masks and tone-mapped images are treated as prepared demo assets.
- This repository does not document how those assets were originally computed.
- Provenance and redistribution notes should be maintained in
  `ASSET_ATTRIBUTION.md`.
