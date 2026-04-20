"""MediaPipe baseline helpers for image loading and landmark inference.

This module wraps the bundled MediaPipe hand-landmarker asset so that baseline
comparisons can be run through the repository evaluation scripts.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(image_path):
    """Load an RGB image as a uint8 NumPy array."""
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _normalize_handedness_entry(entry):
    """Extract handedness label/score from one MediaPipe classification entry."""
    if not entry:
        return "", None
    first = entry[0]
    label = (
        getattr(first, "category_name", None)
        or getattr(first, "display_name", None)
        or ""
    )
    score = getattr(first, "score", None)
    return str(label).strip().lower(), (float(score) if score is not None else None)


class MediaPipeHandLandmarkerRunner:
    """Small wrapper around the MediaPipe Hand Landmarker task API."""

    def __init__(
        self,
        *,
        model_asset_path,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError(
                "MediaPipe is not installed. Install it with `pip install mediapipe` "
                "before running the baseline evaluation."
            ) from exc

        self._mp = mp
        self.model_asset_path = str(Path(model_asset_path))
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_asset_path),
            running_mode=RunningMode.IMAGE,
            num_hands=int(num_hands),
            min_hand_detection_confidence=float(min_hand_detection_confidence),
            min_hand_presence_confidence=float(min_hand_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def close(self):
        """Release the MediaPipe task when present."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def detect_image(self, rgb_image):
        """Run MediaPipe on one RGB uint8 image and return normalized detections."""
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.asarray(rgb_image, dtype=np.uint8),
        )
        result = self._landmarker.detect(mp_image)
        detections = []
        handedness_entries = getattr(result, "handedness", []) or []
        for idx, landmarks in enumerate(getattr(result, "hand_landmarks", []) or []):
            coords = np.array(
                [[float(lm.x), float(lm.y)] for lm in landmarks],
                dtype=np.float32,
            )
            coords = np.clip(coords, 0.0, 1.0)
            handedness, handedness_score = _normalize_handedness_entry(
                handedness_entries[idx] if idx < len(handedness_entries) else []
            )
            detections.append(
                {
                    "coords": coords,
                    "handedness": handedness,
                    "handedness_score": handedness_score,
                }
            )
        return detections
