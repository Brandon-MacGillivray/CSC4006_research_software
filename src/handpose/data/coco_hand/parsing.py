"""Annotation parsing helpers for COCO-style hand datasets.

This module loads COCO-format annotations and converts them into sample
structures used by the repository dataset wrappers.
"""

import json

import numpy as np


def load_annotation_payload(annotation_path):
    """Load the COCO-style annotation JSON."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_samples(payload: dict, image_root, *, total_keypoints: int = 21):
    """Convert COCO image/annotation tables into flat hand samples."""
    images_by_id = {}
    for image_meta in payload.get("images", []):
        if "id" not in image_meta:
            continue
        images_by_id[int(image_meta["id"])] = image_meta

    samples = []
    annotations = sorted(
        payload.get("annotations", []),
        key=lambda ann: (int(ann.get("image_id", -1)), int(ann.get("id", -1))),
    )
    required_values = int(total_keypoints) * 3
    for ann in annotations:
        if int(ann.get("iscrowd", 0)) != 0:
            continue

        image_id = int(ann.get("image_id", -1))
        image_meta = images_by_id.get(image_id)
        if image_meta is None:
            continue

        raw_keypoints = ann.get("keypoints")
        if not isinstance(raw_keypoints, list) or len(raw_keypoints) < required_values:
            continue

        keypoints = np.asarray(raw_keypoints[:required_values], dtype=np.float32).reshape(total_keypoints, 3)
        file_name = str(image_meta.get("file_name", "")).strip()
        if not file_name:
            continue

        samples.append(
            {
                "image_id": image_id,
                "annotation_id": int(ann.get("id", len(samples))),
                "image_path": image_root / file_name,
                "coords": keypoints[:, :2],
                "vis": keypoints[:, 2],
            }
        )
    return samples
