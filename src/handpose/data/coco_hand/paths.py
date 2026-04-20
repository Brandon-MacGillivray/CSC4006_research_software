"""Path and split-resolution helpers for COCO-style hand datasets.

This module resolves annotation and image locations for HK26K, RH8, and other
COCO-style hand datasets used in the repository.
"""

from pathlib import Path


COCO_HAND_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "validation": "val",
    "eval": "val",
    "evaluation": "val",
    "benchmark": "val",
}


def resolve_split_name(split: str):
    """Resolve train/val/eval aliases to the COCO-hand on-disk split name."""
    text = str(split).strip().lower()
    if text in COCO_HAND_SPLIT_ALIASES:
        return COCO_HAND_SPLIT_ALIASES[text]
    raise ValueError(f"Unsupported COCO-hand split: {split!r}")


def annotation_path(root, split: str):
    """Return the COCO annotation JSON path for one split."""
    resolved_split = resolve_split_name(split)
    return Path(root) / "coco_annotation" / resolved_split / "_annotations.coco.json"


def image_dir(root, split: str):
    """Return the image directory for one COCO-hand split."""
    resolved_split = resolve_split_name(split)
    return Path(root) / "images" / resolved_split
