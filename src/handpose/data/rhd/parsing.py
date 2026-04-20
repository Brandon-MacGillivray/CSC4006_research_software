"""Annotation parsing helpers for the RHD dataset.

This module extracts coordinates and hand-selection information from the RHD
annotation format for dataset loading and evaluation.
"""

import pickle


def load_annotations(annotation_path):
    """Load the RHD annotation pickle."""
    with open(annotation_path, "rb") as f:
        return pickle.load(f)


def select_hand(uv_data, hand: str):
    """Select left, right, or most-visible hand landmarks from RHD annotations."""
    if hand == "left":
        return uv_data[0:21]
    if hand == "right":
        return uv_data[21:42]
    if hand == "auto":
        left = uv_data[0:21]
        right = uv_data[21:42]
        left_score = left[:, 2].sum()
        right_score = right[:, 2].sum()
        return right if right_score >= left_score else left
    raise ValueError("hand must be one of: 'left', 'right', 'auto'")


def find_uv_key(anno: dict):
    """Find the UV-coordinate annotation key in a sample dict."""
    return next(k for k in anno.keys() if str(k).lower().startswith("uv"))
