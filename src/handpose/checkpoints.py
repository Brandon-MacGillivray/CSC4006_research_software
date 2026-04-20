"""Checkpoint loading, validation, and persistence helpers.

This module defines the repository checkpoint contract used by training,
evaluation, inference, and the preserved checkpoint archive.
"""

import os

import torch


CHECKPOINT_VERSION = 1


def _as_int_list(values, field_name: str):
    """Normalize a list-like field to a list of ints."""
    if values is None:
        raise ValueError(f"Missing required checkpoint field: {field_name}")
    out = [int(x) for x in values]
    if len(out) == 0:
        raise ValueError(f"Checkpoint field {field_name!r} must not be empty")
    if len(set(out)) != len(out):
        raise ValueError(f"Checkpoint field {field_name!r} must contain unique indices")
    return out


def validate_checkpoint(ckpt: dict):
    """Validate strict checkpoint schema and normalize keypoint metadata."""
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint object must be a dict")

    required_fields = [
        "checkpoint_version",
        "stage",
        "epoch",
        "model_state",
        "num_keypoints",
        "keypoint_indices",
    ]
    for field in required_fields:
        if field not in ckpt:
            raise ValueError(f"Missing required checkpoint field: {field}")

    version = int(ckpt["checkpoint_version"])
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint_version={version}. Expected {CHECKPOINT_VERSION}."
        )

    if not isinstance(ckpt["model_state"], dict):
        raise ValueError("Checkpoint field 'model_state' must be a state_dict (dict)")

    num_keypoints = int(ckpt["num_keypoints"])
    if num_keypoints <= 0:
        raise ValueError("Checkpoint field 'num_keypoints' must be > 0")

    keypoint_indices = _as_int_list(ckpt["keypoint_indices"], "keypoint_indices")
    if len(keypoint_indices) != num_keypoints:
        raise ValueError(
            "Checkpoint fields 'num_keypoints' and 'keypoint_indices' are inconsistent: "
            f"{num_keypoints} vs {len(keypoint_indices)}"
        )

    # Normalize in place.
    ckpt["checkpoint_version"] = version
    ckpt["stage"] = int(ckpt["stage"])
    ckpt["epoch"] = int(ckpt["epoch"])
    ckpt["num_keypoints"] = num_keypoints
    ckpt["keypoint_indices"] = keypoint_indices
    return ckpt


def save_training_checkpoint(state: dict, path: str):
    """Validate and persist a training checkpoint to disk."""
    validate_checkpoint(state)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device):
    """Load and validate a checkpoint, returning metadata and model state."""
    obj = torch.load(path, map_location=device)
    ckpt = validate_checkpoint(obj)
    return ckpt, ckpt["model_state"]


def infer_checkpoint_keypoint_indices(ckpt_meta: dict):
    """Return checkpoint keypoint index layout from strict metadata."""
    ckpt = validate_checkpoint(ckpt_meta)
    return list(ckpt["keypoint_indices"])


def get_training_config(ckpt_meta: dict):
    """Return stored training metadata when present."""
    ckpt = validate_checkpoint(ckpt_meta)
    training_config = ckpt.get("training_config")
    if not isinstance(training_config, dict):
        return {}
    return dict(training_config)
