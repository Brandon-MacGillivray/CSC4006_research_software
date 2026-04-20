"""Tests for checkpoint schema validation helpers.

These tests protect the repository's checkpoint metadata contract used by
training, evaluation, and inference workflows.
"""

import pytest

from handpose.checkpoints import CHECKPOINT_VERSION, validate_checkpoint


def make_ckpt(**overrides):
    ckpt = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "stage": 2,
        "epoch": 7,
        "model_state": {},
        "num_keypoints": 3,
        "keypoint_indices": [0, 1, 2],
    }
    ckpt.update(overrides)
    return ckpt


def test_validate_checkpoint_accepts_and_normalizes_valid_schema():
    ckpt = make_ckpt(stage="2", epoch="7")
    validated = validate_checkpoint(ckpt)

    assert validated["checkpoint_version"] == CHECKPOINT_VERSION
    assert validated["stage"] == 2
    assert validated["epoch"] == 7
    assert validated["num_keypoints"] == 3
    assert validated["keypoint_indices"] == [0, 1, 2]


def test_validate_checkpoint_rejects_duplicate_keypoint_indices():
    with pytest.raises(ValueError, match="unique indices"):
        validate_checkpoint(make_ckpt(keypoint_indices=[0, 1, 1]))


def test_validate_checkpoint_rejects_inconsistent_num_keypoints():
    with pytest.raises(ValueError, match="inconsistent"):
        validate_checkpoint(make_ckpt(num_keypoints=2, keypoint_indices=[0, 1, 2]))
