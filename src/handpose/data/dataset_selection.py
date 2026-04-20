"""Dataset-selection helpers shared across loaders and evaluation code.

This module validates keypoint subsets and re-exports hand-selection helpers
used by dataset-aware parts of the repository.
"""

from handpose.data.rhd.parsing import find_uv_key, select_hand


def validate_keypoint_indices(keypoint_indices, total_keypoints=21):
    """Validate and normalize the requested keypoint index list."""
    if keypoint_indices is None:
        return list(range(total_keypoints))
    if len(keypoint_indices) == 0:
        raise ValueError("keypoint_indices must contain at least one index")
    if len(set(keypoint_indices)) != len(keypoint_indices):
        raise ValueError("keypoint_indices must be unique")
    if min(keypoint_indices) < 0 or max(keypoint_indices) >= total_keypoints:
        raise ValueError(f"keypoint_indices must be in [0, {total_keypoints - 1}]")
    return list(keypoint_indices)


__all__ = [
    "find_uv_key",
    "select_hand",
    "validate_keypoint_indices",
]
