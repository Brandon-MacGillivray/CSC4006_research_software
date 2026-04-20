"""Tests for fusion logic and heatmap decoding helpers.

These tests cover the keypoint-layout handling and argmax coordinate decoding
used by fused inference.
"""

import pytest
import torch

from handpose.data.keypoints import TIP_BASE_KEYPOINT_INDICES
from handpose.inference.fusion import (
    HAND_BONE_EDGES_21,
    heatmaps_to_coords_argmax,
    resolve_fusion_bone_edges,
)


def test_heatmaps_to_coords_argmax_decodes_normalized_xy():
    heatmaps = torch.zeros(1, 1, 3, 4)
    heatmaps[0, 0, 2, 1] = 10.0

    coords = heatmaps_to_coords_argmax(heatmaps)
    expected = torch.tensor([[[1.0 / 3.0, 1.0]]])

    assert torch.allclose(coords, expected)


def test_resolve_fusion_bone_edges_supports_full_21_layout():
    edges = resolve_fusion_bone_edges(list(range(21)))

    assert edges == HAND_BONE_EDGES_21


def test_resolve_fusion_bone_edges_supports_tip_base_10_and_rejects_other_layouts():
    edges = resolve_fusion_bone_edges(TIP_BASE_KEYPOINT_INDICES)

    assert edges == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    with pytest.raises(ValueError, match="Fusion supports only"):
        resolve_fusion_bone_edges([0, 1, 2])
