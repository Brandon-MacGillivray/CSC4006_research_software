"""Fusion and heatmap-decoding helpers for hand pose inference.

This module implements the geometry and selection logic used to combine branch
predictions in fused inference mode.
"""

import torch

from handpose.data.keypoints import TIP_BASE_KEYPOINT_INDICES


# 21-joint hand skeleton edges used to compute fusion alpha.
HAND_BONE_EDGES_21 = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Global-index base->tip pairs used for the 10-joint tip/base fusion variant.
HAND_BONE_EDGES_10_GLOBAL = [
    (1, 4),
    (5, 8),
    (9, 12),
    (13, 16),
    (17, 20),
]


def heatmaps_to_coords_argmax(pred_heatmaps: torch.Tensor):
    """Decode heatmaps to normalized (x, y) coordinates via argmax."""
    n, k, h, w = pred_heatmaps.shape
    flat = pred_heatmaps.view(n, k, -1)
    max_idx = torch.argmax(flat, dim=-1)
    y = (max_idx // w).float()
    x = (max_idx % w).float()
    if w > 1:
        x = x / float(w - 1)
    else:
        x = x * 0.0
    if h > 1:
        y = y / float(h - 1)
    else:
        y = y * 0.0
    return torch.stack([x, y], dim=-1)


def resolve_fusion_bone_edges(model_keypoint_indices):
    """Resolve local fusion edges for full-21 or tip/base-10 checkpoints."""
    keypoints = [int(x) for x in model_keypoint_indices]
    if len(set(keypoints)) != len(keypoints):
        raise ValueError(f"Duplicate keypoint indices in checkpoint: {keypoints}")

    index_of = {kp: i for i, kp in enumerate(keypoints)}
    if set(keypoints) == set(range(21)):
        return [(index_of[a], index_of[b]) for a, b in HAND_BONE_EDGES_21]
    if set(keypoints) == set(TIP_BASE_KEYPOINT_INDICES):
        return [(index_of[a], index_of[b]) for a, b in HAND_BONE_EDGES_10_GLOBAL]

    raise ValueError(
        "Fusion supports only full 21-joint checkpoints or tip/base 10-joint checkpoints. "
        f"Got keypoints: {keypoints}"
    )


def fuse_coords(pred_heatmaps: torch.Tensor, pred_coords: torch.Tensor, bone_edges_local):
    """Fuse heatmap and coordinate predictions using d_i < alpha."""
    hm_coords = heatmaps_to_coords_argmax(pred_heatmaps)

    bone_lengths = []
    for a, b in bone_edges_local:
        vec = pred_coords[:, a, :] - pred_coords[:, b, :]
        bone_lengths.append(torch.sqrt((vec * vec).sum(dim=-1)))
    bone_lengths = torch.stack(bone_lengths, dim=-1)
    alpha = torch.median(bone_lengths, dim=-1).values

    diff_hm_coord = hm_coords - pred_coords
    d = torch.sqrt((diff_hm_coord * diff_hm_coord).sum(dim=-1))
    use_heatmap = d < alpha.unsqueeze(-1)
    fused = torch.where(use_heatmap.unsqueeze(-1), hm_coords, pred_coords)

    return fused, hm_coords, alpha, use_heatmap
