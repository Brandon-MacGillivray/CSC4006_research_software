"""Prediction helpers for fused, heatmap-only, and coord-only inference.

This module provides the shared inference logic used by evaluation, benchmarking,
single-image prediction, and qualitative rendering scripts.
"""

import torch

from handpose.inference.fusion import (
    fuse_coords,
    heatmaps_to_coords_argmax,
    resolve_fusion_bone_edges,
)
from handpose.models.hand_pose_model import HandPoseNet


SUPPORTED_PREDICTION_MODES = ("fusion", "heatmap", "coord")


def _validate_prediction_mode(prediction_mode: str):
    """Validate the requested inference mode."""
    if prediction_mode not in SUPPORTED_PREDICTION_MODES:
        raise ValueError(
            f"prediction_mode must be one of {SUPPORTED_PREDICTION_MODES}, got {prediction_mode!r}"
        )


@torch.no_grad()
def build_fusion_context(model_keypoint_indices):
    """Build reusable fusion context from checkpoint keypoint layout."""
    return {"bone_edges_local": resolve_fusion_bone_edges(model_keypoint_indices)}


@torch.no_grad()
def predict_all_modes(model: HandPoseNet, x: torch.Tensor, fusion_context: dict):
    """Run model inference once and return all branch outputs plus fusion metadata."""
    pred_heatmaps, pred_coords = model(x)
    fused, hm_coords, alpha, use_heatmap = fuse_coords(
        pred_heatmaps=pred_heatmaps,
        pred_coords=pred_coords,
        bone_edges_local=fusion_context["bone_edges_local"],
    )
    diff_hm_coord = hm_coords - pred_coords
    d = torch.sqrt((diff_hm_coord * diff_hm_coord).sum(dim=-1))
    return {
        "fusion": fused,
        "heatmap": hm_coords,
        "coord": pred_coords,
        "pred_heatmaps": pred_heatmaps,
        "alpha": alpha,
        "use_heatmap": use_heatmap,
        "d": d,
    }


@torch.no_grad()
def predict_coords(
    model: HandPoseNet,
    x: torch.Tensor,
    fusion_context: dict,
    prediction_mode: str = "fusion",
):
    """Run model inference and return coordinates for the selected prediction mode."""
    _validate_prediction_mode(prediction_mode)
    pred_heatmaps, pred_coords = model(x)
    if prediction_mode == "coord":
        return pred_coords

    hm_coords = heatmaps_to_coords_argmax(pred_heatmaps)
    if prediction_mode == "heatmap":
        return hm_coords

    fused, _, _, _ = fuse_coords(
        pred_heatmaps=pred_heatmaps,
        pred_coords=pred_coords,
        bone_edges_local=fusion_context["bone_edges_local"],
    )
    return fused


@torch.no_grad()
def infer_coords(
    model: HandPoseNet,
    x: torch.Tensor,
    fusion_context: dict,
    prediction_mode: str = "fusion",
):
    """Run inference for one image and return predicted coordinates on CPU."""
    pred = predict_coords(
        model=model,
        x=x,
        fusion_context=fusion_context,
        prediction_mode=prediction_mode,
    )
    return pred[0].detach().cpu()


@torch.no_grad()
def infer_fused_coords(model: HandPoseNet, x: torch.Tensor, fusion_context: dict):
    """Backward-compatible wrapper for fused single-image inference."""
    return infer_coords(
        model=model,
        x=x,
        fusion_context=fusion_context,
        prediction_mode="fusion",
    )
