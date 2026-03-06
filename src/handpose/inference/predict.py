import torch

from handpose.inference.fusion import fuse_coords, resolve_fusion_bone_edges
from handpose.models.hand_pose_model import HandPoseNet


@torch.no_grad()
def build_fusion_context(model_keypoint_indices):
    """Build reusable fusion context from checkpoint keypoint layout."""
    return {"bone_edges_local": resolve_fusion_bone_edges(model_keypoint_indices)}


@torch.no_grad()
def predict_coords(model: HandPoseNet, x: torch.Tensor, fusion_context: dict):
    """Run model inference and return fused coordinates for a batch."""
    pred_heatmaps, pred_coords = model(x)
    fused, _, _, _ = fuse_coords(
        pred_heatmaps=pred_heatmaps,
        pred_coords=pred_coords,
        bone_edges_local=fusion_context["bone_edges_local"],
    )
    return fused


@torch.no_grad()
def infer_fused_coords(model: HandPoseNet, x: torch.Tensor, fusion_context: dict):
    """Run fused inference and return predicted coordinates for one image."""
    pred = predict_coords(model=model, x=x, fusion_context=fusion_context)
    return pred[0].detach().cpu()
