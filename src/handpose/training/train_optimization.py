"""Optimizer builders for the two-stage training pipeline.

This module configures the stage-specific optimizers used by the main training
entry point, including optional freezing during stage 2.
"""

import torch
import torch.nn as nn

from handpose.models.hand_pose_model import HandPoseNet


def set_requires_grad(module: nn.Module, flag: bool):
    """Enable or disable gradients for all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = flag


def build_optimizer_stage1(model: HandPoseNet, lr: float):
    """Build the stage-1 optimizer for backbone + heatmap head."""
    set_requires_grad(model.backbone, True)
    set_requires_grad(model.heatmapHead, True)
    set_requires_grad(model.coordhead, False)

    params = list(model.backbone.parameters()) + list(model.heatmapHead.parameters())
    return torch.optim.Adam(params, lr=lr)


def build_optimizer_stage2(model: HandPoseNet, lr: float, freeze_backbone: bool, freeze_heatmap: bool):
    """Build the stage-2 optimizer with optional backbone/head freezing."""
    set_requires_grad(model.coordhead, True)
    set_requires_grad(model.backbone, not freeze_backbone)
    set_requires_grad(model.heatmapHead, not freeze_heatmap)

    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr)
