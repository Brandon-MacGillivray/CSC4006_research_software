"""High-level hand pose network wrapper used across the repository.

This module exposes the combined DRHand-style model built from the backbone and
branch components defined in the lower-level architecture module.
"""

import torch.nn as nn

from handpose.models.architecture import Backbone, Heatmap_reg, coord_reg


class HandPoseNet(nn.Module):
    def __init__(self, num_keypoints=21):
        """Initialize backbone, heatmap head, and coordinate head."""
        super().__init__()
        self.num_keypoints = num_keypoints
        self.backbone = Backbone()
        self.heatmapHead = Heatmap_reg(num_keypoints=num_keypoints)
        self.coordhead = coord_reg(num_keypoints=num_keypoints)

    def forward_heatmap(self, x):
        """Run only the heatmap branch (used in stage-1 training)."""
        heatmaps = self.heatmapHead(self.backbone(x), return_feat_64=False)
        return heatmaps

    def forward(self, x):
        """Run both branches and return heatmaps plus coordinates."""
        heatmaps, feat_64 = self.heatmapHead(self.backbone(x), return_feat_64=True)
        coords = self.coordhead(feat_64)  # (N,K,2)
        return heatmaps, coords
