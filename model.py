import torch.nn as nn

from architecture import Backbone, Heatmap_reg, coord_reg


TIP_BASE_KEYPOINT_INDICES = [1, 4, 5, 8, 9, 12, 13, 16, 17, 20]


class HandPoseNet(nn.Module):
    def __init__(self, num_keypoints=21):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.backbone = Backbone()
        self.heatmapHead = Heatmap_reg(num_keypoints=num_keypoints)
        self.coordhead = coord_reg(num_keypoints=num_keypoints)

    def forward_heatmap(self, x):
        heatmaps = self.heatmapHead(self.backbone(x), return_feat_64=False)
        return heatmaps

    def forward(self, x):
        heatmaps, feat_64 = self.heatmapHead(self.backbone(x), return_feat_64=True)
        coords = self.coordhead(feat_64)  # (N,K,2)
        return heatmaps, coords
