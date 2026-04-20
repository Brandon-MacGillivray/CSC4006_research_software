"""Loss functions used by the hand-pose training pipeline.

This module provides the heatmap and coordinate regression losses used during
the two-stage optimisation process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# def soft_argmax_2d(heatmaps, beta=100.0, normalize=True):
#     """
#     heatmaps: (N, J, H, W)
#     returns coords: (N, J, 2) as (x, y)
#     """
#     n, j, h, w = heatmaps.shape
#     flat = heatmaps.view(n, j, -1)
#     prob = F.softmax(flat * beta, dim=-1).view(n, j, h, w)

#     ys = torch.linspace(0, h - 1, h, device=heatmaps.device)
#     xs = torch.linspace(0, w - 1, w, device=heatmaps.device)
#     yy, xx = torch.meshgrid(ys, xs, indexing="ij")

#     x = (prob * xx).sum(dim=(-2, -1))
#     y = (prob * yy).sum(dim=(-2, -1))

#     if normalize:
#         x = x / (w - 1)
#         y = y / (h - 1)

#     return torch.stack([x, y], dim=-1)


def coords_to_heatmaps(coords, H=64, W=64, sigma=2.0):
    """
    coords: (N,J,2) in [0,1] normalized
    returns (N,J,H,W)
    """
    N, J, _ = coords.shape
    device = coords.device
    yy = torch.arange(H, device=device).view(1, 1, H, 1).float()
    xx = torch.arange(W, device=device).view(1, 1, 1, W).float()

    x = coords[..., 0].clamp(0, 1) * (W - 1)
    y = coords[..., 1].clamp(0, 1) * (H - 1)
    x = x.view(N, J, 1, 1)
    y = y.view(N, J, 1, 1)

    return torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


class HeatmapMSELoss(nn.Module):
    def __init__(self, H=64, W=64, sigma=2.0):
        """Configure Gaussian-target heatmap MSE settings."""
        super().__init__()
        self.H, self.W, self.sigma = H, W, sigma
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, pred_heatmaps, target_coords):
        """Compute MSE between predicted and Gaussian target heatmaps."""
        target = coords_to_heatmaps(target_coords, H=self.H, W=self.W, sigma=self.sigma)
        return self.mse(pred_heatmaps, target)


class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        """Configure Wing loss shape parameters."""
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * torch.log(torch.tensor(1 + w / epsilon))

    def forward(self, pred, target):
        """Compute mean Wing loss over all coordinate values."""
        # pred, target: (N, J, 2)
        diff = pred - target
        abs_diff = diff.abs()

        w = self.w
        eps = self.epsilon
        C = self.C.to(pred.device)

        small = w * torch.log(1 + abs_diff / eps)
        large = abs_diff - C

        loss = torch.where(abs_diff < w, small, large)
        denom = loss.numel()

        return loss.sum() / denom
