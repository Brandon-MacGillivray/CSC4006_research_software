"""Per-epoch training and validation steps for model optimisation.

This module contains the lower-level loops used by the repository training
script for stage-1 and stage-2 optimisation.
"""

import torch


def _build_epoch_metrics(total_loss_sum, hm_loss_sum, coord_loss_sum, num_steps):
    """Return averaged loss metrics for one epoch."""
    denom = max(int(num_steps), 1)
    return {
        "loss_total": float(total_loss_sum) / denom,
        "loss_hm": float(hm_loss_sum) / denom,
        "loss_coord": float(coord_loss_sum) / denom,
        "num_steps": int(num_steps),
    }


def train_one_epoch(
    model,
    loader,
    loss_hm_fn,
    optim,
    device,
    stage: int,
    loss_coord_fn=None,
    lambda_hm: float = 1.0,
    lambda_coord: float = 1.0,
    accum_steps: int = 1,
):
    """Run one training epoch for stage 1 or stage 2."""
    model.train()
    total_loss_sum = 0.0
    hm_loss_sum = 0.0
    coord_loss_sum = 0.0

    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")

    if accum_steps < 1:
        raise ValueError("accum_steps must be >= 1")

    optim.zero_grad(set_to_none=True)

    for step, (imgs, coords) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)

        if stage == 1:
            pred_heatmaps = model.forward_heatmap(imgs)
            loss_hm = loss_hm_fn(pred_heatmaps, coords)
            loss_coord = loss_hm.new_tensor(0.0)
            loss_total = loss_hm
        else:
            pred_heatmaps, pred_coords = model(imgs)
            loss_hm = loss_hm_fn(pred_heatmaps, coords)
            if loss_coord_fn is None:
                raise ValueError("stage 2 requires loss_coord_fn")
            loss_coord = loss_coord_fn(pred_coords, coords)
            loss_total = lambda_hm * loss_hm + lambda_coord * loss_coord

        scaled_loss = loss_total / accum_steps
        scaled_loss.backward()

        if (step + 1) % accum_steps == 0:
            optim.step()
            optim.zero_grad(set_to_none=True)

        total_loss_sum += float(loss_total.item())
        hm_loss_sum += float(loss_hm.item())
        coord_loss_sum += float(loss_coord.item())

    if len(loader) % accum_steps != 0:
        optim.step()
        optim.zero_grad(set_to_none=True)

    return _build_epoch_metrics(total_loss_sum, hm_loss_sum, coord_loss_sum, len(loader))


@torch.no_grad()
def validate(
    model,
    loader,
    loss_hm_fn,
    device,
    stage: int,
    loss_coord_fn=None,
    lambda_hm: float = 1.0,
    lambda_coord: float = 1.0,
):
    """Run one validation epoch for stage 1 or stage 2."""
    model.eval()
    total_loss_sum = 0.0
    hm_loss_sum = 0.0
    coord_loss_sum = 0.0

    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")

    for imgs, coords in loader:
        imgs = imgs.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)

        if stage == 1:
            pred_heatmaps = model.forward_heatmap(imgs)
            loss_hm = loss_hm_fn(pred_heatmaps, coords)
            loss_coord = loss_hm.new_tensor(0.0)
            loss_total = loss_hm
        else:
            pred_heatmaps, pred_coords = model(imgs)
            loss_hm = loss_hm_fn(pred_heatmaps, coords)
            if loss_coord_fn is None:
                raise ValueError("stage 2 requires loss_coord_fn")
            loss_coord = loss_coord_fn(pred_coords, coords)
            loss_total = lambda_hm * loss_hm + lambda_coord * loss_coord

        total_loss_sum += float(loss_total.item())
        hm_loss_sum += float(loss_hm.item())
        coord_loss_sum += float(loss_coord.item())

    return _build_epoch_metrics(total_loss_sum, hm_loss_sum, coord_loss_sum, len(loader))
