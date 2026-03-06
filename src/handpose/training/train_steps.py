import torch


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
    total = 0.0

    if accum_steps < 1:
        raise ValueError("accum_steps must be >= 1")

    optim.zero_grad(set_to_none=True)

    for step, (imgs, coords) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)

        if stage == 1:
            pred_heatmaps = model.forward_heatmap(imgs)
            loss = loss_hm_fn(pred_heatmaps, coords)
        else:
            pred_heatmaps, pred_coords = model(imgs)
            loss_hm = loss_hm_fn(pred_heatmaps, coords)
            if loss_coord_fn is None:
                raise ValueError("stage 2 requires loss_coord_fn")
            loss_coord = loss_coord_fn(pred_coords, coords)
            loss = lambda_hm * loss_hm + lambda_coord * loss_coord

        loss = loss / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            optim.step()
            optim.zero_grad(set_to_none=True)

        total += float(loss.item()) * accum_steps

    if len(loader) % accum_steps != 0:
        optim.step()
        optim.zero_grad(set_to_none=True)

    return total / max(len(loader), 1)


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
    total = 0.0

    for imgs, coords in loader:
        imgs = imgs.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)

        if stage == 1:
            pred_heatmaps = model.forward_heatmap(imgs)
            loss = loss_hm_fn(pred_heatmaps, coords)
        else:
            pred_heatmaps, pred_coords = model(imgs)
            loss_hm = loss_hm_fn(pred_heatmaps, coords)
            if loss_coord_fn is None:
                raise ValueError("stage 2 requires loss_coord_fn")
            loss_coord = loss_coord_fn(pred_coords, coords)
            loss = lambda_hm * loss_hm + lambda_coord * loss_coord

        total += float(loss.item())

    return total / max(len(loader), 1)
