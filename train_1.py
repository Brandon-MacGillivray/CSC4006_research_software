
import argparse
import os
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import RHDDatasetCoords  
from losses import WingLoss, HeatmapMSELoss
from architecture import Backbone, Heatmap_reg, coord_reg  
from utils import save_checkpoint, EarlyStopper  

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


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def build_optimizer_stage1(model: HandPoseNet, lr: float):
    set_requires_grad(model.backbone, True)
    set_requires_grad(model.heatmapHead, True)
    set_requires_grad(model.coordhead, False)

    params = list(model.backbone.parameters()) + list(model.heatmapHead.parameters())
    return torch.optim.Adam(params, lr=lr)


def build_optimizer_stage2(model: HandPoseNet, lr: float, freeze_backbone: bool, freeze_heatmap: bool):
    set_requires_grad(model.coordhead, True)
    set_requires_grad(model.backbone, not freeze_backbone)
    set_requires_grad(model.heatmapHead, not freeze_heatmap)

    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr)


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

        # step optimizer 
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


def append_csv_row(csv_path, row):
    file_exists = os.path.exists(csv_path)
    if not file_exists:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stage", "epoch", "train_loss", "val_loss", "seconds"])
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def make_loaders(train_ds, val_ds, batch_size, num_workers):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/RHD_published_v2")
    parser.add_argument("--checkpoint-root", default="training_results")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size-stage1", type=int, default=64)
    parser.add_argument("--batch-size-stage2", type=int, default=64)
    parser.add_argument("--accum-steps-stage1", type=int, default=1)
    parser.add_argument("--accum-steps-stage2", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--lr-stage1", type=float, default=1e-3)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--stage1-patience", type=int, default=5)
    parser.add_argument("--stage1-min-delta", type=float, default=0.0)
    parser.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--train-dataset-length", default="0")
    parser.add_argument("--freeze-backbone-stage2", action="store_true")
    parser.add_argument("--freeze-heatmap-stage2", action="store_true")
    parser.add_argument(
        "--tips-bases-only",
        action="store_true",
        help="Use 10 keypoints only: finger tips + finger bases (per hand).",
    )
    parser.add_argument("--lambda-hm", type=float, default=1.0)
    parser.add_argument("--lambda-coord", type=float, default=1.0)

    args = parser.parse_args()
    if args.tips_bases_only:
        keypoint_indices = list(TIP_BASE_KEYPOINT_INDICES)
    else:
        keypoint_indices = list(range(21))
    num_keypoints = len(keypoint_indices)

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Current device:", torch.cuda.current_device())
    print("Selected keypoints:", keypoint_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    job_id = args.job_id if args.job_id is not None else "local"
    run_dir = os.path.join(args.checkpoint_root, str(job_id))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "losses.csv")

    train_ds = RHDDatasetCoords(
        args.root,
        split="training",
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
        keypoint_indices=keypoint_indices,
    )
    val_ds = RHDDatasetCoords(
        args.root,
        split="evaluation",
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
        keypoint_indices=keypoint_indices,
    )

    if int(args.train_dataset_length) > 0:
        N = min(int(args.train_dataset_length), len(train_ds))
        train_ds = Subset(train_ds, range(N))

    model = HandPoseNet(num_keypoints=num_keypoints).to(device)
    hm_loss_fn = HeatmapMSELoss()
    coord_loss_fn = WingLoss(w=10.0, epsilon=2.0)

    # STAGE 1: heatmap only

    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=args.batch_size_stage1, num_workers=args.num_workers)

    optim = build_optimizer_stage1(model, lr=args.lr_stage1)

    early_stopper = EarlyStopper(patience=args.stage1_patience, min_delta=args.stage1_min_delta)
    best_val = float("inf")

    for epoch in range(args.stage1_epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, hm_loss_fn, optim, device, stage=1, accum_steps=args.accum_steps_stage1)
        val_loss = validate(model, val_loader, hm_loss_fn, device, stage=1)

        elapsed = time.time() - t0
        append_csv_row(csv_path, [1, epoch, float(train_loss), float(val_loss), float(elapsed)])

        ckpt = {
            "stage": 1,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "batch_size": args.batch_size_stage1,
            "accum_steps": args.accum_steps_stage1,
            "keypoint_indices": keypoint_indices,
        }
        save_checkpoint(ckpt, os.path.join(ckpt_dir, f"stage1_epoch_{epoch}.pt"))

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt, os.path.join(ckpt_dir, "stage1_best.pt"))

        if early_stopper.early_stop(val_loss):
            print(f"stage 1 early stop at epoch {epoch} (val={val_loss:.6f})")
            break

    best_path = os.path.join(ckpt_dir, "stage1_best.pt")
    if os.path.exists(best_path):
        best = torch.load(best_path, map_location=device)
        model.load_state_dict(best["model_state"])
        print("loaded stage1_best.pt for stage 2")

    # STAGE 2: coord branch starts 

    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=args.batch_size_stage2, num_workers=args.num_workers)

    optim = build_optimizer_stage2(model, lr=args.lr_stage2, freeze_backbone=args.freeze_backbone_stage2, freeze_heatmap=args.freeze_heatmap_stage2)

    best_val2 = float("inf")
    early_stopper2 = EarlyStopper(patience=10, min_delta=0.0)

    for epoch in range(args.stage2_epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            hm_loss_fn,
            optim,
            device,
            stage=2,
            loss_coord_fn=coord_loss_fn,
            lambda_hm=args.lambda_hm,
            lambda_coord=args.lambda_coord,
            accum_steps=args.accum_steps_stage2,
        )
        val_loss = validate(
            model,
            val_loader,
            hm_loss_fn,
            device,
            stage=2,
            loss_coord_fn=coord_loss_fn,
            lambda_hm=args.lambda_hm,
            lambda_coord=args.lambda_coord,
        )

        elapsed = time.time() - t0
        append_csv_row(csv_path, [2, epoch, float(train_loss), float(val_loss), float(elapsed)])

        ckpt = {
            "stage": 2,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "freeze_backbone_stage2": args.freeze_backbone_stage2,
            "freeze_heatmap_stage2": args.freeze_heatmap_stage2,
            "lambda_hm": args.lambda_hm,
            "lambda_coord": args.lambda_coord,
            "batch_size": args.batch_size_stage2,
            "accum_steps": args.accum_steps_stage2,
            "keypoint_indices": keypoint_indices,
        }
        save_checkpoint(ckpt, os.path.join(ckpt_dir, f"stage2_epoch_{epoch}.pt"))

        if val_loss < best_val2:
            best_val2 = val_loss
            save_checkpoint(ckpt, os.path.join(ckpt_dir, "best.pt"))

        if early_stopper2.early_stop(val_loss):
            print(f"stage 2 early stop at epoch {epoch} (val={val_loss:.6f})")
            break


if __name__ == "__main__":
    main()
