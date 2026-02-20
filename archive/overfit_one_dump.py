# overfit_one_dump.py
import os
import argparse
import numpy as np
import torch

from dataset import RHDDatasetCoords
from losses import HeatmapMSELoss, WingLoss, coords_to_heatmaps, soft_argmax_2d
from train_1 import HandPoseNet, build_optimizer_stage1, build_optimizer_stage2


def save_npz(path, img_chw, gt_coords, pred_coords, gt_hm, pred_hm, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # move to cpu numpy (store float32 to reduce size)
    out = {
        "img_chw": img_chw.detach().cpu().float().numpy().astype(np.float32),          # (3,256,256)
        "gt_coords": gt_coords.detach().cpu().float().numpy().astype(np.float32),      # (21,2) in [0,1]
        "pred_coords": pred_coords.detach().cpu().float().numpy().astype(np.float32),  # (21,2) in [0,1] (or unbounded)
        "gt_hm": gt_hm.detach().cpu().float().numpy().astype(np.float32),              # (21,64,64)
        "pred_hm": pred_hm.detach().cpu().float().numpy().astype(np.float32),          # (21,64,64)
        "meta": meta,
    }
    np.savez_compressed(path, **out)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--root", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--hand", default="left", choices=["left", "right", "auto"])
    p.add_argument("--input-size", type=int, default=256)

    p.add_argument("--stage", type=int, default=1, choices=[1, 2])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--iters-per-epoch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--freeze-heatmap", action="store_true")
    p.add_argument("--lambda-hm", type=float, default=1.0)
    p.add_argument("--lambda-coord", type=float, default=1.0)

    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=100.0)

    p.add_argument("--outdir", default="debug_overfit_one")
    p.add_argument("--dump-every", type=int, default=5)

    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = p.parse_args()

    if args.iters_per_epoch < 1:
        raise ValueError("--iters-per-epoch must be >= 1")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("device:", device)

    ds = RHDDatasetCoords(
        args.root, split="training", input_size=args.input_size, hand=args.hand, normalize=True
    )
    img, coords = ds[args.index]
    img = img.unsqueeze(0).to(device)       # (1,3,256,256)
    coords = coords.unsqueeze(0).to(device) # (1,21,2)

    model = HandPoseNet().to(device)

    hm_loss_fn = HeatmapMSELoss(H=64, W=64, sigma=args.sigma).to(device)
    coord_loss_fn = WingLoss(w=10.0, epsilon=2.0).to(device)

    if args.stage == 1:
        optim = build_optimizer_stage1(model, lr=args.lr)
    else:
        optim = build_optimizer_stage2(
            model, lr=args.lr,
            freeze_backbone=args.freeze_backbone,
            freeze_heatmap=args.freeze_heatmap
        )

    with torch.no_grad():
        gt_hm = coords_to_heatmaps(coords, H=64, W=64, sigma=args.sigma)  # (1,21,64,64)

    run_dir = os.path.join(args.outdir, f"idx{args.index:05d}_stage{args.stage}")
    dump_dir = os.path.join(run_dir, "npz")
    os.makedirs(dump_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()

        last_loss = None
        last_hm = None
        last_coord = None

        for _ in range(args.iters_per_epoch):
            optim.zero_grad(set_to_none=True)

            if args.stage == 1:
                pred_hm = model.forward_heatmap(img)
                loss_hm = hm_loss_fn(pred_hm, coords)
                loss = loss_hm
                loss_coord = None
            else:
                pred_hm, pred_xy = model(img)
                loss_hm = hm_loss_fn(pred_hm, coords)
                loss_coord = coord_loss_fn(pred_xy, coords)
                loss = args.lambda_hm * loss_hm + args.lambda_coord * loss_coord

            loss.backward()
            optim.step()

            last_loss = float(loss.item())
            last_hm = float(loss_hm.item())
            last_coord = float(loss_coord.item()) if loss_coord is not None else None

        # dump + print
        if (epoch % args.dump_every) == 0 or epoch == (args.epochs - 1):
            model.eval()
            with torch.no_grad():
                if args.stage == 1:
                    pred_hm = model.forward_heatmap(img)
                    pred_xy = soft_argmax_2d(pred_hm, beta=args.beta, normalize=True)
                else:
                    pred_hm, pred_xy = model(img)

                if last_coord is None:
                    print(f"epoch {epoch:04d}  loss={last_loss:.6f}  hm={last_hm:.6f}")
                else:
                    print(f"epoch {epoch:04d}  loss={last_loss:.6f}  hm={last_hm:.6f}  coord={last_coord:.6f}")

                # store one file per epoch checkpoint
                meta = {
                    "epoch": epoch,
                    "index": args.index,
                    "stage": args.stage,
                    "loss": last_loss,
                    "hm_loss": last_hm,
                    "coord_loss": last_coord if last_coord is not None else -1.0,
                    "sigma": args.sigma,
                    "beta": args.beta,
                }

                npz_path = os.path.join(dump_dir, f"epoch_{epoch:04d}.npz")
                save_npz(
                    npz_path,
                    img_chw=img[0],
                    gt_coords=coords[0],
                    pred_coords=pred_xy[0],
                    gt_hm=gt_hm[0],
                    pred_hm=pred_hm[0],
                    meta=meta,
                )

    print("done. dumps in:", dump_dir)


if __name__ == "__main__":
    main()
