# render_npz.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def chw_to_hwc(img_chw):
    img = np.clip(img_chw, 0.0, 1.0)
    return np.transpose(img, (1, 2, 0))


def agg_heatmap(hm_jhw, mode="max"):
    if mode == "max":
        return hm_jhw.max(axis=0)
    if mode == "mean":
        return hm_jhw.mean(axis=0)
    raise ValueError("mode must be 'max' or 'mean'")


def save_figure(out_path, img_chw, gt_coords, pred_coords, gt_hm, pred_hm, title_prefix="", agg="max"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = chw_to_hwc(img_chw)

    gt_agg = agg_heatmap(gt_hm, mode=agg)
    pr_agg = agg_heatmap(pred_hm, mode=agg)

    gt_px = np.clip(gt_coords, 0.0, 1.0) * 255.0
    pr_px = np.clip(pred_coords, 0.0, 1.0) * 255.0

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title(f"{title_prefix}Image + coords (GT=green, Pred=red)".strip())
    ax1.imshow(img)
    ax1.scatter(gt_px[:, 0], gt_px[:, 1], s=18, c="lime", label="GT")
    ax1.scatter(pr_px[:, 0], pr_px[:, 1], s=18, c="red", label="Pred")
    ax1.legend(loc="lower right")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title(f"{title_prefix}GT heatmap ({agg} over joints)".strip())
    ax2.imshow(gt_agg)
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title(f"{title_prefix}Pred heatmap ({agg} over joints)".strip())
    ax3.imshow(pr_agg)
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title(f"{title_prefix}Abs diff".strip())
    ax4.imshow(np.abs(pr_agg - gt_agg))
    ax4.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz-dir", required=True, help="directory containing epoch_XXXX.npz")
    p.add_argument("--outdir", default=None, help="where to write PNGs (default: <npz-dir>/png)")
    p.add_argument("--agg", default="max", choices=["max", "mean"])
    args = p.parse_args()

    outdir = args.outdir or os.path.join(args.npz_dir, "png")
    os.makedirs(outdir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.npz_dir) if f.endswith(".npz")])
    if not files:
        raise RuntimeError(f"No .npz files found in {args.npz_dir}")

    for f in files:
        path = os.path.join(args.npz_dir, f)
        data = np.load(path, allow_pickle=True)

        img_chw = data["img_chw"]
        gt_coords = data["gt_coords"]
        pred_coords = data["pred_coords"]
        gt_hm = data["gt_hm"]
        pred_hm = data["pred_hm"]
        meta = data["meta"].item() if "meta" in data else {}

        epoch = meta.get("epoch", None)
        loss = meta.get("loss", None)
        title = ""
        if epoch is not None:
            title += f"e{int(epoch):04d} "
        if loss is not None:
            title += f"loss={float(loss):.4f} "

        out_path = os.path.join(outdir, f.replace(".npz", ".png"))
        save_figure(out_path, img_chw, gt_coords, pred_coords, gt_hm, pred_hm, title_prefix=title, agg=args.agg)

    print("done. pngs in:", outdir)


if __name__ == "__main__":
    main()
