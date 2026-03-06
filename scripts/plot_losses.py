import argparse
import os
import csv

import matplotlib.pyplot as plt


def _maybe_float(value):
    """Parse a CSV field as float, returning None when empty/missing."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def read_losses(csv_path):
    """Load per-epoch losses and optional component losses from CSV."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "stage": int(row["stage"]),
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                    "train_loss_hm": _maybe_float(row.get("train_loss_hm")),
                    "val_loss_hm": _maybe_float(row.get("val_loss_hm")),
                    "train_loss_coord": _maybe_float(row.get("train_loss_coord")),
                    "val_loss_coord": _maybe_float(row.get("val_loss_coord")),
                }
            )
    return rows


def stage_series(rows, stage_id, train_key, val_key):
    """Extract one train/val metric series for a stage."""
    x, y_train, y_val = [], [], []
    for row in rows:
        if row["stage"] != stage_id:
            continue
        train_val = row.get(train_key)
        val_val = row.get(val_key)
        if train_val is None or val_val is None:
            continue
        x.append(row["epoch"])
        y_train.append(train_val)
        y_val.append(val_val)
    return x, y_train, y_val


def _any_nonzero(values):
    """Return True when any value differs from zero."""
    return any(abs(float(v)) > 0.0 for v in values)


def _plot_stage(
    ax,
    stage_id,
    job_id,
    total_series,
    hm_series=None,
    coord_series=None,
):
    """Render one stage panel with total loss and optional components."""
    total_x, total_train, total_val = total_series
    if total_x:
        ax.plot(total_x, total_train, label=f"train_s{stage_id}_total")
        ax.plot(total_x, total_val, label=f"val_s{stage_id}_total")
    else:
        ax.text(0.5, 0.5, f"No stage {stage_id} data", ha="center", va="center", transform=ax.transAxes)

    if hm_series is not None:
        hm_x, hm_train, hm_val = hm_series
        if hm_x:
            ax.plot(hm_x, hm_train, "--", label=f"train_s{stage_id}_hm")
            ax.plot(hm_x, hm_val, "--", label=f"val_s{stage_id}_hm")

    if coord_series is not None:
        coord_x, coord_train, coord_val = coord_series
        if coord_x and (_any_nonzero(coord_train) or _any_nonzero(coord_val)):
            ax.plot(coord_x, coord_train, ":", label=f"train_s{stage_id}_coord")
            ax.plot(coord_x, coord_val, ":", label=f"val_s{stage_id}_coord")

    ax.set_xlabel(f"stage {stage_id} epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Stage {stage_id} Loss (job {job_id})")
    ax.legend()


def main():
    """Plot stage-separated loss curves for a training run."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", default="training_results")
    p.add_argument("--job-id", required=True)
    p.add_argument("--csv-name", default="losses.csv")
    p.add_argument("--out-name", default="loss_plot.png")
    p.add_argument(
        "--plot-components",
        action="store_true",
        help="Overlay heatmap/coordinate component losses when available in losses.csv.",
    )
    args = p.parse_args()

    run_dir = os.path.join(args.checkpoint_root, str(args.job_id))
    csv_path = os.path.join(run_dir, args.csv_name)
    out_path = os.path.join(run_dir, args.out_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    rows = read_losses(csv_path)
    s1_total = stage_series(rows, stage_id=1, train_key="train_loss", val_key="val_loss")
    s2_total = stage_series(rows, stage_id=2, train_key="train_loss", val_key="val_loss")

    s1_hm = None
    s2_hm = None
    s1_coord = None
    s2_coord = None
    if args.plot_components:
        s1_hm = stage_series(rows, stage_id=1, train_key="train_loss_hm", val_key="val_loss_hm")
        s2_hm = stage_series(rows, stage_id=2, train_key="train_loss_hm", val_key="val_loss_hm")
        s1_coord = stage_series(rows, stage_id=1, train_key="train_loss_coord", val_key="val_loss_coord")
        s2_coord = stage_series(rows, stage_id=2, train_key="train_loss_coord", val_key="val_loss_coord")
        has_any_components = bool(s1_hm[0] or s2_hm[0] or s1_coord[0] or s2_coord[0])
        if not has_any_components:
            print("component columns not found in CSV; plotting total loss only")

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

    _plot_stage(
        ax=axes[0],
        stage_id=1,
        job_id=args.job_id,
        total_series=s1_total,
        hm_series=s1_hm,
        coord_series=s1_coord,
    )
    _plot_stage(
        ax=axes[1],
        stage_id=2,
        job_id=args.job_id,
        total_series=s2_total,
        hm_series=s2_hm,
        coord_series=s2_coord,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
