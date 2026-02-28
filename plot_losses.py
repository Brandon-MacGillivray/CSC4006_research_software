import argparse
import os
import csv

import matplotlib.pyplot as plt


def read_losses(csv_path):
    stage, epochs, train_losses, val_losses = [], [], [], []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            stage.append(int(row["stage"]))
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
    return stage, epochs, train_losses, val_losses


def stage_series(stages, epochs, train_losses, val_losses, stage_id):
    x = [e for s, e in zip(stages, epochs) if s == stage_id]
    y_train = [y for s, y in zip(stages, train_losses) if s == stage_id]
    y_val = [y for s, y in zip(stages, val_losses) if s == stage_id]
    return x, y_train, y_val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", default="training_results")
    p.add_argument("--job-id", required=True)
    p.add_argument("--csv-name", default="losses.csv")
    p.add_argument("--out-name", default="loss_plot.png")
    args = p.parse_args()

    run_dir = os.path.join(args.checkpoint_root, str(args.job_id))
    csv_path = os.path.join(run_dir, args.csv_name)
    out_path = os.path.join(run_dir, args.out_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    stages, epochs, train_losses, val_losses = read_losses(csv_path)
    s1_x, s1_train, s1_val = stage_series(stages, epochs, train_losses, val_losses, stage_id=1)
    s2_x, s2_train, s2_val = stage_series(stages, epochs, train_losses, val_losses, stage_id=2)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

    ax1 = axes[0]
    if s1_x:
        ax1.plot(s1_x, s1_train, label="train_s1")
        ax1.plot(s1_x, s1_val, label="val_s1")
    else:
        ax1.text(0.5, 0.5, "No stage 1 data", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel("stage 1 epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(f"Stage 1 Loss (job {args.job_id})")
    ax1.legend()

    ax2 = axes[1]
    if s2_x:
        ax2.plot(s2_x, s2_train, label="train_s2")
        ax2.plot(s2_x, s2_val, label="val_s2")
    else:
        ax2.text(0.5, 0.5, "No stage 2 data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("stage 2 epoch")
    ax2.set_ylabel("loss")
    ax2.set_title(f"Stage 2 Loss (job {args.job_id})")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
