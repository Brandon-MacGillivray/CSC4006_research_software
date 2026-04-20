import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import Subset

from bootstrap_src import bootstrap_src_path

bootstrap_src_path()

from handpose.data import SUPPORTED_DATASETS, build_dataset, resolve_dataset_name, resolve_dataset_split
from handpose.models.losses import WingLoss, HeatmapMSELoss
from handpose.models.hand_pose_model import HandPoseNet
from handpose.data.keypoints import TIP_BASE_KEYPOINT_INDICES
from handpose.checkpoints import CHECKPOINT_VERSION, load_checkpoint, save_training_checkpoint
from handpose.training.train_artifacts import append_csv_row, make_loaders
from handpose.training.train_optimization import build_optimizer_stage1, build_optimizer_stage2
from handpose.training.early_stopper import EarlyStopper
from handpose.training.train_steps import train_one_epoch, validate


EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.0


def infer_experiment_id(job_id: str):
    """Recover the experiment ID from the generated job name when available."""
    parts = str(job_id).split("-")
    if len(parts) >= 2 and parts[0] == "drh":
        return str(parts[1]).upper()
    if len(parts) >= 3 and parts[0] == "trf":
        tail = list(parts[1:])
        if tail and tail[-1].startswith("s") and tail[-1][1:].isdigit():
            tail = tail[:-1]
        if tail:
            return "_".join(str(part).upper() for part in tail)
    return ""


def resolve_experiment_family(experiment_family_arg: str, job_id: str):
    """Resolve experiment-family metadata with sensible defaults."""
    if str(experiment_family_arg or "").strip():
        return str(experiment_family_arg)
    text = str(job_id)
    if text.startswith("drh-"):
        return "ablation"
    if text.startswith("trf-"):
        return "transfer"
    return "custom"


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_effective_hand(dataset_name: str, hand: str):
    """Normalize hand-selection metadata across supported datasets."""
    if dataset_name == "rhd":
        return str(hand)
    return "single"


def infer_parent_job_id(init_ckpt: str, init_ckpt_meta: dict):
    """Recover the parent run identifier from checkpoint metadata or path."""
    if not init_ckpt:
        return ""
    training_config = {}
    if isinstance(init_ckpt_meta, dict):
        training_config = init_ckpt_meta.get("training_config", {})
    job_id = training_config.get("job_id", "")
    if str(job_id).strip():
        return str(job_id)
    norm_path = os.path.normpath(str(init_ckpt))
    ckpt_dir = os.path.dirname(norm_path)
    if os.path.basename(ckpt_dir) == "checkpoints":
        return os.path.basename(os.path.dirname(ckpt_dir))
    return os.path.splitext(os.path.basename(norm_path))[0]


def resolve_training_sequence(training_sequence_arg: str, dataset_name: str, init_ckpt_meta: dict):
    """Resolve the stored training-sequence lineage for the current run."""
    if str(training_sequence_arg or "").strip():
        return str(training_sequence_arg)
    training_config = {}
    if isinstance(init_ckpt_meta, dict):
        training_config = init_ckpt_meta.get("training_config", {})
    parent_sequence = (
        training_config.get("training_sequence")
        or training_config.get("dataset")
        or ""
    )
    if str(parent_sequence).strip() and str(parent_sequence) != str(dataset_name):
        return f"{parent_sequence}->{dataset_name}"
    return str(dataset_name)


def build_training_config(
    args,
    *,
    dataset_name: str,
    train_split: str,
    val_split: str,
    job_id: str,
    experiment_id: str,
    experiment_family: str,
    training_sequence: str,
    num_keypoints: int,
    keypoint_indices,
    init_ckpt_meta: dict,
    init_ckpt_path: str,
):
    """Capture experiment settings for checkpoints and downstream reports."""
    effective_hand = resolve_effective_hand(dataset_name, args.hand)
    parent_training_config = {}
    if isinstance(init_ckpt_meta, dict):
        parent_training_config = init_ckpt_meta.get("training_config", {})
    parent_job_id = infer_parent_job_id(init_ckpt_path, init_ckpt_meta)
    init_mode = "stage2_only" if args.skip_stage1 else ("from_checkpoint" if init_ckpt_path else "scratch")
    return {
        "job_id": str(job_id),
        "experiment_id": str(experiment_id),
        "experiment_family": str(experiment_family),
        "training_sequence": str(training_sequence),
        "dataset": dataset_name,
        "preprocess": "full_image_resize",
        "train_split": str(train_split),
        "val_split": str(val_split),
        "init_ckpt": str(init_ckpt_path or ""),
        "init_checkpoint_stage": init_ckpt_meta.get("stage", "") if isinstance(init_ckpt_meta, dict) else "",
        "init_checkpoint_epoch": init_ckpt_meta.get("epoch", "") if isinstance(init_ckpt_meta, dict) else "",
        "finetune_mode": init_mode,
        "parent_job_id": str(parent_job_id),
        "parent_experiment_id": str(parent_training_config.get("experiment_id", "")),
        "parent_dataset": str(parent_training_config.get("dataset", "")),
        "parent_training_sequence": str(parent_training_config.get("training_sequence", "")),
        "seed": args.seed,
        "hand": effective_hand,
        "input_size": args.input_size,
        "num_keypoints": num_keypoints,
        "keypoint_indices": list(keypoint_indices),
        "tips_bases_only": bool(args.tips_bases_only),
        "lambda_hm": float(args.lambda_hm),
        "lambda_coord": float(args.lambda_coord),
        "heatmap_sigma": float(args.heatmap_sigma),
        "wing_w": float(args.wing_w),
        "wing_epsilon": float(args.wing_epsilon),
        "lr_stage1": float(args.lr_stage1),
        "lr_stage2": float(args.lr_stage2),
        "batch_size_stage1": int(args.batch_size_stage1),
        "batch_size_stage2": int(args.batch_size_stage2),
        "accum_steps_stage1": int(args.accum_steps_stage1),
        "accum_steps_stage2": int(args.accum_steps_stage2),
        "stage1_epochs": int(args.stage1_epochs),
        "stage2_epochs": int(args.stage2_epochs),
        "stage1_patience": int(EARLY_STOPPING_PATIENCE),
        "stage1_min_delta": float(EARLY_STOPPING_MIN_DELTA),
        "stage2_patience": int(EARLY_STOPPING_PATIENCE),
        "stage2_min_delta": float(EARLY_STOPPING_MIN_DELTA),
    }


def main():
    """Train the model with the two-stage DRHand workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/RHD_published_v2")
    parser.add_argument(
        "--dataset",
        default="rhd",
        choices=SUPPORTED_DATASETS,
        help="Dataset loader to use for training and validation.",
    )
    parser.add_argument("--checkpoint-root", default="training_results")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--experiment-family", default=None)
    parser.add_argument("--training-sequence", default=None)
    parser.add_argument("--init-ckpt", default=None)
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip stage 1 and fine-tune stage 2 from --init-ckpt.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size-stage1", type=int, default=64)
    parser.add_argument("--batch-size-stage2", type=int, default=64)
    parser.add_argument("--accum-steps-stage1", type=int, default=1)
    parser.add_argument("--accum-steps-stage2", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=100)
    parser.add_argument("--lr-stage1", type=float, default=1e-3)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--wing-w", type=float, default=10.0)
    parser.add_argument("--wing-epsilon", type=float, default=2.0)

    args = parser.parse_args()
    if args.skip_stage1 and not args.init_ckpt:
        raise ValueError("--skip-stage1 requires --init-ckpt")
    if args.seed is not None:
        set_seed(int(args.seed))
    dataset_name = resolve_dataset_name(args.dataset)
    train_split = resolve_dataset_split(dataset_name, "train")
    val_split = resolve_dataset_split(dataset_name, "val")
    job_id = args.job_id if args.job_id is not None else "local"
    if args.tips_bases_only:
        keypoint_indices = list(TIP_BASE_KEYPOINT_INDICES)
    else:
        keypoint_indices = list(range(21))
    num_keypoints = len(keypoint_indices)
    experiment_id = str(args.experiment_id or infer_experiment_id(job_id))
    experiment_family = resolve_experiment_family(args.experiment_family, job_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_ckpt_meta = {}
    init_state_dict = None
    if args.init_ckpt:
        init_ckpt_meta, init_state_dict = load_checkpoint(args.init_ckpt, device)
        init_keypoint_indices = list(init_ckpt_meta.get("keypoint_indices", []))
        if list(init_keypoint_indices) != list(keypoint_indices):
            raise ValueError(
                "The requested keypoint layout does not match --init-ckpt. "
                f"requested={keypoint_indices} init_ckpt={init_keypoint_indices}"
            )
    training_sequence = resolve_training_sequence(args.training_sequence, dataset_name, init_ckpt_meta)
    training_config = build_training_config(
        args,
        dataset_name=dataset_name,
        train_split=train_split,
        val_split=val_split,
        job_id=job_id,
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        training_sequence=training_sequence,
        num_keypoints=num_keypoints,
        keypoint_indices=keypoint_indices,
        init_ckpt_meta=init_ckpt_meta,
        init_ckpt_path=args.init_ckpt,
    )
    effective_hand = str(training_config["hand"])

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Current device:", torch.cuda.current_device())
    print("Dataset:", dataset_name)
    print("Train split:", train_split)
    print("Val split:", val_split)
    print("Experiment family:", training_config["experiment_family"])
    print("Experiment ID:", training_config["experiment_id"])
    print("Training sequence:", training_config["training_sequence"])
    print("Seed:", args.seed)
    print("Selected keypoints:", keypoint_indices)
    if args.init_ckpt:
        print("Init checkpoint:", args.init_ckpt)
        print("Parent job ID:", training_config["parent_job_id"])

    run_dir = os.path.join(args.checkpoint_root, str(job_id))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "losses.csv")

    train_ds = build_dataset(
        dataset_name=dataset_name,
        root=args.root,
        split=train_split,
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
        keypoint_indices=keypoint_indices,
    )
    val_ds = build_dataset(
        dataset_name=dataset_name,
        root=args.root,
        split=val_split,
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
        keypoint_indices=keypoint_indices,
    )

    if int(args.train_dataset_length) > 0:
        N = min(int(args.train_dataset_length), len(train_ds))
        train_ds = Subset(train_ds, range(N))

    model = HandPoseNet(num_keypoints=num_keypoints).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
        print(
            "loaded init checkpoint "
            f"(stage={init_ckpt_meta.get('stage', '')}, epoch={init_ckpt_meta.get('epoch', '')})"
        )
    hm_loss_fn = HeatmapMSELoss(sigma=args.heatmap_sigma)
    coord_loss_fn = WingLoss(w=args.wing_w, epsilon=args.wing_epsilon)

    # STAGE 1: heatmap only
    if args.skip_stage1:
        print("skipping stage 1; stage 2 starts from the initialized checkpoint weights")
    else:
        train_loader, val_loader = make_loaders(
            train_ds,
            val_ds,
            batch_size=args.batch_size_stage1,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        optim = build_optimizer_stage1(model, lr=args.lr_stage1)

        early_stopper = EarlyStopper(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
        )
        best_val = float("inf")

        for epoch in range(args.stage1_epochs):
            t0 = time.time()

            train_metrics = train_one_epoch(
                model,
                train_loader,
                hm_loss_fn,
                optim,
                device,
                stage=1,
                accum_steps=args.accum_steps_stage1,
            )
            val_metrics = validate(model, val_loader, hm_loss_fn, device, stage=1)
            train_loss = float(train_metrics["loss_total"])
            val_loss = float(val_metrics["loss_total"])

            elapsed = time.time() - t0
            append_csv_row(
                csv_path,
                {
                    "stage": 1,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_loss_hm": float(train_metrics["loss_hm"]),
                    "train_loss_coord": float(train_metrics["loss_coord"]),
                    "val_loss": val_loss,
                    "val_loss_hm": float(val_metrics["loss_hm"]),
                    "val_loss_coord": float(val_metrics["loss_coord"]),
                    "seconds": float(elapsed),
                    "lr": float(optim.param_groups[0]["lr"]),
                    "batch_size": int(args.batch_size_stage1),
                    "accum_steps": int(args.accum_steps_stage1),
                    "num_train_steps": int(train_metrics["num_steps"]),
                    "num_val_steps": int(val_metrics["num_steps"]),
                    "seed": args.seed,
                    "dataset": dataset_name,
                    "experiment_family": training_config["experiment_family"],
                    "training_sequence": training_config["training_sequence"],
                    "parent_job_id": training_config["parent_job_id"],
                    "num_keypoints": num_keypoints,
                    "hand": effective_hand,
                    "input_size": int(args.input_size),
                    "tips_bases_only": int(bool(args.tips_bases_only)),
                    "lambda_hm": float(args.lambda_hm),
                    "lambda_coord": float(args.lambda_coord),
                    "heatmap_sigma": float(args.heatmap_sigma),
                    "wing_w": float(args.wing_w),
                    "wing_epsilon": float(args.wing_epsilon),
                },
            )
            print(
                f"[stage 1][epoch {epoch:03d}] "
                f"train(total={train_loss:.6f}, hm={train_metrics['loss_hm']:.6f}) "
                f"val(total={val_loss:.6f}, hm={val_metrics['loss_hm']:.6f}) "
                f"lr={optim.param_groups[0]['lr']:.2e} "
                f"time={elapsed:.1f}s"
            )

            ckpt = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "stage": 1,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "num_keypoints": num_keypoints,
                "keypoint_indices": keypoint_indices,
                "optim_state": optim.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_loss_hm": float(train_metrics["loss_hm"]),
                "train_loss_coord": float(train_metrics["loss_coord"]),
                "val_loss_hm": float(val_metrics["loss_hm"]),
                "val_loss_coord": float(val_metrics["loss_coord"]),
                "batch_size": args.batch_size_stage1,
                "accum_steps": args.accum_steps_stage1,
                "training_config": dict(training_config),
            }
            save_training_checkpoint(ckpt, os.path.join(ckpt_dir, f"stage1_epoch_{epoch}.pt"))

            if val_loss < best_val:
                best_val = val_loss
                save_training_checkpoint(ckpt, os.path.join(ckpt_dir, "stage1_best.pt"))

            if early_stopper.early_stop(val_loss):
                print(f"stage 1 early stop at epoch {epoch} (val={val_loss:.6f})")
                break

        best_path = os.path.join(ckpt_dir, "stage1_best.pt")
        if os.path.exists(best_path):
            best = torch.load(best_path, map_location=device)
            model.load_state_dict(best["model_state"])
            print("loaded stage1_best.pt for stage 2")

    # STAGE 2: coord branch starts 

    train_loader, val_loader = make_loaders(
        train_ds,
        val_ds,
        batch_size=args.batch_size_stage2,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    optim = build_optimizer_stage2(model, lr=args.lr_stage2, freeze_backbone=args.freeze_backbone_stage2, freeze_heatmap=args.freeze_heatmap_stage2)

    best_val2 = float("inf")
    early_stopper2 = EarlyStopper(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
    )

    for epoch in range(args.stage2_epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
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
        val_metrics = validate(
            model,
            val_loader,
            hm_loss_fn,
            device,
            stage=2,
            loss_coord_fn=coord_loss_fn,
            lambda_hm=args.lambda_hm,
            lambda_coord=args.lambda_coord,
        )
        train_loss = float(train_metrics["loss_total"])
        val_loss = float(val_metrics["loss_total"])

        elapsed = time.time() - t0
        append_csv_row(
            csv_path,
            {
                "stage": 2,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_loss_hm": float(train_metrics["loss_hm"]),
                "train_loss_coord": float(train_metrics["loss_coord"]),
                "val_loss": val_loss,
                "val_loss_hm": float(val_metrics["loss_hm"]),
                "val_loss_coord": float(val_metrics["loss_coord"]),
                "seconds": float(elapsed),
                "lr": float(optim.param_groups[0]["lr"]),
                "batch_size": int(args.batch_size_stage2),
                "accum_steps": int(args.accum_steps_stage2),
                "num_train_steps": int(train_metrics["num_steps"]),
                "num_val_steps": int(val_metrics["num_steps"]),
                "seed": args.seed,
                "dataset": dataset_name,
                "experiment_family": training_config["experiment_family"],
                "training_sequence": training_config["training_sequence"],
                "parent_job_id": training_config["parent_job_id"],
                "num_keypoints": num_keypoints,
                "hand": effective_hand,
                "input_size": int(args.input_size),
                "tips_bases_only": int(bool(args.tips_bases_only)),
                "lambda_hm": float(args.lambda_hm),
                "lambda_coord": float(args.lambda_coord),
                "heatmap_sigma": float(args.heatmap_sigma),
                "wing_w": float(args.wing_w),
                "wing_epsilon": float(args.wing_epsilon),
            },
        )
        print(
            f"[stage 2][epoch {epoch:03d}] "
            f"train(total={train_loss:.6f}, hm={train_metrics['loss_hm']:.6f}, coord={train_metrics['loss_coord']:.6f}) "
            f"val(total={val_loss:.6f}, hm={val_metrics['loss_hm']:.6f}, coord={val_metrics['loss_coord']:.6f}) "
            f"lr={optim.param_groups[0]['lr']:.2e} "
            f"time={elapsed:.1f}s"
        )

        ckpt = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "stage": 2,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "num_keypoints": num_keypoints,
            "keypoint_indices": keypoint_indices,
            "optim_state": optim.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_loss_hm": float(train_metrics["loss_hm"]),
            "train_loss_coord": float(train_metrics["loss_coord"]),
            "val_loss_hm": float(val_metrics["loss_hm"]),
            "val_loss_coord": float(val_metrics["loss_coord"]),
            "freeze_backbone_stage2": args.freeze_backbone_stage2,
            "freeze_heatmap_stage2": args.freeze_heatmap_stage2,
            "lambda_hm": args.lambda_hm,
            "lambda_coord": args.lambda_coord,
            "batch_size": args.batch_size_stage2,
            "accum_steps": args.accum_steps_stage2,
            "training_config": dict(training_config),
        }
        save_training_checkpoint(ckpt, os.path.join(ckpt_dir, f"stage2_epoch_{epoch}.pt"))

        if val_loss < best_val2:
            best_val2 = val_loss
            save_training_checkpoint(ckpt, os.path.join(ckpt_dir, "best.pt"))

        if early_stopper2.early_stop(val_loss):
            print(f"stage 2 early stop at epoch {epoch} (val={val_loss:.6f})")
            break


if __name__ == "__main__":
    main()
