import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import RHDDatasetCoords
from train_1 import HandPoseNet, TIP_BASE_KEYPOINT_INDICES


def build_arg_parser():
    """Create the CLI parser for the skeleton evaluation script."""
    p = argparse.ArgumentParser(description="Skeleton evaluation runner for a trained stage-2 model.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt), typically stage-2 best.pt")
    p.add_argument("--root", default="data/RHD_published_v2")
    p.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--shared-10-eval",
        action="store_true",
        help="Evaluate only the shared tip+base 10-keypoint subset for fair 21-vs-10 comparison.",
    )
    p.add_argument(
        "--debug-coords",
        action="store_true",
        help="Print GT/pred coordinate differences for the first batch (to stderr).",
    )
    p.add_argument(
        "--pck-threshold",
        type=float,
        default=0.2,
        help="Normalized distance threshold sigma used for PCK@sigma (default: 0.2).",
    )
    p.add_argument("--out-json", default=None, help="Optional path to write evaluation results as JSON")
    return p


def resolve_device(device_arg: str):
    """Resolve 'auto' / 'cpu' / 'cuda' into a torch.device with validation."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def load_checkpoint(path: str, device: torch.device):
    """Load a checkpoint and return (metadata_dict, model_state_dict)."""
    obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path!r}")

    if "model_state" in obj:
        return obj, obj["model_state"]

    # Allow raw state_dict as fallback.
    return {}, obj


def infer_num_keypoints_from_state(state_dict: dict):
    """Infer output keypoint count K from the heatmap head weight shape."""
    key = "heatmapHead.convM.net.0.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])

    for k, v in state_dict.items():
        if k.endswith("heatmapHead.convM.net.0.weight"):
            return int(v.shape[0])

    raise KeyError("Could not infer num_keypoints from checkpoint state_dict")


def infer_checkpoint_keypoint_indices(ckpt_meta: dict, state_dict: dict):
    """Recover the model's keypoint index list from checkpoint metadata or shape."""
    if "keypoint_indices" in ckpt_meta and ckpt_meta["keypoint_indices"] is not None:
        return [int(x) for x in ckpt_meta["keypoint_indices"]]

    k = infer_num_keypoints_from_state(state_dict)
    if k == 21:
        return list(range(21))
    if k == len(TIP_BASE_KEYPOINT_INDICES):
        return list(TIP_BASE_KEYPOINT_INDICES)

    raise ValueError(
        "Checkpoint does not contain keypoint_indices metadata and num_keypoints "
        f"={k} is not one of the supported defaults (21 or {len(TIP_BASE_KEYPOINT_INDICES)})."
    )


def resolve_eval_indices(model_keypoint_indices, shared_10_eval: bool):
    """Choose which keypoints to score and map them to model output positions."""
    model_keypoint_indices = [int(x) for x in model_keypoint_indices]

    if not shared_10_eval:
        # Evaluate all keypoints predicted by this checkpoint.
        return list(range(len(model_keypoint_indices))), list(model_keypoint_indices)

    # For fair 21-vs-10 comparisons, evaluate only the shared tip/base subset.
    eval_keypoints = list(TIP_BASE_KEYPOINT_INDICES)
    eval_positions = []
    missing = []
    for kp in eval_keypoints:
        try:
            eval_positions.append(model_keypoint_indices.index(kp))
        except ValueError:
            missing.append(kp)

    if missing:
        raise ValueError(
            "Checkpoint keypoint set does not contain all shared-10 keypoints. "
            f"Missing: {missing}. Checkpoint keypoints: {model_keypoint_indices}"
        )

    return eval_positions, eval_keypoints


def build_model(num_keypoints: int, state_dict: dict, device: torch.device):
    """Recreate the model with the right output size and load checkpoint weights."""
    model = HandPoseNet(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(state_dict)
    return model


def build_loader(args, device: torch.device, model_keypoint_indices):
    """Build the evaluation dataset/loader using the checkpoint keypoint ordering."""
    ds = RHDDatasetCoords(
        args.root,
        split=args.split,
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
        keypoint_indices=model_keypoint_indices,
        return_visibility=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return loader


@torch.no_grad()
def evaluate_checkpoint(
    model,
    loader,
    device,
    eval_positions,
    root_keypoint_local_index,
    pck_threshold: float,
    debug_coords: bool = False,
):
    """Run model inference over the loader and return counts, SSE/EPE/PCK, and timing."""
    model.eval()
    index_tensor = None
    if eval_positions is not None:
        # Prebuild the index tensor once so we can reuse it for every batch.
        index_tensor = torch.tensor(eval_positions, dtype=torch.long, device=device)

    total_samples = 0
    total_points = 0
    total_visible_points = 0
    total_sse = 0.0
    total_epe = 0.0
    total_epe_points = 0
    total_pck_hits = 0
    total_pck_points = 0

    forward_seconds = 0.0
    num_batches = 0
    debug_printed = False

    for imgs, coords, vis in loader:
        # Dataset returns normalized coordinates with the same keypoint order as training.
        imgs = imgs.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)

        # Synchronize CUDA so the timing reflects actual forward-pass duration.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _, pred_coords = model(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - t0
        num_batches += 1

        if index_tensor is not None:
            # Slice both GT and predictions to the evaluation subset (e.g., shared 10).
            coords = coords.index_select(1, index_tensor)
            pred_coords = pred_coords.index_select(1, index_tensor)
            vis = vis.index_select(1, index_tensor)

        # coords/pred_coords are normalized. SSE is computed only on visible joints.
        diff = pred_coords - coords
        sq_per_joint = (diff * diff).sum(dim=-1)   # (N, K_eval), squared 2D error per joint
        l2_per_joint = torch.sqrt(sq_per_joint)
        visible_mask = (vis > 0).float()
        sse_per_sample = (sq_per_joint * visible_mask).sum(dim=-1)   # (N,), sum over visible joints

        # PCK@sigma on visible joints.
        pck_hits = ((l2_per_joint <= pck_threshold).float() * visible_mask).sum()
        pck_points = visible_mask.sum()

        # EPE from Eq. (4): root-relative 2D error, normalized (coords are already normalized).
        if root_keypoint_local_index is not None:
            root_gt = coords[:, root_keypoint_local_index : root_keypoint_local_index + 1, :]
            root_pred = pred_coords[:, root_keypoint_local_index : root_keypoint_local_index + 1, :]
            rel_diff = (pred_coords - root_pred) - (coords - root_gt)
            rel_l2 = torch.sqrt((rel_diff * rel_diff).sum(dim=-1))  # (N, K_eval)
            total_epe += float((rel_l2 * visible_mask).sum().item())
            total_epe_points += int(visible_mask.sum().item())

        if debug_coords and not debug_printed:
            l2 = l2_per_joint
            n_show = min(5, int(coords.shape[1]))
            gt0 = coords[0, :n_show].detach().cpu()
            pred0 = pred_coords[0, :n_show].detach().cpu()
            diff0 = diff[0, :n_show].detach().cpu()
            l20 = l2[0, :n_show].detach().cpu()
            vis0 = vis[0, :n_show].detach().cpu()
            print("[debug-coords] showing first sample, first", n_show, "joints", file=sys.stderr)
            print("[debug-coords] gt[0,:n]   =", gt0, file=sys.stderr)
            print("[debug-coords] pred[0,:n] =", pred0, file=sys.stderr)
            print("[debug-coords] diff[0,:n] =", diff0, file=sys.stderr)
            print("[debug-coords] l2[0,:n]   =", l20, file=sys.stderr)
            print("[debug-coords] vis[0,:n]  =", vis0, file=sys.stderr)
            print(
                "[debug-coords] gt range / pred range =",
                (float(coords.min().item()), float(coords.max().item())),
                (float(pred_coords.min().item()), float(pred_coords.max().item())),
                file=sys.stderr,
            )
            print(
                "[debug-coords] mean l2 / max l2 =",
                float(l2.mean().item()),
                float(l2.max().item()),
                file=sys.stderr,
            )
            debug_printed = True

        batch_size = int(imgs.shape[0])
        total_samples += batch_size
        total_points += int(coords.shape[0] * coords.shape[1])
        total_visible_points += int(visible_mask.sum().item())
        total_sse += float(sse_per_sample.sum().item())
        total_pck_hits += int(pck_hits.item())
        total_pck_points += int(pck_points.item())

    if total_samples == 0 or total_points == 0:
        raise RuntimeError("No samples were evaluated")

    results = {
        "num_samples": total_samples,
        "num_points": total_points,
        "num_visible_points": total_visible_points,
        "num_eval_keypoints": total_points // total_samples,
        "sse_norm": (total_sse / total_samples),
        "epe_norm": (total_epe / total_epe_points) if total_epe_points > 0 else None,
        "epe_root_keypoint_index_in_eval": root_keypoint_local_index,
        "pck_threshold": float(pck_threshold),
        "pck": (float(total_pck_hits) / float(total_pck_points)) if total_pck_points > 0 else None,
        "timing": {
            "forward_seconds": forward_seconds,
            "num_batches": num_batches,
            "ms_per_image_forward_only": (forward_seconds * 1000.0 / total_samples),
            "images_per_second_forward_only": (total_samples / forward_seconds) if forward_seconds > 0 else None,
        },
    }
    return results


def build_results_payload(
    args,
    ckpt_meta: dict,
    device: torch.device,
    model_keypoint_indices,
    eval_keypoint_indices,
    metrics: dict,
):
    """Assemble the final JSON payload printed (and optionally saved) by the CLI."""
    return {
        "checkpoint": str(Path(args.ckpt)),
        "checkpoint_stage": ckpt_meta.get("stage"),
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "device": str(device),
        "dataset": {
            "root": args.root,
            "split": args.split,
            "hand": args.hand,
            "input_size": args.input_size,
        },
        "model_keypoint_indices": model_keypoint_indices,
        "eval_keypoint_indices": eval_keypoint_indices,
        "shared_10_eval": bool(args.shared_10_eval),
        "metrics": metrics,
    }


def save_results_json(results: dict, out_json: str):
    """Write the JSON payload to disk if an output path is provided."""
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


def main():
    """CLI entry point: load checkpoint/model, build loader, run skeleton evaluation."""
    args = build_arg_parser().parse_args()
    if args.pck_threshold < 0:
        raise ValueError("--pck-threshold must be >= 0")
    device = resolve_device(args.device)

    # Load checkpoint and recover which keypoints this model was trained to predict.
    ckpt_meta, state_dict = load_checkpoint(args.ckpt, device)
    model_keypoint_indices = infer_checkpoint_keypoint_indices(ckpt_meta, state_dict)
    num_keypoints = len(model_keypoint_indices)

    # Decide whether to evaluate all keypoints or the shared 10-keypoint subset.
    eval_positions, eval_keypoint_indices = resolve_eval_indices(
        model_keypoint_indices=model_keypoint_indices,
        shared_10_eval=args.shared_10_eval,
    )
    root_keypoint_local_index = eval_keypoint_indices.index(0) if 0 in eval_keypoint_indices else None
    if root_keypoint_local_index is None:
        print(
            "[eval] root keypoint 0 not present in eval set; epe_norm will be null.",
            file=sys.stderr,
        )

    model = build_model(num_keypoints=num_keypoints, state_dict=state_dict, device=device)
    loader = build_loader(args=args, device=device, model_keypoint_indices=model_keypoint_indices)

    # Run the evaluation pass (counts + SSE/EPE/PCK + timing).
    metrics = evaluate_checkpoint(
        model=model,
        loader=loader,
        device=device,
        eval_positions=eval_positions,
        root_keypoint_local_index=root_keypoint_local_index,
        pck_threshold=float(args.pck_threshold),
        debug_coords=bool(args.debug_coords),
    )

    results = build_results_payload(
        args=args,
        ckpt_meta=ckpt_meta,
        device=device,
        model_keypoint_indices=model_keypoint_indices,
        eval_keypoint_indices=eval_keypoint_indices,
        metrics=metrics,
    )

    print(json.dumps(results, indent=2))

    if args.out_json:
        save_results_json(results=results, out_json=args.out_json)


if __name__ == "__main__":
    main()
