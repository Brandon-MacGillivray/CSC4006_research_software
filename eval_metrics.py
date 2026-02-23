import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import RHDDatasetCoords
from train_1 import HandPoseNet, TIP_BASE_KEYPOINT_INDICES


def parse_thresholds(spec: str):
    parts = [p.strip() for p in spec.replace(";", ",").split(",")]
    vals = [float(p) for p in parts if p]
    if not vals:
        raise ValueError("At least one PCK threshold is required")
    if any(v < 0 for v in vals):
        raise ValueError("PCK thresholds must be >= 0")
    return vals


def load_checkpoint(path: str, device: torch.device):
    obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path!r}")

    if "model_state" in obj:
        return obj, obj["model_state"]

    # Allow raw state_dict as fallback.
    return {}, obj


def infer_num_keypoints_from_state(state_dict: dict):
    key = "heatmapHead.convM.net.0.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])

    for k, v in state_dict.items():
        if k.endswith("heatmapHead.convM.net.0.weight"):
            return int(v.shape[0])

    raise KeyError("Could not infer num_keypoints from checkpoint state_dict")


def infer_checkpoint_keypoint_indices(ckpt_meta: dict, state_dict: dict):
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
    model_keypoint_indices = [int(x) for x in model_keypoint_indices]

    if not shared_10_eval:
        return list(range(len(model_keypoint_indices))), list(model_keypoint_indices)

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


@torch.no_grad()
def evaluate_checkpoint(
    model,
    loader,
    device,
    eval_positions,
    thresholds,
    input_size: int,
):
    model.eval()
    index_tensor = None
    if eval_positions is not None:
        index_tensor = torch.tensor(eval_positions, dtype=torch.long, device=device)

    total_samples = 0
    total_points = 0
    total_dist_sq = 0.0  # sum over all samples and evaluated joints (normalized coords)
    total_dist = 0.0     # sum L2 over all samples and evaluated joints (normalized coords)
    pck_counts = [0 for _ in thresholds]

    forward_seconds = 0.0
    num_batches = 0

    for imgs, coords in loader:
        imgs = imgs.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _, pred_coords = model(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - t0
        num_batches += 1

        if index_tensor is not None:
            coords = coords.index_select(1, index_tensor)
            pred_coords = pred_coords.index_select(1, index_tensor)

        diff = pred_coords - coords
        dist_sq = (diff * diff).sum(dim=-1)  # (N, K_eval)
        dist = torch.sqrt(dist_sq)

        batch_size = int(imgs.shape[0])
        total_samples += batch_size
        total_points += int(dist.numel())
        total_dist_sq += float(dist_sq.sum().item())
        total_dist += float(dist.sum().item())

        for i, thr in enumerate(thresholds):
            pck_counts[i] += int((dist <= thr).sum().item())

    if total_samples == 0 or total_points == 0:
        raise RuntimeError("No samples were evaluated")

    epe_norm = total_dist / total_points
    sse_norm = total_dist_sq / total_samples
    pck = {str(thr): (count / total_points) for thr, count in zip(thresholds, pck_counts)}

    results = {
        "num_samples": total_samples,
        "num_points": total_points,
        "num_eval_keypoints": total_points // total_samples,
        "epe_norm": epe_norm,
        "epe_px": epe_norm * float(input_size),
        "sse_norm": sse_norm,
        "pck": pck,
        "timing": {
            "forward_seconds": forward_seconds,
            "num_batches": num_batches,
            "ms_per_image_forward_only": (forward_seconds * 1000.0 / total_samples),
            "images_per_second_forward_only": (total_samples / forward_seconds) if forward_seconds > 0 else None,
        },
    }
    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained stage-2 model with coordinate metrics.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt), typically stage-2 best.pt")
    p.add_argument("--root", default="data/RHD_published_v2")
    p.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--thresholds",
        default="0.02,0.05,0.10,0.20",
        help="Comma-separated PCK thresholds on normalized coordinates (default: 0.02,0.05,0.10,0.20)",
    )
    p.add_argument(
        "--shared-10-eval",
        action="store_true",
        help="Evaluate only the shared tip+base 10-keypoint subset for fair 21-vs-10 comparison.",
    )
    p.add_argument("--out-json", default=None, help="Optional path to write evaluation results as JSON")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

    thresholds = parse_thresholds(args.thresholds)

    ckpt_meta, state_dict = load_checkpoint(args.ckpt, device)
    model_keypoint_indices = infer_checkpoint_keypoint_indices(ckpt_meta, state_dict)
    num_keypoints = len(model_keypoint_indices)

    eval_positions, eval_keypoint_indices = resolve_eval_indices(
        model_keypoint_indices=model_keypoint_indices,
        shared_10_eval=args.shared_10_eval,
    )

    model = HandPoseNet(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(state_dict)

    ds = RHDDatasetCoords(
        args.root,
        split=args.split,
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
        keypoint_indices=model_keypoint_indices,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    metrics = evaluate_checkpoint(
        model=model,
        loader=loader,
        device=device,
        eval_positions=eval_positions,
        thresholds=thresholds,
        input_size=args.input_size,
    )

    results = {
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

    # Convenience aliases for the paper's common threshold.
    if "0.2" in results["metrics"]["pck"]:
        results["metrics"]["pck_at_0_2"] = results["metrics"]["pck"]["0.2"]
    elif "0.20" in results["metrics"]["pck"]:
        results["metrics"]["pck_at_0_2"] = results["metrics"]["pck"]["0.20"]

    print(json.dumps(results, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
