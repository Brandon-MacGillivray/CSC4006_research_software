import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import RHDDatasetCoords
from train_1 import HandPoseNet, TIP_BASE_KEYPOINT_INDICES

# 21-joint hand skeleton edges used to compute fusion alpha.
HAND_BONE_EDGES_21 = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def build_arg_parser():
    """CLI args."""
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
    p.add_argument(
        "--fusion-21-only",
        action="store_true",
        help=(
            "Use paper-faithful post-processing fusion for 21-keypoint evaluation only: "
            "decode heatmap coords, compute per-sample alpha as median predicted knuckle length, "
            "and fuse each joint by d_i < alpha."
        ),
    )
    p.add_argument("--out-json", default=None, help="Optional path to write evaluation results as JSON")
    return p


def resolve_device(device_arg: str):
    """Resolve requested device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def load_checkpoint(path: str, device: torch.device):
    """Load checkpoint and return metadata + state dict."""
    obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path!r}")

    if "model_state" in obj:
        return obj, obj["model_state"]

    # Fallback: checkpoint is a raw state_dict.
    return {}, obj


def infer_num_keypoints_from_state(state_dict: dict):
    """Infer output keypoint count from heatmap head."""
    key = "heatmapHead.convM.net.0.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])

    for k, v in state_dict.items():
        if k.endswith("heatmapHead.convM.net.0.weight"):
            return int(v.shape[0])

    raise KeyError("Could not infer num_keypoints from checkpoint state_dict")


def infer_checkpoint_keypoint_indices(ckpt_meta: dict, state_dict: dict):
    """Get checkpoint keypoint indices from metadata or inferred shape."""
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
    """Pick eval keypoints and map to model output positions."""
    model_keypoint_indices = [int(x) for x in model_keypoint_indices]

    if not shared_10_eval:
        # Default: evaluate all model keypoints.
        return list(range(len(model_keypoint_indices))), list(model_keypoint_indices)

    # Fair comparison mode: evaluate only the shared 10.
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
    """Build model and load checkpoint weights."""
    model = HandPoseNet(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(state_dict)
    return model


def build_loader(args, device: torch.device, model_keypoint_indices):
    """Build evaluation DataLoader with checkpoint keypoint order."""
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


def heatmaps_to_coords_argmax(pred_heatmaps: torch.Tensor):
    """Decode heatmaps to normalized (x, y) via argmax."""
    n, k, h, w = pred_heatmaps.shape
    flat = pred_heatmaps.view(n, k, -1)
    max_idx = torch.argmax(flat, dim=-1)
    y = (max_idx // w).float()
    x = (max_idx % w).float()
    if w > 1:
        x = x / float(w - 1)
    else:
        x = x * 0.0
    if h > 1:
        y = y / float(h - 1)
    else:
        y = y * 0.0
    return torch.stack([x, y], dim=-1)


def fuse_coords_paper(pred_heatmaps: torch.Tensor, pred_coords: torch.Tensor):
    """Fuse heatmap/coord branches using paper rule d_i < alpha."""
    hm_coords = heatmaps_to_coords_argmax(pred_heatmaps)

    # Alpha per sample = median predicted bone length.
    bone_lengths = []
    for a, b in HAND_BONE_EDGES_21:
        vec = pred_coords[:, a, :] - pred_coords[:, b, :]
        bone_lengths.append(torch.sqrt((vec * vec).sum(dim=-1)))
    bone_lengths = torch.stack(bone_lengths, dim=-1)
    alpha = torch.median(bone_lengths, dim=-1).values

    # Choose heatmap coordinate when branch distance is below alpha.
    diff_hm_coord = hm_coords - pred_coords
    d = torch.sqrt((diff_hm_coord * diff_hm_coord).sum(dim=-1))
    use_heatmap = d < alpha.unsqueeze(-1)
    fused = torch.where(use_heatmap.unsqueeze(-1), hm_coords, pred_coords)

    return fused, hm_coords, alpha, use_heatmap


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
    """Evaluate coordinate branch metrics."""
    model.eval()
    index_tensor = None
    if eval_positions is not None:
        # Prebuild index tensor for shared-10 slicing.
        index_tensor = torch.tensor(eval_positions, dtype=torch.long, device=device)

    # Running totals.
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
        # Move batch to device.
        imgs = imgs.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)

        # Time forward pass only.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _, pred_coords = model(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - t0
        num_batches += 1

        if index_tensor is not None:
            # Restrict to eval subset.
            coords = coords.index_select(1, index_tensor)
            pred_coords = pred_coords.index_select(1, index_tensor)
            vis = vis.index_select(1, index_tensor)

        # Visible-joint masked SSE/PCK.
        diff = pred_coords - coords
        sq_per_joint = (diff * diff).sum(dim=-1)
        l2_per_joint = torch.sqrt(sq_per_joint)
        visible_mask = (vis > 0).float()
        sse_per_sample = (sq_per_joint * visible_mask).sum(dim=-1)

        # PCK@sigma on visible joints.
        pck_hits = ((l2_per_joint <= pck_threshold).float() * visible_mask).sum()
        pck_points = visible_mask.sum()

        # Root-relative EPE (normalized coords).
        if root_keypoint_local_index is not None:
            root_gt = coords[:, root_keypoint_local_index : root_keypoint_local_index + 1, :]
            root_pred = pred_coords[:, root_keypoint_local_index : root_keypoint_local_index + 1, :]
            rel_diff = (pred_coords - root_pred) - (coords - root_gt)
            rel_l2 = torch.sqrt((rel_diff * rel_diff).sum(dim=-1))
            total_epe += float((rel_l2 * visible_mask).sum().item())
            total_epe_points += int(visible_mask.sum().item())

        if debug_coords and not debug_printed:
            # One-time sample dump for sanity-checking.
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

        # Update totals.
        batch_size = int(imgs.shape[0])
        total_samples += batch_size
        total_points += int(coords.shape[0] * coords.shape[1])
        total_visible_points += int(visible_mask.sum().item())
        total_sse += float(sse_per_sample.sum().item())
        total_pck_hits += int(pck_hits.item())
        total_pck_points += int(pck_points.item())

    if total_samples == 0 or total_points == 0:
        raise RuntimeError("No samples were evaluated")

    # Final normalized metrics.
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


@torch.no_grad()
def evaluate_checkpoint_paper_fusion(
    model,
    loader,
    device,
    pck_threshold: float,
    debug_coords: bool = False,
):
    """Evaluate full 21-joint paper-style fusion metrics."""
    model.eval()

    # Running totals (masked metrics).
    total_samples = 0
    total_points = 0
    masked_total_visible_points = 0
    masked_total_sse = 0.0
    masked_total_epe = 0.0
    masked_total_pck_hits = 0
    masked_total_pck_points = 0

    alpha_values = []
    heatmap_selected_per_joint = torch.zeros(21, dtype=torch.float64)
    total_heatmap_selected = 0

    forward_seconds = 0.0
    fusion_postprocess_seconds = 0.0
    num_batches = 0
    debug_printed = False

    for imgs, coords, vis in loader:
        # Move batch to device.
        imgs = imgs.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)

        # Time model forward.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        pred_heatmaps, pred_coords = model(imgs)  # full outputs from both branches
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - t0
        num_batches += 1

        # Time fusion post-processing.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tpp = time.perf_counter()
        fused_coords, hm_coords, alpha, use_heatmap = fuse_coords_paper(pred_heatmaps, pred_coords)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fusion_postprocess_seconds += time.perf_counter() - tpp

        # Visible-joint masked SSE/PCK.
        visible_mask = (vis > 0).float()
        diff = fused_coords - coords
        sq_per_joint = (diff * diff).sum(dim=-1)
        l2_per_joint = torch.sqrt(sq_per_joint)
        masked_total_sse += float((sq_per_joint * visible_mask).sum(dim=-1).sum().item())
        masked_total_visible_points += int(visible_mask.sum().item())

        # Root-relative EPE with same visibility mask.
        root_gt = coords[:, 0:1, :]
        root_pred = fused_coords[:, 0:1, :]
        rel_diff = (fused_coords - root_pred) - (coords - root_gt)
        rel_l2 = torch.sqrt((rel_diff * rel_diff).sum(dim=-1))
        masked_total_epe += float((rel_l2 * visible_mask).sum().item())
        masked_total_pck_hits += int(((l2_per_joint <= pck_threshold).float() * visible_mask).sum().item())
        masked_total_pck_points += int(visible_mask.sum().item())

        alpha_values.append(alpha.detach().cpu().to(torch.float64))
        heatmap_selected_per_joint += use_heatmap.sum(dim=0).detach().cpu().to(torch.float64)
        total_heatmap_selected += int(use_heatmap.sum().item())

        if debug_coords and not debug_printed:
            # One-time fusion debug dump.
            n_show = min(5, int(coords.shape[1]))
            print("[debug-fusion] showing first sample, first", n_show, "joints", file=sys.stderr)
            print("[debug-fusion] alpha[0]      =", float(alpha[0].item()), file=sys.stderr)
            print("[debug-fusion] gt[0,:n]      =", coords[0, :n_show].detach().cpu(), file=sys.stderr)
            print("[debug-fusion] coord[0,:n]   =", pred_coords[0, :n_show].detach().cpu(), file=sys.stderr)
            print("[debug-fusion] heatmap[0,:n] =", hm_coords[0, :n_show].detach().cpu(), file=sys.stderr)
            print("[debug-fusion] fused[0,:n]   =", fused_coords[0, :n_show].detach().cpu(), file=sys.stderr)
            print("[debug-fusion] use_hm[0,:n]  =", use_heatmap[0, :n_show].detach().cpu(), file=sys.stderr)
            print("[debug-fusion] l2[0,:n]      =", l2_per_joint[0, :n_show].detach().cpu(), file=sys.stderr)
            debug_printed = True

        # Update totals.
        batch_size = int(imgs.shape[0])
        total_samples += batch_size
        total_points += int(coords.shape[0] * coords.shape[1])

    num_samples = total_samples
    num_points = total_points

    if num_samples <= 0 or num_points <= 0:
        raise RuntimeError("No samples were evaluated")

    # Summarize alpha across all samples.
    alpha_all = torch.cat(alpha_values, dim=0) if alpha_values else torch.empty(0, dtype=torch.float64)
    if alpha_all.numel() > 0:
        alpha_p10 = float(torch.quantile(alpha_all, 0.10).item())
        alpha_p90 = float(torch.quantile(alpha_all, 0.90).item())
        alpha_mean = float(alpha_all.mean().item())
        alpha_median = float(alpha_all.median().item())
        alpha_non_positive_count = int((alpha_all <= 0).sum().item())
    else:
        alpha_p10 = None
        alpha_p90 = None
        alpha_mean = None
        alpha_median = None
        alpha_non_positive_count = 0

    # Final fused metrics.
    results = {
        "num_samples": num_samples,
        "num_points": num_points,
        "num_visible_points": masked_total_visible_points,
        "num_eval_keypoints": (num_points // num_samples),
        "sse_norm": (masked_total_sse / num_samples),
        "epe_norm": (masked_total_epe / masked_total_visible_points) if masked_total_visible_points > 0 else None,
        "epe_root_keypoint_index_in_eval": 0,
        "pck_threshold": float(pck_threshold),
        "pck": (float(masked_total_pck_hits) / float(masked_total_pck_points)) if masked_total_pck_points > 0 else None,
        "fusion": {
            "mode": "fusion_21_only",
            "alpha_stats": {
                "mean": alpha_mean,
                "median": alpha_median,
                "p10": alpha_p10,
                "p90": alpha_p90,
            },
            "alpha_non_positive_count": alpha_non_positive_count,
            "heatmap_selected_ratio": (float(total_heatmap_selected) / float(num_points)),
            "heatmap_selected_ratio_per_joint": (heatmap_selected_per_joint / float(num_samples)).tolist(),
        },
        "timing": {
            "forward_seconds": forward_seconds,
            "fusion_postprocess_seconds": fusion_postprocess_seconds,
            "num_batches": num_batches,
            "ms_per_image_forward_only": (forward_seconds * 1000.0 / num_samples),
            "images_per_second_forward_only": (num_samples / forward_seconds) if forward_seconds > 0 else None,
            "ms_per_image_fusion_postprocess_only": (fusion_postprocess_seconds * 1000.0 / num_samples),
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
    """Build final JSON payload."""
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
        "fusion_21_only": bool(args.fusion_21_only),
        "metrics": metrics,
    }


def save_results_json(results: dict, out_json: str):
    """Save JSON file."""
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


def main():
    """CLI entry point."""
    # Parse and validate args.
    args = build_arg_parser().parse_args()
    if args.pck_threshold < 0:
        raise ValueError("--pck-threshold must be >= 0")
    device = resolve_device(args.device)

    # Load checkpoint and recover keypoint layout.
    ckpt_meta, state_dict = load_checkpoint(args.ckpt, device)
    model_keypoint_indices = infer_checkpoint_keypoint_indices(ckpt_meta, state_dict)
    num_keypoints = len(model_keypoint_indices)

    # Resolve eval subset (full or shared-10).
    eval_positions, eval_keypoint_indices = resolve_eval_indices(
        model_keypoint_indices=model_keypoint_indices,
        shared_10_eval=args.shared_10_eval,
    )
    root_keypoint_local_index = eval_keypoint_indices.index(0) if 0 in eval_keypoint_indices else None
    if root_keypoint_local_index is None and not args.fusion_21_only:
        print(
            "[eval] root keypoint 0 not present in eval set; epe_norm will be null.",
            file=sys.stderr,
        )

    if args.fusion_21_only:
        # Fusion mode only supports full 21-joint setup.
        if args.shared_10_eval:
            raise ValueError("--fusion-21-only requires full 21-joint evaluation (do not use --shared-10-eval)")
        if eval_keypoint_indices != list(range(21)):
            raise ValueError(
                "--fusion-21-only requires checkpoint/eval keypoints to be exactly [0..20] "
                f"(got: {eval_keypoint_indices})"
            )

    model = build_model(num_keypoints=num_keypoints, state_dict=state_dict, device=device)
    loader = build_loader(args=args, device=device, model_keypoint_indices=model_keypoint_indices)

    # Run selected evaluation path.
    if args.fusion_21_only:
        metrics = evaluate_checkpoint_paper_fusion(
            model=model,
            loader=loader,
            device=device,
            pck_threshold=float(args.pck_threshold),
            debug_coords=bool(args.debug_coords),
        )
    else:
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
        # Optional JSON output file.
        save_results_json(results=results, out_json=args.out_json)


if __name__ == "__main__":
    main()
