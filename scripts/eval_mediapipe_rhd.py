"""Evaluate the MediaPipe baseline on the RHD dataset.

This script runs the bundled MediaPipe hand-landmarker asset against RHD and
exports repository-style JSON payloads for comparison and aggregation.
"""

import argparse
import json
import time

import numpy as np
import torch

from bootstrap_src import bootstrap_src_path

bootstrap_src_path()

from handpose.data.keypoints import TIP_BASE_KEYPOINT_INDICES
from handpose.data.rhd.parsing import find_uv_key, load_annotations, select_hand
from handpose.data.rhd.paths import annotation_path, image_dir, resolve_split_name
from handpose.evaluation.eval_outputs import save_results_json
from handpose.evaluation.eval_pipeline import resolve_eval_indices, resolve_root_keypoint_local_index
from handpose.inference.mediapipe_baseline import (
    MediaPipeHandLandmarkerRunner,
    load_rgb_image,
)


MEDIAPIPE_MODEL_KEYPOINT_INDICES = list(range(21))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the MediaPipe Hand Landmarker baseline on the RHD dataset."
    )
    parser.add_argument("--model-asset-path", required=True, help="Path to the MediaPipe .task model asset.")
    parser.add_argument("--root", default="data/RHD_published_v2")
    parser.add_argument("--split", default="evaluation", help="RHD split alias, for example evaluation/val.")
    parser.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    parser.add_argument(
        "--ignore-handedness",
        action="store_true",
        help="Ignore MediaPipe handedness labels and always use the highest-confidence detection.",
    )
    parser.add_argument("--shared-10-eval", action="store_true")
    parser.add_argument("--pck-threshold", type=float, default=0.2)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--job-id", default="mediapipe_rhd_eval")
    parser.add_argument("--experiment-id", default="MEDIAPIPE")
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def choose_detection(detections, *, requested_hand: str, ignore_handedness: bool):
    """Select one MediaPipe detection for evaluation."""
    if not detections:
        return None, "no_detections"

    def det_score(det):
        score = det.get("handedness_score")
        return float(score) if score is not None else 0.0

    if requested_hand == "auto" or ignore_handedness:
        return max(detections, key=det_score), None

    requested = str(requested_hand).strip().lower()
    matching = [
        det for det in detections if str(det.get("handedness", "")).strip().lower() == requested
    ]
    if matching:
        return max(matching, key=det_score), None
    return None, "no_matching_handedness"


def normalize_rhd_coords(coords_px, *, width: int, height: int):
    """Convert pixel coordinates to [0, 1] image coordinates."""
    coords = np.asarray(coords_px, dtype=np.float32).copy()
    coords[:, 0] /= float(width)
    coords[:, 1] /= float(height)
    return coords


def build_results_payload(
    *,
    args,
    metrics: dict,
    eval_keypoint_indices,
):
    """Build a JSON payload aligned with this repo's evaluation outputs."""
    training_config = {
        "job_id": str(args.job_id),
        "experiment_id": str(args.experiment_id),
        "experiment_family": "baseline",
        "training_sequence": "mediapipe",
        "dataset": "mediapipe",
        "preprocess": "mediapipe_hand_landmarker",
        "seed": "",
        "hand": "single",
        "input_size": "",
        "num_keypoints": len(MEDIAPIPE_MODEL_KEYPOINT_INDICES),
        "keypoint_indices": list(MEDIAPIPE_MODEL_KEYPOINT_INDICES),
        "tips_bases_only": False,
    }
    return {
        "checkpoint": str(args.model_asset_path),
        "checkpoint_stage": None,
        "checkpoint_epoch": None,
        "job_id": str(args.job_id),
        "experiment_id": str(args.experiment_id),
        "prediction_mode": "mediapipe",
        "with_fusion_diagnostics": False,
        "device": "cpu",
        "dataset": {
            "name": "rhd",
            "root": str(args.root),
            "split": str(args.split),
            "hand": str(args.hand),
            "input_size": None,
        },
        "training_config": training_config,
        "model_keypoint_indices": list(MEDIAPIPE_MODEL_KEYPOINT_INDICES),
        "eval_keypoint_indices": list(eval_keypoint_indices),
        "shared_10_eval": bool(args.shared_10_eval),
        "metrics": dict(metrics),
        "baseline": {
            "name": "mediapipe_hand_landmarker",
            "ignore_handedness": bool(args.ignore_handedness),
            "num_hands": int(args.num_hands),
            "min_hand_detection_confidence": float(args.min_hand_detection_confidence),
            "min_hand_presence_confidence": float(args.min_hand_presence_confidence),
            "min_tracking_confidence": float(args.min_tracking_confidence),
        },
    }


def main():
    args = parse_args()
    if args.pck_threshold < 0:
        raise ValueError("--pck-threshold must be >= 0")

    args.split = resolve_split_name(args.split)
    eval_positions, eval_keypoint_indices = resolve_eval_indices(
        model_keypoint_indices=MEDIAPIPE_MODEL_KEYPOINT_INDICES,
        shared_10_eval=bool(args.shared_10_eval),
    )
    root_keypoint_local_index = resolve_root_keypoint_local_index(
        eval_keypoint_indices=eval_keypoint_indices,
        model_keypoint_indices=MEDIAPIPE_MODEL_KEYPOINT_INDICES,
    )

    anno_path = annotation_path(args.root, args.split)
    rgb_dir = image_dir(args.root, args.split)
    annotations = load_annotations(anno_path)
    sample_ids = sorted(annotations.keys())

    total_samples = 0
    total_points = 0
    total_visible_points = 0
    total_sse = 0.0
    total_epe = 0.0
    total_epe_points = 0
    total_pck_hits = 0
    total_pck_points = 0
    prediction_seconds = 0.0
    detection_failures = 0
    handedness_misses = 0

    with MediaPipeHandLandmarkerRunner(
        model_asset_path=args.model_asset_path,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as baseline:
        for sample_id in sample_ids:
            anno = annotations[sample_id]
            img_path = rgb_dir / f"{sample_id:05d}.png"
            rgb_image = load_rgb_image(img_path)
            height, width = rgb_image.shape[:2]

            uv_data = anno[find_uv_key(anno)]
            gt_hand = select_hand(uv_data, args.hand)
            gt_coords = normalize_rhd_coords(gt_hand[:, :2], width=width, height=height)
            gt_vis = np.asarray(gt_hand[:, 2], dtype=np.float32)

            t0 = time.perf_counter()
            detections = baseline.detect_image(rgb_image)
            prediction_seconds += time.perf_counter() - t0

            selected, miss_reason = choose_detection(
                detections,
                requested_hand=args.hand,
                ignore_handedness=bool(args.ignore_handedness),
            )

            pred_coords = np.zeros((len(MEDIAPIPE_MODEL_KEYPOINT_INDICES), 2), dtype=np.float32)
            if selected is None:
                detection_failures += 1
                if miss_reason == "no_matching_handedness":
                    handedness_misses += 1
            else:
                pred_coords = np.asarray(selected["coords"], dtype=np.float32)

            gt_coords = gt_coords[eval_positions]
            pred_coords = pred_coords[eval_positions]
            gt_vis = gt_vis[eval_positions]

            gt_coords_t = torch.tensor(gt_coords, dtype=torch.float32)
            pred_coords_t = torch.tensor(pred_coords, dtype=torch.float32)
            vis_t = torch.tensor(gt_vis, dtype=torch.float32)

            diff = pred_coords_t - gt_coords_t
            sq_per_joint = (diff * diff).sum(dim=-1)
            l2_per_joint = torch.sqrt(sq_per_joint)
            visible_mask = (vis_t > 0).float()

            total_samples += 1
            total_points += int(gt_coords_t.shape[0])
            total_visible_points += int(visible_mask.sum().item())
            total_sse += float((sq_per_joint * visible_mask).sum().item())

            total_pck_hits += int((((l2_per_joint <= args.pck_threshold).float()) * visible_mask).sum().item())
            total_pck_points += int(visible_mask.sum().item())

            if root_keypoint_local_index is not None:
                root_gt = gt_coords_t[root_keypoint_local_index : root_keypoint_local_index + 1, :]
                root_pred = pred_coords_t[root_keypoint_local_index : root_keypoint_local_index + 1, :]
                rel_diff = (pred_coords_t - root_pred) - (gt_coords_t - root_gt)
                rel_l2 = torch.sqrt((rel_diff * rel_diff).sum(dim=-1))
                total_epe += float((rel_l2 * visible_mask).sum().item())
                total_epe_points += int(visible_mask.sum().item())

    if total_samples == 0 or total_points == 0:
        raise RuntimeError("No samples were evaluated")

    metrics = {
        "num_samples": total_samples,
        "num_points": total_points,
        "num_visible_points": total_visible_points,
        "num_eval_keypoints": total_points // total_samples,
        "sse_norm": (total_sse / total_samples),
        "epe_norm": (total_epe / total_epe_points) if total_epe_points > 0 else None,
        "epe_root_keypoint_index_in_eval": root_keypoint_local_index,
        "pck_threshold": float(args.pck_threshold),
        "pck": (float(total_pck_hits) / float(total_pck_points)) if total_pck_points > 0 else None,
        "timing": {
            "prediction_seconds": prediction_seconds,
            "num_batches": total_samples,
            "ms_per_image": (prediction_seconds * 1000.0 / total_samples),
            "images_per_second": (total_samples / prediction_seconds) if prediction_seconds > 0 else None,
        },
        "num_detection_failures": int(detection_failures),
        "num_handedness_misses": int(handedness_misses),
        "detection_success_rate": (
            float(total_samples - detection_failures) / float(total_samples)
        ) if total_samples > 0 else None,
    }

    results = build_results_payload(
        args=args,
        metrics=metrics,
        eval_keypoint_indices=eval_keypoint_indices,
    )
    print(json.dumps(results, indent=2))
    if args.out_json:
        save_results_json(results=results, out_json=args.out_json)


if __name__ == "__main__":
    main()
