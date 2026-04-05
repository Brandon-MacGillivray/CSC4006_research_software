import argparse
import json
import time
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from handpose.inference.mediapipe_baseline import (
    MediaPipeHandLandmarkerRunner,
    load_rgb_image,
)


MEDIAPIPE_MODEL_KEYPOINT_INDICES = list(range(21))
TIP_BASE_KEYPOINT_INDICES = [1, 4, 5, 8, 9, 12, 13, 16, 17, 20]
COCO_HAND_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "validation": "val",
    "eval": "val",
    "evaluation": "val",
    "benchmark": "val",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the MediaPipe Hand Landmarker baseline on a COCO-style hand dataset."
    )
    parser.add_argument("--model-asset-path", required=True, help="Path to the MediaPipe .task model asset.")
    parser.add_argument("--root", default="data/hand_keypoint_dataset")
    parser.add_argument("--split", default="val", help="COCO-hand split alias, for example val/evaluation.")
    parser.add_argument(
        "--dataset-name",
        default="coco_hand_custom",
        help="Dataset label stored in the JSON payload for aggregation/reporting.",
    )
    parser.add_argument("--hand", default="auto", choices=["left", "right", "auto"])
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
    parser.add_argument("--job-id", default="mediapipe_coco_hand_eval")
    parser.add_argument("--experiment-id", default="MEDIAPIPE")
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def resolve_split_name(split: str):
    text = str(split).strip().lower()
    if text in COCO_HAND_SPLIT_ALIASES:
        return COCO_HAND_SPLIT_ALIASES[text]
    raise ValueError(f"Unsupported COCO-hand split: {split!r}")


def annotation_path(root, split: str):
    return Path(root) / "coco_annotation" / resolve_split_name(split) / "_annotations.coco.json"


def image_dir(root, split: str):
    return Path(root) / "images" / resolve_split_name(split)


def load_annotation_payload(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_samples(payload: dict, image_root, *, total_keypoints: int = 21):
    images_by_id = {}
    for image_meta in payload.get("images", []):
        if "id" not in image_meta:
            continue
        images_by_id[int(image_meta["id"])] = image_meta

    samples = []
    annotations = sorted(
        payload.get("annotations", []),
        key=lambda ann: (int(ann.get("image_id", -1)), int(ann.get("id", -1))),
    )
    required_values = int(total_keypoints) * 3
    for ann in annotations:
        if int(ann.get("iscrowd", 0)) != 0:
            continue

        image_id = int(ann.get("image_id", -1))
        image_meta = images_by_id.get(image_id)
        if image_meta is None:
            continue

        raw_keypoints = ann.get("keypoints")
        if not isinstance(raw_keypoints, list) or len(raw_keypoints) < required_values:
            continue

        keypoints = np.asarray(raw_keypoints[:required_values], dtype=np.float32).reshape(total_keypoints, 3)
        file_name = str(image_meta.get("file_name", "")).strip()
        if not file_name:
            continue

        samples.append(
            {
                "image_id": image_id,
                "annotation_id": int(ann.get("id", len(samples))),
                "image_path": image_root / file_name,
                "coords": keypoints[:, :2],
                "vis": keypoints[:, 2],
            }
        )
    return samples


def save_results_json(results: dict, out_json: str):
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


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


def resolve_root_keypoint_local_index(eval_keypoint_indices, model_keypoint_indices):
    tip_base_set = set(int(x) for x in TIP_BASE_KEYPOINT_INDICES)
    model_keypoint_set = set(int(x) for x in model_keypoint_indices)
    is_tip_base_10_model = (
        len(model_keypoint_indices) == len(TIP_BASE_KEYPOINT_INDICES)
        and model_keypoint_set == tip_base_set
    )

    if 0 in eval_keypoint_indices:
        return eval_keypoint_indices.index(0)
    if is_tip_base_10_model and 1 in eval_keypoint_indices:
        return eval_keypoint_indices.index(1)
    if set(int(x) for x in eval_keypoint_indices) == tip_base_set and 1 in eval_keypoint_indices:
        return eval_keypoint_indices.index(1)
    return None


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


def normalize_coords(coords_px, *, width: int, height: int):
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
            "name": str(args.dataset_name),
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
    payload = load_annotation_payload(anno_path)
    samples = build_samples(payload, rgb_dir, total_keypoints=21)

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
        for sample in samples:
            rgb_image = load_rgb_image(sample["image_path"])
            height, width = rgb_image.shape[:2]

            gt_coords = normalize_coords(sample["coords"], width=width, height=height)
            gt_vis = np.asarray(sample["vis"], dtype=np.float32)

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

            diff = pred_coords - gt_coords
            sq_per_joint = (diff * diff).sum(axis=-1)
            l2_per_joint = np.sqrt(sq_per_joint)
            visible_mask = (gt_vis > 0).astype(np.float32)

            total_samples += 1
            total_points += int(gt_coords.shape[0])
            total_visible_points += int(float(visible_mask.sum()))
            total_sse += float((sq_per_joint * visible_mask).sum())

            total_pck_hits += int(float((((l2_per_joint <= args.pck_threshold).astype(np.float32)) * visible_mask).sum()))
            total_pck_points += int(float(visible_mask.sum()))

            if root_keypoint_local_index is not None:
                root_gt = gt_coords[root_keypoint_local_index : root_keypoint_local_index + 1, :]
                root_pred = pred_coords[root_keypoint_local_index : root_keypoint_local_index + 1, :]
                rel_diff = (pred_coords - root_pred) - (gt_coords - root_gt)
                rel_l2 = np.sqrt((rel_diff * rel_diff).sum(axis=-1))
                total_epe += float((rel_l2 * visible_mask).sum())
                total_epe_points += int(float(visible_mask.sum()))

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
