import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from handpose.checkpoints import get_training_config, infer_checkpoint_keypoint_indices, load_checkpoint
from handpose.data.coco_hand.parsing import build_samples, load_annotation_payload
from handpose.data.coco_hand.paths import annotation_path, image_dir, resolve_split_name
from handpose.data.keypoints import TIP_BASE_KEYPOINT_INDICES
from handpose.data.transforms import load_rgb_image, preprocess_full_image_from_pil
from handpose.evaluation.eval_pipeline import resolve_device, resolve_eval_indices
from handpose.inference.fusion import HAND_BONE_EDGES_10_GLOBAL, HAND_BONE_EDGES_21
from handpose.inference.mediapipe_baseline import MediaPipeHandLandmarkerRunner
from handpose.inference.predict import SUPPORTED_PREDICTION_MODES, build_fusion_context, infer_coords
from handpose.models.hand_pose_model import HandPoseNet


TIP_BASE_SET = set(int(x) for x in TIP_BASE_KEYPOINT_INDICES)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a qualitative figure showing ground truth and one or more prediction overlays "
            "for a COCO-style hand dataset image."
        )
    )
    parser.add_argument("--root", required=True, help="COCO-style dataset root, for example docs/hand_frame_small")
    parser.add_argument("--split", default="val", help="COCO-hand split alias, for example val/evaluation")
    parser.add_argument(
        "--image",
        default="",
        help="Exact image file name to render. If omitted, --index is used.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="1-based sample index used when --image is omitted.",
    )
    parser.add_argument("--out", required=True, help="Output figure path, for example qual_figures/img01_compare.png")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Repeated model spec in the form Label=/path/to/checkpoint.pt",
    )
    parser.add_argument(
        "--prediction-mode",
        default="fusion",
        choices=SUPPORTED_PREDICTION_MODES,
        help="Prediction mode used for all checkpoint panels.",
    )
    parser.add_argument(
        "--shared-10-eval",
        action="store_true",
        help="Render all methods on the shared 10-joint subset for fair 21-vs-10 comparison.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the output figure grid.")
    parser.add_argument(
        "--include-mediapipe",
        action="store_true",
        help="Add a MediaPipe panel using the hand landmarker baseline.",
    )
    parser.add_argument("--mediapipe-model-asset-path", default=None, help="Path to the MediaPipe .task model asset.")
    parser.add_argument("--mediapipe-hand", default="auto", choices=["left", "right", "auto"])
    parser.add_argument("--ignore-handedness", action="store_true")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def parse_model_spec(text: str):
    if "=" not in str(text):
        raise ValueError(f"--model must use Label=/path/to/checkpoint.pt, got {text!r}")
    label, path = str(text).split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"--model must use Label=/path/to/checkpoint.pt, got {text!r}")
    return {"label": label, "ckpt": path}


def choose_detection(detections, *, requested_hand: str, ignore_handedness: bool):
    if not detections:
        return None

    def det_score(det):
        score = det.get("handedness_score")
        return float(score) if score is not None else 0.0

    if requested_hand == "auto" or ignore_handedness:
        return max(detections, key=det_score)

    requested = str(requested_hand).strip().lower()
    matching = [
        det for det in detections if str(det.get("handedness", "")).strip().lower() == requested
    ]
    if matching:
        return max(matching, key=det_score)
    return None


def load_sample(*, root: str, split: str, image_name: str, index: int):
    resolved_split = resolve_split_name(split)
    anno = load_annotation_payload(annotation_path(root, resolved_split))
    samples = build_samples(anno, image_dir(root, resolved_split), total_keypoints=21)
    if not samples:
        raise ValueError(f"No samples found under {root!r} split={resolved_split!r}")

    if image_name:
        for sample in samples:
            if sample["image_path"].name == image_name:
                return sample
        raise ValueError(f"Image {image_name!r} not found in {root!r} split={resolved_split!r}")

    idx = int(index) - 1
    if idx < 0 or idx >= len(samples):
        raise ValueError(f"--index must be in [1, {len(samples)}], got {index}")
    return samples[idx]


def resolve_edges_for_keypoints(global_keypoint_indices):
    keys = [int(x) for x in global_keypoint_indices]
    key_set = set(keys)
    index_of = {kp: i for i, kp in enumerate(keys)}

    if key_set == set(range(21)):
        return [(index_of[a], index_of[b]) for a, b in HAND_BONE_EDGES_21]
    if key_set == TIP_BASE_SET:
        return [(index_of[a], index_of[b]) for a, b in HAND_BONE_EDGES_10_GLOBAL]
    return [(index_of[a], index_of[b]) for a, b in HAND_BONE_EDGES_21 if a in key_set and b in key_set]


def draw_hand(ax, coords_px, *, vis, global_keypoint_indices, point_color, edge_color, alpha=1.0):
    coords = np.asarray(coords_px, dtype=np.float32)
    vis = np.asarray(vis, dtype=np.float32)
    edges = resolve_edges_for_keypoints(global_keypoint_indices)

    for a, b in edges:
        if vis[a] > 0 and vis[b] > 0:
            ax.plot(
                [coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                color=edge_color,
                linewidth=1.8,
                alpha=alpha,
            )
    visible = vis > 0
    if np.any(visible):
        ax.scatter(
            coords[visible, 0],
            coords[visible, 1],
            s=22,
            c=point_color,
            edgecolors="black",
            linewidths=0.5,
            alpha=alpha,
        )


def load_checkpoint_panel(
    *,
    sample,
    label: str,
    ckpt_path: str,
    device: torch.device,
    prediction_mode: str,
    shared_10_eval: bool,
):
    ckpt_meta, state_dict = load_checkpoint(ckpt_path, device)
    training_config = get_training_config(ckpt_meta)
    model_keypoint_indices = infer_checkpoint_keypoint_indices(ckpt_meta)
    input_size = int(training_config.get("input_size", 256) or 256)
    eval_positions, eval_keypoint_indices = resolve_eval_indices(
        model_keypoint_indices=model_keypoint_indices,
        shared_10_eval=bool(shared_10_eval),
    )

    image = load_rgb_image(sample["image_path"])
    x, _, orig_w, orig_h = preprocess_full_image_from_pil(
        image=image,
        coords_px=sample["coords"],
        input_size=input_size,
        normalize=True,
    )
    x = x.unsqueeze(0).to(device)

    model = HandPoseNet(num_keypoints=len(model_keypoint_indices)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    fusion_context = build_fusion_context(model_keypoint_indices=model_keypoint_indices)

    pred_norm = infer_coords(
        model=model,
        x=x,
        fusion_context=fusion_context,
        prediction_mode=prediction_mode,
    )
    pred_norm = pred_norm.index_select(
        0,
        torch.tensor(eval_positions, dtype=torch.long),
    )
    pred_px = pred_norm.clone()
    pred_px[:, 0] = pred_px[:, 0] * float(orig_w)
    pred_px[:, 1] = pred_px[:, 1] * float(orig_h)

    gt_coords = np.asarray(sample["coords"], dtype=np.float32)[eval_keypoint_indices]
    gt_vis = np.asarray(sample["vis"], dtype=np.float32)[eval_keypoint_indices]

    return {
        "label": label,
        "global_keypoint_indices": list(eval_keypoint_indices),
        "gt_coords": gt_coords,
        "gt_vis": gt_vis,
        "pred_coords": pred_px.detach().cpu().numpy(),
    }


def load_mediapipe_panel(*, sample, args, shared_10_eval: bool):
    if not args.mediapipe_model_asset_path:
        raise ValueError("--include-mediapipe requires --mediapipe-model-asset-path")

    eval_positions, eval_keypoint_indices = resolve_eval_indices(
        model_keypoint_indices=MEDIAPIPE_MODEL_KEYPOINT_INDICES,
        shared_10_eval=bool(shared_10_eval),
    )
    rgb_image = np.asarray(load_rgb_image(sample["image_path"]).convert("RGB"), dtype=np.uint8)
    height, width = rgb_image.shape[:2]

    with MediaPipeHandLandmarkerRunner(
        model_asset_path=args.mediapipe_model_asset_path,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as baseline:
        detections = baseline.detect_image(rgb_image)

    selected = choose_detection(
        detections,
        requested_hand=args.mediapipe_hand,
        ignore_handedness=bool(args.ignore_handedness),
    )
    pred_coords = np.zeros((len(MEDIAPIPE_MODEL_KEYPOINT_INDICES), 2), dtype=np.float32)
    pred_vis = np.zeros((len(MEDIAPIPE_MODEL_KEYPOINT_INDICES),), dtype=np.float32)
    if selected is not None:
        pred_coords = np.asarray(selected["coords"], dtype=np.float32)
        pred_coords[:, 0] *= float(width)
        pred_coords[:, 1] *= float(height)
        pred_vis[:] = 1.0

    gt_coords = np.asarray(sample["coords"], dtype=np.float32)[eval_keypoint_indices]
    gt_vis = np.asarray(sample["vis"], dtype=np.float32)[eval_keypoint_indices]
    pred_coords = pred_coords[eval_positions]
    pred_vis = pred_vis[eval_positions]

    return {
        "label": "MediaPipe",
        "global_keypoint_indices": list(eval_keypoint_indices),
        "gt_coords": gt_coords,
        "gt_vis": gt_vis,
        "pred_coords": pred_coords,
        "pred_vis": pred_vis,
    }


def render_figure(*, image, sample_name: str, panels, out_path: Path, cols: int):
    panel_count = 1 + len(panels)
    cols = max(1, min(int(cols), panel_count))
    rows = int(math.ceil(float(panel_count) / float(cols)))

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    axes_flat = list(axes.flat)

    gt_panel = panels[0]
    axes_flat[0].imshow(image)
    axes_flat[0].set_title("Ground Truth", fontsize=11)
    draw_hand(
        axes_flat[0],
        gt_panel["gt_coords"],
        vis=gt_panel["gt_vis"],
        global_keypoint_indices=gt_panel["global_keypoint_indices"],
        point_color="lime",
        edge_color="lime",
        alpha=0.95,
    )
    axes_flat[0].axis("off")

    for ax, panel in zip(axes_flat[1:], panels):
        ax.imshow(image)
        draw_hand(
            ax,
            panel["gt_coords"],
            vis=panel["gt_vis"],
            global_keypoint_indices=panel["global_keypoint_indices"],
            point_color="lime",
            edge_color="lime",
            alpha=0.9,
        )
        draw_hand(
            ax,
            panel["pred_coords"],
            vis=panel.get("pred_vis", panel["gt_vis"]),
            global_keypoint_indices=panel["global_keypoint_indices"],
            point_color="red",
            edge_color="red",
            alpha=0.8,
        )
        ax.set_title(panel["label"], fontsize=11)
        ax.axis("off")

    for ax in axes_flat[1 + len(panels):]:
        ax.axis("off")

    fig.suptitle(
        f"{sample_name}\nGT: green  |  Prediction: red",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.model and not args.include_mediapipe:
        raise ValueError("Provide at least one --model or set --include-mediapipe")

    device = resolve_device(args.device)
    sample = load_sample(
        root=args.root,
        split=args.split,
        image_name=args.image,
        index=args.index,
    )
    image = load_rgb_image(sample["image_path"])

    panels = []
    for spec_text in args.model:
        spec = parse_model_spec(spec_text)
        panels.append(
            load_checkpoint_panel(
                sample=sample,
                label=spec["label"],
                ckpt_path=spec["ckpt"],
                device=device,
                prediction_mode=args.prediction_mode,
                shared_10_eval=bool(args.shared_10_eval),
            )
        )

    if args.include_mediapipe:
        panels.append(
            load_mediapipe_panel(
                sample=sample,
                args=args,
                shared_10_eval=bool(args.shared_10_eval),
            )
        )

    out_path = Path(args.out)
    render_figure(
        image=image,
        sample_name=sample["image_path"].name,
        panels=panels,
        out_path=out_path,
        cols=args.cols,
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
