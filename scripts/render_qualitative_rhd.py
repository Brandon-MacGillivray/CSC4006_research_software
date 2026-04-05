import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from handpose.checkpoints import infer_checkpoint_keypoint_indices, load_checkpoint
from handpose.data.keypoints import TIP_BASE_KEYPOINT_INDICES
from handpose.data.rhd.parsing import find_uv_key, load_annotations, select_hand
from handpose.data.rhd.paths import annotation_path, image_dir, resolve_split_name
from handpose.inference.fusion import HAND_BONE_EDGES_10_GLOBAL, HAND_BONE_EDGES_21
from handpose.inference.image_io import load_image_tensor
from handpose.inference.mediapipe_baseline import (
    MediaPipeHandLandmarkerRunner,
    load_rgb_image,
)
from handpose.inference.predict import (
    SUPPORTED_PREDICTION_MODES,
    build_fusion_context,
    infer_coords,
)
from handpose.models.hand_pose_model import HandPoseNet


MEDIAPIPE_MODEL_KEYPOINT_INDICES = list(range(21))
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal standalone qualitative renderer for the RHD dataset. "
            "Writes one comparison figure per selected image."
        )
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where rendered figures will be written.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Repeated model spec in the form Label=/path/to/checkpoint.pt",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional explicit list of RHD sample IDs or file names, for example 3 143 00435.png",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of images to render. 0 means all matching images.",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Use native keypoint layout instead of the default shared-10 comparison.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of columns in the figure grid. Defaults to all panels in one row.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for checkpoint-based models.",
    )
    parser.add_argument(
        "--root",
        default="data/RHD_published_v2",
        help="RHD dataset root. Defaults to data/RHD_published_v2.",
    )
    parser.add_argument(
        "--split",
        default="evaluation",
        help="RHD split alias. Defaults to evaluation.",
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right", "auto"],
        help="Which RHD hand to use as ground truth. Defaults to right.",
    )
    parser.add_argument(
        "--prediction-mode",
        default="fusion",
        choices=SUPPORTED_PREDICTION_MODES,
        help="Prediction mode for checkpoint-based models.",
    )
    parser.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="Do not add the MediaPipe baseline panel.",
    )
    parser.add_argument(
        "--mediapipe-model-asset-path",
        default="src/handpose/models/hand_landmarker.task",
        help="Path to the MediaPipe .task asset. Defaults to the repo copy.",
    )
    parser.add_argument(
        "--mediapipe-hand",
        default="match",
        choices=["left", "right", "auto", "match"],
        help="Requested hand for the MediaPipe panel. 'match' uses --hand.",
    )
    parser.add_argument(
        "--ignore-handedness",
        action="store_true",
        help="Ignore MediaPipe handedness labels and use the highest-score detection.",
    )
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def resolve_device(device_name: str):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_user_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    repo_relative = REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative.resolve()
    return repo_relative


def parse_model_spec(spec: str):
    if "=" not in str(spec):
        raise ValueError(f"Invalid --model spec {spec!r}. Expected Label=/path/to/checkpoint.pt")
    label, ckpt = str(spec).split("=", 1)
    label = label.strip()
    ckpt = ckpt.strip()
    if not label or not ckpt:
        raise ValueError(f"Invalid --model spec {spec!r}. Expected Label=/path/to/checkpoint.pt")
    return label, Path(ckpt).expanduser()


def parse_requested_ids(values):
    requested = set()
    for value in values or []:
        text = str(value).strip()
        if not text:
            continue
        if text.lower().endswith(".png"):
            text = Path(text).stem
        requested.add(int(text))
    return requested


def choose_detection(detections, *, requested_hand: str, ignore_handedness: bool):
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


def resolve_eval_positions(model_keypoint_indices, shared_10_eval: bool):
    model_keypoint_indices = [int(x) for x in model_keypoint_indices]
    if not shared_10_eval:
        return list(range(len(model_keypoint_indices))), list(model_keypoint_indices)

    eval_positions = []
    missing = []
    for kp in TIP_BASE_KEYPOINT_INDICES:
        try:
            eval_positions.append(model_keypoint_indices.index(int(kp)))
        except ValueError:
            missing.append(int(kp))
    if missing:
        raise ValueError(
            "Checkpoint keypoint set does not contain all shared-10 keypoints. "
            f"Missing: {missing}. Checkpoint keypoints: {model_keypoint_indices}"
        )
    return eval_positions, list(TIP_BASE_KEYPOINT_INDICES)


def resolve_draw_edges(keypoint_indices):
    keypoint_indices = [int(x) for x in keypoint_indices]
    keypoint_set = set(keypoint_indices)
    index_of = {kp: i for i, kp in enumerate(keypoint_indices)}
    if keypoint_set == set(range(21)):
        source_edges = HAND_BONE_EDGES_21
    elif keypoint_set == set(TIP_BASE_KEYPOINT_INDICES):
        source_edges = HAND_BONE_EDGES_10_GLOBAL
    else:
        return []
    return [(index_of[a], index_of[b]) for a, b in source_edges if a in index_of and b in index_of]


def project_normalized_to_pixels(coords_norm, *, width: int, height: int):
    coords_px = np.asarray(coords_norm, dtype=np.float32).copy()
    scale_x = float(max(width - 1, 1))
    scale_y = float(max(height - 1, 1))
    coords_px[:, 0] *= scale_x
    coords_px[:, 1] *= scale_y
    return coords_px


def draw_hand(ax, coords_px, vis_mask, edges, *, color: str, linewidth: float, point_size: float):
    coords_px = np.asarray(coords_px, dtype=np.float32)
    vis_mask = np.asarray(vis_mask, dtype=bool)
    for a, b in edges:
        if a >= len(coords_px) or b >= len(coords_px):
            continue
        if not (vis_mask[a] and vis_mask[b]):
            continue
        ax.plot(
            [coords_px[a, 0], coords_px[b, 0]],
            [coords_px[a, 1], coords_px[b, 1]],
            color=color,
            linewidth=linewidth,
        )
    if np.any(vis_mask):
        ax.scatter(
            coords_px[vis_mask, 0],
            coords_px[vis_mask, 1],
            s=point_size,
            c=color,
            edgecolors="white",
            linewidths=0.7,
        )


def load_rhd_samples(root: Path, split: str, hand: str):
    resolved_split = resolve_split_name(split)
    annotations = load_annotations(annotation_path(root, resolved_split))
    color_dir = image_dir(root, resolved_split)

    samples = []
    for sample_id in sorted(annotations.keys()):
        sample_meta = annotations[sample_id]
        uv_data = sample_meta[find_uv_key(sample_meta)]
        hand_uv = select_hand(uv_data, hand)
        samples.append(
            {
                "sample_id": int(sample_id),
                "file_name": f"{int(sample_id):05d}.png",
                "image_path": color_dir / f"{int(sample_id):05d}.png",
                "coords": np.asarray(hand_uv[:, :2], dtype=np.float32),
                "vis": np.asarray(hand_uv[:, 2], dtype=np.float32),
            }
        )
    return samples


class CheckpointPanel:
    def __init__(self, *, label: str, ckpt_path: Path, device: torch.device, prediction_mode: str, shared_10_eval: bool):
        self.label = str(label)
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        self.prediction_mode = str(prediction_mode)
        ckpt_meta, state_dict = load_checkpoint(str(self.ckpt_path), device)
        self.model_keypoint_indices = infer_checkpoint_keypoint_indices(ckpt_meta)
        self.eval_positions, self.eval_keypoint_indices = resolve_eval_positions(
            self.model_keypoint_indices,
            shared_10_eval=bool(shared_10_eval),
        )
        self.draw_edges = resolve_draw_edges(self.eval_keypoint_indices)
        self.model = HandPoseNet(num_keypoints=len(self.model_keypoint_indices))
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.fusion_context = build_fusion_context(self.model_keypoint_indices)

    def predict(self, image_path: Path):
        original_image, width, height, x = load_image_tensor(Path(image_path), self.device)
        pred_norm = infer_coords(
            model=self.model,
            x=x,
            fusion_context=self.fusion_context,
            prediction_mode=self.prediction_mode,
        ).numpy()
        pred_norm = pred_norm[self.eval_positions]
        pred_px = project_normalized_to_pixels(pred_norm, width=width, height=height)
        return np.asarray(original_image.convert("RGB"), dtype=np.uint8), pred_px


class MediaPipePanel:
    def __init__(
        self,
        *,
        model_asset_path: Path,
        shared_10_eval: bool,
        requested_hand: str,
        ignore_handedness: bool,
        num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float,
        min_tracking_confidence: float,
    ):
        self.label = "MediaPipe"
        self.eval_positions, self.eval_keypoint_indices = resolve_eval_positions(
            MEDIAPIPE_MODEL_KEYPOINT_INDICES,
            shared_10_eval=bool(shared_10_eval),
        )
        self.draw_edges = resolve_draw_edges(self.eval_keypoint_indices)
        self.requested_hand = str(requested_hand)
        self.ignore_handedness = bool(ignore_handedness)
        self.runner = MediaPipeHandLandmarkerRunner(
            model_asset_path=str(model_asset_path),
            num_hands=int(num_hands),
            min_hand_detection_confidence=float(min_hand_detection_confidence),
            min_hand_presence_confidence=float(min_hand_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def close(self):
        self.runner.close()

    def predict(self, image_path: Path):
        rgb_image = load_rgb_image(image_path)
        height, width = rgb_image.shape[:2]
        detections = self.runner.detect_image(rgb_image)
        selected, miss_reason = choose_detection(
            detections,
            requested_hand=self.requested_hand,
            ignore_handedness=self.ignore_handedness,
        )
        if selected is None:
            return rgb_image, None, miss_reason
        coords_norm = np.asarray(selected["coords"], dtype=np.float32)[self.eval_positions]
        coords_px = project_normalized_to_pixels(coords_norm, width=width, height=height)
        return rgb_image, coords_px, None


def render_sample(*, sample: dict, gt_keypoint_indices, gt_edges, panels, out_path: Path, cols: int):
    gt_vis = np.asarray(sample["vis"], dtype=np.float32) > 0
    gt_positions = [int(kp) for kp in gt_keypoint_indices]
    gt_coords = np.asarray(sample["coords"], dtype=np.float32)[gt_positions]
    gt_mask = gt_vis[gt_positions]

    total_panels = 1 + len(panels)
    rows = int(math.ceil(total_panels / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    base_image = load_rgb_image(sample["image_path"])
    axes[0].imshow(base_image)
    draw_hand(
        axes[0],
        gt_coords,
        gt_mask,
        gt_edges,
        color="lime",
        linewidth=2.0,
        point_size=46.0,
    )
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    for ax, panel in zip(axes[1:], panels):
        ax.imshow(panel["image"])
        draw_hand(
            ax,
            gt_coords,
            gt_mask,
            gt_edges,
            color="lime",
            linewidth=1.8,
            point_size=42.0,
        )
        pred_coords = panel.get("pred_coords")
        if pred_coords is not None:
            draw_hand(
                ax,
                pred_coords,
                np.ones(len(pred_coords), dtype=bool),
                panel.get("draw_edges", []),
                color="red",
                linewidth=1.8,
                point_size=36.0,
            )
            title = panel["label"]
        else:
            title = f"{panel['label']}\n(no detection)"
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[total_panels:]:
        ax.axis("off")

    fig.suptitle(sample["file_name"], fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    shared_10_eval = not bool(args.native)

    if not args.model and args.no_mediapipe:
        raise ValueError("Provide at least one --model unless MediaPipe is included.")

    dataset_root = resolve_user_path(args.root)
    samples = load_rhd_samples(dataset_root, args.split, args.hand)

    requested_ids = parse_requested_ids(args.ids)
    if requested_ids:
        samples = [sample for sample in samples if int(sample["sample_id"]) in requested_ids]
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]
    if not samples:
        raise ValueError("No samples matched the requested split/ids selection.")

    device = resolve_device(str(args.device))

    checkpoint_panels = []
    for spec in args.model:
        label, ckpt_path = parse_model_spec(spec)
        checkpoint_panels.append(
            CheckpointPanel(
                label=label,
                ckpt_path=ckpt_path,
                device=device,
                prediction_mode=args.prediction_mode,
                shared_10_eval=shared_10_eval,
            )
        )

    mediapipe_panel = None
    if not args.no_mediapipe:
        requested_mp_hand = args.hand if args.mediapipe_hand == "match" else args.mediapipe_hand
        mediapipe_panel = MediaPipePanel(
            model_asset_path=resolve_user_path(args.mediapipe_model_asset_path),
            shared_10_eval=shared_10_eval,
            requested_hand=requested_mp_hand,
            ignore_handedness=bool(args.ignore_handedness),
            num_hands=int(args.num_hands),
            min_hand_detection_confidence=float(args.min_hand_detection_confidence),
            min_hand_presence_confidence=float(args.min_hand_presence_confidence),
            min_tracking_confidence=float(args.min_tracking_confidence),
        )

    gt_keypoint_indices = list(TIP_BASE_KEYPOINT_INDICES) if shared_10_eval else list(range(21))
    gt_edges = resolve_draw_edges(gt_keypoint_indices)
    out_dir = Path(args.out_dir).expanduser()

    try:
        for sample in samples:
            panels = []
            for runner in checkpoint_panels:
                image_arr, pred_coords = runner.predict(sample["image_path"])
                panels.append(
                    {
                        "label": runner.label,
                        "image": image_arr,
                        "pred_coords": pred_coords,
                        "draw_edges": runner.draw_edges,
                    }
                )
            if mediapipe_panel is not None:
                image_arr, pred_coords, _ = mediapipe_panel.predict(sample["image_path"])
                panels.append(
                    {
                        "label": mediapipe_panel.label,
                        "image": image_arr,
                        "pred_coords": pred_coords,
                        "draw_edges": mediapipe_panel.draw_edges,
                    }
                )

            out_path = out_dir / f"{Path(sample['file_name']).stem}_qualitative.png"
            total_panels = 1 + len(panels)
            cols = int(args.cols) if args.cols is not None else total_panels
            render_sample(
                sample=sample,
                gt_keypoint_indices=gt_keypoint_indices,
                gt_edges=gt_edges,
                panels=panels,
                out_path=out_path,
                cols=max(1, cols),
            )
            print(f"saved: {out_path}")
    finally:
        if mediapipe_panel is not None:
            mediapipe_panel.close()


if __name__ == "__main__":
    main()
