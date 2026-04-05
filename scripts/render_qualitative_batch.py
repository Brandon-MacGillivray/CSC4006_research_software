import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()


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
        description=(
            "Batch wrapper for render_qualitative_comparison.py. "
            "Renders one qualitative comparison figure per image in a COCO-style split."
        )
    )
    parser.add_argument("--root", required=True, help="COCO-style dataset root, for example docs/hand_frame_small")
    parser.add_argument("--split", default="val", help="COCO-hand split alias, for example val/evaluation")
    parser.add_argument("--out-dir", required=True, help="Directory where output figures will be written")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Repeated model spec in the form Label=/path/to/checkpoint.pt",
    )
    parser.add_argument(
        "--prediction-mode",
        default="fusion",
        choices=("fusion", "heatmap", "coord"),
        help="Prediction mode used for all checkpoint panels.",
    )
    parser.add_argument("--shared-10-eval", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--include-mediapipe", action="store_true")
    parser.add_argument("--mediapipe-model-asset-path", default=None)
    parser.add_argument("--mediapipe-hand", default="auto", choices=["left", "right", "auto"])
    parser.add_argument("--ignore-handedness", action="store_true")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of images to render. 0 means all images.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Optional explicit list of image file names to render. Overrides --limit filtering.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix added to each output file name.",
    )
    parser.add_argument(
        "--renderer-script",
        default="scripts/render_qualitative_comparison.py",
        help="Path to the single-image renderer script.",
    )
    return parser.parse_args()


def resolve_split_name(split: str):
    text = str(split).strip().lower()
    if text in COCO_HAND_SPLIT_ALIASES:
        return COCO_HAND_SPLIT_ALIASES[text]
    raise ValueError(f"Unsupported COCO-hand split: {split!r}")


def load_image_names(root: str, split: str):
    split_name = resolve_split_name(split)
    anno_path = Path(root) / "coco_annotation" / split_name / "_annotations.coco.json"
    with open(anno_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    images = sorted(
        str(item.get("file_name", "")).strip()
        for item in payload.get("images", [])
        if str(item.get("file_name", "")).strip()
    )
    if not images:
        raise ValueError(f"No images listed in annotation JSON: {anno_path}")
    return images


def build_command(args, *, image_name: str, out_path: Path):
    cmd = [
        sys.executable,
        str(Path(args.renderer_script)),
        "--root",
        str(args.root),
        "--split",
        str(args.split),
        "--image",
        image_name,
        "--out",
        str(out_path),
        "--prediction-mode",
        str(args.prediction_mode),
        "--device",
        str(args.device),
        "--cols",
        str(args.cols),
    ]
    if args.shared_10_eval:
        cmd.append("--shared-10-eval")
    for model_spec in args.model:
        cmd.extend(["--model", str(model_spec)])
    if args.include_mediapipe:
        cmd.append("--include-mediapipe")
        if args.mediapipe_model_asset_path:
            cmd.extend(["--mediapipe-model-asset-path", str(args.mediapipe_model_asset_path)])
        cmd.extend(["--mediapipe-hand", str(args.mediapipe_hand)])
        if args.ignore_handedness:
            cmd.append("--ignore-handedness")
        cmd.extend(
            [
                "--num-hands",
                str(args.num_hands),
                "--min-hand-detection-confidence",
                str(args.min_hand_detection_confidence),
                "--min-hand-presence-confidence",
                str(args.min_hand_presence_confidence),
                "--min-tracking-confidence",
                str(args.min_tracking_confidence),
            ]
        )
    return cmd


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_names = load_image_names(args.root, args.split)
    if args.images:
        requested = set(str(name) for name in args.images)
        image_names = [name for name in image_names if name in requested]
    elif int(args.limit) > 0:
        image_names = image_names[: int(args.limit)]

    if not image_names:
        raise ValueError("No images selected for rendering")

    prefix = str(args.prefix)
    for idx, image_name in enumerate(image_names, start=1):
        stem = Path(image_name).stem
        out_name = f"{prefix}{stem}.png" if prefix else f"{stem}.png"
        out_path = out_dir / out_name
        cmd = build_command(args, image_name=image_name, out_path=out_path)
        print(f"[{idx}/{len(image_names)}] rendering {image_name} -> {out_path}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
