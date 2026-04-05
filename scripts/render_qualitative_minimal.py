import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal wrapper for qualitative rendering on the hand_frame_small dataset. "
            "This is a thin preset around render_qualitative_batch.py with sensible defaults."
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
        "--images",
        nargs="*",
        default=None,
        help="Optional explicit list of image file names to render.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of images to render. 0 means all images.",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Use native keypoint evaluation instead of the default shared-10 comparison.",
    )
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the figure grid.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--root",
        default="docs/hand_frame_small",
        help="Dataset root. Defaults to docs/hand_frame_small.",
    )
    parser.add_argument("--split", default="val", help="Dataset split alias. Defaults to val.")
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
        "--batch-script",
        default="scripts/render_qualitative_batch.py",
        help="Path to the full batch renderer script.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model and args.no_mediapipe:
        raise ValueError("Provide at least one --model unless MediaPipe is included.")

    cmd = [
        sys.executable,
        str(Path(args.batch_script)),
        "--root",
        str(args.root),
        "--split",
        str(args.split),
        "--out-dir",
        str(args.out_dir),
        "--device",
        str(args.device),
        "--cols",
        str(args.cols),
    ]

    if not args.native:
        cmd.append("--shared-10-eval")

    if int(args.limit) > 0:
        cmd.extend(["--limit", str(args.limit)])

    if args.images:
        cmd.append("--images")
        cmd.extend([str(name) for name in args.images])

    for model_spec in args.model:
        cmd.extend(["--model", str(model_spec)])

    if not args.no_mediapipe:
        cmd.append("--include-mediapipe")
        cmd.extend(
            [
                "--mediapipe-model-asset-path",
                str(args.mediapipe_model_asset_path),
            ]
        )

    print("running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
