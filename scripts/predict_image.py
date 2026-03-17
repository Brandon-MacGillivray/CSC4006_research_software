import argparse
import json
from pathlib import Path

import torch

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from handpose.inference.image_io import load_image_tensor
from handpose.inference.predict import (
    SUPPORTED_PREDICTION_MODES,
    build_fusion_context,
    infer_coords,
)
from handpose.checkpoints import (
    get_training_config,
    infer_checkpoint_keypoint_indices,
    load_checkpoint,
)
from handpose.models.hand_pose_model import HandPoseNet
from handpose.inference.visualization import save_overlay


def parse_args():
    """Parse CLI arguments for single-image inference."""
    p = argparse.ArgumentParser(description="Minimal single-image inference for DRHand checkpoints.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt), typically stage-2 best.pt")
    p.add_argument("--image", required=True, help="Path to an input RGB image")
    p.add_argument(
        "--prediction-mode",
        default="fusion",
        choices=SUPPORTED_PREDICTION_MODES,
        help="Select fused, heatmap-only, or coord-only predictions.",
    )
    p.add_argument("--overlay", action="store_true", help="If set, save image with predicted points drawn")
    p.add_argument("--overlay-out", default=None, help="Optional output path for overlay image")
    return p.parse_args()


def main():
    """Run checkpoint loading, inference, and optional overlay export."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_meta, state_dict = load_checkpoint(args.ckpt, device)
    keypoint_indices = infer_checkpoint_keypoint_indices(ckpt_meta)
    training_config = get_training_config(ckpt_meta)
    fusion_context = build_fusion_context(model_keypoint_indices=keypoint_indices)

    model = HandPoseNet(num_keypoints=len(keypoint_indices)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    image_path = Path(args.image)
    img, orig_w, orig_h, x = load_image_tensor(image_path=image_path, device=device)
    pred_norm = infer_coords(
        model=model,
        x=x,
        fusion_context=fusion_context,
        prediction_mode=args.prediction_mode,
    )

    pred_orig_px = pred_norm.clone()
    pred_orig_px[:, 0] = pred_orig_px[:, 0] * float(orig_w)
    pred_orig_px[:, 1] = pred_orig_px[:, 1] * float(orig_h)

    results = {
        "checkpoint": str(Path(args.ckpt)),
        "image": str(image_path),
        "prediction_mode": args.prediction_mode,
        "training_config": training_config,
        "model_keypoint_indices": keypoint_indices,
        "pred_coords_pixels_original": pred_orig_px.tolist(),
    }
    print(json.dumps(results, indent=2))

    if args.overlay:
        if args.overlay_out:
            overlay_path = Path(args.overlay_out)
        else:
            overlay_path = image_path.with_name(f"{image_path.stem}_pred_overlay{image_path.suffix}")
        save_overlay(image=img, coords_px=pred_orig_px.tolist(), out_path=overlay_path)
        print(f"saved: {overlay_path}")


if __name__ == "__main__":
    main()
