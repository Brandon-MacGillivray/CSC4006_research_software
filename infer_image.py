import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from train_1 import HandPoseNet, TIP_BASE_KEYPOINT_INDICES


INPUT_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser(description="Minimal single-image inference for DRHand checkpoints.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt), typically stage-2 best.pt")
    p.add_argument("--image", required=True, help="Path to an input RGB image")
    p.add_argument("--overlay", action="store_true", help="If set, save image with predicted points drawn")
    p.add_argument("--overlay-out", default=None, help="Optional output path for overlay image")
    return p.parse_args()


def load_checkpoint(path: str, device: torch.device):
    obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path!r}")
    if "model_state" in obj:
        return obj, obj["model_state"]
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
    raise ValueError(f"Unsupported checkpoint keypoint count: {k}")


def load_image_tensor(image_path: Path, device: torch.device):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    resized = img.resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(resized)
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(device)
    return img, w, h, x


@torch.no_grad()
def infer_coords(model: HandPoseNet, x: torch.Tensor):
    _, pred = model(x)  # (1, K, 2)
    return pred[0].detach().cpu()


def save_overlay(image: Image.Image, coords_px, out_path: Path):
    draw = ImageDraw.Draw(image)
    r = 3
    for x, y in coords_px:
        x = float(x)
        y = float(y)
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", fill="red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_meta, state_dict = load_checkpoint(args.ckpt, device)
    keypoint_indices = infer_checkpoint_keypoint_indices(ckpt_meta, state_dict)

    model = HandPoseNet(num_keypoints=len(keypoint_indices)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    image_path = Path(args.image)
    img, orig_w, orig_h, x = load_image_tensor(image_path=image_path, device=device)
    pred_norm = infer_coords(model=model, x=x)

    pred_orig_px = pred_norm.clone()
    pred_orig_px[:, 0] = pred_orig_px[:, 0] * float(orig_w)
    pred_orig_px[:, 1] = pred_orig_px[:, 1] * float(orig_h)

    results = {
        "checkpoint": str(Path(args.ckpt)),
        "image": str(image_path),
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
