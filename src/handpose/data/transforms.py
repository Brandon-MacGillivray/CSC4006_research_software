from pathlib import Path

import numpy as np
from PIL import Image
import torch


def load_rgb_image(image_path):
    """Load one RGB image from disk."""
    return Image.open(Path(image_path)).convert("RGB")


def image_to_tensor(image: Image.Image):
    """Convert a PIL RGB image to a float tensor in CHW layout."""
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


def preprocess_full_image(image_path, coords_px, *, input_size: int, normalize: bool = True):
    """Resize the full image and rescale keypoints into the resized frame."""
    image = load_rgb_image(image_path)
    return preprocess_full_image_from_pil(
        image=image,
        coords_px=coords_px,
        input_size=input_size,
        normalize=normalize,
    )


def preprocess_full_image_from_pil(
    image: Image.Image,
    coords_px,
    *,
    input_size: int,
    normalize: bool = True,
):
    """Resize a PIL image and transform pixel keypoints to the resized space."""
    if int(input_size) <= 0:
        raise ValueError("input_size must be > 0")

    orig_w, orig_h = image.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image size: {(orig_w, orig_h)}")

    coords = np.asarray(coords_px, dtype=np.float32).copy()
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords_px must have shape (K, 2), got {coords.shape}")

    resized = image.resize((input_size, input_size))
    scale_x = float(input_size) / float(orig_w)
    scale_y = float(input_size) / float(orig_h)
    coords[:, 0] *= scale_x
    coords[:, 1] *= scale_y

    if normalize:
        coords[:, 0] /= float(input_size)
        coords[:, 1] /= float(input_size)

    return image_to_tensor(resized), coords, orig_w, orig_h
