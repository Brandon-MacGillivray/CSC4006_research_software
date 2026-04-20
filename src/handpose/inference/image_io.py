"""Image-loading helpers for repository inference workflows.

This module converts input images into the tensor format expected by the hand
pose prediction pipeline.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch


INPUT_SIZE = 256


def load_image_tensor(image_path: Path, device: torch.device):
    """Load, resize, and convert one RGB image to a model-ready tensor."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    resized = img.resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(resized)
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(device)
    return img, w, h, x
