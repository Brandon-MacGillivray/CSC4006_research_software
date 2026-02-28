import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class RHDDatasetCoords(Dataset):
    def __init__(
        self,
        root,
        split="training",
        input_size=256,
        hand="right",
        normalize=True,
        keypoint_indices=None,
        return_visibility=False,
    ):
        self.root = root
        self.split = split
        self.input_size = input_size
        self.hand = hand
        self.normalize = normalize
        if keypoint_indices is None:
            keypoint_indices = list(range(21))
        if len(keypoint_indices) == 0:
            raise ValueError("keypoint_indices must contain at least one index")
        if len(set(keypoint_indices)) != len(keypoint_indices):
            raise ValueError("keypoint_indices must be unique")
        if min(keypoint_indices) < 0 or max(keypoint_indices) >= 21:
            raise ValueError("keypoint_indices must be in [0, 20]")
        self.keypoint_indices = list(keypoint_indices)
        self.return_visibility = return_visibility

        anno_path = os.path.join(root, split, f"anno_{split}.pickle")
        with open(anno_path, "rb") as f:
            self.anno = pickle.load(f)

        self.ids = sorted(self.anno.keys())

    def __len__(self):
        return len(self.ids)

    def _select_hand(self, uv_data):
        if self.hand == "left":
            return uv_data[0:21]
        if self.hand == "right":
            return uv_data[21:42]
        if self.hand == "auto":
            left = uv_data[0:21]
            right = uv_data[21:42]
            left_score = left[:, 2].sum()
            right_score = right[:, 2].sum()
            return right if right_score >= left_score else left
        raise ValueError("hand must be one of: 'left', 'right', 'auto'")

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        anno = self.anno[sample_id]

        img_path = os.path.join(self.root, self.split, "color", f"{sample_id:05d}.png")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.input_size, self.input_size))
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        uv_key = next(k for k in anno.keys() if str(k).lower().startswith("uv"))
        uv_data = anno[uv_key]  # expected shape (42, 3)
        hand = self._select_hand(uv_data)
        coords = hand[:, :2]  # (21, 2) in 320x320 pixels
        coords = coords[self.keypoint_indices]  # (K, 2), K=len(keypoint_indices)
        vis = hand[:, 2]
        vis = vis[self.keypoint_indices]

        scale = self.input_size / 320.0
        coords = coords * scale
        if self.normalize:
            coords = coords / self.input_size

        coords = torch.tensor(coords, dtype=torch.float32)
        vis = torch.tensor(vis, dtype=torch.float32)

        if self.return_visibility:
            return img, coords, vis
        return img, coords
