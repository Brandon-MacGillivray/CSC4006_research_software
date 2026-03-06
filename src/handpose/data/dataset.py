import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from handpose.data.dataset_selection import find_uv_key, select_hand, validate_keypoint_indices

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
        """Load an RHD split and configure preprocessing/keypoint selection."""
        self.root = root
        self.split = split
        self.input_size = input_size
        self.hand = hand
        self.normalize = normalize
        self.keypoint_indices = validate_keypoint_indices(keypoint_indices, total_keypoints=21)
        self.return_visibility = return_visibility

        anno_path = os.path.join(root, split, f"anno_{split}.pickle")
        with open(anno_path, "rb") as f:
            self.anno = pickle.load(f)

        self.ids = sorted(self.anno.keys())

    def __len__(self):
        """Return the number of annotated samples in the split."""
        return len(self.ids)

    def __getitem__(self, idx):
        """Return image tensor, normalized keypoints, and optional visibility."""
        sample_id = self.ids[idx]
        anno = self.anno[sample_id]

        img_path = os.path.join(self.root, self.split, "color", f"{sample_id:05d}.png")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.input_size, self.input_size))
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        uv_key = find_uv_key(anno)
        uv_data = anno[uv_key]  # expected shape (42, 3)
        hand = select_hand(uv_data, self.hand)
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
