"""PyTorch dataset wrapper for the Rendered Handpose Dataset.

This module loads RHD samples, applies repository preprocessing, and exposes
the tensors used by training and evaluation pipelines.
"""

import torch
from torch.utils.data import Dataset

from handpose.data.dataset_selection import validate_keypoint_indices
from handpose.data.rhd.parsing import find_uv_key, load_annotations, select_hand
from handpose.data.rhd.paths import annotation_path, image_dir, resolve_split_name
from handpose.data.transforms import preprocess_full_image


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
        self.split = resolve_split_name(split)
        self.input_size = input_size
        self.hand = hand
        self.normalize = normalize
        self.keypoint_indices = validate_keypoint_indices(keypoint_indices, total_keypoints=21)
        self.return_visibility = return_visibility
        self.annotation_path = annotation_path(root, self.split)
        self.image_dir = image_dir(root, self.split)
        self.anno = load_annotations(self.annotation_path)
        self.ids = sorted(self.anno.keys())

    def __len__(self):
        """Return the number of annotated samples in the split."""
        return len(self.ids)

    def __getitem__(self, idx):
        """Return image tensor, normalized keypoints, and optional visibility."""
        sample_id = self.ids[idx]
        anno = self.anno[sample_id]
        img_path = self.image_dir / f"{sample_id:05d}.png"

        uv_data = anno[find_uv_key(anno)]  # expected shape (42, 3)
        hand = select_hand(uv_data, self.hand)
        coords = hand[:, :2]
        coords = coords[self.keypoint_indices]
        vis = hand[:, 2]
        vis = vis[self.keypoint_indices]

        img, coords, _, _ = preprocess_full_image(
            image_path=img_path,
            coords_px=coords,
            input_size=self.input_size,
            normalize=self.normalize,
        )

        coords = torch.tensor(coords, dtype=torch.float32)
        vis = torch.tensor(vis, dtype=torch.float32)

        if self.return_visibility:
            return img, coords, vis
        return img, coords
