"""PyTorch dataset wrapper for COCO-style hand-pose datasets.

This module loads HK26K, RH8, and related COCO-style samples into the tensor
format expected by the repository training and evaluation pipelines.
"""

import torch
from torch.utils.data import Dataset

from handpose.data.coco_hand.parsing import build_samples, load_annotation_payload
from handpose.data.coco_hand.paths import annotation_path, image_dir, resolve_split_name
from handpose.data.dataset_selection import validate_keypoint_indices
from handpose.data.transforms import preprocess_full_image


class COCOHandKeypointsDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        input_size=256,
        hand="single",
        normalize=True,
        keypoint_indices=None,
        return_visibility=False,
    ):
        """Load a COCO-style hand-keypoint split using full-image resize preprocessing."""
        self.root = root
        self.split = resolve_split_name(split)
        self.input_size = input_size
        self.hand = hand
        self.normalize = normalize
        self.keypoint_indices = validate_keypoint_indices(keypoint_indices, total_keypoints=21)
        self.return_visibility = return_visibility
        self.annotation_path = annotation_path(root, self.split)
        self.image_dir = image_dir(root, self.split)
        payload = load_annotation_payload(self.annotation_path)
        self.samples = build_samples(payload, self.image_dir, total_keypoints=21)

    def __len__(self):
        """Return the number of annotated hand instances in the split."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return image tensor, normalized keypoints, and optional visibility."""
        sample = self.samples[idx]
        coords = sample["coords"][self.keypoint_indices]
        vis = sample["vis"][self.keypoint_indices]
        img, coords, _, _ = preprocess_full_image(
            image_path=sample["image_path"],
            coords_px=coords,
            input_size=self.input_size,
            normalize=self.normalize,
        )

        coords = torch.tensor(coords, dtype=torch.float32)
        vis = torch.tensor(vis, dtype=torch.float32)
        if self.return_visibility:
            return img, coords, vis
        return img, coords
