"""Common dataset exports for the repository data layer."""

from handpose.data.coco_hand.dataset import COCOHandKeypointsDataset
from handpose.data.rhd.dataset import RHDDatasetCoords

__all__ = [
    "COCOHandKeypointsDataset",
    "RHDDatasetCoords",
]
