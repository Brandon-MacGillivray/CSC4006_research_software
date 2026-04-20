"""Public data-layer exports for the supported hand-pose datasets."""

from handpose.data.dataset import COCOHandKeypointsDataset, RHDDatasetCoords
from handpose.data.factory import (
    AUTO_DATASET,
    SUPPORTED_DATASETS,
    build_dataset,
    discover_dataset_images,
    normalize_dataset_name,
    resolve_dataset_name,
    resolve_dataset_split,
)

__all__ = [
    "AUTO_DATASET",
    "COCOHandKeypointsDataset",
    "RHDDatasetCoords",
    "SUPPORTED_DATASETS",
    "build_dataset",
    "discover_dataset_images",
    "normalize_dataset_name",
    "resolve_dataset_name",
    "resolve_dataset_split",
]
