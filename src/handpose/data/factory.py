"""Dataset selection and split-resolution helpers for supported datasets.

This module maps dataset names and aliases to the concrete dataset loaders used
throughout training, evaluation, and inference.
"""

from pathlib import Path

from handpose.data.coco_hand.dataset import COCOHandKeypointsDataset
from handpose.data.coco_hand.paths import image_dir as coco_hand_image_dir
from handpose.data.coco_hand.paths import resolve_split_name as resolve_coco_hand_split
from handpose.data.rhd.dataset import RHDDatasetCoords
from handpose.data.rhd.paths import image_dir as rhd_image_dir
from handpose.data.rhd.paths import resolve_split_name as resolve_rhd_split


SUPPORTED_DATASETS = ("rhd", "coco_hand")
AUTO_DATASET = "auto"
_DATASET_ALIASES = {
    "auto": AUTO_DATASET,
    "rhd": "rhd",
    "coco": "coco_hand",
    "coco_hand": "coco_hand",
    "hand_keypoint": "coco_hand",
    "hand_keypoint_dataset": "coco_hand",
}


def normalize_dataset_name(dataset_name, *, allow_auto: bool = False):
    """Normalize supported dataset names and aliases."""
    if dataset_name is None:
        return AUTO_DATASET if allow_auto else "rhd"

    text = str(dataset_name).strip().lower()
    if text == "":
        return AUTO_DATASET if allow_auto else "rhd"

    resolved = _DATASET_ALIASES.get(text)
    if resolved is None:
        raise ValueError(
            f"dataset must be one of {SUPPORTED_DATASETS}"
            + (" or 'auto'" if allow_auto else "")
            + f", got {dataset_name!r}"
        )
    if resolved == AUTO_DATASET and not allow_auto:
        return "rhd"
    return resolved


def resolve_dataset_name(dataset_name, training_config=None):
    """Resolve 'auto' to the checkpoint dataset when present, else default to RHD."""
    resolved = normalize_dataset_name(dataset_name, allow_auto=True)
    if resolved != AUTO_DATASET:
        return resolved

    if isinstance(training_config, dict):
        candidate = training_config.get("dataset") or training_config.get("dataset_name")
        if candidate:
            return normalize_dataset_name(candidate, allow_auto=False)
    return "rhd"


def resolve_dataset_split(dataset_name, split: str):
    """Resolve train/val/eval aliases for one supported dataset."""
    resolved_dataset = normalize_dataset_name(dataset_name, allow_auto=False)
    if resolved_dataset == "rhd":
        return resolve_rhd_split(split)
    if resolved_dataset == "coco_hand":
        return resolve_coco_hand_split(split)
    raise ValueError(f"Unsupported dataset: {resolved_dataset!r}")


def build_dataset(
    *,
    dataset_name,
    root,
    split,
    input_size,
    hand="right",
    normalize=True,
    keypoint_indices=None,
    return_visibility=False,
):
    """Instantiate one supported dataset through a shared interface."""
    resolved_dataset = normalize_dataset_name(dataset_name, allow_auto=False)
    resolved_split = resolve_dataset_split(resolved_dataset, split)
    common_kwargs = {
        "root": root,
        "split": resolved_split,
        "input_size": input_size,
        "hand": hand,
        "normalize": normalize,
        "keypoint_indices": keypoint_indices,
        "return_visibility": return_visibility,
    }
    if resolved_dataset == "rhd":
        return RHDDatasetCoords(**common_kwargs)
    if resolved_dataset == "coco_hand":
        return COCOHandKeypointsDataset(**common_kwargs)
    raise ValueError(f"Unsupported dataset: {resolved_dataset!r}")


def discover_dataset_images(dataset_name, root, split="benchmark"):
    """Discover benchmark/eval images for the selected dataset."""
    resolved_dataset = normalize_dataset_name(dataset_name, allow_auto=False)
    resolved_split = resolve_dataset_split(resolved_dataset, split)
    root_path = Path(root)
    if resolved_dataset == "rhd":
        images_root = rhd_image_dir(root_path, resolved_split)
    elif resolved_dataset == "coco_hand":
        images_root = coco_hand_image_dir(root_path, resolved_split)
    else:
        raise ValueError(f"Unsupported dataset: {resolved_dataset!r}")

    if not images_root.exists():
        raise FileNotFoundError(f"Missing image directory: {images_root}")

    images = sorted(
        path
        for path in images_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        raise ValueError(f"No images found under {images_root}")
    return resolved_split, images_root, images
