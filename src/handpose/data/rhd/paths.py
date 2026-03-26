from pathlib import Path


RHD_SPLIT_ALIASES = {
    "train": "training",
    "training": "training",
    "val": "evaluation",
    "validation": "evaluation",
    "eval": "evaluation",
    "evaluation": "evaluation",
    "benchmark": "evaluation",
}


def resolve_split_name(split: str):
    """Resolve train/val/eval aliases to the RHD on-disk split name."""
    text = str(split).strip().lower()
    if text in RHD_SPLIT_ALIASES:
        return RHD_SPLIT_ALIASES[text]
    raise ValueError(f"Unsupported RHD split: {split!r}")


def annotation_path(root, split: str):
    """Return the annotation pickle path for one RHD split."""
    resolved_split = resolve_split_name(split)
    return Path(root) / resolved_split / f"anno_{resolved_split}.pickle"


def image_dir(root, split: str):
    """Return the RGB image directory for one RHD split."""
    resolved_split = resolve_split_name(split)
    return Path(root) / resolved_split / "color"
