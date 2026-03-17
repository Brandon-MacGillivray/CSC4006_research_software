import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


LOSS_CSV_COLUMNS = [
    "stage",
    "epoch",
    "train_loss",
    "train_loss_hm",
    "train_loss_coord",
    "val_loss",
    "val_loss_hm",
    "val_loss_coord",
    "seconds",
    "lr",
    "batch_size",
    "accum_steps",
    "num_train_steps",
    "num_val_steps",
    "seed",
    "num_keypoints",
    "hand",
    "input_size",
    "tips_bases_only",
    "lambda_hm",
    "lambda_coord",
    "heatmap_sigma",
    "wing_w",
    "wing_epsilon",
]


def _read_existing_header(csv_path):
    """Return existing CSV header row if the file already has one."""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        return next(r, None)


def _normalize_row(row):
    """Normalize row payloads to the new detailed CSV schema."""
    if isinstance(row, dict):
        return dict(row)
    if isinstance(row, (list, tuple)):
        if len(row) != 5:
            raise ValueError("Legacy list rows must contain 5 values: stage, epoch, train_loss, val_loss, seconds")
        stage, epoch, train_loss, val_loss, seconds = row
        return {
            "stage": stage,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": seconds,
        }
    raise TypeError("row must be a dict or a 5-value legacy list/tuple")


def append_csv_row(csv_path, row):
    """Append one metrics row to the loss CSV, creating headers if needed."""
    existing_header = _read_existing_header(csv_path)
    file_exists = existing_header is not None
    fieldnames = existing_header if file_exists else LOSS_CSV_COLUMNS
    row = _normalize_row(row)
    row = {k: row.get(k, "") for k in fieldnames}

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def _build_worker_init_fn(seed):
    """Seed DataLoader workers deterministically when a run seed is provided."""
    if seed is None:
        return None

    def _init_fn(worker_id):
        worker_seed = (int(seed) + int(worker_id)) % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn


def make_loaders(train_ds, val_ds, batch_size, num_workers, seed=None):
    """Create training and validation DataLoaders with shared settings."""
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    worker_init_fn = _build_worker_init_fn(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader
