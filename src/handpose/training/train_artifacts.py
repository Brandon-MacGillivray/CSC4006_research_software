import csv
import os

from torch.utils.data import DataLoader


def append_csv_row(csv_path, row):
    """Append one metrics row to the loss CSV, creating headers if needed."""
    file_exists = os.path.exists(csv_path)
    if not file_exists:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stage", "epoch", "train_loss", "val_loss", "seconds"])
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def make_loaders(train_ds, val_ds, batch_size, num_workers):
    """Create training and validation DataLoaders with shared settings."""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
