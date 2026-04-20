"""Overlay rendering helpers for visualising predicted hand keypoints.

This module writes simple image overlays used by single-image prediction and
qualitative figure generation.
"""

from pathlib import Path

from PIL import Image, ImageDraw


def save_overlay(image: Image.Image, coords_px, out_path: Path):
    """Draw predicted keypoints on an image and save the result."""
    draw = ImageDraw.Draw(image)
    r = 3
    for x, y in coords_px:
        x = float(x)
        y = float(y)
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", fill="red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
