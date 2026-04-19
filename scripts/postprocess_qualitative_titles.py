import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TITLE_BAND_RATIO = 0.100
TITLE_Y_RATIO = 0.040
PANEL_WIDTH_ESTIMATE = 742.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite qualitative figure title bands to remove debug filenames and normalize panel labels."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing source PNGs.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed PNGs.")
    return parser.parse_args()


def load_font(size: int):
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in font_candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def panel_titles(panel_count: int):
    if panel_count == 8:
        return [
            "Ground Truth",
            "Baseline-21",
            "Reduced-10",
            "RHD-only",
            "HK26K-only",
            "HK26K→RHD",
            "RHD→HK26K",
            "MediaPipe",
        ]
    if panel_count == 7:
        return [
            "Ground Truth",
            "Baseline-21",
            "Reduced-10",
            "HK26K-only",
            "HK26K→RHD",
            "RHD→HK26K",
            "MediaPipe",
        ]
    raise ValueError(f"Unsupported panel count inferred from image width: {panel_count}")


def infer_panel_count(width: int):
    return int(round(width / PANEL_WIDTH_ESTIMATE))


def process_image(src_path: Path, dst_path: Path):
    image = Image.open(src_path).convert("RGB")
    width, height = image.size
    panel_count = infer_panel_count(width)
    titles = panel_titles(panel_count)

    band_height = int(round(height * TITLE_BAND_RATIO))
    title_y = int(round(height * TITLE_Y_RATIO))
    font_size = max(24, int(round(height * 0.042)))
    font = load_font(font_size)

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, band_height), fill="white")

    for idx, title in enumerate(titles):
        center_x = ((idx + 0.5) / panel_count) * width
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (center_x - (text_w / 2), title_y - (text_h / 2)),
            title,
            fill="black",
            font=font,
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst_path)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    for src_path in sorted(input_dir.glob("*.png")):
        process_image(src_path, output_dir / src_path.name)


if __name__ == "__main__":
    main()
