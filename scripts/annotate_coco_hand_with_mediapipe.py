import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from handpose.inference.mediapipe_baseline import (
    MediaPipeHandLandmarkerRunner,
    load_rgb_image,
)


KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_VISIBILITY = 2
COCO_HAND_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "validation": "val",
    "eval": "val",
    "evaluation": "val",
    "benchmark": "val",
}
HAND_BONE_EDGES_21 = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively annotate a small COCO-style hand dataset using MediaPipe "
            "landmarks as the initial coordinates."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root of the qualitative dataset, for example data/qual_hand_small",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split alias. Images are read from images/<split> and JSON is written to coco_annotation/<split>.",
    )
    parser.add_argument(
        "--model-asset-path",
        required=True,
        help="Path to the MediaPipe Hand Landmarker .task asset.",
    )
    parser.add_argument(
        "--hand",
        default="auto",
        choices=["left", "right", "auto"],
        help="Requested hand to initialize from MediaPipe detections.",
    )
    parser.add_argument(
        "--ignore-handedness",
        action="store_true",
        help="Ignore MediaPipe handedness and always pick the highest-confidence detection.",
    )
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--start-at",
        default="",
        help="Optional 1-based image index or exact file name to open first.",
    )
    return parser.parse_args()


def resolve_split_name(split: str):
    text = str(split).strip().lower()
    if text in COCO_HAND_SPLIT_ALIASES:
        return COCO_HAND_SPLIT_ALIASES[text]
    raise ValueError(f"Unsupported COCO-hand split: {split!r}")


def choose_detection(detections, *, requested_hand: str, ignore_handedness: bool):
    """Select one MediaPipe detection using the same policy as baseline evaluation."""
    if not detections:
        return None, "no_detections"

    def det_score(det):
        score = det.get("handedness_score")
        return float(score) if score is not None else 0.0

    if requested_hand == "auto" or ignore_handedness:
        return max(detections, key=det_score), None

    requested = str(requested_hand).strip().lower()
    matching = [
        det for det in detections if str(det.get("handedness", "")).strip().lower() == requested
    ]
    if matching:
        return max(matching, key=det_score), None
    return None, "no_matching_handedness"


def list_images(image_dir: Path):
    paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported images found under {image_dir}")
    return paths


def load_existing_annotations(annotation_path: Path):
    if not annotation_path.exists():
        return {}, {}

    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    images_by_id = {}
    for image_meta in payload.get("images", []):
        image_id = image_meta.get("id")
        if image_id is None:
            continue
        images_by_id[int(image_id)] = image_meta

    annotation_by_file_name = {}
    for ann in payload.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        image_meta = images_by_id.get(int(image_id))
        if image_meta is None:
            continue
        file_name = str(image_meta.get("file_name", "")).strip()
        if not file_name:
            continue
        annotation_by_file_name[file_name] = ann

    return payload, annotation_by_file_name


def decode_keypoints(raw_keypoints):
    values = list(raw_keypoints or [])
    if len(values) < 63:
        values = values + [0.0] * (63 - len(values))
    arr = np.asarray(values[:63], dtype=np.float32).reshape(21, 3)
    return arr[:, :2].copy(), arr[:, 2].copy()


def encode_keypoints(coords: np.ndarray, vis: np.ndarray):
    flat = []
    for (x, y), v in zip(coords.tolist(), vis.tolist()):
        flat.extend([float(x), float(y), int(v)])
    return flat


def empty_annotation(width: int, height: int):
    center = np.array([float(width) * 0.5, float(height) * 0.5], dtype=np.float32)
    coords = np.repeat(center[None, :], 21, axis=0)
    vis = np.zeros((21,), dtype=np.int32)
    return coords, vis


class AnnotatorApp:
    def __init__(self, args):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on local runtime package
            raise RuntimeError(
                "matplotlib is required for interactive annotation. "
                "Install it with `pip install matplotlib`."
            ) from exc

        self.plt = plt
        self.args = args
        self.dataset_root = Path(args.dataset_root)
        self.split = resolve_split_name(args.split)
        self.image_dir = self.dataset_root / "images" / self.split
        self.annotation_path = self.dataset_root / "coco_annotation" / self.split / "_annotations.coco.json"
        self.annotation_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")

        self.image_paths = list_images(self.image_dir)
        self.image_index_by_name = {path.name: idx for idx, path in enumerate(self.image_paths)}

        self.existing_payload, self.existing_annotations = load_existing_annotations(self.annotation_path)
        self.image_records = []
        for idx, path in enumerate(self.image_paths, start=1):
            rgb = load_rgb_image(path)
            height, width = rgb.shape[:2]
            self.image_records.append(
                {
                    "id": idx,
                    "file_name": path.name,
                    "width": int(width),
                    "height": int(height),
                }
            )

        self.states = {}
        self.current_index = self._resolve_start_index(args.start_at)
        self.current_joint = 0

        self.runner = MediaPipeHandLandmarkerRunner(
            model_asset_path=args.model_asset_path,
            num_hands=args.num_hands,
            min_hand_detection_confidence=args.min_hand_detection_confidence,
            min_hand_presence_confidence=args.min_hand_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )

        self.fig, self.ax = self.plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(bottom=0.16)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.help_text = self.fig.text(
            0.01,
            0.02,
            (
                "Left click: set point + next  |  Right click or 0: mark hidden + next  |  "
                "[ / ]: prev/next joint  |  a / d: prev/next image  |  arrows: nudge  |  "
                "r: reset from MediaPipe  |  s: save  |  q: save + quit"
            ),
            fontsize=9,
        )

    def _resolve_start_index(self, start_at: str):
        text = str(start_at or "").strip()
        if not text:
            return 0
        if text.isdigit():
            idx = int(text) - 1
            return min(max(idx, 0), len(self.image_paths) - 1)
        return self.image_index_by_name.get(text, 0)

    def _build_state_from_existing(self, file_name: str):
        ann = self.existing_annotations.get(file_name)
        if ann is None:
            return None
        coords, vis = decode_keypoints(ann.get("keypoints", []))
        return {
            "coords": coords,
            "vis": vis.astype(np.int32),
            "source": "existing",
        }

    def _build_state_from_mediapipe(self, image_path: Path):
        rgb = load_rgb_image(image_path)
        height, width = rgb.shape[:2]
        detections = self.runner.detect_image(rgb)
        selected, reason = choose_detection(
            detections,
            requested_hand=self.args.hand,
            ignore_handedness=bool(self.args.ignore_handedness),
        )
        if selected is None:
            coords, vis = empty_annotation(width=width, height=height)
            return {
                "coords": coords,
                "vis": vis,
                "source": f"mediapipe:{reason or 'empty'}",
            }

        coords_norm = np.asarray(selected["coords"], dtype=np.float32)
        coords = np.zeros_like(coords_norm)
        coords[:, 0] = coords_norm[:, 0] * float(width)
        coords[:, 1] = coords_norm[:, 1] * float(height)
        vis = np.full((21,), DEFAULT_VISIBILITY, dtype=np.int32)
        return {
            "coords": coords.astype(np.float32),
            "vis": vis,
            "source": "mediapipe",
        }

    def ensure_state(self, index: int):
        key = self.image_paths[index].name
        state = self.states.get(key)
        if state is not None:
            return state

        state = self._build_state_from_existing(key)
        if state is None:
            state = self._build_state_from_mediapipe(self.image_paths[index])
        self.states[key] = state
        return state

    def current_path(self):
        return self.image_paths[self.current_index]

    def current_state(self):
        return self.ensure_state(self.current_index)

    def write_payload(self):
        payload = {
            "images": list(self.image_records),
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "hand",
                    "supercategory": "person",
                    "keypoints": list(KEYPOINT_NAMES),
                    "skeleton": [[a + 1, b + 1] for a, b in HAND_BONE_EDGES_21],
                }
            ],
        }

        for record in self.image_records:
            state = self.states.get(record["file_name"])
            if state is None:
                continue
            vis = np.asarray(state["vis"], dtype=np.int32)
            coords = np.asarray(state["coords"], dtype=np.float32)
            payload["annotations"].append(
                {
                    "id": int(record["id"]),
                    "image_id": int(record["id"]),
                    "category_id": 1,
                    "iscrowd": 0,
                    "num_keypoints": int((vis > 0).sum()),
                    "keypoints": encode_keypoints(coords, vis),
                }
            )

        with open(self.annotation_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def advance_joint(self, delta: int):
        self.current_joint = min(max(self.current_joint + int(delta), 0), 20)

    def move_image(self, delta: int):
        self.write_payload()
        self.current_index = min(max(self.current_index + int(delta), 0), len(self.image_paths) - 1)
        self.current_joint = 0
        self.render()

    def set_current_point(self, x: float, y: float, *, visibility: int = DEFAULT_VISIBILITY):
        state = self.current_state()
        width = float(self.image_records[self.current_index]["width"])
        height = float(self.image_records[self.current_index]["height"])
        state["coords"][self.current_joint, 0] = np.clip(float(x), 0.0, max(width - 1.0, 0.0))
        state["coords"][self.current_joint, 1] = np.clip(float(y), 0.0, max(height - 1.0, 0.0))
        state["vis"][self.current_joint] = int(visibility)
        self.write_payload()

    def hide_current_point(self):
        state = self.current_state()
        state["vis"][self.current_joint] = 0
        self.write_payload()

    def reset_current_from_mediapipe(self):
        image_path = self.current_path()
        self.states[image_path.name] = self._build_state_from_mediapipe(image_path)
        self.current_joint = 0
        self.write_payload()
        self.render()

    def nudge_current_point(self, dx: float, dy: float):
        state = self.current_state()
        x, y = state["coords"][self.current_joint].tolist()
        self.set_current_point(x + dx, y + dy, visibility=max(int(state["vis"][self.current_joint]), DEFAULT_VISIBILITY))
        self.render()

    def render(self):
        path = self.current_path()
        state = self.current_state()
        rgb = load_rgb_image(path)
        coords = np.asarray(state["coords"], dtype=np.float32)
        vis = np.asarray(state["vis"], dtype=np.int32)

        self.ax.clear()
        self.ax.imshow(rgb)
        self.ax.axis("off")

        for a, b in HAND_BONE_EDGES_21:
            if vis[a] > 0 and vis[b] > 0:
                self.ax.plot(
                    [coords[a, 0], coords[b, 0]],
                    [coords[a, 1], coords[b, 1]],
                    color="cyan",
                    linewidth=1.0,
                    alpha=0.8,
                )

        for idx, (name, xy, v) in enumerate(zip(KEYPOINT_NAMES, coords, vis)):
            color = "lime" if v > 0 else "gray"
            size = 45
            edge = "black"
            alpha = 1.0 if v > 0 else 0.35
            if idx == self.current_joint:
                color = "red" if v > 0 else "orange"
                size = 80
                edge = "yellow"
                alpha = 1.0
            self.ax.scatter([xy[0]], [xy[1]], s=size, c=color, edgecolors=edge, linewidths=1.2, alpha=alpha)
            self.ax.text(
                float(xy[0]) + 4.0,
                float(xy[1]) + 4.0,
                str(idx),
                color="white" if idx == self.current_joint else "black",
                fontsize=8,
                bbox={"facecolor": "black" if idx == self.current_joint else "white", "alpha": 0.55, "pad": 1},
            )

        labeled = int((vis > 0).sum())
        current_xy = coords[self.current_joint]
        current_vis = int(vis[self.current_joint])
        status = "\n".join(
            [
                f"Image {self.current_index + 1}/{len(self.image_paths)}: {path.name}",
                f"Source: {state.get('source', '')}  |  labeled joints: {labeled}/21",
                (
                    f"Joint {self.current_joint}: {KEYPOINT_NAMES[self.current_joint]}  |  "
                    f"x={current_xy[0]:.1f}, y={current_xy[1]:.1f}, vis={current_vis}"
                ),
            ]
        )
        self.ax.text(
            0.01,
            0.99,
            status,
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 6},
        )
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            self.set_current_point(event.xdata, event.ydata, visibility=DEFAULT_VISIBILITY)
            self.advance_joint(1)
            self.render()
            return

        if event.button == 3:
            self.hide_current_point()
            self.advance_joint(1)
            self.render()

    def on_key(self, event):
        key = str(event.key or "").lower()
        if key in {"q"}:
            self.write_payload()
            self.runner.close()
            self.plt.close(self.fig)
            return
        if key in {"s"}:
            self.write_payload()
            print(f"saved: {self.annotation_path}")
            return
        if key in {"d", "n", "pagedown", "enter"}:
            self.move_image(+1)
            return
        if key in {"a", "b", "pageup", "backspace"}:
            self.move_image(-1)
            return
        if key in {"]", "."}:
            self.advance_joint(+1)
            self.render()
            return
        if key in {"[", ","}:
            self.advance_joint(-1)
            self.render()
            return
        if key in {"0"}:
            self.hide_current_point()
            self.advance_joint(+1)
            self.render()
            return
        if key in {"2", "v"}:
            state = self.current_state()
            x, y = state["coords"][self.current_joint].tolist()
            self.set_current_point(x, y, visibility=DEFAULT_VISIBILITY)
            self.render()
            return
        if key in {"r", "m"}:
            self.reset_current_from_mediapipe()
            return
        if key == "left":
            self.nudge_current_point(-1.0, 0.0)
            return
        if key == "right":
            self.nudge_current_point(+1.0, 0.0)
            return
        if key == "up":
            self.nudge_current_point(0.0, -1.0)
            return
        if key == "down":
            self.nudge_current_point(0.0, +1.0)
            return

    def on_close(self, _event):
        self.write_payload()
        self.runner.close()

    def run(self):
        self.render()
        self.plt.show()


def main():
    args = parse_args()
    app = AnnotatorApp(args)
    print(f"images: {len(app.image_paths)}")
    print(f"dataset root: {app.dataset_root}")
    print(f"annotation json: {app.annotation_path}")
    app.run()


if __name__ == "__main__":
    main()
