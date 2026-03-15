import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Missing dependency: opencv-python. Install it with `pip install opencv-python`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from handpose.checkpoints import infer_checkpoint_keypoint_indices, load_checkpoint
from handpose.inference.predict import build_fusion_context, infer_fused_coords
from handpose.models.hand_pose_model import HandPoseNet


INPUT_SIZE = 256
WINDOW_NAME = "DRHand Webcam Demo"


@dataclass
class DemoSession:
    model: HandPoseNet
    device: torch.device
    keypoint_indices: list[int]
    fusion_context: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal webcam demo for DRHand checkpoints."
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint (.pt), typically a stage-2 best.pt file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. Default: auto.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0.",
    )
    parser.add_argument(
        "--roi-fraction",
        type=float,
        default=0.6,
        help="Center square ROI side length as a fraction of min(frame_h, frame_w).",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the preview horizontally before inference.",
    )
    parser.add_argument(
        "--show-roi",
        action="store_true",
        help="Draw the center ROI box used for inference.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def load_demo_session(ckpt_path: str, device: torch.device):
    ckpt_meta, state_dict = load_checkpoint(ckpt_path, device)
    keypoint_indices = infer_checkpoint_keypoint_indices(ckpt_meta)
    model = HandPoseNet(num_keypoints=len(keypoint_indices)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    fusion_context = build_fusion_context(model_keypoint_indices=keypoint_indices)
    return DemoSession(
        model=model,
        device=device,
        keypoint_indices=keypoint_indices,
        fusion_context=fusion_context,
    )


def compute_center_roi(frame_shape, roi_fraction: float):
    frame_h, frame_w = frame_shape[:2]
    side = int(min(frame_h, frame_w) * roi_fraction)
    side = max(32, min(side, frame_h, frame_w))
    x0 = max((frame_w - side) // 2, 0)
    y0 = max((frame_h - side) // 2, 0)
    return x0, y0, side


def preprocess_roi(roi_bgr, device: torch.device):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_rgb = cv2.resize(roi_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(roi_rgb).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(device)
    return x


def draw_points(frame, coords_px):
    for x, y in coords_px:
        cv2.circle(frame, (int(round(x)), int(round(y))), 4, (0, 0, 255), thickness=-1)


def draw_status(frame, fps: float, keypoint_count: int, device: torch.device):
    status = f"fps: {fps:5.1f} | keypoints: {keypoint_count} | device: {device}"
    cv2.putText(
        frame,
        status,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "q: quit",
        (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def run_demo(args):
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    session = load_demo_session(ckpt_path=args.ckpt, device=device)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    fps = 0.0
    last_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            if args.mirror:
                frame = cv2.flip(frame, 1)

            x0, y0, side = compute_center_roi(frame.shape, args.roi_fraction)
            roi = frame[y0 : y0 + side, x0 : x0 + side]
            x = preprocess_roi(roi_bgr=roi, device=device)

            pred_norm = infer_fused_coords(
                model=session.model,
                x=x,
                fusion_context=session.fusion_context,
            )

            pred_px = pred_norm.clone()
            pred_px[:, 0] = pred_px[:, 0] * float(side) + float(x0)
            pred_px[:, 1] = pred_px[:, 1] * float(side) + float(y0)

            if args.show_roi:
                cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
            draw_points(frame, pred_px.tolist())

            now = time.perf_counter()
            frame_time = max(now - last_time, 1e-6)
            fps = 1.0 / frame_time
            last_time = now
            draw_status(
                frame,
                fps=fps,
                keypoint_count=len(session.keypoint_indices),
                device=device,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    if not (0.0 < args.roi_fraction <= 1.0):
        raise ValueError("--roi-fraction must be in the range (0, 1]")
    run_demo(args)


if __name__ == "__main__":
    main()
