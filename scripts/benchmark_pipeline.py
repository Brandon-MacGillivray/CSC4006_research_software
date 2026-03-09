import argparse
import csv
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import threading
import time

import numpy as np
from PIL import Image
import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from handpose.checkpoints import infer_checkpoint_keypoint_indices, load_checkpoint
from handpose.inference.predict import build_fusion_context, infer_fused_coords
from handpose.models.hand_pose_model import HandPoseNet


INPUT_SIZE = 256
DEFAULT_TEGRASTATS_INTERVAL_MS = 200

# One summary CSV row is written for each benchmark run.
SUMMARY_COLUMNS = [
    "timestamp_utc",
    "run_label",
    "checkpoint_path",
    "checkpoint_stage",
    "checkpoint_epoch",
    "num_keypoints",
    "model_keypoint_indices",
    "device",
    "device_name",
    "jetson_model",
    "jetpack_version",
    "nvpmodel_mode",
    "jetson_clocks",
    "python_version",
    "torch_version",
    "dataset_root",
    "dataset_split",
    "image_dir",
    "num_images",
    "session_setup_ms",
    "image_read_ms_mean",
    "image_read_ms_median",
    "image_read_ms_p95",
    "preprocess_ms_mean",
    "preprocess_ms_median",
    "preprocess_ms_p95",
    "host_to_device_ms_mean",
    "host_to_device_ms_median",
    "host_to_device_ms_p95",
    "forward_fusion_ms_mean",
    "forward_fusion_ms_median",
    "forward_fusion_ms_p95",
    "write_json_ms_mean",
    "write_json_ms_median",
    "write_json_ms_p95",
    "total_e2e_ms_mean",
    "total_e2e_ms_median",
    "total_e2e_ms_p95",
    "proc_cpu_pct_mean",
    "proc_cpu_pct_max",
    "proc_rss_mb_peak",
    "torch_cuda_allocated_mb_peak",
    "torch_cuda_reserved_mb_peak",
    "tegrastats_ram_mb_peak",
    "tegrastats_gr3d_pct_mean",
    "tegrastats_gr3d_pct_max",
    "tegrastats_temp_c_peak",
    "tegrastats_power_mw_mean",
    "tegrastats_power_mw_peak",
    "failures",
]

TIMING_FIELDS = [
    "image_read_ms",
    "preprocess_ms",
    "host_to_device_ms",
    "forward_fusion_ms",
    "write_json_ms",
    "total_e2e_ms",
]

RESOURCE_FIELDS = [
    "proc_cpu_pct_mean",
    "proc_cpu_pct_max",
    "proc_rss_mb_peak",
    "torch_cuda_allocated_mb_peak",
    "torch_cuda_reserved_mb_peak",
    "tegrastats_ram_mb_peak",
    "tegrastats_gr3d_pct_mean",
    "tegrastats_gr3d_pct_max",
    "tegrastats_temp_c_peak",
    "tegrastats_power_mw_mean",
    "tegrastats_power_mw_peak",
]


class BenchmarkSession:
    def __init__(self, ckpt_path: Path, device: torch.device, ckpt_meta: dict, model, keypoint_indices, fusion_context):
        self.ckpt_path = ckpt_path
        self.device = device
        self.ckpt_meta = ckpt_meta
        self.model = model
        self.keypoint_indices = keypoint_indices
        self.fusion_context = fusion_context


class ProcessSampler:
    # Lightweight process-level sampling for CPU% and resident memory.
    def __init__(self, interval_s: float):
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread = None
        self._process = None
        self._cpu_values = []
        self._rss_values_mb = []

    def start(self):
        if psutil is None:
            return
        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                cpu = float(self._process.cpu_percent(interval=None))
                rss_mb = float(self._process.memory_info().rss) / (1024.0 * 1024.0)
                self._cpu_values.append(cpu)
                self._rss_values_mb.append(rss_mb)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(self.interval_s * 2.0, 1.0))

    def summary(self):
        return {
            "proc_cpu_pct_mean": mean_or_blank(self._cpu_values),
            "proc_cpu_pct_max": max_or_blank(self._cpu_values),
            "proc_rss_mb_peak": max_or_blank(self._rss_values_mb),
        }


class TegraStatsSampler:
    # Jetson-wide sampling is optional; blank fields are expected off-device.
    def __init__(self, interval_ms: int):
        self.interval_ms = int(interval_ms)
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._samples = []
        self.available = shutil.which("tegrastats") is not None

    def start(self):
        if not self.available:
            return
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                sample = parse_tegrastats_line(line)
                if sample:
                    self._samples.append(sample)
        except Exception:
            pass

    def stop(self):
        if self._proc is None:
            return
        self._stop.set()
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self):
        ram_peaks = [sample["ram_used_mb"] for sample in self._samples if sample.get("ram_used_mb") is not None]
        gr3d_values = [sample["gr3d_pct"] for sample in self._samples if sample.get("gr3d_pct") is not None]
        temp_values = [sample["temp_c_max"] for sample in self._samples if sample.get("temp_c_max") is not None]
        power_values = [sample["power_mw"] for sample in self._samples if sample.get("power_mw") is not None]
        return {
            "tegrastats_ram_mb_peak": max_or_blank(ram_peaks),
            "tegrastats_gr3d_pct_mean": mean_or_blank(gr3d_values),
            "tegrastats_gr3d_pct_max": max_or_blank(gr3d_values),
            "tegrastats_temp_c_peak": max_or_blank(temp_values),
            "tegrastats_power_mw_mean": mean_or_blank(power_values),
            "tegrastats_power_mw_peak": max_or_blank(power_values),
        }


def parse_tegrastats_line(line: str):
    sample = {}

    ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if ram_match:
        sample["ram_used_mb"] = float(ram_match.group(1))

    gr3d_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    if gr3d_match:
        sample["gr3d_pct"] = float(gr3d_match.group(1))

    power_match = re.search(r"VDD_IN\s+(\d+)mW", line)
    if power_match:
        sample["power_mw"] = float(power_match.group(1))

    temps = [float(value) for _, value in re.findall(r"([A-Za-z0-9_]+)@([0-9]+(?:\.[0-9]+)?)C", line)]
    if temps:
        sample["temp_c_max"] = max(temps)

    return sample or None


def checkpoint_label(ckpt_path: Path):
    if ckpt_path.parent.name == "checkpoints" and ckpt_path.parent.parent.name:
        return ckpt_path.parent.parent.name
    return ckpt_path.stem


def percentile(values, pct: float):
    if not values:
        return ""
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean_or_blank(values):
    if not values:
        return ""
    return float(statistics.fmean(float(v) for v in values))


def max_or_blank(values):
    if not values:
        return ""
    return float(max(float(v) for v in values))


def timestamp_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def discover_images(root: Path):
    # Benchmark against the full evaluation split so runs are comparable.
    split = "evaluation"
    image_dir = root / split / "color"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise ValueError(f"No images found under {image_dir}")
    return split, image_dir, images


def collect_runtime_metadata(device: torch.device):
    metadata = {
        "device": str(device),
        "device_name": "",
        "jetson_model": "",
        "jetpack_version": "",
        "nvpmodel_mode": "",
        "jetson_clocks": "",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        metadata["device_name"] = torch.cuda.get_device_name(device)

    device_tree_model = Path("/proc/device-tree/model")
    if device_tree_model.exists():
        try:
            metadata["jetson_model"] = device_tree_model.read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
        except Exception:
            pass

    nv_tegra_release = Path("/etc/nv_tegra_release")
    if nv_tegra_release.exists():
        try:
            metadata["jetpack_version"] = nv_tegra_release.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
        except Exception:
            pass

    metadata["nvpmodel_mode"] = run_command_output(["nvpmodel", "-q"])
    metadata["jetson_clocks"] = run_command_output(["jetson_clocks", "--show"])
    return metadata


def run_command_output(command):
    if shutil.which(command[0]) is None:
        return ""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=5.0,
        )
    except Exception:
        return ""
    text = (completed.stdout or completed.stderr or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def reset_cuda_memory_peaks(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def collect_cuda_memory_peaks(device: torch.device):
    if device.type != "cuda":
        return {
            "torch_cuda_allocated_mb_peak": "",
            "torch_cuda_reserved_mb_peak": "",
        }
    return {
        "torch_cuda_allocated_mb_peak": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
        "torch_cuda_reserved_mb_peak": float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0),
    }


def load_benchmark_session(ckpt_path: Path, device: torch.device):
    # Load the checkpoint once and reuse the same model for the whole run.
    ckpt_meta, state_dict = load_checkpoint(str(ckpt_path), device)
    keypoint_indices = infer_checkpoint_keypoint_indices(ckpt_meta)
    model = HandPoseNet(num_keypoints=len(keypoint_indices)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    fusion_context = build_fusion_context(model_keypoint_indices=keypoint_indices)
    return BenchmarkSession(
        ckpt_path=ckpt_path,
        device=device,
        ckpt_meta=ckpt_meta,
        model=model,
        keypoint_indices=keypoint_indices,
        fusion_context=fusion_context,
    )


def benchmark_image(session: BenchmarkSession, image_path: Path, output_dir: Path):
    timings = {}
    t_total = time.perf_counter()

    # Stage 1: read the source image from disk.
    t0 = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    timings["image_read_ms"] = (time.perf_counter() - t0) * 1000.0

    # Stage 2: resize and convert to a normalized CPU tensor.
    t0 = time.perf_counter()
    resized = image.resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(resized)
    x_cpu = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    x_cpu = x_cpu.unsqueeze(0)
    timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    # Stage 3: move the tensor onto the execution device.
    maybe_sync(session.device)
    t0 = time.perf_counter()
    x = x_cpu.to(session.device)
    maybe_sync(session.device)
    timings["host_to_device_ms"] = (time.perf_counter() - t0) * 1000.0

    # Stage 4: run inference and the existing fusion rule together.
    maybe_sync(session.device)
    t0 = time.perf_counter()
    pred_norm = infer_fused_coords(model=session.model, x=x, fusion_context=session.fusion_context)
    maybe_sync(session.device)
    timings["forward_fusion_ms"] = (time.perf_counter() - t0) * 1000.0

    pred_px = pred_norm.clone()
    pred_px[:, 0] = pred_px[:, 0] * float(orig_w)
    pred_px[:, 1] = pred_px[:, 1] * float(orig_h)

    payload = {
        "checkpoint": str(session.ckpt_path),
        "image": str(image_path),
        "prediction_mode": "fusion",
        "model_keypoint_indices": session.keypoint_indices,
        "pred_coords_pixels_original": pred_px.tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "pred.json"

    # Stage 5: persist the final prediction payload for this image.
    t0 = time.perf_counter()
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    timings["write_json_ms"] = (time.perf_counter() - t0) * 1000.0

    timings["total_e2e_ms"] = (time.perf_counter() - t_total) * 1000.0
    return timings


def write_resolved_image_list(base_dir: Path, selected_images, image_dir: Path):
    # Save the exact file order used so the run is reproducible later.
    lines = [str(path.relative_to(image_dir.parent.parent)) for path in selected_images]
    resolved_path = base_dir / "resolved_images.txt"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_timings(rows):
    summary = {}
    for field in TIMING_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
        summary[f"{field}_mean"] = mean_or_blank(values)
        summary[f"{field}_median"] = percentile(values, 0.5)
        summary[f"{field}_p95"] = percentile(values, 0.95)
    return summary


def append_summary_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def build_summary_row(
    *,
    args,
    ckpt_path: Path,
    image_dir: Path,
    num_images: int,
    session_setup_ms,
    session: BenchmarkSession,
    timing_summary: dict,
    resource_summary: dict,
    runtime_metadata: dict,
    failures: int,
):
    row = {
        "timestamp_utc": timestamp_utc(),
        "run_label": args.run_label,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_stage": session.ckpt_meta.get("stage", ""),
        "checkpoint_epoch": session.ckpt_meta.get("epoch", ""),
        "num_keypoints": len(session.keypoint_indices),
        "model_keypoint_indices": json.dumps(session.keypoint_indices),
        "device": runtime_metadata.get("device", ""),
        "device_name": runtime_metadata.get("device_name", ""),
        "jetson_model": runtime_metadata.get("jetson_model", ""),
        "jetpack_version": runtime_metadata.get("jetpack_version", ""),
        "nvpmodel_mode": runtime_metadata.get("nvpmodel_mode", ""),
        "jetson_clocks": runtime_metadata.get("jetson_clocks", ""),
        "python_version": runtime_metadata.get("python_version", ""),
        "torch_version": runtime_metadata.get("torch_version", ""),
        "dataset_root": str(Path(args.root)),
        "dataset_split": "evaluation",
        "image_dir": str(image_dir),
        "num_images": num_images,
        "session_setup_ms": session_setup_ms,
        "failures": failures,
    }
    row.update(timing_summary)
    row.update(resource_summary)
    return row


def run_benchmark_for_checkpoint(args, ckpt_path: Path, selected_images, image_dir: Path, runtime_metadata: dict):
    device = resolve_device(args.device)

    # Keep model setup out of per-image timings and report it separately.
    t0 = time.perf_counter()
    session = load_benchmark_session(ckpt_path=ckpt_path, device=device)
    session_setup_ms = (time.perf_counter() - t0) * 1000.0

    base_dir = Path(args.output_root) / args.run_label / checkpoint_label(ckpt_path)
    write_resolved_image_list(base_dir=base_dir, selected_images=selected_images, image_dir=image_dir)

    # Resource sampling spans the image loop only, not checkpoint setup.
    reset_cuda_memory_peaks(device)
    process_sampler = ProcessSampler(interval_s=max(args.tegrastats_interval_ms / 1000.0, 0.05))
    tegra_sampler = TegraStatsSampler(interval_ms=args.tegrastats_interval_ms)
    process_sampler.start()
    tegra_sampler.start()

    rows = []
    failures = 0
    for image_path in selected_images:
        try:
            out_dir = base_dir / image_path.stem
            rows.append(benchmark_image(session=session, image_path=image_path, output_dir=out_dir))
        except Exception:
            failures += 1

    tegra_sampler.stop()
    process_sampler.stop()

    resource_summary = {
        **process_sampler.summary(),
        **collect_cuda_memory_peaks(device),
        **tegra_sampler.summary(),
    }
    timing_summary = summarize_timings(rows)
    return build_summary_row(
        args=args,
        ckpt_path=ckpt_path,
        image_dir=image_dir,
        num_images=len(rows),
        session_setup_ms=session_setup_ms,
        session=session,
        timing_summary=timing_summary,
        resource_summary=resource_summary,
        runtime_metadata=runtime_metadata,
        failures=failures,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Simple end-to-end latency benchmark for DRHand checkpoints.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (.pt).")
    parser.add_argument("--root", required=True, help="Dataset root containing the RHD split folders.")
    parser.add_argument("--summary-csv", required=True, help="Output CSV path for benchmark summaries.")
    parser.add_argument("--output-root", required=True, help="Directory for per-image JSON benchmark outputs.")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-label", default=time.strftime("benchmark_%Y%m%d_%H%M%S", time.localtime()))
    parser.add_argument("--tegrastats-interval-ms", type=int, default=DEFAULT_TEGRASTATS_INTERVAL_MS)
    return parser


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    # Discover the full evaluation set once and benchmark it in sorted order.
    _, image_dir, selected_images = discover_images(root=Path(args.root))
    runtime_metadata = collect_runtime_metadata(device)

    summary_row = run_benchmark_for_checkpoint(
        args=args,
        ckpt_path=Path(args.ckpt),
        selected_images=selected_images,
        image_dir=image_dir,
        runtime_metadata=runtime_metadata,
    )
    append_summary_row(Path(args.summary_csv), summary_row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
