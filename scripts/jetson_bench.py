import argparse
import csv
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    """Parse CLI args for the Jetson benchmark driver."""
    p = argparse.ArgumentParser(
        description=(
            "Run cold-start inference and eval_metrics benchmarks on Jetson, "
            "capture tegrastats, and write a summary CSV."
        )
    )
    p.add_argument(
        "--cases-json",
        required=True,
        help="Path to a JSON file containing benchmark cases.",
    )
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the RHD dataset root.",
    )
    p.add_argument(
        "--image",
        required=True,
        help="Path to one image used for cold-start predict_image testing.",
    )
    p.add_argument(
        "--results-root",
        default="jetson_results",
        help="Directory where timestamped benchmark results will be written.",
    )
    p.add_argument(
        "--python-bin",
        default=sys.executable or "python",
        help="Python executable used to launch predict_image.py and eval_metrics.py.",
    )
    p.add_argument(
        "--split",
        default="evaluation",
        choices=["training", "evaluation"],
        help="Dataset split used for eval_metrics.py.",
    )
    p.add_argument(
        "--hand",
        default="right",
        choices=["left", "right", "auto"],
        help="Hand selection used for eval_metrics.py.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cpu", "cuda"],
        help="Device passed to eval_metrics.py.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size passed to eval_metrics.py.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="num_workers passed to eval_metrics.py.",
    )
    p.add_argument(
        "--power-mode",
        type=int,
        default=0,
        help="nvpmodel mode used before benchmarking (default: 0).",
    )
    p.add_argument(
        "--tegrastats-interval-ms",
        type=int,
        default=1000,
        help="Sampling interval passed to tegrastats.",
    )
    p.add_argument(
        "--skip-power-setup",
        action="store_true",
        help="Skip running nvpmodel and jetson_clocks.",
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to later cases when one case fails.",
    )
    return p.parse_args()


def load_cases(path: Path):
    """Load benchmark cases from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("--cases-json must contain a non-empty JSON list")

    cases = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"case {i} must be a JSON object")
        name = item.get("name")
        ckpt = item.get("ckpt")
        eval_args = item.get("eval_args", [])
        if not name or not isinstance(name, str):
            raise ValueError(f"case {i} is missing string field 'name'")
        if not ckpt or not isinstance(ckpt, str):
            raise ValueError(f"case {i} is missing string field 'ckpt'")
        if not isinstance(eval_args, list) or not all(isinstance(x, str) for x in eval_args):
            raise ValueError(f"case {i} field 'eval_args' must be a list of strings")
        cases.append(
            {
                "name": name,
                "ckpt": ckpt,
                "eval_args": list(eval_args),
            }
        )
    return cases


def run_subprocess(cmd, cwd: Path, stdout_path: Path, stderr_path: Path):
    """Run one subprocess, log stdout/stderr to files, and return elapsed seconds."""
    t0 = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=out_f,
            stderr=err_f,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - t0
    return completed.returncode, elapsed


def try_power_setup(skip: bool, power_mode: int):
    """Best-effort Jetson power setup."""
    if skip:
        return []

    outcomes = []
    for cmd in (
        ["sudo", "-n", "nvpmodel", "-m", str(power_mode)],
        ["sudo", "-n", "jetson_clocks"],
    ):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            outcomes.append({"cmd": cmd, "status": "ok"})
        except Exception as exc:
            outcomes.append({"cmd": cmd, "status": "skipped", "error": str(exc)})
    return outcomes


def start_tegrastats(log_path: Path, interval_ms: int):
    """Start tegrastats logging if available."""
    try:
        log_f = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc, log_f
    except Exception as exc:
        log_path.write_text(f"failed to start tegrastats: {exc}\n", encoding="utf-8")
        return None, None


def stop_tegrastats(proc, log_f):
    """Stop tegrastats logging cleanly."""
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    if log_f is not None:
        log_f.close()


def build_predict_cmd(args, repo_root: Path, ckpt: str):
    """Build cold-start predict_image.py command."""
    return [
        args.python_bin,
        str(repo_root / "scripts" / "predict_image.py"),
        "--ckpt",
        ckpt,
        "--image",
        str(Path(args.image)),
    ]


def build_eval_cmd(args, repo_root: Path, ckpt: str, out_json: Path, extra_args):
    """Build eval_metrics.py command."""
    return [
        args.python_bin,
        str(repo_root / "scripts" / "eval_metrics.py"),
        "--ckpt",
        ckpt,
        "--root",
        str(Path(args.dataset_root)),
        "--split",
        args.split,
        "--hand",
        args.hand,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--out-json",
        str(out_json),
        *extra_args,
    ]


def load_eval_metrics(path: Path):
    """Load key metrics from a saved eval_metrics JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    timing = metrics.get("timing", {})
    return {
        "sse_norm": metrics.get("sse_norm"),
        "epe_norm": metrics.get("epe_norm"),
        "pck": metrics.get("pck"),
        "ms_per_image": timing.get("ms_per_image"),
        "images_per_second": timing.get("images_per_second"),
    }


def write_session_metadata(path: Path, args, repo_root: Path, cases, power_setup):
    """Persist one JSON file describing the benchmark session."""
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "repo_root": str(repo_root),
        "dataset_root": str(Path(args.dataset_root)),
        "image": str(Path(args.image)),
        "python_bin": args.python_bin,
        "split": args.split,
        "hand": args.hand,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "tegrastats_interval_ms": args.tegrastats_interval_ms,
        "power_setup": power_setup,
        "cases": cases,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_summary_row(writer, row):
    """Write one row to the summary CSV."""
    writer.writerow(row)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cases_path = Path(args.cases_json).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cases_path)
    power_setup = try_power_setup(skip=bool(args.skip_power_setup), power_mode=int(args.power_mode))

    session_dir = results_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    write_session_metadata(session_dir / "session.json", args=args, repo_root=repo_root, cases=cases, power_setup=power_setup)

    summary_path = session_dir / "summary.csv"
    fieldnames = [
        "name",
        "ckpt",
        "eval_args",
        "status",
        "cold_start_seconds",
        "eval_wall_seconds",
        "sse_norm",
        "epe_norm",
        "pck",
        "ms_per_image",
        "images_per_second",
        "cold_start_stdout",
        "cold_start_stderr",
        "eval_json",
        "eval_stdout",
        "eval_stderr",
        "tegrastats_log",
        "error",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            case_dir = session_dir / case["name"]
            case_dir.mkdir(parents=True, exist_ok=True)

            tegrastats_log = case_dir / "tegrastats.log"
            tegrastats_proc, tegrastats_file = start_tegrastats(
                log_path=tegrastats_log,
                interval_ms=int(args.tegrastats_interval_ms),
            )

            cold_stdout = case_dir / "cold_start_stdout.json"
            cold_stderr = case_dir / "cold_start_stderr.txt"
            eval_json = case_dir / "eval.json"
            eval_stdout = case_dir / "eval_stdout.txt"
            eval_stderr = case_dir / "eval_stderr.txt"

            row = {
                "name": case["name"],
                "ckpt": case["ckpt"],
                "eval_args": " ".join(case["eval_args"]),
                "status": "ok",
                "cold_start_seconds": "",
                "eval_wall_seconds": "",
                "sse_norm": "",
                "epe_norm": "",
                "pck": "",
                "ms_per_image": "",
                "images_per_second": "",
                "cold_start_stdout": str(cold_stdout),
                "cold_start_stderr": str(cold_stderr),
                "eval_json": str(eval_json),
                "eval_stdout": str(eval_stdout),
                "eval_stderr": str(eval_stderr),
                "tegrastats_log": str(tegrastats_log),
                "error": "",
            }

            try:
                predict_cmd = build_predict_cmd(args=args, repo_root=repo_root, ckpt=case["ckpt"])
                cold_rc, cold_secs = run_subprocess(
                    cmd=predict_cmd,
                    cwd=repo_root,
                    stdout_path=cold_stdout,
                    stderr_path=cold_stderr,
                )
                row["cold_start_seconds"] = f"{cold_secs:.6f}"
                if cold_rc != 0:
                    raise RuntimeError(f"cold-start command failed with exit code {cold_rc}")

                eval_cmd = build_eval_cmd(
                    args=args,
                    repo_root=repo_root,
                    ckpt=case["ckpt"],
                    out_json=eval_json,
                    extra_args=case["eval_args"],
                )
                eval_rc, eval_secs = run_subprocess(
                    cmd=eval_cmd,
                    cwd=repo_root,
                    stdout_path=eval_stdout,
                    stderr_path=eval_stderr,
                )
                row["eval_wall_seconds"] = f"{eval_secs:.6f}"
                if eval_rc != 0:
                    raise RuntimeError(f"eval command failed with exit code {eval_rc}")
                if not eval_json.exists():
                    raise FileNotFoundError(f"expected eval output JSON not found: {eval_json}")

                metrics = load_eval_metrics(eval_json)
                row["sse_norm"] = metrics["sse_norm"]
                row["epe_norm"] = metrics["epe_norm"]
                row["pck"] = metrics["pck"]
                row["ms_per_image"] = metrics["ms_per_image"]
                row["images_per_second"] = metrics["images_per_second"]

            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                append_summary_row(writer, row)
                csv_f.flush()
                stop_tegrastats(tegrastats_proc, tegrastats_file)
                if not args.keep_going:
                    raise
                continue

            append_summary_row(writer, row)
            csv_f.flush()
            stop_tegrastats(tegrastats_proc, tegrastats_file)

    print(f"done: {session_dir}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
