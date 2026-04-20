import argparse
import csv
import json
from pathlib import Path


DEFAULT_SEEDS = [101, 202, 303, 404, 505]
DEFAULT_PREDICTION_MODES = ["fusion", "heatmap", "coord"]
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
HOME_ROOT = REPO_ROOT.parent
DEFAULT_DATA_ROOT = HOME_ROOT / "data" / "RHD_published_v2"
DEFAULT_SHARED_ROOT = HOME_ROOT / "sharedscratch"
DEFAULT_CHECKPOINT_ROOT = DEFAULT_SHARED_ROOT / "training_results"
DEFAULT_EVAL_ROOT = DEFAULT_SHARED_ROOT / "eval_results"
DEFAULT_BENCHMARK_ROOT = DEFAULT_SHARED_ROOT / "benchmark_results"
DEFAULT_BENCHMARK_SUMMARY_CSV = DEFAULT_BENCHMARK_ROOT / "jetson_orin_nano_maxn_summary.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "ablation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the DRHand experiment matrix and runnable command files."
    )
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--benchmark-summary-csv", default=str(DEFAULT_BENCHMARK_SUMMARY_CSV))
    parser.add_argument("--benchmark-output-root", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def base_rows():
    return [
        {"experiment_id": "B0", "num_keypoints": 21, "lambda_hm": 1.0, "lambda_coord": 1.0, "heatmap_sigma": 2.0, "wing_w": 10.0, "wing_epsilon": 2.0},
        {"experiment_id": "B1", "num_keypoints": 10, "lambda_hm": 1.0, "lambda_coord": 1.0, "heatmap_sigma": 2.0, "wing_w": 10.0, "wing_epsilon": 2.0},
        {"experiment_id": "L1", "num_keypoints": 21, "lambda_hm": 1.5, "lambda_coord": 0.5, "heatmap_sigma": 2.0, "wing_w": 10.0, "wing_epsilon": 2.0},
        {"experiment_id": "L2", "num_keypoints": 21, "lambda_hm": 0.5, "lambda_coord": 1.5, "heatmap_sigma": 2.0, "wing_w": 10.0, "wing_epsilon": 2.0},
        {"experiment_id": "S1", "num_keypoints": 21, "lambda_hm": 1.0, "lambda_coord": 1.0, "heatmap_sigma": 1.5, "wing_w": 10.0, "wing_epsilon": 2.0},
        {"experiment_id": "S2", "num_keypoints": 21, "lambda_hm": 1.0, "lambda_coord": 1.0, "heatmap_sigma": 3.0, "wing_w": 10.0, "wing_epsilon": 2.0},
        {"experiment_id": "W1", "num_keypoints": 21, "lambda_hm": 1.0, "lambda_coord": 1.0, "heatmap_sigma": 2.0, "wing_w": 5.0, "wing_epsilon": 2.0},
        {"experiment_id": "W2", "num_keypoints": 21, "lambda_hm": 1.0, "lambda_coord": 1.0, "heatmap_sigma": 2.0, "wing_w": 15.0, "wing_epsilon": 2.0},
        {"experiment_id": "E1", "num_keypoints": 21, "lambda_hm": 1.0, "lambda_coord": 1.0, "heatmap_sigma": 2.0, "wing_w": 10.0, "wing_epsilon": 1.0},
        {"experiment_id": "E2", "num_keypoints": 21, "lambda_hm": 1.0, "lambda_coord": 1.0, "heatmap_sigma": 2.0, "wing_w": 10.0, "wing_epsilon": 4.0},
    ]


def build_runs(seeds):
    runs = []
    for row in base_rows():
        for seed in seeds:
            tips_bases_only = row["num_keypoints"] == 10
            job_id = f"drh-{row['experiment_id'].lower()}-k{row['num_keypoints']}-right-s{seed}"
            runs.append(
                {
                    **row,
                    "seed": int(seed),
                    "job_id": job_id,
                    "hand": "right",
                    "tips_bases_only": tips_bases_only,
                    "prediction_modes": list(DEFAULT_PREDICTION_MODES),
                }
            )
    return runs


def quote(value):
    text = str(value)
    if " " in text:
        return f"\"{text}\""
    return text


def build_train_command(run, args):
    parts = [
        "python scripts/train.py",
        f"--root {quote(args.root)}",
        f"--checkpoint-root {quote(args.checkpoint_root)}",
        f"--job-id {quote(run['job_id'])}",
        f"--seed {run['seed']}",
        "--input-size 256",
        "--batch-size-stage1 64",
        "--batch-size-stage2 64",
        "--accum-steps-stage1 1",
        "--accum-steps-stage2 4",
        "--stage1-epochs 100",
        "--stage2-epochs 100",
        "--lr-stage1 1e-3",
        "--lr-stage2 1e-4",
        "--hand right",
        f"--lambda-hm {run['lambda_hm']}",
        f"--lambda-coord {run['lambda_coord']}",
        f"--heatmap-sigma {run['heatmap_sigma']}",
        f"--wing-w {run['wing_w']}",
        f"--wing-epsilon {run['wing_epsilon']}",
    ]
    if run["tips_bases_only"]:
        parts.append("--tips-bases-only")
    return " ".join(parts)


def build_eval_commands(run, args):
    ckpt = f"{args.checkpoint_root}/{run['job_id']}/checkpoints/best.pt"
    commands = []
    for prediction_mode in run["prediction_modes"]:
        out_json = f"{args.eval_root}/{run['job_id']}.{prediction_mode}.native.json"
        commands.append(
            " ".join(
                [
                    "python scripts/eval_metrics.py",
                    f"--ckpt {quote(ckpt)}",
                    f"--root {quote(args.root)}",
                    "--split evaluation",
                    "--hand right",
                    f"--prediction-mode {prediction_mode}",
                    f"--out-json {quote(out_json)}",
                ]
            )
        )

    shared10_out = f"{args.eval_root}/{run['job_id']}.fusion.shared10.json"
    commands.append(
        " ".join(
            [
                "python scripts/eval_metrics.py",
                f"--ckpt {quote(ckpt)}",
                f"--root {quote(args.root)}",
                "--split evaluation",
                "--hand right",
                "--prediction-mode fusion",
                "--shared-10-eval",
                "--out-json " + quote(shared10_out),
            ]
        )
    )

    fusion_diag_out = f"{args.eval_root}/{run['job_id']}.fusion.native_diag.json"
    commands.append(
        " ".join(
            [
                "python scripts/eval_metrics.py",
                f"--ckpt {quote(ckpt)}",
                f"--root {quote(args.root)}",
                "--split evaluation",
                "--hand right",
                "--prediction-mode fusion",
                "--with-fusion-diagnostics",
                f"--out-json {quote(fusion_diag_out)}",
            ]
        )
    )
    return commands


def build_benchmark_commands(run, args):
    ckpt = f"{args.checkpoint_root}/{run['job_id']}/checkpoints/best.pt"
    commands = []
    for prediction_mode in run["prediction_modes"]:
        commands.append(
            " ".join(
                [
                    "python scripts/benchmark_pipeline.py",
                    f"--ckpt {quote(ckpt)}",
                    f"--root {quote(args.root)}",
                    f"--summary-csv {quote(args.benchmark_summary_csv)}",
                    f"--output-root {quote(args.benchmark_output_root)}",
                    "--device cuda",
                    f"--prediction-mode {prediction_mode}",
                    f"--run-label {quote(run['job_id'] + '-' + prediction_mode)}",
                ]
            )
        )
    return commands


def write_text(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = build_runs(args.seeds)
    (out_dir / "experiment_matrix.json").write_text(
        json.dumps(runs, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "experiment_matrix.csv", runs)

    train_commands = [build_train_command(run, args) for run in runs]
    eval_commands = []
    benchmark_commands = []
    for run in runs:
        eval_commands.extend(build_eval_commands(run, args))
        benchmark_commands.extend(build_benchmark_commands(run, args))

    write_text(out_dir / "train_commands.txt", train_commands)
    write_text(out_dir / "eval_commands.txt", eval_commands)
    write_text(out_dir / "benchmark_commands.txt", benchmark_commands)


if __name__ == "__main__":
    main()
