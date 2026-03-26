import argparse
import csv
import json
from pathlib import Path


DEFAULT_SEEDS = [101, 202, 303, 404, 505]
DEFAULT_EVAL_DATASETS = ["rhd", "coco_hand"]
DEFAULT_EVAL_PREDICTION_MODES = ["fusion"]
DEFAULT_BENCHMARK_DATASETS = ["rhd", "coco_hand"]
DEFAULT_BENCHMARK_PREDICTION_MODES = ["fusion"]
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
HOME_ROOT = REPO_ROOT.parent
DEFAULT_RHD_ROOT = HOME_ROOT / "data" / "RHD_published_v2"
DEFAULT_COCO_ROOT = HOME_ROOT / "data" / "hand_keypoint_dataset"
DEFAULT_SHARED_ROOT = HOME_ROOT / "sharedscratch"
DEFAULT_CHECKPOINT_ROOT = DEFAULT_SHARED_ROOT / "transfer_training_results"
DEFAULT_EVAL_ROOT = DEFAULT_SHARED_ROOT / "transfer_eval_results"
DEFAULT_BENCHMARK_ROOT = DEFAULT_SHARED_ROOT / "transfer_benchmark_results"
DEFAULT_BENCHMARK_SUMMARY_CSV = DEFAULT_BENCHMARK_ROOT / "transfer_summary.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "experiment_plan_transfer"
SUPPORTED_DATASETS = ("rhd", "coco_hand")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate transfer-learning experiment matrices and command files."
    )
    parser.add_argument("--rhd-root", default=str(DEFAULT_RHD_ROOT))
    parser.add_argument("--coco-root", default=str(DEFAULT_COCO_ROOT))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--benchmark-summary-csv", default=str(DEFAULT_BENCHMARK_SUMMARY_CSV))
    parser.add_argument("--benchmark-output-root", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--eval-datasets",
        nargs="*",
        default=DEFAULT_EVAL_DATASETS,
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--eval-prediction-modes",
        nargs="*",
        default=DEFAULT_EVAL_PREDICTION_MODES,
    )
    parser.add_argument(
        "--benchmark-datasets",
        nargs="*",
        default=DEFAULT_BENCHMARK_DATASETS,
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--benchmark-prediction-modes",
        nargs="*",
        default=DEFAULT_BENCHMARK_PREDICTION_MODES,
    )
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size-stage1", type=int, default=64)
    parser.add_argument("--batch-size-stage2", type=int, default=64)
    parser.add_argument("--accum-steps-stage1", type=int, default=1)
    parser.add_argument("--accum-steps-stage2", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=100)
    parser.add_argument(
        "--transfer-stage2-epochs",
        type=int,
        default=50,
        help="Stage-2 fine-tuning epochs for transfer runs that skip stage 1.",
    )
    parser.add_argument("--lr-stage1", type=float, default=1e-3)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    return parser.parse_args()


def transfer_rows():
    return [
        {
            "experiment_id": "R_ONLY",
            "job_slug": "r-only",
            "dataset": "rhd",
            "training_sequence": "rhd",
            "parent_job_slug": "",
            "skip_stage1": False,
        },
        {
            "experiment_id": "C_ONLY",
            "job_slug": "c-only",
            "dataset": "coco_hand",
            "training_sequence": "coco_hand",
            "parent_job_slug": "",
            "skip_stage1": False,
        },
        {
            "experiment_id": "C_TO_R",
            "job_slug": "c-to-r",
            "dataset": "rhd",
            "training_sequence": "coco_hand->rhd",
            "parent_job_slug": "c-only",
            "skip_stage1": True,
        },
        {
            "experiment_id": "R_TO_C",
            "job_slug": "r-to-c",
            "dataset": "coco_hand",
            "training_sequence": "rhd->coco_hand",
            "parent_job_slug": "r-only",
            "skip_stage1": True,
        },
    ]


def build_runs(seeds, *, eval_datasets, benchmark_datasets):
    runs = []
    for row in transfer_rows():
        for seed in seeds:
            job_id = f"trf-{row['job_slug']}-s{seed}"
            parent_job_id = (
                f"trf-{row['parent_job_slug']}-s{seed}" if row["parent_job_slug"] else ""
            )
            runs.append(
                {
                    **row,
                    "experiment_family": "transfer",
                    "job_id": job_id,
                    "parent_job_id": parent_job_id,
                    "seed": int(seed),
                    "eval_datasets": ",".join(eval_datasets),
                    "benchmark_datasets": ",".join(benchmark_datasets),
                }
            )
    return runs


def quote(value):
    text = str(value)
    if " " in text:
        return f"\"{text}\""
    return text


def dataset_root(args, dataset_name: str):
    if dataset_name == "rhd":
        return args.rhd_root
    if dataset_name == "coco_hand":
        return args.coco_root
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def checkpoint_path(checkpoint_root: str, job_id: str):
    return f"{checkpoint_root}/{job_id}/checkpoints/best.pt"


def build_train_command(run, args):
    stage2_epochs = args.transfer_stage2_epochs if run["skip_stage1"] else args.stage2_epochs
    parts = [
        "python scripts/train.py",
        f"--dataset {run['dataset']}",
        f"--root {quote(dataset_root(args, run['dataset']))}",
        f"--checkpoint-root {quote(args.checkpoint_root)}",
        f"--job-id {quote(run['job_id'])}",
        f"--experiment-id {run['experiment_id']}",
        f"--experiment-family {run['experiment_family']}",
        f"--training-sequence {quote(run['training_sequence'])}",
        f"--seed {run['seed']}",
        f"--input-size {args.input_size}",
        f"--batch-size-stage1 {args.batch_size_stage1}",
        f"--batch-size-stage2 {args.batch_size_stage2}",
        f"--accum-steps-stage1 {args.accum_steps_stage1}",
        f"--accum-steps-stage2 {args.accum_steps_stage2}",
        f"--stage1-epochs {args.stage1_epochs}",
        f"--stage2-epochs {stage2_epochs}",
        f"--lr-stage1 {args.lr_stage1}",
        f"--lr-stage2 {args.lr_stage2}",
        f"--hand {args.hand}",
    ]
    if run["parent_job_id"]:
        parts.append(f"--init-ckpt {quote(checkpoint_path(args.checkpoint_root, run['parent_job_id']))}")
    if run["skip_stage1"]:
        parts.append("--skip-stage1")
    return " ".join(parts)


def build_eval_commands(run, args):
    ckpt = checkpoint_path(args.checkpoint_root, run["job_id"])
    commands = []
    for eval_dataset in args.eval_datasets:
        for prediction_mode in args.eval_prediction_modes:
            out_json = f"{args.eval_root}/{run['job_id']}.on-{eval_dataset}.{prediction_mode}.native.json"
            commands.append(
                " ".join(
                    [
                        "python scripts/eval_metrics.py",
                        f"--ckpt {quote(ckpt)}",
                        f"--dataset {eval_dataset}",
                        f"--root {quote(dataset_root(args, eval_dataset))}",
                        "--split evaluation",
                        f"--hand {args.hand}",
                        f"--prediction-mode {prediction_mode}",
                        f"--out-json {quote(out_json)}",
                    ]
                )
            )
    return commands


def build_benchmark_commands(run, args):
    ckpt = checkpoint_path(args.checkpoint_root, run["job_id"])
    commands = []
    for benchmark_dataset in args.benchmark_datasets:
        for prediction_mode in args.benchmark_prediction_modes:
            run_label = f"{run['job_id']}-on-{benchmark_dataset}-{prediction_mode}"
            commands.append(
                " ".join(
                    [
                        "python scripts/benchmark_pipeline.py",
                        f"--ckpt {quote(ckpt)}",
                        f"--dataset {benchmark_dataset}",
                        f"--root {quote(dataset_root(args, benchmark_dataset))}",
                        f"--summary-csv {quote(args.benchmark_summary_csv)}",
                        f"--output-root {quote(args.benchmark_output_root)}",
                        "--device cuda",
                        f"--prediction-mode {prediction_mode}",
                        f"--run-label {quote(run_label)}",
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

    runs = build_runs(
        args.seeds,
        eval_datasets=args.eval_datasets,
        benchmark_datasets=args.benchmark_datasets,
    )
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
