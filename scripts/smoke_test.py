from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

# Exclude scripts with optional runtime dependencies such as MediaPipe or OpenCV.
COMMANDS = [
    [sys.executable, "scripts/train.py", "--help"],
    [sys.executable, "scripts/eval_metrics.py", "--help"],
    [sys.executable, "scripts/benchmark_pipeline.py", "--help"],
    [sys.executable, "scripts/predict_image.py", "--help"],
    [sys.executable, "scripts/aggregate_results.py", "--help"],
    [sys.executable, "scripts/generate_experiment_matrix.py", "--help"],
    [sys.executable, "scripts/generate_transfer_experiment_matrix.py", "--help"],
    [sys.executable, "scripts/plot_losses.py", "--help"],
]


def run_command(command):
    print("Running:", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(
            f"Smoke test failed for command: {' '.join(command)} "
            f"(exit_code={completed.returncode})"
        )


def main():
    for command in COMMANDS:
        run_command(command)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
