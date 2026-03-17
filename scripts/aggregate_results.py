import argparse
import csv
import json
import math
from pathlib import Path
import statistics


EVAL_GROUP_FIELDS = [
    "experiment_id",
    "prediction_mode",
    "shared_10_eval",
    "train_num_keypoints",
    "eval_num_keypoints",
    "train_hand",
    "train_input_size",
    "tips_bases_only",
    "lambda_hm",
    "lambda_coord",
    "heatmap_sigma",
    "wing_w",
    "wing_epsilon",
]

EVAL_METRIC_FIELDS = [
    "metrics.sse_norm",
    "metrics.epe_norm",
    "metrics.pck",
    "metrics.timing.ms_per_image",
    "metrics.timing.images_per_second",
]

FUSION_DIAGNOSTIC_FIELDS = [
    "fusion_diagnostics.heatmap_selection_rate",
    "fusion_diagnostics.coord_selection_rate",
    "fusion_diagnostics.alpha_mean",
    "fusion_diagnostics.alpha_median",
    "fusion_diagnostics.disagreement_mean",
    "fusion_diagnostics.disagreement_median",
    "fusion_diagnostics.fusion_matches_lower_error_branch_rate",
]

LATENCY_GROUP_FIELDS = [
    "experiment_id",
    "prediction_mode",
    "num_keypoints",
    "train_hand",
    "train_input_size",
    "lambda_hm",
    "lambda_coord",
    "heatmap_sigma",
    "wing_w",
    "wing_epsilon",
]

LATENCY_METRIC_FIELDS = [
    "forward_predict_ms_mean",
    "total_e2e_ms_mean",
    "session_setup_ms",
]

EVAL_ID_FIELDS = [
    "job_id",
    "seed",
    "prediction_mode",
    "shared_10_eval",
    "with_fusion_diagnostics",
]

BENCHMARK_ID_FIELDS = [
    "job_id",
    "seed",
    "prediction_mode",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation JSONs and benchmark CSVs into mean/std tables."
    )
    parser.add_argument(
        "--eval-json-dir",
        default="eval_results",
        help="Directory containing evaluation JSON outputs.",
    )
    parser.add_argument(
        "--benchmark-csv",
        default=None,
        help="Optional benchmark summary CSV to aggregate.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for aggregated CSV outputs.",
    )
    return parser.parse_args()


def flatten_dict(data, parent_key="", out=None):
    if out is None:
        out = {}
    for key, value in data.items():
        flat_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flatten_dict(value, parent_key=flat_key, out=out)
        else:
            out[flat_key] = value
    return out


def infer_job_id_from_checkpoint(checkpoint_path: str):
    if not checkpoint_path:
        return ""
    ckpt_path = Path(str(checkpoint_path))
    if ckpt_path.parent.name == "checkpoints" and ckpt_path.parent.parent.name:
        return ckpt_path.parent.parent.name
    return ckpt_path.stem


def infer_experiment_id(job_id: str):
    parts = str(job_id).split("-")
    if len(parts) >= 2 and parts[0] == "drh":
        return str(parts[1]).upper()
    return ""


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return value


def parse_number(value):
    if value in ("", None):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value):
    number = parse_number(value)
    if number is None or math.isnan(number):
        return None
    return int(number)


def safe_mean(values):
    if not values:
        return ""
    return float(statistics.fmean(values))


def safe_std(values):
    if len(values) <= 1:
        return 0.0 if values else ""
    return float(statistics.stdev(values))


def normalize_identity_value(value):
    normalized = parse_bool(value)
    if isinstance(normalized, bool):
        return normalized
    return str(normalized)


def format_identity(fields, key):
    return ", ".join(f"{field}={value!r}" for field, value in zip(fields, key))


def assert_unique_rows(rows, *, identity_fields, label, source_field):
    duplicates = {}
    for row in rows:
        key = tuple(normalize_identity_value(row.get(field, "")) for field in identity_fields)
        duplicates.setdefault(key, []).append(str(row.get(source_field, "")))

    duplicate_messages = []
    for key, sources in duplicates.items():
        if len(sources) > 1:
            duplicate_messages.append(
                f"{label} duplicate for {format_identity(identity_fields, key)}: {sources}"
            )

    if duplicate_messages:
        preview = "\n".join(duplicate_messages[:10])
        raise ValueError(
            f"Duplicate {label} results detected. Clean old outputs before aggregating.\n{preview}"
        )


def load_eval_rows(eval_json_dir: Path):
    rows = []
    if not eval_json_dir.exists():
        return rows

    for path in sorted(eval_json_dir.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        flat = flatten_dict(payload)
        training_config = payload.get("training_config", {})
        job_id = (
            payload.get("job_id")
            or training_config.get("job_id")
            or infer_job_id_from_checkpoint(payload.get("checkpoint", ""))
        )
        experiment_id = (
            payload.get("experiment_id")
            or training_config.get("experiment_id")
            or infer_experiment_id(job_id)
        )
        row = {
            "source_path": str(path),
            "job_id": job_id,
            "experiment_id": experiment_id,
            "prediction_mode": payload.get("prediction_mode", ""),
            "shared_10_eval": bool(payload.get("shared_10_eval", False)),
            "with_fusion_diagnostics": bool(payload.get("with_fusion_diagnostics", False)),
            "train_num_keypoints": training_config.get(
                "num_keypoints",
                len(payload.get("model_keypoint_indices", [])),
            ),
            "eval_num_keypoints": len(payload.get("eval_keypoint_indices", [])),
            "train_hand": training_config.get(
                "hand",
                payload.get("dataset", {}).get("hand", ""),
            ),
            "train_input_size": training_config.get(
                "input_size",
                payload.get("dataset", {}).get("input_size", ""),
            ),
            "tips_bases_only": bool(training_config.get("tips_bases_only", False)),
            "lambda_hm": training_config.get("lambda_hm", ""),
            "lambda_coord": training_config.get("lambda_coord", ""),
            "heatmap_sigma": training_config.get("heatmap_sigma", ""),
            "wing_w": training_config.get("wing_w", ""),
            "wing_epsilon": training_config.get("wing_epsilon", ""),
            "seed": training_config.get("seed", ""),
        }
        for field in EVAL_METRIC_FIELDS + FUSION_DIAGNOSTIC_FIELDS:
            row[field] = flat.get(field)
        rows.append(row)
    return rows


def load_benchmark_rows(csv_path: Path):
    rows = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            job_id = row.get("job_id", "") or infer_job_id_from_checkpoint(
                row.get("checkpoint_path", "")
            )
            row["job_id"] = job_id
            row["experiment_id"] = row.get("experiment_id", "") or infer_experiment_id(job_id)
            row["prediction_mode"] = row.get("prediction_mode", "")
            row["num_keypoints"] = row.get("num_keypoints", "")
            row["train_hand"] = row.get("train_hand", "")
            row["train_input_size"] = row.get("train_input_size", "")
            row["lambda_hm"] = row.get("lambda_hm", "")
            row["lambda_coord"] = row.get("lambda_coord", "")
            row["heatmap_sigma"] = row.get("heatmap_sigma", "")
            row["wing_w"] = row.get("wing_w", "")
            row["wing_epsilon"] = row.get("wing_epsilon", "")
            row["seed"] = row.get("seed", "")
            row["status"] = row.get("status", "")
            row["source_path"] = str(csv_path)
            rows.append(row)
    return rows


def validate_benchmark_rows(rows):
    invalid = []
    for row in rows:
        failures = parse_int(row.get("failures"))
        expected_num_images = parse_int(row.get("expected_num_images"))
        completed_num_images = parse_int(row.get("completed_num_images"))
        status = str(row.get("status", "")).strip().lower()
        if (
            status != "ok"
            or failures not in (0, None)
            or expected_num_images is None
            or completed_num_images is None
            or completed_num_images != expected_num_images
        ):
            invalid.append(
                {
                    "job_id": row.get("job_id", ""),
                    "seed": row.get("seed", ""),
                    "prediction_mode": row.get("prediction_mode", ""),
                    "status": row.get("status", ""),
                    "failures": row.get("failures", ""),
                    "completed_num_images": row.get("completed_num_images", ""),
                    "expected_num_images": row.get("expected_num_images", ""),
                }
            )

    if invalid:
        preview = "\n".join(str(entry) for entry in invalid[:10])
        raise ValueError(
            "Invalid or partial benchmark rows detected. Clean or rerun them before aggregating.\n"
            f"{preview}"
        )


def group_rows(rows, group_fields):
    grouped = {}
    for row in rows:
        key = tuple(parse_bool(row.get(field, "")) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    return grouped


def aggregate_rows(rows, *, group_fields, metric_fields):
    grouped = group_rows(rows, group_fields)
    out_rows = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        out = {field: value for field, value in zip(group_fields, key)}
        out["n"] = len(group)
        out["num_seeds"] = len(
            {str(row.get("seed", "")) for row in group if str(row.get("seed", "")) != ""}
        )
        for field in metric_fields:
            values = [
                value
                for value in (parse_number(row.get(field)) for row in group)
                if value is not None and not math.isnan(value)
            ]
            out[f"{field}_mean"] = safe_mean(values)
            out[f"{field}_std"] = safe_std(values)
        out_rows.append(out)
    return out_rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_branch_ablation_rows(rows):
    branch_fields = [
        "experiment_id",
        "prediction_mode",
        "train_num_keypoints",
        "shared_10_eval",
        "lambda_hm",
        "lambda_coord",
        "heatmap_sigma",
        "wing_w",
        "wing_epsilon",
    ]
    metric_fields = [
        "metrics.sse_norm",
        "metrics.epe_norm",
        "metrics.pck",
    ]
    return aggregate_rows(rows, group_fields=branch_fields, metric_fields=metric_fields)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_eval_rows(Path(args.eval_json_dir))
    assert_unique_rows(
        eval_rows,
        identity_fields=EVAL_ID_FIELDS,
        label="evaluation",
        source_field="source_path",
    )

    primary_eval_rows = [row for row in eval_rows if not row.get("with_fusion_diagnostics", False)]
    diagnostic_eval_rows = [row for row in eval_rows if row.get("with_fusion_diagnostics", False)]

    accuracy_rows = aggregate_rows(
        primary_eval_rows,
        group_fields=EVAL_GROUP_FIELDS,
        metric_fields=EVAL_METRIC_FIELDS,
    )
    write_csv(out_dir / "aggregated_accuracy.csv", accuracy_rows)
    write_csv(
        out_dir / "aggregated_branch_ablation.csv",
        build_branch_ablation_rows(primary_eval_rows),
    )

    fusion_diag_rows = aggregate_rows(
        diagnostic_eval_rows,
        group_fields=EVAL_GROUP_FIELDS,
        metric_fields=FUSION_DIAGNOSTIC_FIELDS,
    )
    write_csv(out_dir / "aggregated_fusion_diagnostics.csv", fusion_diag_rows)

    if args.benchmark_csv:
        latency_rows = load_benchmark_rows(Path(args.benchmark_csv))
        assert_unique_rows(
            latency_rows,
            identity_fields=BENCHMARK_ID_FIELDS,
            label="benchmark",
            source_field="source_path",
        )
        validate_benchmark_rows(latency_rows)
        aggregated_latency = aggregate_rows(
            latency_rows,
            group_fields=LATENCY_GROUP_FIELDS,
            metric_fields=LATENCY_METRIC_FIELDS,
        )
        write_csv(out_dir / "aggregated_latency.csv", aggregated_latency)


if __name__ == "__main__":
    main()
