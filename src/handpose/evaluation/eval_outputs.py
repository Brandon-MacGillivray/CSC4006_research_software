import json
from pathlib import Path


def build_results_payload(
    args,
    ckpt_meta: dict,
    device,
    model_keypoint_indices,
    eval_keypoint_indices,
    metrics: dict,
):
    """Build final JSON payload."""
    metrics_payload = dict(metrics)
    fusion_diagnostics = metrics_payload.pop("fusion_diagnostics", None)
    return {
        "checkpoint": str(Path(args.ckpt)),
        "checkpoint_stage": ckpt_meta.get("stage"),
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "job_id": ckpt_meta.get("training_config", {}).get("job_id", ""),
        "experiment_id": ckpt_meta.get("training_config", {}).get("experiment_id", ""),
        "prediction_mode": args.prediction_mode,
        "with_fusion_diagnostics": bool(args.with_fusion_diagnostics),
        "device": str(device),
        "dataset": {
            "name": getattr(args, "dataset", ""),
            "root": args.root,
            "split": args.split,
            "hand": args.hand,
            "input_size": args.input_size,
        },
        "training_config": ckpt_meta.get("training_config", {}),
        "model_keypoint_indices": model_keypoint_indices,
        "eval_keypoint_indices": eval_keypoint_indices,
        "shared_10_eval": bool(args.shared_10_eval),
        "metrics": metrics_payload,
        **({"fusion_diagnostics": fusion_diagnostics} if fusion_diagnostics is not None else {}),
    }


def save_results_json(results: dict, out_json: str):
    """Save JSON file."""
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
