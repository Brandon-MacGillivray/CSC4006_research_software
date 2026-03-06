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
    return {
        "checkpoint": str(Path(args.ckpt)),
        "checkpoint_stage": ckpt_meta.get("stage"),
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "prediction_mode": "fusion",
        "device": str(device),
        "dataset": {
            "root": args.root,
            "split": args.split,
            "hand": args.hand,
            "input_size": args.input_size,
        },
        "model_keypoint_indices": model_keypoint_indices,
        "eval_keypoint_indices": eval_keypoint_indices,
        "shared_10_eval": bool(args.shared_10_eval),
        "metrics": metrics,
    }


def save_results_json(results: dict, out_json: str):
    """Save JSON file."""
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
