import argparse
import json

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from handpose.data import AUTO_DATASET, SUPPORTED_DATASETS, resolve_dataset_name, resolve_dataset_split
from handpose.evaluation.eval_metrics_core import evaluate_checkpoint
from handpose.evaluation.eval_outputs import build_results_payload, save_results_json
from handpose.evaluation.eval_pipeline import (
    build_loader,
    build_model,
    resolve_device,
    resolve_eval_indices,
    resolve_root_keypoint_local_index,
)
from handpose.checkpoints import load_checkpoint, infer_checkpoint_keypoint_indices
from handpose.inference.predict import SUPPORTED_PREDICTION_MODES


def build_arg_parser():
    """Build the CLI parser for checkpoint evaluation."""
    p = argparse.ArgumentParser(description="Evaluate a trained stage-2 model using fused predictions.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt), typically stage-2 best.pt")
    p.add_argument("--root", default="data/RHD_published_v2")
    p.add_argument(
        "--dataset",
        default=AUTO_DATASET,
        choices=[AUTO_DATASET, *SUPPORTED_DATASETS],
        help="Dataset loader to use. 'auto' uses checkpoint metadata and falls back to RHD.",
    )
    p.add_argument(
        "--split",
        default="evaluation",
        help="Split alias resolved per dataset, for example evaluation/val.",
    )
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--prediction-mode",
        default="fusion",
        choices=SUPPORTED_PREDICTION_MODES,
        help="Select fused, heatmap-only, or coord-only predictions.",
    )
    p.add_argument(
        "--shared-10-eval",
        action="store_true",
        help="Evaluate only the shared tip+base 10-keypoint subset for fair 21-vs-10 comparison.",
    )
    p.add_argument(
        "--debug-coords",
        action="store_true",
        help="Print GT/pred coordinate differences for the first batch (to stderr).",
    )
    p.add_argument(
        "--pck-threshold",
        type=float,
        default=0.2,
        help="Normalized distance threshold sigma used for PCK@sigma (default: 0.2).",
    )
    p.add_argument(
        "--with-fusion-diagnostics",
        action="store_true",
        help="Include fusion-selection diagnostics in the output payload.",
    )
    p.add_argument("--out-json", default=None, help="Optional path to write evaluation results as JSON")
    return p


def main():
    """Run evaluation and print/save the resulting metrics payload."""
    args = build_arg_parser().parse_args()
    if args.pck_threshold < 0:
        raise ValueError("--pck-threshold must be >= 0")
    device = resolve_device(args.device)

    # Load checkpoint and recover keypoint layout.
    ckpt_meta, state_dict = load_checkpoint(args.ckpt, device)
    training_config = ckpt_meta.get("training_config", {})
    dataset_name = resolve_dataset_name(args.dataset, training_config=training_config)
    args.dataset = dataset_name
    args.split = resolve_dataset_split(dataset_name, args.split)
    model_keypoint_indices = infer_checkpoint_keypoint_indices(ckpt_meta)
    num_keypoints = len(model_keypoint_indices)

    # Resolve eval subset (full or shared-10).
    eval_positions, eval_keypoint_indices = resolve_eval_indices(
        model_keypoint_indices=model_keypoint_indices,
        shared_10_eval=args.shared_10_eval,
    )
    root_keypoint_local_index = resolve_root_keypoint_local_index(
        eval_keypoint_indices=eval_keypoint_indices,
        model_keypoint_indices=model_keypoint_indices,
    )

    model = build_model(num_keypoints=num_keypoints, state_dict=state_dict, device=device)
    loader = build_loader(
        args=args,
        device=device,
        model_keypoint_indices=model_keypoint_indices,
        dataset_name=dataset_name,
    )

    metrics = evaluate_checkpoint(
        model=model,
        loader=loader,
        device=device,
        model_keypoint_indices=model_keypoint_indices,
        eval_positions=eval_positions,
        root_keypoint_local_index=root_keypoint_local_index,
        pck_threshold=float(args.pck_threshold),
        prediction_mode=args.prediction_mode,
        debug_coords=bool(args.debug_coords),
        with_fusion_diagnostics=bool(args.with_fusion_diagnostics),
    )

    results = build_results_payload(
        args=args,
        ckpt_meta=ckpt_meta,
        device=device,
        model_keypoint_indices=model_keypoint_indices,
        eval_keypoint_indices=eval_keypoint_indices,
        metrics=metrics,
    )

    print(json.dumps(results, indent=2))

    if args.out_json:
        save_results_json(results=results, out_json=args.out_json)


if __name__ == "__main__":
    main()
