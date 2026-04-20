#!/usr/bin/env bash

# Fair-comparison evaluation commands for the manually annotated RH8 set.
#
# Assumptions:
# - Repo on Kelvin: clone this repository and run the script from anywhere
# - Qualitative dataset root: <repo>/docs/artefacts/datasets/rh8
# - Ablation checkpoints: ~/sharedscratch/training_results
# - Transfer checkpoints: ~/sharedscratch/training_results
#
# Earlier notes used ~/sharedscratch/transfer_training_results, but the
# observed RH8 eval outputs point at ~/sharedscratch/training_results
# for the transfer family as well.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
QUAL_ROOT="${REPO_ROOT}/docs/artefacts/datasets/rh8"
OUT_ROOT=~/qual_eval_results/rh8
ABL_ROOT=~/sharedscratch/training_results
TRF_ROOT=~/sharedscratch/training_results

mkdir -p "$OUT_ROOT"

# B0 baseline: 21-joint model evaluated on the shared-10 subset for fair comparison.
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$ABL_ROOT/drh-b0-k21-right-s101/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/drh-b0-k21-right-s101.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$ABL_ROOT/drh-b0-k21-right-s202/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/drh-b0-k21-right-s202.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$ABL_ROOT/drh-b0-k21-right-s303/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/drh-b0-k21-right-s303.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$ABL_ROOT/drh-b0-k21-right-s404/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/drh-b0-k21-right-s404.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$ABL_ROOT/drh-b0-k21-right-s505/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/drh-b0-k21-right-s505.fusion.shared10.rh8.json"

# B1 reduced: 10-joint model evaluated natively.
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$ABL_ROOT/drh-b1-k10-right-s101/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --out-json "$OUT_ROOT/drh-b1-k10-right-s101.fusion.native10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$ABL_ROOT/drh-b1-k10-right-s202/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --out-json "$OUT_ROOT/drh-b1-k10-right-s202.fusion.native10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$ABL_ROOT/drh-b1-k10-right-s303/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --out-json "$OUT_ROOT/drh-b1-k10-right-s303.fusion.native10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$ABL_ROOT/drh-b1-k10-right-s404/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --out-json "$OUT_ROOT/drh-b1-k10-right-s404.fusion.native10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$ABL_ROOT/drh-b1-k10-right-s505/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --out-json "$OUT_ROOT/drh-b1-k10-right-s505.fusion.native10.rh8.json"

# Transfer family: all are 21-joint checkpoints, so evaluate on the shared-10 subset for fair comparison with B1.
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-r-only-s101/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-r-only-s101.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-r-only-s202/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-r-only-s202.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-r-only-s303/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-r-only-s303.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-r-only-s404/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-r-only-s404.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-r-only-s505/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-r-only-s505.fusion.shared10.rh8.json"

python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-c-only-s101/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-c-only-s101.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-c-only-s202/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-c-only-s202.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-c-only-s303/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-c-only-s303.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-c-only-s404/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-c-only-s404.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-c-only-s505/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-c-only-s505.fusion.shared10.rh8.json"

python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-c-to-r-s101/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-c-to-r-s101.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-c-to-r-s202/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-c-to-r-s202.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-c-to-r-s303/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-c-to-r-s303.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-c-to-r-s404/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-c-to-r-s404.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-c-to-r-s505/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-c-to-r-s505.fusion.shared10.rh8.json"

python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-r-to-c-s101/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-r-to-c-s101.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-r-to-c-s202/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-r-to-c-s202.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-r-to-c-s303/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-r-to-c-s303.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-r-to-c-s404/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-r-to-c-s404.fusion.shared10.rh8.json"
python "$REPO_ROOT/scripts/eval_metrics.py" --dataset coco_hand --ckpt "$TRF_ROOT/trf-r-to-c-s505/checkpoints/best.pt" --root "$QUAL_ROOT" --split val --prediction-mode fusion --shared-10-eval --out-json "$OUT_ROOT/trf-r-to-c-s505.fusion.shared10.rh8.json"

# MediaPipe baseline on the same custom dataset.
python "$REPO_ROOT/scripts/eval_mediapipe_coco_hand.py" --model-asset-path "$REPO_ROOT/src/handpose/models/hand_landmarker.task" --root "$QUAL_ROOT" --split val --dataset-name rh8 --hand auto --out-json "$OUT_ROOT/mediapipe.rh8.native21.json"
python "$REPO_ROOT/scripts/eval_mediapipe_coco_hand.py" --model-asset-path "$REPO_ROOT/src/handpose/models/hand_landmarker.task" --root "$QUAL_ROOT" --split val --dataset-name rh8 --hand auto --shared-10-eval --out-json "$OUT_ROOT/mediapipe.rh8.shared10.json"
