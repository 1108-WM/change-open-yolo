#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
RUNNER="$ROOT_DIR/tools/run_scannet200_even48_seed_merge_policy_eval.sh"

echo "[RUNNER] even96 seed_topk2 预计耗时: 17-25 min on RTX 4090"
MODE=topk2 \
SPLIT_NAME=even96 \
RUN_LABEL_PREFIX=even96_ \
SCENE_LIST="$ROOT_DIR/output/scannet200/scene_splits/even96.txt" \
OUT_DIR="$ROOT_DIR/output/scannet200/subset_sweeps/seed_topk2_96_confirm" \
"$RUNNER"

echo "[RUNNER] odd96 seed_topk2 预计耗时: 17-25 min on RTX 4090"
MODE=topk2 \
SPLIT_NAME=odd96 \
RUN_LABEL_PREFIX=odd96_ \
SCENE_LIST="$ROOT_DIR/output/scannet200/scene_splits/odd96.txt" \
OUT_DIR="$ROOT_DIR/output/scannet200/subset_sweeps/seed_topk2_96_confirm" \
"$RUNNER"

echo "[DONE] Wrote seed_topk2 96-scene confirmation outputs to $ROOT_DIR/output/scannet200/subset_sweeps/seed_topk2_96_confirm"
