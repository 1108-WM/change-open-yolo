#!/usr/bin/env bash
set -euo pipefail

# 新冻结版本：启用独立重观测的软确认；旧 frozen 脚本的默认行为不变。
ROOT_DIR="${ROOT_DIR:-/home/jia/Wm/wm_open-yolo/OpenYOLO3D}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/output/sam2_details_even48_reobserve_20260721}"
EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/output/scannet200/sam2_details_even48_reobserve_20260721_eval}"

exec env \
  RUN_DIR="$RUN_DIR" \
  EVAL_DIR="$EVAL_DIR" \
  REOBSERVATION_STRIDE="${REOBSERVATION_STRIDE:-5}" \
  REOBSERVATION_MIN_IOU="${REOBSERVATION_MIN_IOU:-0.30}" \
  REOBSERVATION_REJECTED_FRAME_WEIGHT="${REOBSERVATION_REJECTED_FRAME_WEIGHT:-0.50}" \
  bash "$ROOT_DIR/tools/run_scannet200_even48_sam2_details_frozen.sh"
