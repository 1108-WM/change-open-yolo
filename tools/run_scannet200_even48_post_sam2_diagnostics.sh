#!/usr/bin/env bash
# 仅运行离线 GT 诊断；不产生或修改推理候选、模型权重和评测结果。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/docs/diagnostics/post_sam2_decision_even48_20260722}"
SCENE_LIST="${SCENE_LIST:-$ROOT/output/scannet200/scene_splits/even48.txt}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$ROOT/output/sam2_details_even48_reobserve_20260721/mvpdist_candidates}"

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/driver.log") 2>&1
cd "$ROOT"
"$PYTHON_BIN" tools/diagnose_yoloworld_mask3d_missed_coverage_gt.py \
  --scene_list "$SCENE_LIST" \
  --baseline_masks_root output/scannet200/scannet200_masks \
  --bboxes_2d_root output/scannet200/bboxes_2d \
  --dataset_root data/scannet200 \
  --output_dir "$OUT_DIR/yoloworld_coverage" \
  --max_frames 30 \
  --allow_gt_diagnostics
"$PYTHON_BIN" tools/diagnose_mask3d_sam2_local_correction_gt.py \
  --scene_list "$SCENE_LIST" \
  --baseline_masks_root output/scannet200/scannet200_masks \
  --candidate_root "$CANDIDATE_ROOT" \
  --output_dir "$OUT_DIR/local_correction" \
  --allow_gt_diagnostics
