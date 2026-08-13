#!/usr/bin/env bash
set -euo pipefail

# 仅复现历史强基线 B 并保存最终预测缓存，供 GT-only 误差归因读取。
# SCENES 可传入逗号分隔的子集；分段运行不会改变任何单场景预测。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SCENES="${SCENES:?必须传入逗号分隔的场景名}"
TAG="${TAG:-subset}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/final_strong_baseline_error_attribution_even48_20260724}"

mkdir -p "$OUT_DIR/prediction_cache" "$OUT_DIR/reports"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENYOLO3D_ALLOW_LEGACY_2D_CACHE="${OPENYOLO3D_ALLOW_LEGACY_2D_CACHE:-1}"

"$PYTHON" run_evaluation.py \
  --dataset_name scannet200 \
  --path_to_3d_masks ./output/scannet200/scannet200_masks \
  --path_to_2d_preds ./output/scannet200/bboxes_2d \
  --scene_list "$SCENES" \
  --backprojection_candidates ./output/sam_fused_proposals_scannet200_s5_m30_prefilter,./output/backprojection_candidates_scannet200_mv_m20 \
  --backprojection_min_score 0.50 \
  --backprojection_min_seed_points 80 \
  --backprojection_max_existing_iou 0.30 \
  --backprojection_max_seed_in_existing_mask_ratio 0.70 \
  --backprojection_max_candidates_per_scene 15 \
  --backprojection_score_scale 2.00 \
  --no-backprojection_use_candidate_fusion_score \
  --backprojection_blocked_classes rug \
  --backprojection_source_score_scales sam_fused=1.2,bpr=1.0 \
  --backprojection_source_priorities sam_fused=2.0,bpr=1.0 \
  --backprojection_source_max_candidates sam_fused=12,bpr=3 \
  --backprojection_superpoint_refine \
  --backprojection_superpoint_min_coverage 0.30 \
  --backprojection_superpoint_max_expansion_ratio 3.0 \
  --backprojection_superpoint_min_view_siou 0.60 \
  --backprojection_report_path "$OUT_DIR/reports/B_original_superpoints_${TAG}.json" \
  --eval_prediction_cache_dir "$OUT_DIR/prediction_cache" \
  --eval_output_file "$OUT_DIR/B_original_superpoints_${TAG}.csv"
