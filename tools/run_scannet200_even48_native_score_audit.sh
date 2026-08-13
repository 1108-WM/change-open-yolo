#!/usr/bin/env bash
set -euo pipefail

# 冻结历史强基线 B 的候选、融合与类别，仅保留原生预测分数做最终排序。
# 该脚本用于判断评分问题是已有分数未接入，还是需要新的质量评分方法。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/final_strong_baseline_native_score_even48_20260724}"

SAM_FUSED_IN="${SAM_FUSED_IN:-$ROOT_DIR/output/sam_fused_proposals_scannet200_s5_m30_prefilter}"
BPR_IN="${BPR_IN:-$ROOT_DIR/output/backprojection_candidates_scannet200_mv_m20}"

mkdir -p "$OUT_DIR/reports"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENYOLO3D_ALLOW_LEGACY_2D_CACHE="${OPENYOLO3D_ALLOW_LEGACY_2D_CACHE:-1}"

"$PYTHON" tools/verify_native_experiment_entry.py \
  --scene-list "$SCENE_LIST" \
  --mask-root ./output/scannet200/scannet200_masks \
  --bbox-root ./output/scannet200/bboxes_2d \
  --candidate-roots "$SAM_FUSED_IN,$BPR_IN" \
  --output-dir "$OUT_DIR" \
  --score-mode native

"$PYTHON" run_evaluation.py \
  --dataset_name scannet200 \
  --path_to_3d_masks ./output/scannet200/scannet200_masks \
  --path_to_2d_preds ./output/scannet200/bboxes_2d \
  --scene_list "$SCENE_LIST" \
  --eval_score_mode native \
  --backprojection_candidates "$SAM_FUSED_IN,$BPR_IN" \
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
  --backprojection_report_path "$OUT_DIR/reports/B_native_score_original_superpoints.json" \
  --eval_prediction_cache_dir "$OUT_DIR/prediction_cache" \
  --eval_output_file "$OUT_DIR/B_native_score_original_superpoints.csv"

"$PYTHON" - "$OUT_DIR/B_native_score_original_superpoints.csv" <<'PY'
import csv
import math
import sys

values = {key: [] for key in ("ap", "ap50", "ap25")}
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        for key in values:
            value = float(row[key])
            if not math.isnan(value):
                values[key].append(value)
print("[RESULT] B_native_score_original_superpoints: " + " ".join(
    f"{key.upper()}={sum(items) / len(items):.6f}" for key, items in values.items()
))
PY

echo "[DONE] Outputs: $OUT_DIR"
