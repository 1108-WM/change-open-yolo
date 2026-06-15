#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_selectable_sam_refine}"
MODE="${MODE:-all}"  # fast | sam_top1 | sam_top3 | all

BPR_IN="${BPR_IN:-./output/backprojection_candidates_scannet200_mv_m20}"
SAM_FUSED_IN="${SAM_FUSED_IN:-./output/sam_fused_proposals_scannet200_s5_m30_prefilter}"
BPR_SAM_TOP1="${BPR_SAM_TOP1:-./output/backprojection_candidates_scannet200_even48_mv_sam_top1_m5}"
BPR_SAM_TOP3="${BPR_SAM_TOP3:-./output/backprojection_candidates_scannet200_even48_mv_sam_top3_m5}"
MAX_SAM_PER_SCENE="${MAX_SAM_PER_SCENE:-5}"

mkdir -p "$OUT_DIR/reports"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

BASE_EVAL_ARGS=(
  run_evaluation.py
  --dataset_name scannet200
  --path_to_3d_masks ./output/scannet200/scannet200_masks
  --path_to_2d_preds ./output/scannet200/bboxes_2d
  --scene_list "$SCENE_LIST"
)

COMMON_BPR_ARGS=(
  --backprojection_min_score 0.50
  --backprojection_min_seed_points 80
  --backprojection_max_existing_iou 0.30
  --backprojection_max_seed_in_existing_mask_ratio 0.70
  --backprojection_max_candidates_per_scene 15
  --backprojection_score_scale 2.00
  --no-backprojection_use_candidate_fusion_score
  --backprojection_blocked_classes rug
  --backprojection_source_score_scales sam_fused=1.2,bpr=1.0
  --backprojection_source_priorities sam_fused=2.0,bpr=1.0
  --backprojection_source_max_candidates sam_fused=12,bpr=3
  --backprojection_superpoint_refine
  --backprojection_superpoint_min_coverage 0.30
  --backprojection_superpoint_max_expansion_ratio 3.0
  --backprojection_superpoint_min_view_siou 0.60
)

summarize_csv() {
  local name="$1"
  local csv_path="$2"
  "$PYTHON" - "$name" "$csv_path" <<'PY'
import csv
import math
import sys

name, csv_path = sys.argv[1], sys.argv[2]
vals = {"ap": [], "ap50": [], "ap25": []}
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        for key in vals:
            value = float(row[key])
            if not math.isnan(value):
                vals[key].append(value)
print(
    f"[RESULT] {name}: "
    f"AP={sum(vals['ap']) / len(vals['ap']):.6f} "
    f"AP50={sum(vals['ap50']) / len(vals['ap50']):.6f} "
    f"AP25={sum(vals['ap25']) / len(vals['ap25']):.6f}"
)
PY
}

run_eval() {
  local name="$1"
  local candidates="$2"
  local eta="$3"
  local csv_path="$OUT_DIR/${name}.csv"
  local log_path="$OUT_DIR/${name}.log"
  local report_path="$OUT_DIR/reports/${name}.json"
  echo "[RUN] $name 预计耗时: $eta"
  "$PYTHON" "${BASE_EVAL_ARGS[@]}" \
    --backprojection_candidates "$candidates" \
    "${COMMON_BPR_ARGS[@]}" \
    --backprojection_report_path "$report_path" \
    --eval_output_file "$csv_path" >"$log_path" 2>&1
  summarize_csv "$name" "$csv_path"
}

refine_top1() {
  if [[ -f "$BPR_SAM_TOP1/sam_multiview_refine_summary.json" ]]; then
    echo "[SKIP] top1 SAM refinement cache exists: $BPR_SAM_TOP1"
    return
  fi
  echo "[EXPORT] top1 SAM-refined BPR candidates 预计耗时: 4-8 min on RTX 4090"
  "$PYTHON" tools/refine_backprojection_candidates_sam_multiview.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --path_to_2d_preds ./output/scannet200/bboxes_2d \
    --candidates_dir "$BPR_IN" \
    --output_dir "$BPR_SAM_TOP1" \
    --scene_list "$SCENE_LIST" \
    --max_per_scene "$MAX_SAM_PER_SCENE" \
    --top_views 1 \
    --min_vote_count 1 \
    --min_vote_ratio 0.50 \
    --min_visible_seed_points 30 \
    --min_refined_seed_points 30 \
    --fallback_to_best_view
}

refine_top3() {
  if [[ -f "$BPR_SAM_TOP3/sam_multiview_refine_summary.json" ]]; then
    echo "[SKIP] top3 SAM refinement cache exists: $BPR_SAM_TOP3"
    return
  fi
  echo "[EXPORT] top3 SAM-refined BPR candidates 预计耗时: 8-18 min on RTX 4090"
  "$PYTHON" tools/refine_backprojection_candidates_sam_multiview.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --path_to_2d_preds ./output/scannet200/bboxes_2d \
    --candidates_dir "$BPR_IN" \
    --output_dir "$BPR_SAM_TOP3" \
    --scene_list "$SCENE_LIST" \
    --max_per_scene "$MAX_SAM_PER_SCENE" \
    --top_views 3 \
    --min_vote_count 2 \
    --min_vote_ratio 0.50 \
    --min_visible_seed_points 30 \
    --min_refined_seed_points 30 \
    --fallback_to_best_view
}

case "$MODE" in
  fast)
    run_eval "fast_yoloworld_bpr" "$SAM_FUSED_IN,$BPR_IN" "3-6 min on RTX 4090"
    ;;
  sam_top1)
    refine_top1
    run_eval "sam_top1_refined_bpr" "$SAM_FUSED_IN,$BPR_SAM_TOP1" "3-6 min on RTX 4090"
    ;;
  sam_top3)
    refine_top3
    run_eval "sam_top3_refined_bpr" "$SAM_FUSED_IN,$BPR_SAM_TOP3" "3-6 min on RTX 4090"
    ;;
  all)
    run_eval "fast_yoloworld_bpr" "$SAM_FUSED_IN,$BPR_IN" "3-6 min on RTX 4090"
    refine_top1
    run_eval "sam_top1_refined_bpr" "$SAM_FUSED_IN,$BPR_SAM_TOP1" "3-6 min on RTX 4090"
    refine_top3
    run_eval "sam_top3_refined_bpr" "$SAM_FUSED_IN,$BPR_SAM_TOP3" "3-6 min on RTX 4090"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use fast, sam_top1, sam_top3, or all." >&2
    exit 2
    ;;
esac

echo "[DONE] Wrote selectable SAM refinement outputs to $OUT_DIR"
