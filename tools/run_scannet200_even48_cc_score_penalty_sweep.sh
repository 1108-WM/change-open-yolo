#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_cc_score_penalty_sweep}"
MODE="${MODE:-all}"  # w050 | w100 | w100k060 | all

BPR_IN="${BPR_IN:-./output/backprojection_candidates_scannet200_mv_m20}"
SAM_FUSED_IN="${SAM_FUSED_IN:-./output/sam_fused_proposals_scannet200_s5_m30_prefilter}"

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
  --backprojection_candidates "$SAM_FUSED_IN,$BPR_IN"
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
  --backprojection_cc_cleanup
  --backprojection_cc_radius 0.03
  --backprojection_cc_min_component_points 50
  --backprojection_cc_keep_topk 1
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
  local eta="$2"
  shift 2
  local csv_path="$OUT_DIR/${name}.csv"
  local log_path="$OUT_DIR/${name}.log"
  local report_path="$OUT_DIR/reports/${name}.json"
  echo "[RUN] $name 预计耗时: $eta"
  "$PYTHON" "${BASE_EVAL_ARGS[@]}" \
    "${COMMON_BPR_ARGS[@]}" \
    "$@" \
    --backprojection_report_path "$report_path" \
    --eval_output_file "$csv_path" >"$log_path" 2>&1
  summarize_csv "$name" "$csv_path"
}

case "$MODE" in
  w050)
    run_eval "cc_score_w050" "3-6 min on RTX 4090" --backprojection_cc_keep_ratio_score_weight 0.50
    ;;
  w100)
    run_eval "cc_score_w100" "3-6 min on RTX 4090" --backprojection_cc_keep_ratio_score_weight 1.00
    ;;
  w100k060)
    run_eval "cc_score_w100_keep060" "3-6 min on RTX 4090" \
      --backprojection_cc_keep_ratio_score_weight 1.00 \
      --backprojection_cc_min_keep_ratio 0.60
    ;;
  all)
    run_eval "cc_score_w050" "3-6 min on RTX 4090" --backprojection_cc_keep_ratio_score_weight 0.50
    run_eval "cc_score_w100" "3-6 min on RTX 4090" --backprojection_cc_keep_ratio_score_weight 1.00
    run_eval "cc_score_w100_keep060" "3-6 min on RTX 4090" \
      --backprojection_cc_keep_ratio_score_weight 1.00 \
      --backprojection_cc_min_keep_ratio 0.60
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

echo "[DONE] Wrote even48 CC score-penalty sweep outputs to $OUT_DIR"
