#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_hierarchy_score}"
BPR_IN="${BPR_IN:-./output/backprojection_candidates_scannet200_mv_m20}"
SAM_OUT="${SAM_OUT:-$ROOT_DIR/output/sam_fused_proposals_scannet200_even48_maskpath}"
MODE="${MODE:-all}"

mkdir -p "$OUT_DIR/reports"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

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

summarize_report() {
  local report_path="$1"
  "$PYTHON" - "$report_path" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
applied = []
for report in data.get("scene_reports", {}).values():
    applied.extend(report.get("applied", []))
factors = [
    float(item.get("hierarchy_score_factor", 1.0))
    for item in applied
    if "hierarchy_score_factor" in item
]
downweighted = sum(1 for value in factors if value < 0.999)
mean_factor = sum(factors) / max(1, len(factors))
print(f"[REPORT] applied={len(applied)} hierarchy_downweighted={downweighted} mean_factor={mean_factor:.6f}")
PY
}

run_eval() {
  local name="$1"
  local weight="$2"
  local threshold="$3"
  local min_factor="$4"
  local csv_path="$OUT_DIR/${name}.csv"
  local log_path="$OUT_DIR/${name}.log"
  local report_path="$OUT_DIR/reports/${name}.json"
  local cache_dir="$OUT_DIR/cache_${name}"
  echo "[RUN] $name"
  "$PYTHON" run_evaluation.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --path_to_2d_preds ./output/scannet200/bboxes_2d \
    --scene_list "$SCENE_LIST" \
    --backprojection_candidates "$SAM_OUT,$BPR_IN" \
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
    --backprojection_superpoint_min_box_positive_ratio 0.50 \
    --backprojection_superpoint_max_box_negative_ratio 0.50 \
    --backprojection_superpoint_box_min_visible_points 5 \
    --backprojection_superpoint_box_min_views 1 \
    --backprojection_cc_cleanup \
    --backprojection_cc_radius 0.03 \
    --backprojection_cc_min_component_points 50 \
    --backprojection_cc_keep_topk 1 \
    --backprojection_hierarchy_score_weight "$weight" \
    --backprojection_hierarchy_low_occupancy_threshold "$threshold" \
    --backprojection_hierarchy_min_score_factor "$min_factor" \
    --backprojection_report_path "$report_path" \
    --eval_output_file "$csv_path" \
    --eval_prediction_cache_dir "$cache_dir" \
    --eval_cleanup_prediction_cache >"$log_path" 2>&1
  summarize_csv "$name" "$csv_path"
  summarize_report "$report_path"
}

case "$MODE" in
  conservative)
    run_eval hierarchy_w050_t025_min070 0.50 0.25 0.70
    ;;
  stronger)
    run_eval hierarchy_w075_t025_min060 0.75 0.25 0.60
    ;;
  threshold035)
    run_eval hierarchy_w050_t035_min070 0.50 0.35 0.70
    ;;
  all)
    run_eval hierarchy_w050_t025_min070 0.50 0.25 0.70
    run_eval hierarchy_w075_t025_min060 0.75 0.25 0.60
    run_eval hierarchy_w050_t035_min070 0.50 0.35 0.70
    ;;
  *)
    echo "Unknown MODE=$MODE; expected conservative, stronger, threshold035, or all" >&2
    exit 2
    ;;
esac

echo "[DONE] Wrote outputs to $OUT_DIR"
