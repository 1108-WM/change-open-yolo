#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_hierarchy_substitution}"
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
post = [report.get("postprocess", {}) for report in data.get("scene_reports", {}).values()]
removed = sum(int(item.get("hierarchy_removed_parents", 0)) for item in post)
events = sum(len(item.get("hierarchy_substitution_events", [])) for item in post)
input_appended = sum(int(item.get("input_appended", 0)) for item in post)
output_appended = sum(int(item.get("output_appended", 0)) for item in post)
print(
    f"[REPORT] hierarchy_removed_parents={removed} events={events} "
    f"appended={input_appended}->{output_appended}"
)
PY
}

run_eval() {
  local name="$1"
  local child_coverage="$2"
  local parent_exclusive="$3"
  local min_children="$4"
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
    --backprojection_hierarchy_substitution_action remove_parent \
    --backprojection_hierarchy_substitution_min_child_coverage "$child_coverage" \
    --backprojection_hierarchy_substitution_max_parent_exclusive_ratio "$parent_exclusive" \
    --backprojection_hierarchy_substitution_min_area_ratio 1.2 \
    --backprojection_hierarchy_substitution_min_children "$min_children" \
    --backprojection_report_path "$report_path" \
    --eval_output_file "$csv_path" \
    --eval_prediction_cache_dir "$cache_dir" \
    --eval_cleanup_prediction_cache >"$log_path" 2>&1
  summarize_csv "$name" "$csv_path"
  summarize_report "$report_path"
}

case "$MODE" in
  strict)
    run_eval hierarchy_remove_parent_cov080_ex020_min2 0.80 0.20 2
    ;;
  conservative)
    run_eval hierarchy_remove_parent_cov085_ex015_min1 0.85 0.15 1
    ;;
  relaxed)
    run_eval hierarchy_remove_parent_cov075_ex030_min1 0.75 0.30 1
    ;;
  all)
    run_eval hierarchy_remove_parent_cov080_ex020_min2 0.80 0.20 2
    run_eval hierarchy_remove_parent_cov085_ex015_min1 0.85 0.15 1
    run_eval hierarchy_remove_parent_cov075_ex030_min1 0.75 0.30 1
    ;;
  *)
    echo "Unknown MODE=$MODE; expected strict, conservative, relaxed, or all" >&2
    exit 2
    ;;
esac

echo "[DONE] Wrote outputs to $OUT_DIR"
