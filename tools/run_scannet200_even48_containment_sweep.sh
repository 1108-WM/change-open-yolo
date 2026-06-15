#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_containment}"
MODE="${MODE:-all}"  # current | downweight | carve | remove_large | downweight_cross | remove_large_cross | all

BPR_IN="${BPR_IN:-./output/backprojection_candidates_scannet200_mv_m20}"
SAM_FUSED_IN="${SAM_FUSED_IN:-./output/sam_fused_proposals_scannet200_even48_maskpath}"

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
  --backprojection_superpoint_min_box_positive_ratio 0.50
  --backprojection_superpoint_max_box_negative_ratio 0.50
  --backprojection_superpoint_box_min_visible_points 5
  --backprojection_superpoint_box_min_views 1
  --backprojection_cc_cleanup
  --backprojection_cc_radius 0.03
  --backprojection_cc_min_component_points 50
  --backprojection_cc_keep_topk 1
)

CONTAINMENT_BASE_ARGS=(
  --backprojection_containment_threshold 0.85
  --backprojection_containment_min_area_ratio 1.5
  --backprojection_containment_score_ratio 0.75
  --backprojection_containment_quality_margin 0.0
  --backprojection_containment_min_points 50
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

summarize_report() {
  local name="$1"
  local report_path="$2"
  "$PYTHON" - "$name" "$report_path" <<'PY'
import json
import sys

name, report_path = sys.argv[1], sys.argv[2]
with open(report_path) as f:
    data = json.load(f)
totals = {
    "events": 0,
    "downweighted": 0,
    "carved": 0,
    "removed": 0,
    "carved_points": 0,
    "input_appended": 0,
    "output_appended": 0,
}
for report in data.get("scene_reports", {}).values():
    post = report.get("postprocess", {})
    totals["events"] += len(post.get("containment_events", []))
    totals["downweighted"] += int(post.get("downweighted_containing", 0))
    totals["carved"] += int(post.get("carved_containing", 0))
    totals["removed"] += int(post.get("removed_containing", 0))
    totals["carved_points"] += int(post.get("carved_points", 0))
    totals["input_appended"] += int(post.get("input_appended", 0))
    totals["output_appended"] += int(post.get("output_appended", 0))
print(
    f"[POSTPROCESS] {name}: "
    f"events={totals['events']} downweighted={totals['downweighted']} "
    f"carved={totals['carved']} removed={totals['removed']} "
    f"carved_points={totals['carved_points']} "
    f"appended={totals['input_appended']}->{totals['output_appended']}"
)
PY
}

run_eval() {
  local name="$1"
  shift
  local csv_path="$OUT_DIR/${name}.csv"
  local log_path="$OUT_DIR/${name}.log"
  local report_path="$OUT_DIR/reports/${name}.json"
  local cache_dir="$OUT_DIR/cache_${name}"
  echo "[RUN] $name"
  "$PYTHON" "${BASE_EVAL_ARGS[@]}" \
    "${COMMON_BPR_ARGS[@]}" \
    "$@" \
    --backprojection_report_path "$report_path" \
    --eval_output_file "$csv_path" \
    --eval_prediction_cache_dir "$cache_dir" \
    --eval_cleanup_prediction_cache >"$log_path" 2>&1
  summarize_csv "$name" "$csv_path"
  summarize_report "$name" "$report_path"
}

case "$MODE" in
  current)
    run_eval current
    ;;
  downweight)
    run_eval containment_downweight \
      "${CONTAINMENT_BASE_ARGS[@]}" \
      --backprojection_containment_action downweight \
      --backprojection_containment_score_factor 0.50
    ;;
  carve)
    run_eval containment_carve \
      "${CONTAINMENT_BASE_ARGS[@]}" \
      --backprojection_containment_action carve
    ;;
  remove_large)
    run_eval containment_remove_large \
      "${CONTAINMENT_BASE_ARGS[@]}" \
      --backprojection_containment_action remove_large
    ;;
  downweight_cross)
    run_eval containment_downweight_cross \
      "${CONTAINMENT_BASE_ARGS[@]}" \
      --no-backprojection_postprocess_same_class_only \
      --backprojection_containment_action downweight \
      --backprojection_containment_score_factor 0.50
    ;;
  remove_large_cross)
    run_eval containment_remove_large_cross \
      "${CONTAINMENT_BASE_ARGS[@]}" \
      --no-backprojection_postprocess_same_class_only \
      --backprojection_containment_action remove_large
    ;;
  all)
    run_eval current
    run_eval containment_downweight \
      "${CONTAINMENT_BASE_ARGS[@]}" \
      --backprojection_containment_action downweight \
      --backprojection_containment_score_factor 0.50
    run_eval containment_carve \
      "${CONTAINMENT_BASE_ARGS[@]}" \
      --backprojection_containment_action carve
    run_eval containment_remove_large \
      "${CONTAINMENT_BASE_ARGS[@]}" \
      --backprojection_containment_action remove_large
    run_eval containment_downweight_cross \
      "${CONTAINMENT_BASE_ARGS[@]}" \
      --no-backprojection_postprocess_same_class_only \
      --backprojection_containment_action downweight \
      --backprojection_containment_score_factor 0.50
    run_eval containment_remove_large_cross \
      "${CONTAINMENT_BASE_ARGS[@]}" \
      --no-backprojection_postprocess_same_class_only \
      --backprojection_containment_action remove_large
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

echo "[DONE] Wrote containment sweep outputs to $OUT_DIR"
