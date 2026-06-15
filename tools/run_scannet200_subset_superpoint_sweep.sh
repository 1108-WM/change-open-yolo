#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even96.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even96_superpoint}"

mkdir -p "$OUT_DIR/reports"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

BASE_ARGS=(
  run_evaluation.py
  --dataset_name scannet200
  --path_to_3d_masks ./output/scannet200/scannet200_masks
  --path_to_2d_preds ./output/scannet200/bboxes_2d
  --scene_list "$SCENE_LIST"
)

COMMON_ARGS=(
  --backprojection_candidates ./output/sam_fused_proposals_scannet200_s5_m30_prefilter,./output/backprojection_candidates_scannet200_mv_m20
  --backprojection_min_score 0.50
  --backprojection_min_seed_points 80
  --backprojection_max_existing_iou 0.30
  --backprojection_max_seed_in_existing_mask_ratio 0.70
  --backprojection_max_candidates_per_scene 15
  --backprojection_score_scale 2.00
  --no-backprojection_use_candidate_fusion_score
  --backprojection_blocked_classes rug
  --backprojection_source_score_scales sam_fused=1.2,bpr=1.0
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
  if [[ ! -f "$report_path" ]]; then
    return
  fi
  "$PYTHON" - "$name" "$report_path" <<'PY'
import collections
import json
import sys

name, report_path = sys.argv[1], sys.argv[2]
with open(report_path) as f:
    data = json.load(f)
total = collections.Counter()
fallback = collections.Counter()
for report in data.get("scene_reports", {}).values():
    for applied in report.get("applied", []):
        sp = applied.get("superpoint_refine", {})
        if not sp.get("enabled"):
            continue
        total["enabled"] += 1
        total["input_points"] += int(sp.get("input_points", 0))
        total["output_points"] += int(sp.get("output_points", 0))
        total["selected_segments"] += int(sp.get("selected_segments", 0))
        if sp.get("fallback"):
            fallback[sp.get("fallback")] += 1
print(
    f"[SP] {name}: enabled={total['enabled']} "
    f"input_pts={total['input_points']} output_pts={total['output_points']} "
    f"segments={total['selected_segments']} fallbacks={dict(fallback)}"
)
PY
}

run_eval() {
  local name="$1"
  shift
  local csv_path="$OUT_DIR/${name}.csv"
  local log_path="$OUT_DIR/${name}.log"
  local report_path="$OUT_DIR/reports/${name}.json"

  echo "[RUN] $name"
  if [[ "$name" == "baseline" ]]; then
    "$PYTHON" "${BASE_ARGS[@]}" "$@" --eval_output_file "$csv_path" >"$log_path" 2>&1
  else
    "$PYTHON" "${BASE_ARGS[@]}" "$@" --backprojection_report_path "$report_path" --eval_output_file "$csv_path" >"$log_path" 2>&1
  fi
  summarize_csv "$name" "$csv_path"
  summarize_report "$name" "$report_path"
}

run_eval baseline

run_eval sam_bpr_scale200 \
  "${COMMON_ARGS[@]}"

run_eval sam_bpr_sp_cov020_x2 \
  "${COMMON_ARGS[@]}" \
  --backprojection_superpoint_refine \
  --backprojection_superpoint_min_coverage 0.20 \
  --backprojection_superpoint_max_expansion_ratio 2.0

run_eval sam_bpr_sp_cov030_x2 \
  "${COMMON_ARGS[@]}" \
  --backprojection_superpoint_refine \
  --backprojection_superpoint_min_coverage 0.30 \
  --backprojection_superpoint_max_expansion_ratio 2.0

run_eval sam_bpr_sp_cov040_x2 \
  "${COMMON_ARGS[@]}" \
  --backprojection_superpoint_refine \
  --backprojection_superpoint_min_coverage 0.40 \
  --backprojection_superpoint_max_expansion_ratio 2.0

run_eval sam_bpr_sp_cov030_x3 \
  "${COMMON_ARGS[@]}" \
  --backprojection_superpoint_refine \
  --backprojection_superpoint_min_coverage 0.30 \
  --backprojection_superpoint_max_expansion_ratio 3.0

run_eval sam_bpr_cap12_3_sp_cov030_x2 \
  "${COMMON_ARGS[@]}" \
  --backprojection_source_priorities sam_fused=2.0,bpr=1.0 \
  --backprojection_source_max_candidates sam_fused=12,bpr=3 \
  --backprojection_superpoint_refine \
  --backprojection_superpoint_min_coverage 0.30 \
  --backprojection_superpoint_max_expansion_ratio 2.0

echo "[DONE] Wrote superpoint sweep outputs to $OUT_DIR"
