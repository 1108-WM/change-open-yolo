#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even96.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even96_merge}"

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
import json
import sys

name, report_path = sys.argv[1], sys.argv[2]
with open(report_path) as f:
    data = json.load(f)
total = {"input_appended": 0, "merged": 0, "removed_included": 0, "output_appended": 0}
enabled = 0
for report in data.get("scene_reports", {}).values():
    post = report.get("postprocess", {})
    if post.get("enabled"):
        enabled += 1
    for key in total:
        total[key] += int(post.get(key, 0))
print(
    f"[POST] {name}: scenes={enabled} "
    f"input={total['input_appended']} merged={total['merged']} "
    f"removed={total['removed_included']} output={total['output_appended']}"
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

run_eval sam_bpr_merge030_incl099 \
  "${COMMON_ARGS[@]}" \
  --backprojection_merge_iou 0.30 \
  --backprojection_inclusion_threshold 0.99

run_eval sam_bpr_merge020_incl099 \
  "${COMMON_ARGS[@]}" \
  --backprojection_merge_iou 0.20 \
  --backprojection_inclusion_threshold 0.99

run_eval sam_bpr_merge040_incl099 \
  "${COMMON_ARGS[@]}" \
  --backprojection_merge_iou 0.40 \
  --backprojection_inclusion_threshold 0.99

run_eval sam_bpr_incl099_only \
  "${COMMON_ARGS[@]}" \
  --backprojection_inclusion_threshold 0.99

run_eval sam_bpr_cap12_3_merge030_incl099 \
  "${COMMON_ARGS[@]}" \
  --backprojection_source_priorities sam_fused=2.0,bpr=1.0 \
  --backprojection_source_max_candidates sam_fused=12,bpr=3 \
  --backprojection_merge_iou 0.30 \
  --backprojection_inclusion_threshold 0.99

run_eval sam_bpr_m30_merge030_incl099 \
  "${COMMON_ARGS[@]}" \
  --backprojection_max_candidates_per_scene 30 \
  --backprojection_merge_iou 0.30 \
  --backprojection_inclusion_threshold 0.99

echo "[DONE] Wrote merge sweep outputs to $OUT_DIR"
