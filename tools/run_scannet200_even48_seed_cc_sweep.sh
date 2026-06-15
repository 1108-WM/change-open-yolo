#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_seed_cc}"
SAM_FUSED_IN="${SAM_FUSED_IN:-./output/sam_fused_proposals_scannet200_s5_m30_prefilter}"
BPR_IN="${BPR_IN:-./output/backprojection_candidates_scannet200_mv_m20}"
MODE="${MODE:-all}"

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
)

POST_CC_ARGS=(
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

summarize_report() {
  local name="$1"
  local report_path="$2"
  "$PYTHON" - "$name" "$report_path" <<'PY'
import collections
import json
import sys

name, report_path = sys.argv[1], sys.argv[2]
with open(report_path) as f:
    data = json.load(f)
applied = []
skipped = collections.Counter()
seed_changed = 0
seed_enabled = 0
for report in data.get("scene_reports", {}).values():
    applied.extend(report.get("applied", []))
    skipped.update(item.get("reason") for item in report.get("skipped", []))
for item in applied:
    info = item.get("seed_cc_cleanup") or {}
    if info.get("enabled"):
        seed_enabled += 1
        if int(info.get("output_points", 0) or 0) != int(info.get("input_points", 0) or 0):
            seed_changed += 1
print(
    f"[REPORT] {name}: applied={len(applied)} "
    f"seed_cc_enabled={seed_enabled} seed_cc_changed={seed_changed} "
    f"skipped={skipped.most_common(6)}"
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
  local cache_dir="$OUT_DIR/cache_${name}"
  echo "[RUN] $name 预计耗时: $eta"
  "$PYTHON" "${BASE_ARGS[@]}" "$@" \
    --backprojection_report_path "$report_path" \
    --eval_output_file "$csv_path" \
    --eval_prediction_cache_dir "$cache_dir" \
    --eval_cleanup_prediction_cache >"$log_path" 2>&1
  summarize_csv "$name" "$csv_path"
  summarize_report "$name" "$report_path"
}

case "$MODE" in
  seed)
    run_eval "seedcc_k1_m30" "3-6 min on RTX 4090" \
      --backprojection_seed_cc_cleanup \
      --backprojection_seed_cc_radius 0.03 \
      --backprojection_seed_cc_min_component_points 30 \
      --backprojection_seed_cc_keep_topk 1
    run_eval "seedcc_k2_m30" "3-6 min on RTX 4090" \
      --backprojection_seed_cc_cleanup \
      --backprojection_seed_cc_radius 0.03 \
      --backprojection_seed_cc_min_component_points 30 \
      --backprojection_seed_cc_keep_topk 2
    ;;
  seed_post)
    run_eval "seedcc_k1_postcc" "3-6 min on RTX 4090" \
      --backprojection_seed_cc_cleanup \
      --backprojection_seed_cc_radius 0.03 \
      --backprojection_seed_cc_min_component_points 30 \
      --backprojection_seed_cc_keep_topk 1 \
      "${POST_CC_ARGS[@]}"
    run_eval "seedcc_k2_postcc" "3-6 min on RTX 4090" \
      --backprojection_seed_cc_cleanup \
      --backprojection_seed_cc_radius 0.03 \
      --backprojection_seed_cc_min_component_points 30 \
      --backprojection_seed_cc_keep_topk 2 \
      "${POST_CC_ARGS[@]}"
    ;;
  all)
    run_eval "seedcc_k1_m30" "3-6 min on RTX 4090" \
      --backprojection_seed_cc_cleanup \
      --backprojection_seed_cc_radius 0.03 \
      --backprojection_seed_cc_min_component_points 30 \
      --backprojection_seed_cc_keep_topk 1
    run_eval "seedcc_k2_m30" "3-6 min on RTX 4090" \
      --backprojection_seed_cc_cleanup \
      --backprojection_seed_cc_radius 0.03 \
      --backprojection_seed_cc_min_component_points 30 \
      --backprojection_seed_cc_keep_topk 2
    run_eval "seedcc_k1_postcc" "3-6 min on RTX 4090" \
      --backprojection_seed_cc_cleanup \
      --backprojection_seed_cc_radius 0.03 \
      --backprojection_seed_cc_min_component_points 30 \
      --backprojection_seed_cc_keep_topk 1 \
      "${POST_CC_ARGS[@]}"
    run_eval "seedcc_k2_postcc" "3-6 min on RTX 4090" \
      --backprojection_seed_cc_cleanup \
      --backprojection_seed_cc_radius 0.03 \
      --backprojection_seed_cc_min_component_points 30 \
      --backprojection_seed_cc_keep_topk 2 \
      "${POST_CC_ARGS[@]}"
    ;;
  *)
    echo "Unknown MODE=$MODE; expected seed, seed_post, or all" >&2
    exit 2
    ;;
esac

echo "[DONE] Wrote seed-CC sweep outputs to $OUT_DIR"
