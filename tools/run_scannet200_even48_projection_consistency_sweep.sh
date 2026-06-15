#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_projection_consistency}"
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
import statistics
import sys

name, report_path = sys.argv[1], sys.argv[2]
with open(report_path) as f:
    data = json.load(f)
applied = []
skipped = collections.Counter()
for report in data.get("scene_reports", {}).values():
    applied.extend(report.get("applied", []))
    skipped.update(item.get("reason") for item in report.get("skipped", []))
usable = []
box_iou = []
point_ratio = []
score_factor = []
for item in applied:
    info = item.get("projected_box_consistency") or {}
    if info.get("enabled") and info.get("reason") == "ok":
        usable.append(int(info.get("usable_view_count", 0)))
        box_iou.append(float(info.get("mean_box_iou", 0.0)))
        point_ratio.append(float(info.get("mean_point_in_box_ratio", 0.0)))
    score_factor.append(float(item.get("projection_score_factor", 1.0)))
def mean(values):
    return sum(values) / len(values) if values else 0.0
print(
    f"[REPORT] {name}: applied={len(applied)} "
    f"projection_ok={len(box_iou)} "
    f"mean_box_iou={mean(box_iou):.3f} "
    f"mean_point_ratio={mean(point_ratio):.3f} "
    f"mean_proj_score_factor={mean(score_factor):.3f} "
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
  baseline)
    run_eval "cc_clean" "3-6 min on RTX 4090"
    ;;
  soft)
    run_eval "proj_score_w050" "3-6 min on RTX 4090" \
      --backprojection_projection_consistency_score_weight 0.50
    run_eval "proj_score_w100" "3-6 min on RTX 4090" \
      --backprojection_projection_consistency_score_weight 1.00
    ;;
  hard)
    run_eval "proj_ratio080" "3-6 min on RTX 4090" \
      --backprojection_projection_consistency_min_point_ratio 0.80 \
      --backprojection_projection_consistency_min_views 1
    run_eval "proj_iou010_ratio080" "3-6 min on RTX 4090" \
      --backprojection_projection_consistency_min_box_iou 0.10 \
      --backprojection_projection_consistency_min_point_ratio 0.80 \
      --backprojection_projection_consistency_min_views 1
    ;;
  all)
    run_eval "cc_clean" "3-6 min on RTX 4090"
    run_eval "proj_score_w050" "3-6 min on RTX 4090" \
      --backprojection_projection_consistency_score_weight 0.50
    run_eval "proj_score_w100" "3-6 min on RTX 4090" \
      --backprojection_projection_consistency_score_weight 1.00
    run_eval "proj_ratio080" "3-6 min on RTX 4090" \
      --backprojection_projection_consistency_min_point_ratio 0.80 \
      --backprojection_projection_consistency_min_views 1
    run_eval "proj_iou010_ratio080" "3-6 min on RTX 4090" \
      --backprojection_projection_consistency_min_box_iou 0.10 \
      --backprojection_projection_consistency_min_point_ratio 0.80 \
      --backprojection_projection_consistency_min_views 1
    ;;
  *)
    echo "Unknown MODE=$MODE; expected baseline, soft, hard, or all" >&2
    exit 2
    ;;
esac

echo "[DONE] Wrote projection-consistency sweep outputs to $OUT_DIR"
