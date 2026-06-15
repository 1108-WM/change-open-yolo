#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even96.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even96_refine}"

mkdir -p "$OUT_DIR"
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

run_eval() {
  local name="$1"
  shift
  local csv_path="$OUT_DIR/${name}.csv"
  local log_path="$OUT_DIR/${name}.log"

  echo "[RUN] $name"
  "$PYTHON" "${BASE_ARGS[@]}" "$@" --eval_output_file "$csv_path" >"$log_path" 2>&1
  summarize_csv "$name" "$csv_path"
}

run_eval baseline

run_eval sam_bpr_scale200 \
  "${COMMON_ARGS[@]}"

run_eval sam_bpr_samprio \
  "${COMMON_ARGS[@]}" \
  --backprojection_source_priorities sam_fused=2.0,bpr=1.0

run_eval sam_bpr_samprio_cap10_5 \
  "${COMMON_ARGS[@]}" \
  --backprojection_source_priorities sam_fused=2.0,bpr=1.0 \
  --backprojection_source_max_candidates sam_fused=10,bpr=5

run_eval sam_bpr_samprio_cap12_3 \
  "${COMMON_ARGS[@]}" \
  --backprojection_source_priorities sam_fused=2.0,bpr=1.0 \
  --backprojection_source_max_candidates sam_fused=12,bpr=3

run_eval sam_bpr_seed120 \
  "${COMMON_ARGS[@]}" \
  --backprojection_min_seed_points 120

run_eval sam_bpr_seed160 \
  "${COMMON_ARGS[@]}" \
  --backprojection_min_seed_points 160

run_eval sam_bpr_novel050 \
  "${COMMON_ARGS[@]}" \
  --backprojection_max_seed_in_existing_mask_ratio 0.50

run_eval sam_bpr_iou020 \
  "${COMMON_ARGS[@]}" \
  --backprojection_max_existing_iou 0.20

run_eval sam_bpr_quality008 \
  "${COMMON_ARGS[@]}" \
  --backprojection_quality_sort \
  --backprojection_min_quality_score 0.08

run_eval sam_bpr_quality010 \
  "${COMMON_ARGS[@]}" \
  --backprojection_quality_sort \
  --backprojection_min_quality_score 0.10

run_eval sam_bpr_fusion250 \
  "${COMMON_ARGS[@]}" \
  --backprojection_score_scale 2.50 \
  --backprojection_use_candidate_fusion_score

run_eval sam_bpr_fusion300 \
  "${COMMON_ARGS[@]}" \
  --backprojection_score_scale 3.00 \
  --backprojection_use_candidate_fusion_score

echo "[DONE] Wrote refine outputs to $OUT_DIR"
