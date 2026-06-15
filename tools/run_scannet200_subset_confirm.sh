#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even96.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even96_confirm}"

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
  --backprojection_min_score 0.50
  --backprojection_min_seed_points 80
  --backprojection_max_existing_iou 0.30
  --backprojection_max_seed_in_existing_mask_ratio 0.70
  --backprojection_max_candidates_per_scene 15
  --backprojection_score_scale 2.00
  --no-backprojection_use_candidate_fusion_score
  --backprojection_blocked_classes rug
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

run_eval sam_only_scale200 \
  "${COMMON_ARGS[@]}" \
  --backprojection_candidates ./output/sam_fused_proposals_scannet200_s5_m30_prefilter \
  --backprojection_source_score_scales sam_fused=1.2

run_eval sam_bpr_scale200 \
  "${COMMON_ARGS[@]}" \
  --backprojection_candidates ./output/sam_fused_proposals_scannet200_s5_m30_prefilter,./output/backprojection_candidates_scannet200_mv_m20 \
  --backprojection_source_score_scales sam_fused=1.2,bpr=1.0

echo "[DONE] Wrote confirm outputs to $OUT_DIR"
