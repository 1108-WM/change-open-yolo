#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_sam_mask_support}"
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

summarize_export() {
  local summary_path="$1"
  "$PYTHON" - "$summary_path" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path) as f:
    data = json.load(f)
num_candidates = sum(int(item.get("num_candidates", 0)) for item in data.get("scenes", []))
raw = sum(int(item.get("raw_observations", 0)) for item in data.get("scenes", []))
print(f"[EXPORT_RESULT] candidates={num_candidates} raw_observations={raw}")
PY
}

export_sam_with_masks() {
  if [[ -f "$SAM_OUT/sam_fused_proposals_summary.json" ]]; then
    echo "[EXPORT] Reusing $SAM_OUT"
    summarize_export "$SAM_OUT/sam_fused_proposals_summary.json"
    return
  fi

  echo "[EXPORT] Writing SAM-fused candidates with saved 2D masks to $SAM_OUT"
  "$PYTHON" tools/export_sam_fused_proposals.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --path_to_2d_preds ./output/scannet200/bboxes_2d \
    --scene_list "$SCENE_LIST" \
    --output_dir "$SAM_OUT" \
    --detection_score_th 0.45 \
    --min_seed_points 80 \
    --max_box_area_ratio 0.30 \
    --frame_stride 5 \
    --max_detections_per_frame 8 \
    --merge_iou 0.15 \
    --max_candidates_per_scene 30 \
    --blocked_classes rug \
    --ranking_policy support_priority \
    --sam_multimask_topk 1 \
    --export_max_existing_iou 0.30 \
    --export_max_seed_in_existing_mask_ratio 0.70 \
    >"$OUT_DIR/export_maskpath.log" 2>&1
  summarize_export "$SAM_OUT/sam_fused_proposals_summary.json"
}

run_eval() {
  local name="$1"
  local positive="$2"
  local negative="$3"
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
    --backprojection_superpoint_min_box_positive_ratio "$positive" \
    --backprojection_superpoint_max_box_negative_ratio "$negative" \
    --backprojection_superpoint_box_min_visible_points 5 \
    --backprojection_superpoint_box_min_views 1 \
    --backprojection_cc_cleanup \
    --backprojection_cc_radius 0.03 \
    --backprojection_cc_min_component_points 50 \
    --backprojection_cc_keep_topk 1 \
    --backprojection_report_path "$report_path" \
    --eval_output_file "$csv_path" \
    --eval_prediction_cache_dir "$cache_dir" \
    --eval_cleanup_prediction_cache >"$log_path" 2>&1
  summarize_csv "$name" "$csv_path"
}

export_sam_with_masks

case "$MODE" in
  mask050)
    run_eval mask_support_pos050_neg050 0.50 0.50
    ;;
  mask065)
    run_eval mask_support_pos065_neg035 0.65 0.35
    ;;
  all)
    run_eval mask_support_pos050_neg050 0.50 0.50
    run_eval mask_support_pos065_neg035 0.65 0.35
    ;;
  *)
    echo "Unknown MODE=$MODE; expected mask050, mask065, or all" >&2
    exit 2
    ;;
esac

echo "[DONE] Wrote outputs to $OUT_DIR"
