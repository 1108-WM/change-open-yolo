#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/sam_erode5_96_confirm}"
BPR_IN="${BPR_IN:-./output/backprojection_candidates_scannet200_mv_m20}"
EVEN_LIST="${EVEN_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even96.txt}"
ODD_LIST="${ODD_LIST:-$ROOT_DIR/output/scannet200/scene_splits/odd96.txt}"
ERODE_PIXELS="${ERODE_PIXELS:-5}"

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
  local name="$1"
  local summary_path="$2"
  "$PYTHON" - "$name" "$summary_path" <<'PY'
import json
import sys

name, summary_path = sys.argv[1], sys.argv[2]
with open(summary_path) as f:
    data = json.load(f)
num_candidates = sum(int(item.get("num_candidates", 0)) for item in data.get("scenes", []))
raw = sum(int(item.get("raw_observations", 0)) for item in data.get("scenes", []))
print(f"[EXPORT_RESULT] {name}: candidates={num_candidates} raw_observations={raw}")
PY
}

export_sam_erode() {
  local split_name="$1"
  local scene_list="$2"
  local export_dir="$ROOT_DIR/output/sam_fused_proposals_scannet200_${split_name}_erode${ERODE_PIXELS}"
  echo "[EXPORT] ${split_name}_sam_erode${ERODE_PIXELS} 预计耗时: 8-15 min on RTX 4090"
  "$PYTHON" tools/export_sam_fused_proposals.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --path_to_2d_preds ./output/scannet200/bboxes_2d \
    --scene_list "$scene_list" \
    --output_dir "$export_dir" \
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
    --sam_mask_erode_pixels "$ERODE_PIXELS" \
    --sam_mask_erode_min_area_ratio 0.15 \
    >"$OUT_DIR/${split_name}_sam_erode${ERODE_PIXELS}_export.log" 2>&1
  summarize_export "${split_name}_sam_erode${ERODE_PIXELS}" "$export_dir/sam_fused_proposals_summary.json"
}

run_eval() {
  local split_name="$1"
  local scene_list="$2"
  local sam_dir="$ROOT_DIR/output/sam_fused_proposals_scannet200_${split_name}_erode${ERODE_PIXELS}"
  local name="${split_name}_sam_erode${ERODE_PIXELS}_cc"
  local csv_path="$OUT_DIR/${name}.csv"
  local log_path="$OUT_DIR/${name}.log"
  local report_path="$OUT_DIR/reports/${name}.json"
  local cache_dir="$OUT_DIR/cache_${name}"
  echo "[RUN] $name 预计耗时: 6-10 min on RTX 4090"
  "$PYTHON" run_evaluation.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --path_to_2d_preds ./output/scannet200/bboxes_2d \
    --scene_list "$scene_list" \
    --backprojection_candidates "$sam_dir,$BPR_IN" \
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

export_sam_erode even96 "$EVEN_LIST"
run_eval even96 "$EVEN_LIST"
export_sam_erode odd96 "$ODD_LIST"
run_eval odd96 "$ODD_LIST"

echo "[DONE] Wrote SAM erode${ERODE_PIXELS} 96-scene confirmation outputs to $OUT_DIR"
