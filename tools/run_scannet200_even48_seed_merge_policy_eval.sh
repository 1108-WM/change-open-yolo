#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
SPLIT_NAME="${SPLIT_NAME:-even48}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_seed_merge_policy}"
BPR_IN="${BPR_IN:-./output/backprojection_candidates_scannet200_mv_m20}"
MODE="${MODE:-all}"
RUN_LABEL_PREFIX="${RUN_LABEL_PREFIX:-}"

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

export_sam() {
  local name="$1"
  local policy="$2"
  local topk="$3"
  shift 3
  local extra_export_args=("$@")
  local export_dir="$ROOT_DIR/output/sam_fused_proposals_scannet200_${SPLIT_NAME}_${name}_seed_${policy}_k${topk}"
  echo "[EXPORT] $name 预计耗时: 4-8 min on RTX 4090" >&2
  "$PYTHON" tools/export_sam_fused_proposals.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --path_to_2d_preds ./output/scannet200/bboxes_2d \
    --scene_list "$SCENE_LIST" \
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
    --seed_merge_policy "$policy" \
    --seed_merge_topk "$topk" \
    --export_max_existing_iou 0.30 \
    --export_max_seed_in_existing_mask_ratio 0.70 \
    "${extra_export_args[@]}" \
    >"$OUT_DIR/${name}_export.log" 2>&1
  summarize_export "$name" "$export_dir/sam_fused_proposals_summary.json" >&2
  echo "$export_dir"
}

run_eval() {
  local name="$1"
  local sam_dir="$2"
  shift 2
  local extra_eval_args=("$@")
  local csv_path="$OUT_DIR/${name}.csv"
  local log_path="$OUT_DIR/${name}.log"
  local report_path="$OUT_DIR/reports/${name}.json"
  local cache_dir="$OUT_DIR/cache_${name}"
  echo "[RUN] $name 预计耗时: 3-6 min on RTX 4090"
  "$PYTHON" run_evaluation.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --path_to_2d_preds ./output/scannet200/bboxes_2d \
    --scene_list "$SCENE_LIST" \
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
    "${extra_eval_args[@]}" \
    --backprojection_report_path "$report_path" \
    --eval_output_file "$csv_path" \
    --eval_prediction_cache_dir "$cache_dir" \
    --eval_cleanup_prediction_cache >"$log_path" 2>&1
  summarize_csv "$name" "$csv_path"
}

case "$MODE" in
  best)
    sam_dir="$(export_sam seed_best_view best_view 1)"
    run_eval "${RUN_LABEL_PREFIX}seed_best_view" "$sam_dir"
    ;;
  topk2)
    sam_dir="$(export_sam seed_topk2 topk_priority 2)"
    run_eval "${RUN_LABEL_PREFIX}seed_topk2" "$sam_dir"
    ;;
  topk3)
    sam_dir="$(export_sam seed_topk3 topk_priority 3)"
    run_eval "${RUN_LABEL_PREFIX}seed_topk3" "$sam_dir"
    ;;
  depthc)
    sam_dir="$(export_sam seed_depthc union 1 --seed_depth_cluster --seed_depth_cluster_bin_size 0.10 --seed_depth_cluster_window_bins 1 --seed_depth_cluster_min_keep_ratio 0.25)"
    run_eval "${RUN_LABEL_PREFIX}seed_depthc" "$sam_dir"
    ;;
  depthc_adapt)
    sam_dir="$(export_sam seed_depthc_adapt union 1 --seed_depth_cluster --seed_depth_cluster_bin_size 0.10 --seed_depth_cluster_window_bins 1 --seed_depth_cluster_min_keep_ratio 0.25 --seed_depth_cluster_min_removed_ratio 0.20 --seed_depth_cluster_max_removed_ratio 0.50)"
    run_eval "${RUN_LABEL_PREFIX}seed_depthc_adapt" "$sam_dir"
    ;;
  spbox)
    sam_dir="$(export_sam seed_spbox union 1)"
    run_eval "${RUN_LABEL_PREFIX}seed_spbox" "$sam_dir" \
      --backprojection_superpoint_min_box_positive_ratio 0.30 \
      --backprojection_superpoint_max_box_negative_ratio 0.70 \
      --backprojection_superpoint_box_min_visible_points 5 \
      --backprojection_superpoint_box_min_views 1 \
      --backprojection_superpoint_box_padding_ratio 0.05
    ;;
  all)
    sam_dir="$(export_sam seed_best_view best_view 1)"
    run_eval "${RUN_LABEL_PREFIX}seed_best_view" "$sam_dir"
    sam_dir="$(export_sam seed_topk2 topk_priority 2)"
    run_eval "${RUN_LABEL_PREFIX}seed_topk2" "$sam_dir"
    ;;
  *)
    echo "Unknown MODE=$MODE; expected best, topk2, topk3, depthc, depthc_adapt, spbox, or all" >&2
    exit 2
    ;;
esac

echo "[DONE] Wrote seed merge policy outputs to $OUT_DIR"
