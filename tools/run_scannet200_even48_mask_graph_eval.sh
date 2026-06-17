#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_mask_graph}"
BPR_IN="${BPR_IN:-./output/backprojection_candidates_scannet200_mv_m20}"
MASK_GRAPH_OUT="${MASK_GRAPH_OUT:-$ROOT_DIR/output/mask_graph_proposals_scannet200_even48_s5_m30}"
MODE="${MODE:-default}"
EXPORT_MAX_CANDIDATES="${EXPORT_MAX_CANDIDATES:-30}"
GRAPH_MIN_SEED_IOU="${GRAPH_MIN_SEED_IOU:-0.03}"
GRAPH_MIN_SEED_CONTAINMENT="${GRAPH_MIN_SEED_CONTAINMENT:-0.18}"
GRAPH_MIN_REFERENCE_COVERAGE="${GRAPH_MIN_REFERENCE_COVERAGE:-0.20}"
GRAPH_SPATIAL_SIGMA="${GRAPH_SPATIAL_SIGMA:-0.35}"
GRAPH_EDGE_SCORE_THRESHOLD="${GRAPH_EDGE_SCORE_THRESHOLD:-0.35}"
GRAPH_MIN_CLUSTER_OBSERVATIONS="${GRAPH_MIN_CLUSTER_OBSERVATIONS:-2}"
GRAPH_KEEP_SINGLETONS="${GRAPH_KEEP_SINGLETONS:-0}"
GRAPH_MAX_VIEWS_PER_CLUSTER="${GRAPH_MAX_VIEWS_PER_CLUSTER:-4}"
GRAPH_MIN_NEW_SEED_RATIO="${GRAPH_MIN_NEW_SEED_RATIO:-0.05}"
SOURCE_LIMITS_GRAPH_BPR="${SOURCE_LIMITS_GRAPH_BPR:-mask_graph=12,bpr=3}"
SOURCE_LIMITS_GRAPH_ONLY="${SOURCE_LIMITS_GRAPH_ONLY:-mask_graph=15}"

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
scenes = data.get("scenes", [])
num_candidates = sum(int(item.get("num_candidates", 0)) for item in scenes)
raw = sum(int(item.get("raw_observations", 0)) for item in scenes)
edges = sum(int(item.get("graph_edges", 0)) for item in scenes)
components = sum(int(item.get("graph_components", 0)) for item in scenes)
print(
    f"[EXPORT_RESULT] candidates={num_candidates} raw_observations={raw} "
    f"graph_edges={edges} graph_components={components}"
)
PY
}

export_mask_graph() {
  if [[ -f "$MASK_GRAPH_OUT/mask_graph_proposals_summary.json" ]]; then
    echo "[EXPORT] Reusing $MASK_GRAPH_OUT"
    summarize_export "$MASK_GRAPH_OUT/mask_graph_proposals_summary.json"
    return
  fi

  echo "[EXPORT] Writing mask-graph candidates to $MASK_GRAPH_OUT"
  local singleton_flag="--no-graph_keep_singletons"
  if [[ "$GRAPH_KEEP_SINGLETONS" == "1" || "$GRAPH_KEEP_SINGLETONS" == "true" ]]; then
    singleton_flag="--graph_keep_singletons"
  fi
  "$PYTHON" tools/export_mask_graph_proposals.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --path_to_2d_preds ./output/scannet200/bboxes_2d \
    --scene_list "$SCENE_LIST" \
    --output_dir "$MASK_GRAPH_OUT" \
    --detection_score_th 0.45 \
    --min_seed_points 80 \
    --max_box_area_ratio 0.30 \
    --frame_stride 5 \
    --max_detections_per_frame 8 \
    --max_candidates_per_scene "$EXPORT_MAX_CANDIDATES" \
    --blocked_classes rug \
    --ranking_policy graph_priority \
    --sam_multimask_topk 1 \
    --graph_same_class_only \
    --graph_min_seed_iou "$GRAPH_MIN_SEED_IOU" \
    --graph_min_seed_containment "$GRAPH_MIN_SEED_CONTAINMENT" \
    --graph_min_reference_coverage "$GRAPH_MIN_REFERENCE_COVERAGE" \
    --graph_spatial_sigma "$GRAPH_SPATIAL_SIGMA" \
    --graph_edge_score_threshold "$GRAPH_EDGE_SCORE_THRESHOLD" \
    --graph_min_cluster_observations "$GRAPH_MIN_CLUSTER_OBSERVATIONS" \
    "$singleton_flag" \
    --graph_max_views_per_cluster "$GRAPH_MAX_VIEWS_PER_CLUSTER" \
    --graph_min_new_seed_ratio "$GRAPH_MIN_NEW_SEED_RATIO" \
    --export_max_existing_iou 0.30 \
    --export_max_seed_in_existing_mask_ratio 0.70 \
    >"$OUT_DIR/export_mask_graph.log" 2>&1
  summarize_export "$MASK_GRAPH_OUT/mask_graph_proposals_summary.json"
}

run_eval() {
  local name="$1"
  local candidates="$2"
  local source_scales="$3"
  local source_priorities="$4"
  local source_limits="$5"
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
    --backprojection_candidates "$candidates" \
    --backprojection_min_score 0.50 \
    --backprojection_min_seed_points 80 \
    --backprojection_max_existing_iou 0.30 \
    --backprojection_max_seed_in_existing_mask_ratio 0.70 \
    --backprojection_max_candidates_per_scene 15 \
    --backprojection_score_scale 2.00 \
    --no-backprojection_use_candidate_fusion_score \
    --backprojection_blocked_classes rug \
    --backprojection_source_score_scales "$source_scales" \
    --backprojection_source_priorities "$source_priorities" \
    --backprojection_source_max_candidates "$source_limits" \
    --backprojection_superpoint_refine \
    --backprojection_superpoint_min_coverage 0.30 \
    --backprojection_superpoint_max_expansion_ratio 3.0 \
    --backprojection_superpoint_min_view_siou 0.60 \
    --backprojection_superpoint_min_box_positive_ratio 0.50 \
    --backprojection_superpoint_max_box_negative_ratio 0.50 \
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

export_mask_graph

case "$MODE" in
  graph_only)
    run_eval mask_graph_only \
      "$MASK_GRAPH_OUT" \
      "mask_graph=1.2" \
      "mask_graph=2.0" \
      "$SOURCE_LIMITS_GRAPH_ONLY"
    ;;
  graph_bpr)
    run_eval mask_graph_bpr \
      "$MASK_GRAPH_OUT,$BPR_IN" \
      "mask_graph=1.2,bpr=1.0" \
      "mask_graph=2.0,bpr=1.0" \
      "$SOURCE_LIMITS_GRAPH_BPR"
    ;;
  default|all)
    run_eval mask_graph_bpr \
      "$MASK_GRAPH_OUT,$BPR_IN" \
      "mask_graph=1.2,bpr=1.0" \
      "mask_graph=2.0,bpr=1.0" \
      "$SOURCE_LIMITS_GRAPH_BPR"
    ;;
  *)
    echo "Unknown MODE=$MODE; expected graph_only, graph_bpr, default, or all" >&2
    exit 2
    ;;
esac
