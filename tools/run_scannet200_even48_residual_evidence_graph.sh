#!/usr/bin/env bash
set -euo pipefail

# 只从已冻结的帧级残差读取证据，构图并生成轨迹；不运行 SAM、不读取 GT、不写最终候选。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
RESIDUAL_ROOT="${RESIDUAL_ROOT:-$ROOT_DIR/output/residual_sam_annotations_even48_f30_20260725}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/scannet200}"
PROCESSED_SCENE_ROOT="${PROCESSED_SCENE_ROOT:-$ROOT_DIR/data/scannet200}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/pretrained/config_scannet200.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/output/residual_evidence_graph_even48_f30_v0_20260726}"
RESIDUAL_MODE="${RESIDUAL_MODE:-after_any}"

cd "$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

"$PYTHON" tools/verify_residual_evidence_graph_entry.py \
  --scene_list "$SCENE_LIST" \
  --residual_root "$RESIDUAL_ROOT" \
  --dataset_root "$DATASET_ROOT" \
  --processed_scene_root "$PROCESSED_SCENE_ROOT" \
  --config_path "$CONFIG_PATH" \
  --output_root "$OUTPUT_ROOT" \
  --residual_mode "$RESIDUAL_MODE"

"$PYTHON" tools/build_residual_evidence_graph.py \
  --scene_list "$SCENE_LIST" \
  --residual_root "$RESIDUAL_ROOT" \
  --dataset_root "$DATASET_ROOT" \
  --processed_scene_root "$PROCESSED_SCENE_ROOT" \
  --config_path "$CONFIG_PATH" \
  --output_root "$OUTPUT_ROOT" \
  --residual_mode "$RESIDUAL_MODE" \
  --knn 12 \
  --max_centroid_distance 0.35 \
  --max_nodes_per_point 12 \
  --min_projection_visible_points 8 \
  --min_mutual_projection_support 0.50 \
  --min_direct_point_iou 0.05 \
  --min_track_views 2 \
  --min_track_support_edges 2

echo "[完成] 残差专属多视角互证图：$OUTPUT_ROOT"
