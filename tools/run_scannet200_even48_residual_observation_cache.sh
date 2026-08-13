#!/usr/bin/env bash
set -euo pipefail

# 仅构建无 GT 的帧级 SAM 观测与残差归因；不产生最终候选或 AP 结果。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
MASK_ROOT="${MASK_ROOT:-$ROOT_DIR/output/scannet200/scannet200_masks}"
BBOX_ROOT="${BBOX_ROOT:-$ROOT_DIR/output/scannet200/bboxes_2d}"
NATIVE_CACHE="${NATIVE_CACHE:-$ROOT_DIR/output/scannet200/final_strong_baseline_native_score_even48_20260724/prediction_cache}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-$ROOT_DIR/pretrained/checkpoints/sam_vit_b_01ec64.pth}"
SAM_SOURCE="${SAM_SOURCE:-$ROOT_DIR/_external/segment-anything/segment-anything-main}"
OBSERVATION_OUT="${OBSERVATION_OUT:-$ROOT_DIR/output/residual_sam_observations_even48_f30_20260725}"
RESIDUAL_OUT="${RESIDUAL_OUT:-$ROOT_DIR/output/residual_sam_annotations_even48_f30_20260725}"

cd "$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENYOLO3D_ALLOW_LEGACY_2D_CACHE="${OPENYOLO3D_ALLOW_LEGACY_2D_CACHE:-1}"

"$PYTHON" tools/verify_residual_observation_entry.py \
  --scene_list "$SCENE_LIST" \
  --mask_root "$MASK_ROOT" \
  --bbox_root "$BBOX_ROOT" \
  --prediction_cache_dir "$NATIVE_CACHE" \
  --sam_checkpoint "$SAM_CHECKPOINT" \
  --sam_source "$SAM_SOURCE" \
  --observation_output_root "$OBSERVATION_OUT" \
  --residual_output_root "$RESIDUAL_OUT"

"$PYTHON" tools/export_dense_frame_instance_observations.py \
  --dataset scannet200 \
  --path_to_3d_masks "$MASK_ROOT" \
  --output_root "$OBSERVATION_OUT" \
  --sam_checkpoint "$SAM_CHECKPOINT" \
  --sam_source "$SAM_SOURCE" \
  --sam_model_type vit_b \
  --path_to_2d_preds "$BBOX_ROOT" \
  --allow_legacy_2d_cache \
  --scene_split "$SCENE_LIST" \
  --max_frames 30 \
  --detection_score_th 0.08 \
  --max_detections_per_frame 20 \
  --max_box_area_ratio 0.85 \
  --sam_multimask_topk 1 \
  --min_mask_area 64 \
  --min_visible_points 8 \
  --max_masks_per_frame 20

"$PYTHON" tools/annotate_dense_sam_residuals.py \
  --scene_list "$SCENE_LIST" \
  --observation_root "$OBSERVATION_OUT" \
  --prediction_cache_dir "$NATIVE_CACHE" \
  --output_root "$RESIDUAL_OUT" \
  --min_saved_residual_points 20

echo "[完成] SAM 观测：$OBSERVATION_OUT"
echo "[完成] 残差归因：$RESIDUAL_OUT"
