#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
BPR_OUT="${BPR_OUT:-$ROOT_DIR/output/backprojection_candidates_scannet200_even48_mv_m20_boxnms070}"
SAM_OUT="${SAM_OUT:-$ROOT_DIR/output/sam_fused_proposals_scannet200_even48_s5_m30_prefilter_boxnms070}"
EVAL_OUT="${EVAL_OUT:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_overlap_nms}"

mkdir -p "$EVAL_OUT"
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

if [[ ! -f "$BPR_OUT/summary.json" ]]; then
  echo "[EXPORT] BPR box NMS candidates -> $BPR_OUT"
  "$PYTHON" tools/export_backprojection_candidates.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --output_dir "$BPR_OUT" \
    --scene_list "$SCENE_LIST" \
    --detection_score_th 0.40 \
    --min_seed_points 80 \
    --max_box_area_ratio 0.35 \
    --max_existing_iou 0.30 \
    --max_seed_in_existing_mask_ratio 0.70 \
    --seed_nms_iou 0.50 \
    --max_candidates_per_scene 20 \
    --max_candidates_per_class 3 \
    --min_support_views 1 \
    --support_iou_th 0.25 \
    --min_support_visible_points 30 \
    --box_nms_iou 0.70 \
    --path_to_2d_preds ./output/scannet200/bboxes_2d
fi

if [[ ! -f "$SAM_OUT/sam_fused_proposals_summary.json" ]]; then
  echo "[EXPORT] SAM-fused box NMS candidates -> $SAM_OUT"
  "$PYTHON" tools/export_sam_fused_proposals.py \
    --dataset_name scannet200 \
    --path_to_3d_masks ./output/scannet200/scannet200_masks \
    --output_dir "$SAM_OUT" \
    --sam_checkpoint ./pretrained/checkpoints/sam_vit_b_01ec64.pth \
    --sam_source ./_external/segment-anything/segment-anything-main \
    --sam_model_type vit_b \
    --scene_list "$SCENE_LIST" \
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
    --box_nms_iou 0.70 \
    --path_to_2d_preds ./output/scannet200/bboxes_2d
fi

run_eval() {
  local name="$1"
  local candidates="$2"
  local csv_path="$EVAL_OUT/${name}.csv"
  local log_path="$EVAL_OUT/${name}.log"
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
    --backprojection_source_score_scales sam_fused=1.2,bpr=1.0 \
    --backprojection_source_priorities sam_fused=2.0,bpr=1.0 \
    --backprojection_source_max_candidates sam_fused=12,bpr=3 \
    --backprojection_superpoint_refine \
    --backprojection_superpoint_min_coverage 0.30 \
    --backprojection_superpoint_max_expansion_ratio 3.0 \
    --backprojection_superpoint_min_view_siou 0.60 \
    --eval_output_file "$csv_path" >"$log_path" 2>&1
  summarize_csv "$name" "$csv_path"
}

run_eval current_full_candidates \
  ./output/sam_fused_proposals_scannet200_s5_m30_prefilter,./output/backprojection_candidates_scannet200_mv_m20

run_eval boxnms070_even48_candidates \
  "$SAM_OUT,$BPR_OUT"

echo "[DONE] Wrote overlap-NMS export/eval outputs to $EVAL_OUT"
