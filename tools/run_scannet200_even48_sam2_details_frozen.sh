#!/usr/bin/env bash
set -euo pipefail

# Frozen even48 comparison for the current SAM2 + Details Matter prototype.
# It uses no GT and compares the same strong baseline with and without the
# newly exported SAM2 candidates. Do not change parameters during this run.

ROOT_DIR="${ROOT_DIR:-/home/jia/Wm/wm_open-yolo/OpenYOLO3D}"
SAM2_PYTHON="${SAM2_PYTHON:-/home/jia/anaconda3/envs/sam2/bin/python}"
OPENYOLO_PYTHON="${OPENYOLO_PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/output/sam2_details_even48_frozen_20260720}"
EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/output/scannet200/sam2_details_even48_frozen_20260720_eval}"

SUPERPOINT_ROOT="${SUPERPOINT_ROOT:-$ROOT_DIR/output/mesh_normal_ibsp_dense_even48_f30}"
BASELINE_MASKS="${BASELINE_MASKS:-$ROOT_DIR/output/scannet200/scannet200_masks}"
BBOXES_2D="${BBOXES_2D:-$ROOT_DIR/output/scannet200/bboxes_2d}"
SAM_FUSED_IN="${SAM_FUSED_IN:-$ROOT_DIR/output/sam_fused_proposals_scannet200_s5_m30_prefilter}"
BPR_IN="${BPR_IN:-$ROOT_DIR/output/backprojection_candidates_scannet200_mv_m20}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-$ROOT_DIR/pretrained/sam2/sam2.1_hiera_small.pt}"
REOBSERVATION_STRIDE="${REOBSERVATION_STRIDE:-0}"
REOBSERVATION_MIN_IOU="${REOBSERVATION_MIN_IOU:-0.30}"
REOBSERVATION_REJECTED_FRAME_WEIGHT="${REOBSERVATION_REJECTED_FRAME_WEIGHT:-0.50}"

mkdir -p "$RUN_DIR" "$EVAL_DIR" "$RUN_DIR/matplotlib"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_DIR/matplotlib}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENYOLO3D_ALLOW_LEGACY_2D_CACHE="${OPENYOLO3D_ALLOW_LEGACY_2D_CACHE:-1}"

track_reobservation_args=()
lift_reobservation_args=()
if (( REOBSERVATION_STRIDE > 0 )); then
  track_reobservation_args=(
    --reobservation_stride "$REOBSERVATION_STRIDE"
    --reobservation_min_iou "$REOBSERVATION_MIN_IOU"
  )
  lift_reobservation_args=(
    --use_reobservation_confirmation
    --reobservation_rejected_frame_weight "$REOBSERVATION_REJECTED_FRAME_WEIGHT"
  )
fi

for required in "$SCENE_LIST" "$SUPERPOINT_ROOT" "$BASELINE_MASKS" "$BBOXES_2D" "$SAM_FUSED_IN" "$BPR_IN" "$SAM2_CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "[ERROR] Missing required input: $required" >&2; exit 2; }
done

SCENE_NAMES="$(paste -sd, "$SCENE_LIST")"
EMPTY_FEATURES="$RUN_DIR/empty_semantic_features"

"$OPENYOLO_PYTHON" - "$SCENE_LIST" "$EMPTY_FEATURES" <<'PY'
import json
import sys
from pathlib import Path

scene_list = Path(sys.argv[1])
output_root = Path(sys.argv[2])
for scene_name in (line.strip() for line in scene_list.read_text().splitlines()):
    if not scene_name:
        continue
    scene_dir = output_root / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "multiview_object_clip_features.json").write_text(
        json.dumps({"scene_name": scene_name, "features": []}) + "\n"
    )
PY

track_round1="$RUN_DIR/tracks_round1"
lift_round1="$RUN_DIR/lift_round1"
post_round1="$RUN_DIR/post_round1"
seed_round2="$RUN_DIR/seeds_round2"
track_round2="$RUN_DIR/tracks_round2"
lift_round2="$RUN_DIR/lift_round2"
tracks_round12="$RUN_DIR/tracks_round12"
lift_round12="$RUN_DIR/lift_round12"
post_round12="$RUN_DIR/post_round12"
seed_round3="$RUN_DIR/seeds_round3_baseline_novel"
track_round3="$RUN_DIR/tracks_round3"
lift_round3="$RUN_DIR/lift_round3"
tracks_round123="$RUN_DIR/tracks_round123"
lift_round123="$RUN_DIR/lift_round123"
post_round123="$RUN_DIR/post_round123"
quality_guard="$RUN_DIR/quality_guard"
mvpdist_candidates="$RUN_DIR/mvpdist_candidates"

echo "[STEP] Round 1 SAM2 tracks"
"$SAM2_PYTHON" tools/export_any3dis_sam2_tracks.py \
  --superpoint_root "$SUPERPOINT_ROOT" \
  --output_root "$track_round1" \
  --sam2_checkpoint "$SAM2_CHECKPOINT" \
  --scene_split "$SCENE_LIST" \
  --frame_stride 10 --max_frames 30 --max_tracks 8 \
  --prompt_points 3 --min_superpoint_points 40 --min_visible_frames 3 \
  --min_prompt_points 3 --neighbor_superpoints 32 --depth_tolerance 0.10 \
  --min_mask_area 64 --min_track_frames 2 --initialization_mode image_mask \
  --reappearance_memory_window 7 "${track_reobservation_args[@]}"

echo "[STEP] Round 1 lifting and cleanup"
"$OPENYOLO_PYTHON" tools/lift_sam2_tracks_to_superpoints.py \
  --track_root "$track_round1" --superpoint_root "$SUPERPOINT_ROOT" \
  --output_root "$lift_round1" --scene_split "$SCENE_LIST" \
  --mask_optimization any3dis_dp --same_frame_overlap_cleanup "${lift_reobservation_args[@]}"
"$OPENYOLO_PYTHON" tools/postprocess_sam2_superpoint_instances.py \
  --lift_root "$lift_round1" --track_root "$track_round1" \
  --superpoint_root "$SUPERPOINT_ROOT" --output_root "$post_round1" \
  --scene_split "$SCENE_LIST"

echo "[STEP] Round 2 uncovered seeds and tracks"
"$OPENYOLO_PYTHON" tools/select_uncovered_any3dis_superpoints.py \
  --track_root "$track_round1" --postprocess_root "$post_round1" \
  --output_root "$seed_round2" --scene_split "$SCENE_LIST"
"$SAM2_PYTHON" tools/export_any3dis_sam2_tracks.py \
  --superpoint_root "$SUPERPOINT_ROOT" --output_root "$track_round2" \
  --sam2_checkpoint "$SAM2_CHECKPOINT" --scene_split "$SCENE_LIST" \
  --seed_ids_root "$seed_round2" --frame_stride 10 --max_frames 30 --max_tracks 8 \
  --prompt_points 3 --min_superpoint_points 40 --min_visible_frames 3 \
  --min_prompt_points 3 --neighbor_superpoints 32 --depth_tolerance 0.10 \
  --min_mask_area 64 --min_track_frames 2 --initialization_mode image_mask \
  --reappearance_memory_window 7 "${track_reobservation_args[@]}"
"$OPENYOLO_PYTHON" tools/lift_sam2_tracks_to_superpoints.py \
  --track_root "$track_round2" --superpoint_root "$SUPERPOINT_ROOT" \
  --output_root "$lift_round2" --scene_split "$SCENE_LIST" \
  --mask_optimization any3dis_dp --same_frame_overlap_cleanup "${lift_reobservation_args[@]}"
"$OPENYOLO_PYTHON" tools/merge_any3dis_rounds.py \
  --track_roots "$track_round1,$track_round2" --lift_roots "$lift_round1,$lift_round2" \
  --output_track_root "$tracks_round12" --output_lift_root "$lift_round12" \
  --scene_split "$SCENE_LIST" --overwrite
"$OPENYOLO_PYTHON" tools/postprocess_sam2_superpoint_instances.py \
  --lift_root "$lift_round12" --track_root "$tracks_round12" \
  --superpoint_root "$SUPERPOINT_ROOT" --output_root "$post_round12" \
  --scene_split "$SCENE_LIST"

echo "[STEP] Round 3 baseline-novel seeds and tracks"
"$OPENYOLO_PYTHON" tools/select_uncovered_any3dis_superpoints.py \
  --track_root "$tracks_round12" --postprocess_root "$post_round12" \
  --output_root "$seed_round3" --scene_split "$SCENE_LIST" \
  --baseline_masks_root "$BASELINE_MASKS" --superpoint_root "$SUPERPOINT_ROOT" \
  --max_baseline_superpoint_coverage 0.70
"$SAM2_PYTHON" tools/export_any3dis_sam2_tracks.py \
  --superpoint_root "$SUPERPOINT_ROOT" --output_root "$track_round3" \
  --sam2_checkpoint "$SAM2_CHECKPOINT" --scene_split "$SCENE_LIST" \
  --seed_ids_root "$seed_round3" --frame_stride 10 --max_frames 30 --max_tracks 16 \
  --prompt_points 3 --min_superpoint_points 40 --min_visible_frames 3 \
  --min_prompt_points 3 --neighbor_superpoints 32 --depth_tolerance 0.10 \
  --min_mask_area 64 --min_track_frames 2 --initialization_mode image_mask \
  --reappearance_memory_window 7 "${track_reobservation_args[@]}"
"$OPENYOLO_PYTHON" tools/lift_sam2_tracks_to_superpoints.py \
  --track_root "$track_round3" --superpoint_root "$SUPERPOINT_ROOT" \
  --output_root "$lift_round3" --scene_split "$SCENE_LIST" \
  --mask_optimization any3dis_dp --same_frame_overlap_cleanup "${lift_reobservation_args[@]}"
"$OPENYOLO_PYTHON" tools/merge_any3dis_rounds.py \
  --track_roots "$tracks_round12,$track_round3" --lift_roots "$lift_round12,$lift_round3" \
  --output_track_root "$tracks_round123" --output_lift_root "$lift_round123" \
  --scene_split "$SCENE_LIST" --overwrite
"$OPENYOLO_PYTHON" tools/postprocess_sam2_superpoint_instances.py \
  --lift_root "$lift_round123" --track_root "$tracks_round123" \
  --superpoint_root "$SUPERPOINT_ROOT" --output_root "$post_round123" \
  --scene_split "$SCENE_LIST"

echo "[STEP] GT-free quality guard and MVPDist labels"
"$OPENYOLO_PYTHON" tools/filter_sam2_refined_instances_gtfree.py \
  --refined_root "$post_round123" --output_root "$quality_guard" \
  --superpoint_root "$SUPERPOINT_ROOT" --scene_names "$SCENE_NAMES" \
  --max_scene_point_fraction 0.10 --max_superpoints 40 \
  --min_component_points 100 --min_component_point_fraction 0.02
"$OPENYOLO_PYTHON" tools/export_sam2_refined_mvpdist_candidates.py \
  --refined_root "$quality_guard" --baseline_masks_root "$BASELINE_MASKS" \
  --bboxes_2d_root "$BBOXES_2D" --scene_names "$SCENE_NAMES" \
  --output_root "$mvpdist_candidates"

eval_base=(
  tools/evaluate_multiview_object_clip_correction.py
  --dataset_name scannet200
  --scene_names "$SCENE_NAMES"
  --path_to_3d_masks "$BASELINE_MASKS"
  --path_to_2d_preds "$BBOXES_2D"
  --processed_scene_root "$SUPERPOINT_ROOT"
  --reuse_2d_preds
  --score_threshold 0.20
  --base_eval_score_mode baseline
  --multiview_clip_features "$EMPTY_FEATURES"
  --clip_apply_source_kinds __disabled__
  --clip_score_policy keep
  --backprojection_min_score 0.50
  --backprojection_min_seed_points 80
  --backprojection_max_existing_iou 0.30
  --backprojection_max_seed_in_existing_mask_ratio 0.70
  --backprojection_max_candidates_per_scene 20
  --backprojection_score_scale 2.00
  --no-backprojection_use_candidate_fusion_score
  --backprojection_blocked_classes rug
  --backprojection_source_priorities sam_fused=2.0,bpr=1.0,sam2_details_mvpdist=1.0
  --backprojection_source_max_candidates sam_fused=12,bpr=3,sam2_details_mvpdist=16
  --backprojection_source_score_scales sam_fused=1.2,bpr=1.0,sam2_details_mvpdist=1.0
  --backprojection_source_min_scores sam_fused=0.50,bpr=0.50,sam2_details_mvpdist=0.00
  --backprojection_superpoint_refine
  --backprojection_superpoint_min_coverage 0.30
  --backprojection_superpoint_max_expansion_ratio 3.0
  --backprojection_superpoint_min_view_siou 0.60
)

echo "[STEP] Strong baseline evaluation"
"$OPENYOLO_PYTHON" "${eval_base[@]}" \
  --backprojection_candidates "$SAM_FUSED_IN,$BPR_IN" \
  --eval_output_file "$EVAL_DIR/strong_baseline.csv" \
  --report_path "$EVAL_DIR/strong_baseline_report.json"

echo "[STEP] Strong baseline plus SAM2 evaluation"
"$OPENYOLO_PYTHON" "${eval_base[@]}" \
  --backprojection_candidates "$SAM_FUSED_IN,$BPR_IN,$mvpdist_candidates" \
  --eval_output_file "$EVAL_DIR/strong_baseline_plus_sam2.csv" \
  --report_path "$EVAL_DIR/strong_baseline_plus_sam2_report.json"

"$OPENYOLO_PYTHON" - "$EVAL_DIR/strong_baseline_report.json" "$EVAL_DIR/strong_baseline_plus_sam2_report.json" <<'PY'
import json
import sys

for path in map(str, sys.argv[1:]):
    metrics = json.load(open(path))["inst_ap"][0]
    print(
        f"[RESULT] {path}: AP={metrics['all_ap']:.6f} "
        f"AP50={metrics['all_ap_50%']:.6f} AP25={metrics['all_ap_25%']:.6f}"
    )
PY

echo "[DONE] Run directory: $RUN_DIR"
echo "[DONE] Evaluation directory: $EVAL_DIR"
