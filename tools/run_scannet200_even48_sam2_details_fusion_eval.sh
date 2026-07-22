#!/usr/bin/env bash
set -euo pipefail

# 只重做已完成 even48 冻结输出的融合与评测，不重跑 SAM2 轨迹。
# 强基线继续采用 0.50 分数门槛；SAM2 候选独立采用 0.00，
# 仍受既有实例重叠、superpoint 细化和每场景候选预算约束。

ROOT_DIR="${ROOT_DIR:-/home/jia/Wm/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/output/sam2_details_even48_frozen_20260720_run1}"
EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/output/scannet200/sam2_details_even48_frozen_20260720_run1_source_min_v1_eval}"

SUPERPOINT_ROOT="${SUPERPOINT_ROOT:-$ROOT_DIR/output/mesh_normal_ibsp_dense_even48_f30}"
BASELINE_MASKS="${BASELINE_MASKS:-$ROOT_DIR/output/scannet200/scannet200_masks}"
BBOXES_2D="${BBOXES_2D:-$ROOT_DIR/output/scannet200/bboxes_2d}"
SAM_FUSED_IN="${SAM_FUSED_IN:-$ROOT_DIR/output/sam_fused_proposals_scannet200_s5_m30_prefilter}"
BPR_IN="${BPR_IN:-$ROOT_DIR/output/backprojection_candidates_scannet200_mv_m20}"
SAM2_IN="${SAM2_IN:-$RUN_DIR/mvpdist_candidates}"

mkdir -p "$EVAL_DIR"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$EVAL_DIR/matplotlib}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENYOLO3D_ALLOW_LEGACY_2D_CACHE="${OPENYOLO3D_ALLOW_LEGACY_2D_CACHE:-1}"

for required in "$SCENE_LIST" "$SUPERPOINT_ROOT" "$BASELINE_MASKS" "$BBOXES_2D" "$SAM_FUSED_IN" "$BPR_IN" "$SAM2_IN"; do
  [[ -e "$required" ]] || { echo "[ERROR] Missing required input: $required" >&2; exit 2; }
done

SCENE_NAMES="$(paste -sd, "$SCENE_LIST")"
EMPTY_FEATURES="$RUN_DIR/empty_semantic_features"
mkdir -p "$EMPTY_FEATURES"

"$PYTHON" - "$SCENE_LIST" "$EMPTY_FEATURES" <<'PY'
import json
import sys
from pathlib import Path

scene_list = Path(sys.argv[1])
output_root = Path(sys.argv[2])
for scene_name in (line.strip() for line in scene_list.read_text().splitlines()):
    if scene_name:
        scene_dir = output_root / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "multiview_object_clip_features.json").write_text(
            json.dumps({"scene_name": scene_name, "features": []}) + "\n"
        )
PY

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
"$PYTHON" "${eval_base[@]}" \
  --backprojection_candidates "$SAM_FUSED_IN,$BPR_IN" \
  --eval_output_file "$EVAL_DIR/strong_baseline.csv" \
  --report_path "$EVAL_DIR/strong_baseline_report.json"

echo "[STEP] Strong baseline plus SAM2 evaluation with source-specific minimum scores"
"$PYTHON" "${eval_base[@]}" \
  --backprojection_candidates "$SAM_FUSED_IN,$BPR_IN,$SAM2_IN" \
  --eval_output_file "$EVAL_DIR/strong_baseline_plus_sam2.csv" \
  --report_path "$EVAL_DIR/strong_baseline_plus_sam2_report.json"

"$PYTHON" - "$EVAL_DIR/strong_baseline_report.json" "$EVAL_DIR/strong_baseline_plus_sam2_report.json" <<'PY'
import json
import sys

for path in map(str, sys.argv[1:]):
    metrics = json.load(open(path))["inst_ap"][0]
    print(
        f"[RESULT] {path}: AP={metrics['all_ap']:.6f} "
        f"AP50={metrics['all_ap_50%']:.6f} AP25={metrics['all_ap_25%']:.6f}"
    )
PY

echo "[DONE] Evaluation directory: $EVAL_DIR"
