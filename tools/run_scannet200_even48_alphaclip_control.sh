#!/usr/bin/env bash
set -euo pipefail

# Controlled Alpha-CLIP comparison using the existing even48 feature cache.
# The baseline and Alpha-CLIP runs differ only in semantic correction thresholds.
ROOT_DIR="${ROOT_DIR:-/home/jia/Wm/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
FEATURES="${FEATURES:-$ROOT_DIR/output/multiview_object_alphaclip_scannet200_even48_current_best_low055}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/alphaclip_control_even48_20260720}"

SAM_FUSED_IN="${SAM_FUSED_IN:-$ROOT_DIR/output/sam_fused_proposals_scannet200_s5_m30_prefilter}"
BPR_IN="${BPR_IN:-$ROOT_DIR/output/backprojection_candidates_scannet200_mv_m20}"

mkdir -p "$OUT_DIR/reports"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENYOLO3D_ALLOW_LEGACY_2D_CACHE="${OPENYOLO3D_ALLOW_LEGACY_2D_CACHE:-1}"

for required in "$SCENE_LIST" "$FEATURES" "$SAM_FUSED_IN" "$BPR_IN"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

"$PYTHON" - "$SCENE_LIST" "$FEATURES" <<'PY'
from pathlib import Path
import sys

scene_list, feature_root = map(Path, sys.argv[1:])
expected = {line.strip() for line in scene_list.read_text().splitlines() if line.strip()}
actual = {
    path.parent.name
    for path in feature_root.glob("scene*/multiview_object_clip_features.json")
}
if actual != expected:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise SystemExit(
        "Alpha-CLIP feature scenes do not match the evaluation split: "
        f"missing={missing[:5]} ({len(missing)} total), "
        f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
    )
print(f"[CHECK] Alpha-CLIP cache exactly matches {len(expected)} evaluation scenes.")
PY

SCENE_NAMES="$(paste -sd, "$SCENE_LIST")"
BASE_ARGS=(
  tools/evaluate_multiview_object_clip_correction.py
  --dataset_name scannet200
  --scene_names "$SCENE_NAMES"
  --path_to_3d_masks ./output/scannet200/scannet200_masks
  --path_to_2d_preds ./output/scannet200/bboxes_2d
  --reuse_2d_preds
  --score_threshold 0.20
  --base_eval_score_mode baseline
  --multiview_clip_features "$FEATURES"
  --backprojection_candidates "$SAM_FUSED_IN,$BPR_IN"
  --backprojection_min_score 0.50
  --backprojection_min_seed_points 80
  --backprojection_max_existing_iou 0.30
  --backprojection_max_seed_in_existing_mask_ratio 0.70
  --backprojection_max_candidates_per_scene 15
  --backprojection_score_scale 2.00
  --no-backprojection_use_candidate_fusion_score
  --backprojection_blocked_classes rug
  --backprojection_source_priorities sam_fused=2.0,bpr=1.0
  --backprojection_source_max_candidates sam_fused=12,bpr=3
  --backprojection_source_score_scales sam_fused=1.2,bpr=1.0
  --backprojection_superpoint_refine
  --backprojection_superpoint_min_coverage 0.30
  --backprojection_superpoint_max_expansion_ratio 3.0
  --backprojection_superpoint_min_view_siou 0.60
  --clip_blocked_classes rug
  --clip_score_policy keep
)

run_eval() {
  local name="$1"
  shift
  echo "[RUN] $name"
  "$PYTHON" "${BASE_ARGS[@]}" "$@" \
    --eval_output_file "$OUT_DIR/${name}.csv" \
    --report_path "$OUT_DIR/reports/${name}.json" \
    >"$OUT_DIR/${name}.log" 2>&1
  "$PYTHON" - "$name" "$OUT_DIR/reports/${name}.json" <<'PY'
import json
import sys

name, path = sys.argv[1:]
metrics = json.load(open(path))["inst_ap"][0]
print(
    f"[RESULT] {name}: AP={metrics['all_ap']:.6f} "
    f"AP50={metrics['all_ap_50%']:.6f} AP25={metrics['all_ap_25%']:.6f}"
)
PY
}

# A confidence above 1.0 makes correction impossible while keeping the same entry point.
run_eval baseline_no_semantic --clip_min_confidence 1.10 --clip_min_margin 0.10 --clip_min_gain_over_current 0.10 --clip_max_base_score 1.10
run_eval alphaclip_low055 --clip_min_confidence 0.60 --clip_min_margin 0.10 --clip_min_gain_over_current 0.10 --clip_max_base_score 1.10

echo "[DONE] Outputs: $OUT_DIR"
