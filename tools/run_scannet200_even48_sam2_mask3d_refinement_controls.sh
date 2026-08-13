#!/usr/bin/env bash
set -euo pipefail

# A/B/C：只读取已有 SAM2 候选，评估它们对原始 Mask3D 的局部并集、交集和自适应修正。
# 不读取 GT，不重跑 SAM2，不改写原始 Mask3D。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
BASELINE_MASKS="${BASELINE_MASKS:-$ROOT_DIR/output/scannet200/scannet200_masks}"
SAM2_IN="${SAM2_IN:-$ROOT_DIR/output/sam2_details_even48_reobserve_20260721/mvpdist_candidates}"
VARIANT_ROOT="${VARIANT_ROOT:-$ROOT_DIR/output/mask3d_sam2_refinement_variants_even48_20260722}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/sam2_mask3d_refinement_controls_even48_20260722}"
SAM_FUSED_IN="${SAM_FUSED_IN:-$ROOT_DIR/output/sam_fused_proposals_scannet200_s5_m30_prefilter}"
BPR_IN="${BPR_IN:-$ROOT_DIR/output/backprojection_candidates_scannet200_mv_m20}"

mkdir -p "$OUT_DIR/reports"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/matplotlib}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENYOLO3D_ALLOW_LEGACY_2D_CACHE="${OPENYOLO3D_ALLOW_LEGACY_2D_CACHE:-1}"

for required in "$SCENE_LIST" "$BASELINE_MASKS" "$SAM2_IN" "$SAM_FUSED_IN" "$BPR_IN"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

for mode in union intersection adaptive; do
  echo "[BUILD] $mode"
  "$PYTHON" tools/generate_mask3d_sam2_refinement_variants.py \
    --scene_list "$SCENE_LIST" \
    --baseline_masks_root "$BASELINE_MASKS" \
    --candidate_root "$SAM2_IN" \
    --output_root "$VARIANT_ROOT" \
    --mode "$mode"
done

BASE_ARGS=(
  run_evaluation.py
  --dataset_name scannet200
  --path_to_2d_preds ./output/scannet200/bboxes_2d
  --scene_list "$SCENE_LIST"
  --backprojection_candidates "$SAM_FUSED_IN,$BPR_IN"
  --backprojection_min_score 0.50
  --backprojection_min_seed_points 80
  --backprojection_max_existing_iou 0.30
  --backprojection_max_seed_in_existing_mask_ratio 0.70
  --backprojection_max_candidates_per_scene 15
  --backprojection_score_scale 2.00
  --no-backprojection_use_candidate_fusion_score
  --backprojection_blocked_classes rug
  --backprojection_source_score_scales sam_fused=1.2,bpr=1.0
  --backprojection_source_priorities sam_fused=2.0,bpr=1.0
  --backprojection_source_max_candidates sam_fused=12,bpr=3
  --backprojection_superpoint_refine
  --backprojection_superpoint_min_coverage 0.30
  --backprojection_superpoint_max_expansion_ratio 3.0
  --backprojection_superpoint_min_view_siou 0.60
)

run_eval() {
  local mode="$1"
  echo "[RUN] $mode"
  "$PYTHON" "${BASE_ARGS[@]}" \
    --path_to_3d_masks "$VARIANT_ROOT/$mode" \
    --backprojection_report_path "$OUT_DIR/reports/${mode}.json" \
    --eval_output_file "$OUT_DIR/${mode}.csv" \
    >"$OUT_DIR/${mode}.log" 2>&1
  "$PYTHON" - "$mode" "$OUT_DIR/${mode}.csv" <<'PY'
import csv
import math
import sys

name, path = sys.argv[1:]
values = {key: [] for key in ("ap", "ap50", "ap25")}
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        for key in values:
            value = float(row[key])
            if not math.isnan(value):
                values[key].append(value)
print("[RESULT] " + name + ": " + " ".join(
    f"{key.upper()}={sum(items) / len(items):.6f}" for key, items in values.items()
))
PY
}

run_eval union
run_eval intersection
run_eval adaptive

echo "[DONE] Outputs: $OUT_DIR"
