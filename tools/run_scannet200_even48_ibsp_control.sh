#!/usr/bin/env bash
set -euo pipefail

# Controlled B/C comparison: only the processed superpoint labels may differ.
ROOT_DIR="${ROOT_DIR:-/home/jia/Wm/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
IBSP_ROOT="${IBSP_ROOT:-$ROOT_DIR/output/mesh_normal_ibsp_dense_even48_f30}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/ibsp_control_even48_20260719}"

SAM_FUSED_IN="${SAM_FUSED_IN:-$ROOT_DIR/output/sam_fused_proposals_scannet200_s5_m30_prefilter}"
BPR_IN="${BPR_IN:-$ROOT_DIR/output/backprojection_candidates_scannet200_mv_m20}"

mkdir -p "$OUT_DIR/reports"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENYOLO3D_ALLOW_LEGACY_2D_CACHE="${OPENYOLO3D_ALLOW_LEGACY_2D_CACHE:-1}"

for required in "$SCENE_LIST" "$IBSP_ROOT" "$SAM_FUSED_IN" "$BPR_IN"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

missing=0
while IFS= read -r scene || [[ -n "$scene" ]]; do
  [[ -z "$scene" ]] && continue
  scene_id="${scene#scene}"
  if [[ ! -f "$IBSP_ROOT/$scene/$scene_id.npy" ]]; then
    echo "Missing IBSp processed scene: $IBSP_ROOT/$scene/$scene_id.npy" >&2
    missing=1
  fi
done < "$SCENE_LIST"
[[ "$missing" -eq 0 ]] || exit 2

# run_evaluation.py reads column 9 as ScanNet200 superpoint ids. Refuse a C run
# unless f30 IBSp differs from the original processed data in that column only.
"$PYTHON" - "$SCENE_LIST" "$ROOT_DIR/data/scannet200" "$IBSP_ROOT" <<'PY'
from pathlib import Path
import sys

import numpy as np

scene_list, original_root, ibsp_root = map(Path, sys.argv[1:])
changed_scenes = 0
for scene in (line.strip() for line in scene_list.read_text().splitlines()):
    if not scene:
        continue
    scene_id = scene.removeprefix("scene")
    original = np.load(original_root / scene / f"{scene_id}.npy", mmap_mode="r")
    ibsp = np.load(ibsp_root / scene / f"{scene_id}.npy", mmap_mode="r")
    if original.shape != ibsp.shape:
        raise SystemExit(f"Shape mismatch for {scene}: {original.shape} vs {ibsp.shape}")
    if not np.array_equal(original[:, :9], ibsp[:, :9]) or not np.array_equal(original[:, 10:], ibsp[:, 10:]):
        raise SystemExit(f"Non-superpoint columns changed for {scene}")
    changed_scenes += int(np.any(original[:, 9] != ibsp[:, 9]))
if changed_scenes == 0:
    raise SystemExit("IBSp did not change any scene's superpoint ids")
print(f"[CHECK] IBSp differs only in column 9; changed scenes: {changed_scenes}")
PY

BASE_ARGS=(
  run_evaluation.py
  --dataset_name scannet200
  --path_to_3d_masks ./output/scannet200/scannet200_masks
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
  local name="$1"
  shift
  echo "[RUN] $name"
  "$PYTHON" "${BASE_ARGS[@]}" "$@" \
    --backprojection_report_path "$OUT_DIR/reports/${name}.json" \
    --eval_output_file "$OUT_DIR/${name}.csv" \
    >"$OUT_DIR/${name}.log" 2>&1
  "$PYTHON" - "$name" "$OUT_DIR/${name}.csv" <<'PY'
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
print(
    f"[RESULT] {name}: "
    + " ".join(f"{key.upper()}={sum(items) / len(items):.6f}" for key, items in values.items())
)
PY
}

run_eval B_original_superpoints
run_eval C_f30_ibsp_superpoints --processed_scene_root "$IBSP_ROOT"

echo "[DONE] Outputs: $OUT_DIR"
