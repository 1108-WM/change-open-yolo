#!/usr/bin/env bash
set -euo pipefail

# 原始 superpoint 仅可切分的 f30 IBSp：不重新进行跨区域 Felzenszwalb 合并。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/even48.txt}"
LABEL_ROOT="${LABEL_ROOT:-$ROOT_DIR/output/dense_frame_instance_observations_even48_f30}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/output/original_anchored_ibsp_dense_even48_f30}"

[[ -f "$SCENE_LIST" ]] || { echo "缺少场景列表：$SCENE_LIST" >&2; exit 2; }
[[ -d "$LABEL_ROOT" ]] || { echo "缺少 f30 二维标签：$LABEL_ROOT" >&2; exit 2; }
[[ ! -e "$OUT_ROOT/geometric_superpoints_summary.json" ]] || {
  echo "输出已存在，为避免覆盖请改用新的 OUT_ROOT：$OUT_ROOT" >&2
  exit 2
}

cd "$ROOT_DIR"
"$PYTHON" tools/generate_geometric_superpoints.py \
  --input_root data/scannet200 \
  --output_root "$OUT_ROOT" \
  --scene_split "$SCENE_LIST" \
  --graph_type mesh_normal \
  --anchor_original_superpoints \
  --boundary_mask_root "$LABEL_ROOT" \
  --boundary_mask_subdir frame_label_maps \
  --boundary_frame_stride 1 \
  --boundary_max_frames 30 \
  --boundary_min_observations 1 \
  --boundary_min_conflict_ratio 1.0 \
  --boundary_visibility_tolerance 0.08

echo "[完成] 原始约束 IBSp：$OUT_ROOT"
