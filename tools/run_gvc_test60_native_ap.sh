#!/usr/bin/env bash
set -euo pipefail

# 固定的正式对照：仅比较 native 强基线与 native + GVC append-only。
# 运行前必须完成 test60 的无 GT 候选导出和预检；本脚本是唯一允许读取 test60 GT 的步骤。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/gvc_holdout_20260803/gvc_test60.txt}"
TEST_ROOT="${TEST_ROOT:-$ROOT_DIR/output/gvc_test60_uniform30_20260803}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/gvc_test60_native_ap_20260803}"
SAM_FUSED_ROOT="${SAM_FUSED_ROOT:-$ROOT_DIR/output/sam_fused_proposals_scannet200_s5_m30_prefilter}"
BPR_ROOT="${BPR_ROOT:-$ROOT_DIR/output/backprojection_candidates_scannet200_mv_m20}"

if [[ ! -f "$TEST_ROOT/gvc_append_only_preflight/gvc_append_only_preflight_manifest.json" ]]; then
  echo "缺少 test60 无 GT 预检产物：$TEST_ROOT/gvc_append_only_preflight" >&2
  exit 1
fi
if [[ -e "$OUT_DIR" ]] && find "$OUT_DIR" -type f -print -quit | grep -q .; then
  echo "正式 AP 输出目录已含文件，拒绝覆盖：$OUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/reports" "$OUT_DIR/native_prediction_cache" "$OUT_DIR/gvc_prediction_cache"
cd "$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENYOLO3D_ALLOW_LEGACY_2D_CACHE="${OPENYOLO3D_ALLOW_LEGACY_2D_CACHE:-1}"

BASE_ARGS=(
  --dataset_name scannet200
  --path_to_3d_masks "$ROOT_DIR/output/scannet200/scannet200_masks"
  --path_to_2d_preds "$ROOT_DIR/output/scannet200/bboxes_2d"
  --processed_scene_root "$ROOT_DIR/data/scannet200"
  --scene_list "$SCENE_LIST"
  --eval_score_mode native
  --backprojection_min_score 0.50
  --backprojection_min_seed_points 80
  --backprojection_max_existing_iou 0.30
  --backprojection_max_seed_in_existing_mask_ratio 0.70
  --backprojection_max_proposal_iou 0.50
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

"$PYTHON" run_evaluation.py "${BASE_ARGS[@]}" \
  --backprojection_candidates "$SAM_FUSED_ROOT,$BPR_ROOT" \
  --backprojection_report_path "$OUT_DIR/reports/native_backprojection.json" \
  --eval_prediction_cache_dir "$OUT_DIR/native_prediction_cache" \
  --eval_output_file "$OUT_DIR/native.csv"

"$PYTHON" run_evaluation.py "${BASE_ARGS[@]}" \
  --backprojection_candidates "$SAM_FUSED_ROOT,$BPR_ROOT,$TEST_ROOT/gvc_append_only_candidates" \
  --backprojection_append_only_source_kinds gvc_append_only \
  --backprojection_append_only_same_class_dedup_iou 0.50 \
  --backprojection_report_path "$OUT_DIR/reports/native_plus_gvc_append_only_backprojection.json" \
  --eval_prediction_cache_dir "$OUT_DIR/gvc_prediction_cache" \
  --eval_output_file "$OUT_DIR/native_plus_gvc_append_only.csv"

echo "[完成] gvc_test60 正式 native AP 对照：$OUT_DIR"
