#!/usr/bin/env bash
set -euo pipefail

# 只为 gvc_safety60 重建冻结的无 GT 输入并导出 append-only 候选。
# 本脚本不读取 GT，不调用官方评测，也不修改 native 候选缓存。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/gvc_holdout_20260803/gvc_safety60.txt}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/output/gvc_safety60_uniform30_20260803}"
RESUME="${RESUME:-0}"

MASK_ROOT="${MASK_ROOT:-$ROOT_DIR/output/scannet200/scannet200_masks}"
BBOX_ROOT="${BBOX_ROOT:-$ROOT_DIR/output/scannet200/bboxes_2d}"
SAM_FUSED_ROOT="${SAM_FUSED_ROOT:-$ROOT_DIR/output/sam_fused_proposals_scannet200_s5_m30_prefilter}"
BPR_ROOT="${BPR_ROOT:-$ROOT_DIR/output/backprojection_candidates_scannet200_mv_m20}"
SAM_SOURCE="${SAM_SOURCE:-$ROOT_DIR/_external/segment-anything/segment-anything-main}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-$ROOT_DIR/pretrained/checkpoints/sam_vit_b_01ec64.pth}"

NATIVE_ROOT="$RUN_ROOT/native_cache_no_gt"
AUTOMATIC_ROOT="$RUN_ROOT/sam_automatic_observations_uniform30"
DENSE_ROOT="$RUN_ROOT/yoloworld_sam_observations_uniform30"
TRACK_ROOT="$RUN_ROOT/automatic_mask_tracks_uniform30"
SEMANTIC_ROOT="$RUN_ROOT/automatic_track_yoloworld_semantics_uniform30"
GVC_ROOT="$RUN_ROOT/track_gvc_feature_ledger_uniform30"
CANDIDATE_ROOT="$RUN_ROOT/gvc_append_only_candidates"
PREFLIGHT_ROOT="$RUN_ROOT/gvc_append_only_preflight"

cd "$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENYOLO3D_ALLOW_LEGACY_2D_CACHE="${OPENYOLO3D_ALLOW_LEGACY_2D_CACHE:-1}"

for required in "$SCENE_LIST" "$MASK_ROOT" "$BBOX_ROOT" "$SAM_FUSED_ROOT" "$BPR_ROOT" "$SAM_SOURCE" "$SAM_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "缺少输入：$required" >&2
    exit 1
  fi
done

if [[ -e "$RUN_ROOT" && "$RESUME" != "1" ]]; then
  echo "输出根目录已存在，拒绝混用：$RUN_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"

# 将阶段状态落盘，便于长任务被外部会话中断后精确续跑和诊断。
STATE_FILE="$RUN_ROOT/no_gt_pipeline_state.txt"
CURRENT_STAGE="initializing"
record_state() {
  printf 'time=%s\nstate=%s\nstage=%s\n' "$(date '+%F %T %z')" "$1" "$CURRENT_STAGE" > "$STATE_FILE"
}
on_exit() {
  exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    record_state completed
  else
    printf 'time=%s\nstate=failed\nstage=%s\nexit_code=%s\n' \
      "$(date '+%F %T %z')" "$CURRENT_STAGE" "$exit_code" > "$STATE_FILE"
  fi
}
record_state running
trap on_exit EXIT

NATIVE_RESUME=()
if [[ "$RESUME" == "1" ]]; then
  NATIVE_RESUME+=(--resume)
fi

CURRENT_STAGE="native_cache"
record_state running
"$PYTHON" tools/export_native_baseline_cache_no_gt.py \
  --scene_list "$SCENE_LIST" \
  --mask_root "$MASK_ROOT" \
  --bboxes_2d_root "$BBOX_ROOT" \
  --sam_fused_root "$SAM_FUSED_ROOT" \
  --bpr_root "$BPR_ROOT" \
  --output_dir "$NATIVE_ROOT" \
  --expected_scene_count 60 \
  "${NATIVE_RESUME[@]}"

CURRENT_STAGE="automatic_sam_observations"
record_state running
"$PYTHON" tools/export_sam_automatic_observations.py \
  --scene_list "$SCENE_LIST" \
  --sam_source "$SAM_SOURCE" \
  --sam_checkpoint "$SAM_CHECKPOINT" \
  --device cuda \
  --output_root "$AUTOMATIC_ROOT" \
  --max_frames 30 \
  --frame_selection uniform \
  --points_per_side 16

CURRENT_STAGE="yoloworld_sam_observations"
record_state running
"$PYTHON" tools/export_dense_frame_instance_observations.py \
  --dataset scannet200 \
  --path_to_3d_masks "$MASK_ROOT" \
  --output_root "$DENSE_ROOT" \
  --sam_checkpoint "$SAM_CHECKPOINT" \
  --sam_source "$SAM_SOURCE" \
  --sam_model_type vit_b \
  --path_to_2d_preds "$BBOX_ROOT" \
  --allow_legacy_2d_cache \
  --scene_split "$SCENE_LIST" \
  --max_frames 30 \
  --frame_selection uniform \
  --detection_score_th 0.08 \
  --max_detections_per_frame 20 \
  --max_box_area_ratio 0.85 \
  --sam_multimask_topk 1 \
  --min_mask_area 64 \
  --min_visible_points 8 \
  --max_masks_per_frame 20

CURRENT_STAGE="automatic_mask_tracks"
record_state running
"$PYTHON" tools/build_automatic_mask_tracks.py \
  --scene_list "$SCENE_LIST" \
  --automatic_root "$AUTOMATIC_ROOT" \
  --output_root "$TRACK_ROOT"

CURRENT_STAGE="track_yoloworld_semantics"
record_state running
"$PYTHON" tools/annotate_automatic_mask_tracks_yoloworld.py \
  --scene_list "$SCENE_LIST" \
  --track_root "$TRACK_ROOT" \
  --bboxes_2d_root "$BBOX_ROOT" \
  --output_root "$SEMANTIC_ROOT"

CURRENT_STAGE="gvc_feature_ledger"
record_state running
"$PYTHON" tools/build_track_gvc_feature_ledger.py \
  --scene_list "$SCENE_LIST" \
  --track_root "$TRACK_ROOT" \
  --automatic_root "$AUTOMATIC_ROOT" \
  --semantic_root "$SEMANTIC_ROOT" \
  --native_prediction_cache "$NATIVE_ROOT" \
  --yoloworld_sam_root "$DENSE_ROOT" \
  --output_root "$GVC_ROOT"

CURRENT_STAGE="gvc_candidate_export"
record_state running
"$PYTHON" tools/export_gvc_append_only_candidates.py \
  --scene-list "$SCENE_LIST" \
  --track-root "$TRACK_ROOT" \
  --semantic-root "$SEMANTIC_ROOT" \
  --gvc-root "$GVC_ROOT" \
  --output-root "$CANDIDATE_ROOT" \
  --same-class-dedup-iou 0.50

CURRENT_STAGE="gvc_preflight"
record_state running
"$PYTHON" tools/verify_gvc_append_only_entry.py \
  --scene-list "$SCENE_LIST" \
  --track-root "$TRACK_ROOT" \
  --semantic-root "$SEMANTIC_ROOT" \
  --gvc-root "$GVC_ROOT" \
  --candidate-root "$CANDIDATE_ROOT" \
  --output-dir "$PREFLIGHT_ROOT" \
  --expected-scenes 60

echo "[完成] gvc_safety60 无 GT 管线：$RUN_ROOT"
