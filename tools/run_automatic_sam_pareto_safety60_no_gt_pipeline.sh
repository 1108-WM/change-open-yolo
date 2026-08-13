#!/usr/bin/env bash
set -euo pipefail

# 基于已冻结的 gvc_safety60 输入运行新自动 SAM Pareto 候选链路。
# 不读取 GT、不运行 AP；所有新产物位于独立 OUTPUT_ROOT，原 GVC 与 native 输出不改写。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SCENE_LIST="${SCENE_LIST:-$ROOT_DIR/output/scannet200/scene_splits/gvc_holdout_20260803/gvc_safety60.txt}"
INPUT_ROOT="${INPUT_ROOT:-$ROOT_DIR/output/gvc_safety60_uniform30_20260803}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/output/gvc_safety60_automatic_sam_pareto_20260803}"

AUTOMATIC_ROOT="$INPUT_ROOT/sam_automatic_observations_uniform30"
TRACK_ROOT="$INPUT_ROOT/automatic_mask_tracks_uniform30"
YOLOWORLD_SAM_ROOT="$INPUT_ROOT/yoloworld_sam_observations_uniform30"
NATIVE_ROOT="$INPUT_ROOT/native_cache_no_gt"
GRAPH_ROOT="$OUTPUT_ROOT/evidence_graph"
GROWTH_ROOT="$OUTPUT_ROOT/track_growth_ledger"
PLAN_ROOT="$OUTPUT_ROOT/growth_variant_plan"
QUALITY_ROOT="$OUTPUT_ROOT/variant_quality_ledger"
SEMANTIC_ROOT="$OUTPUT_ROOT/variant_semantic_ledger"
COMPETITION_ROOT="$OUTPUT_ROOT/variant_competition_ledger"
PARETO_ROOT="$OUTPUT_ROOT/variant_pareto_ledger"
CANDIDATE_ROOT="$OUTPUT_ROOT/pareto_append_only_candidates"
PREFLIGHT_ROOT="$OUTPUT_ROOT/pareto_append_only_preflight"

cd "$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

for required in "$SCENE_LIST" "$AUTOMATIC_ROOT" "$TRACK_ROOT" "$YOLOWORLD_SAM_ROOT" "$NATIVE_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "缺少输入：$required" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "新 Pareto 输出根目录已存在，拒绝混用或覆盖：$OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

"$PYTHON" tools/build_automatic_sam_evidence_graph.py \
  --scene_list "$SCENE_LIST" --automatic_root "$AUTOMATIC_ROOT" --track_root "$TRACK_ROOT" \
  --processed_scene_root "$ROOT_DIR/data/scannet200" --dataset_root "$ROOT_DIR/data/scannet200" \
  --output_root "$GRAPH_ROOT"

"$PYTHON" tools/build_automatic_sam_track_growth_ledger.py \
  --scene_list "$SCENE_LIST" --evidence_graph_root "$GRAPH_ROOT" --track_root "$TRACK_ROOT" \
  --processed_scene_root "$ROOT_DIR/data/scannet200" --output_root "$GROWTH_ROOT"

"$PYTHON" tools/build_automatic_sam_growth_variant_plan.py \
  --scene_list "$SCENE_LIST" --growth_ledger_root "$GROWTH_ROOT" --output_root "$PLAN_ROOT"

"$PYTHON" tools/build_automatic_sam_variant_quality_ledger.py \
  --scene_list "$SCENE_LIST" --variant_plan_root "$PLAN_ROOT" --automatic_root "$AUTOMATIC_ROOT" \
  --native_prediction_cache "$NATIVE_ROOT" --dataset_root "$ROOT_DIR/data/scannet200" \
  --processed_scene_root "$ROOT_DIR/data/scannet200" --output_root "$QUALITY_ROOT"

"$PYTHON" tools/build_automatic_sam_variant_semantic_ledger.py \
  --scene_list "$SCENE_LIST" --variant_plan_root "$PLAN_ROOT" --quality_ledger_root "$QUALITY_ROOT" \
  --automatic_root "$AUTOMATIC_ROOT" --yoloworld_sam_root "$YOLOWORLD_SAM_ROOT" \
  --dataset_root "$ROOT_DIR/data/scannet200" --processed_scene_root "$ROOT_DIR/data/scannet200" \
  --output_root "$SEMANTIC_ROOT"

"$PYTHON" tools/build_automatic_sam_variant_competition_ledger.py \
  --scene_list "$SCENE_LIST" --quality_ledger_root "$QUALITY_ROOT" \
  --semantic_ledger_root "$SEMANTIC_ROOT" --output_root "$COMPETITION_ROOT"

"$PYTHON" tools/build_automatic_sam_variant_pareto_ledger.py \
  --scene_list "$SCENE_LIST" --quality_ledger_root "$QUALITY_ROOT" \
  --semantic_ledger_root "$SEMANTIC_ROOT" --output_root "$PARETO_ROOT"

"$PYTHON" tools/export_automatic_sam_pareto_candidates.py \
  --scene-list "$SCENE_LIST" --variant-plan-root "$PLAN_ROOT" --quality-ledger-root "$QUALITY_ROOT" \
  --semantic-ledger-root "$SEMANTIC_ROOT" --pareto-ledger-root "$PARETO_ROOT" \
  --processed-scene-root "$ROOT_DIR/data/scannet200" --output-root "$CANDIDATE_ROOT"

"$PYTHON" tools/verify_automatic_sam_pareto_candidate_entry.py \
  --scene-list "$SCENE_LIST" --variant-plan-root "$PLAN_ROOT" --candidate-root "$CANDIDATE_ROOT" \
  --processed-scene-root "$ROOT_DIR/data/scannet200" --output-dir "$PREFLIGHT_ROOT" --expected-scenes 60

echo "[完成] gvc_safety60 自动 SAM Pareto 无 GT 管线：$OUTPUT_ROOT"
