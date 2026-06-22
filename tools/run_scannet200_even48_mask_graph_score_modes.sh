#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jia/wm_open-yolo/OpenYOLO3D}"
PYTHON="${PYTHON:-/home/jia/anaconda3/envs/openyolo3d/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/scannet200/subset_sweeps/even48_mask_graph_score_modes_full_core}"
DATASET_ROOT="${DATASET_ROOT:-./data/scannet200}"
MASK_GRAPH_OUT="${MASK_GRAPH_OUT:-$ROOT_DIR/output/mask_graph_proposals_scannet200_even48_full_core_gpu}"
PATH_TO_2D_PREDS="${PATH_TO_2D_PREDS:-$ROOT_DIR/output/scannet200/bboxes_2d}"
EXPORT_REUSE_EXISTING="${EXPORT_REUSE_EXISTING:-1}"

mkdir -p "$OUT_DIR"
cd "$ROOT_DIR"

run_one() {
  local score_mode="$1"
  local mode="$2"
  local name="$3"
  local run_dir="$OUT_DIR/$name"
  mkdir -p "$run_dir"
  echo "[SCORE_MODE] Running $name"
  OUT_DIR="$run_dir" \
  MODE="$mode" \
  EVAL_SCORE_MODE="$score_mode" \
  DATASET_ROOT="$DATASET_ROOT" \
  MASK_GRAPH_OUT="$MASK_GRAPH_OUT" \
  PATH_TO_2D_PREDS="$PATH_TO_2D_PREDS" \
  EXPORT_REUSE_EXISTING="$EXPORT_REUSE_EXISTING" \
  bash tools/run_scannet200_even48_mask_graph_eval.sh
}

run_one uniform no_graph_refill uniform_no_graph
run_one uniform graph_refill uniform_with_graph
run_one native no_graph_refill native_no_graph
run_one native graph_refill native_with_graph
run_one calibrated no_graph_refill calibrated_no_graph
run_one calibrated graph_refill calibrated_with_graph

"$PYTHON" - "$OUT_DIR" <<'PY'
import csv
import json
import math
import os
import sys

out_dir = sys.argv[1]
cases = [
    ("uniform", "no_graph", "uniform_no_graph/no_graph_refill.csv", "uniform_no_graph/reports/no_graph_refill.json"),
    ("uniform", "with_graph", "uniform_with_graph/mask_graph_refill.csv", "uniform_with_graph/reports/mask_graph_refill.json"),
    ("native", "no_graph", "native_no_graph/no_graph_refill.csv", "native_no_graph/reports/no_graph_refill.json"),
    ("native", "with_graph", "native_with_graph/mask_graph_refill.csv", "native_with_graph/reports/mask_graph_refill.json"),
    ("calibrated", "no_graph", "calibrated_no_graph/no_graph_refill.csv", "calibrated_no_graph/reports/no_graph_refill.json"),
    ("calibrated", "with_graph", "calibrated_with_graph/mask_graph_refill.csv", "calibrated_with_graph/reports/mask_graph_refill.json"),
]

def mean_metrics(path):
    vals = {"ap": [], "ap50": [], "ap25": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in vals:
                value = float(row[key])
                if not math.isnan(value):
                    vals[key].append(value)
    return {key: sum(items) / max(1, len(items)) for key, items in vals.items()}

def source_counts(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        payload = json.load(f)
    counts = {}
    for report in payload.get("scene_reports", {}).values():
        for item in report.get("applied", []):
            key = str(item.get("source_kind") or "unknown")
            counts[key] = counts.get(key, 0) + 1
    return counts

rows = []
by_mode = {}
for score_mode, graph_state, csv_rel, report_rel in cases:
    metrics = mean_metrics(os.path.join(out_dir, csv_rel))
    counts = source_counts(os.path.join(out_dir, report_rel))
    row = {
        "score_mode": score_mode,
        "graph_state": graph_state,
        "ap": metrics["ap"],
        "ap50": metrics["ap50"],
        "ap25": metrics["ap25"],
        "applied_sam_fused": counts.get("sam_fused", 0),
        "applied_bpr": counts.get("bpr", 0),
        "applied_mask_graph_multi_view": counts.get("mask_graph_multi_view", 0),
        "applied_mask_graph_single_view": counts.get("mask_graph_single_view", 0),
    }
    rows.append(row)
    by_mode[(score_mode, graph_state)] = row

for score_mode in ("uniform", "native", "calibrated"):
    base = by_mode[(score_mode, "no_graph")]
    graph = by_mode[(score_mode, "with_graph")]
    graph["delta_ap_vs_no_graph"] = graph["ap"] - base["ap"]
    graph["delta_ap50_vs_no_graph"] = graph["ap50"] - base["ap50"]
    graph["delta_ap25_vs_no_graph"] = graph["ap25"] - base["ap25"]
    base["delta_ap_vs_no_graph"] = 0.0
    base["delta_ap50_vs_no_graph"] = 0.0
    base["delta_ap25_vs_no_graph"] = 0.0

csv_path = os.path.join(out_dir, "score_mode_summary.csv")
fieldnames = [
    "score_mode",
    "graph_state",
    "ap",
    "ap50",
    "ap25",
    "delta_ap_vs_no_graph",
    "delta_ap50_vs_no_graph",
    "delta_ap25_vs_no_graph",
    "applied_sam_fused",
    "applied_bpr",
    "applied_mask_graph_multi_view",
    "applied_mask_graph_single_view",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
json_path = os.path.join(out_dir, "score_mode_summary.json")
with open(json_path, "w") as f:
    json.dump(rows, f, indent=2)

print(f"[SUMMARY] Wrote {csv_path}")
for row in rows:
    print(
        "[SUMMARY] "
        f"{row['score_mode']} {row['graph_state']}: "
        f"AP={row['ap']:.6f} AP50={row['ap50']:.6f} AP25={row['ap25']:.6f} "
        f"dAP={row['delta_ap_vs_no_graph']:.6f} "
        f"graph_multi={row['applied_mask_graph_multi_view']}"
    )
PY
