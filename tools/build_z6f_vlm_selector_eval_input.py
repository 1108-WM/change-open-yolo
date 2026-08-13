#!/usr/bin/env python3
"""Convert strict symmetric VLM decisions into evaluator selection rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=Path("docs/diagnostics/z6f_qwen25vl_symmetric_review_official100_20260812"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/diagnostics/z6f_vlm_selector_eval_input_official100_20260812"))
    args = parser.parse_args()
    args.review_root = _resolve(args.review_root); args.output_dir = _resolve(args.output_dir)
    if args.output_dir.exists(): raise SystemExit(f"refusing to overwrite {args.output_dir}")
    rows = []
    for row in _read_jsonl(args.review_root / "review_outputs.jsonl"):
        if row["model_decision"] != "PROPOSED":
            continue
        rows.append({
            "scene_name": str(row["scene_name"]), "prediction_index": int(row["prediction_index"]),
            "candidate_source": str(row["candidate_source"]), "candidate_id": int(row["candidate_id"]),
            "semantic_evidence_node_key": str(row["semantic_evidence_node_key"]),
            "current_class_index": int(row["current_class_index"]),
            "current_class_valid": True,
            "selectors": {"vlm_symmetric": {
                "selected_class_index": int(row["proposed_class_index"]),
                "proposed_class_index": int(row["proposed_class_index"]),
                "kept_current": False, "accepted": True,
            }},
        })
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "oof_selections.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"diagnostic_type": "official100 strict symmetric VLM temporary evaluator input", "selection_count": len(rows), "ground_truth_usage": "none", "candidate_mutation": False, "geometry_mutation": False, "score_mutation": False, "inference_plan_written": False, "safety60_read": False, "even48_read": False, "test60_read": False}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()
