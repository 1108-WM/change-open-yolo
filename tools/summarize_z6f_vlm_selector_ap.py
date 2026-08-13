#!/usr/bin/env python3
"""Summarize frozen Z6f symmetric VLM AP against the current control."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")


def delta(current: dict, control: dict) -> dict:
    return {name: float(current[name] - control[name]) for name in METRICS}


def main() -> None:
    output_dir = ROOT / "docs/diagnostics/z6f_vlm_selector_ap_summary_official100_20260812"
    if output_dir.exists(): raise SystemExit(f"refusing to overwrite {output_dir}")
    control_payload = json.loads((ROOT / "docs/diagnostics/z6c_candidate_selector_oof_ap_official100_20260812/summary.json").read_text())
    control = control_payload["systems"]["current_control"]
    controls = {int(row["fold_index"]): row["systems"]["current_control"] for row in control_payload["folds"]}
    full = json.loads((ROOT / "docs/diagnostics/z6f_vlm_selector_ap_official100_20260812_v2/result.json").read_text())["metrics"]
    folds = []
    for index in range(5):
        metrics = json.loads((ROOT / f"docs/diagnostics/z6f_vlm_selector_ap_folds_official100_20260812/fold{index}/result.json").read_text())["metrics"]
        folds.append({"fold_index": index, "current_control": controls[index], "vlm_symmetric": metrics, "delta": delta(metrics, controls[index])})
    review = json.loads((ROOT / "docs/diagnostics/z6f_qwen25vl_symmetric_review_official100_20260812/summary.json").read_text())
    payload = {
        "diagnostic_type": "official100 frozen symmetric-order selective Qwen2.5-VL AP summary",
        "current_control": control, "vlm_symmetric": full, "delta": delta(full, control),
        "folds": folds,
        "positive_fold_count_main_ap": sum(row["delta"]["ap"] > 0 for row in folds),
        "negative_fold_count_main_ap": sum(row["delta"]["ap"] < 0 for row in folds),
        "no_op_fold_count_main_ap": sum(row["delta"]["ap"] == 0 for row in folds),
        "review_count": review["review_count"], "decision_counts": review["decision_counts"],
        "frozen_method": {
            "router": "Z6d semantic-only accepted proposal + YOLO/Alpha top1 conflict + exactly 3 views + DINO cosine mean >= 0.855297 + node cap 1",
            "reviewer": "Qwen2.5-VL-7B-Instruct revision cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "image_contract": "three frozen bbox crops, min_pixels=100352, max_pixels=200704",
            "decision": "current-first and proposed-first A/B prompts; mutate only when both semantic choices are PROPOSED; otherwise keep current",
            "prompt_or_threshold_scan": False,
        },
        "decision": "freeze Z6f for one-way safety60 transfer audit; no official100 action pruning",
        "ground_truth_usage": "evaluation_only_after_method_freeze",
        "candidate_mutation": False, "geometry_mutation": False, "score_mutation": False,
        "inference_plan_written": False, "safety60_read": False, "even48_read": False, "test60_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()
