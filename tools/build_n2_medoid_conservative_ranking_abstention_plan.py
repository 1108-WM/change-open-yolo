#!/usr/bin/env python3
"""Build a frozen no-GT conservative N2 ranking/abstention plan.

The formula is pre-registered here before any GT diagnostic reads it.  It uses
only within-scene ranks of evidence already present in the frozen quality
ledger.  It emits diagnostic budgets, never final proposals or scores.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    "No-GT conservative N2 ranking/abstention plan only; fixed candidates are not "
    "deleted, materialized, re-scored for inference, or evaluated."
)
BUDGETS = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def scenes(path):
    rows = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or duplicated")
    return rows


def percentile(values):
    """Average-tie percentile, with invalid evidence assigned zero."""
    values = np.asarray(values, dtype=float)
    output = np.zeros(len(values), dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return output
    order = np.argsort(values[valid], kind="mergesort")
    valid_indices = np.flatnonzero(valid)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[valid_indices[order[end]]] == values[valid_indices[order[cursor]]]:
            end += 1
        output[valid_indices[order[cursor:end]]] = (cursor + end) / (2.0 * len(order))
        cursor = end
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--quality-ledger-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "quality_ledger_root", "output_root"):
        setattr(args, name, resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("output root is non-empty")
    selected_scenes = scenes(args.scene_list)[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for ordinal, scene in enumerate(selected_scenes, 1):
        rows = [json.loads(line) for line in (args.quality_ledger_root / scene / "candidate_quality_competition_ledger.jsonl").read_text().splitlines() if line]
        component_rows = defaultdict(list)
        for row in rows:
            component_rows[int(row["near_duplicate_component_id"])].append(row)
        evidence = {
            "visible_core": percentile([row.get("source_visible_core_support_ratio") for row in rows]),
            "core_purity": percentile([row.get("source_observation_core_purity_ratio") for row in rows]),
            "normal_coherence": percentile([row.get("normal_mean_resultant_length") for row in rows]),
            "backprojection_extent": percentile([np.log1p(max(0.0, row.get("source_backprojected_point_count") or 0.0)) for row in rows]),
            "candidate_extent": percentile([np.log1p(max(0.0, row.get("point_count") or 0.0)) for row in rows]),
            "cross_view_agreement": percentile([row.get("medoid_cross_view_mean_jaccard") for row in rows]),
        }
        plans = []
        for index, row in enumerate(rows):
            multi_view_support = min(1.0, max(0.0, (float(row.get("eligible_view_count") or 0.0) - 1.0) / 2.0))
            # Fixed pre-registered confidence: direct 3-D support receives 2/3,
            # multi-view support and agreement the remaining 1/3.
            score = (
                0.18 * evidence["visible_core"][index]
                + 0.18 * evidence["core_purity"][index]
                + 0.12 * evidence["normal_coherence"][index]
                + 0.10 * evidence["backprojection_extent"][index]
                + 0.10 * evidence["candidate_extent"][index]
                + 0.17 * multi_view_support
                + 0.15 * evidence["cross_view_agreement"][index]
            )
            plans.append({
                "scene_name": scene,
                "candidate_id": int(row["candidate_id"]),
                "near_duplicate_component_id": int(row["near_duplicate_component_id"]),
                "evidence_percentile_visible_core": float(evidence["visible_core"][index]),
                "evidence_percentile_core_purity": float(evidence["core_purity"][index]),
                "evidence_percentile_normal_coherence": float(evidence["normal_coherence"][index]),
                "evidence_percentile_backprojection_extent": float(evidence["backprojection_extent"][index]),
                "evidence_percentile_candidate_extent": float(evidence["candidate_extent"][index]),
                "evidence_percentile_cross_view_agreement": float(evidence["cross_view_agreement"][index]),
                "multi_view_support": multi_view_support,
                "pre_registered_conservative_score": float(score),
                "ground_truth_usage": "none",
                "proposal_materialization_applied": False,
                "ap_computed": False,
                "decision_constraint": CONTRACT,
            })
        by_id = {row["candidate_id"]: row for row in plans}
        representatives = set()
        for component_id, items in component_rows.items():
            chosen = min(items, key=lambda row: (-by_id[int(row["candidate_id"])]["pre_registered_conservative_score"], int(row["candidate_id"])))
            representatives.add(int(chosen["candidate_id"]))
        kept = [row for row in plans if row["candidate_id"] in representatives]
        kept.sort(key=lambda row: (-row["pre_registered_conservative_score"], row["candidate_id"]))
        for rank, row in enumerate(kept, 1):
            row["component_representative"] = True
            row["component_representative_rank_within_scene"] = rank
            row["component_representative_percentile_within_scene"] = rank / max(1, len(kept))
            for budget in BUDGETS:
                row[f"diagnostic_select_top_{str(budget).replace('.', '_')}"] = rank <= max(1, int(np.ceil(budget * len(kept))))
        for row in plans:
            if row["candidate_id"] not in representatives:
                row["component_representative"] = False
                row["component_representative_rank_within_scene"] = None
                row["component_representative_percentile_within_scene"] = None
                for budget in BUDGETS:
                    row[f"diagnostic_select_top_{str(budget).replace('.', '_')}"] = False
        plans.sort(key=lambda row: row["candidate_id"])
        stage = args.output_root / f".{scene}.tmp.{os.getpid()}"
        stage.mkdir()
        with (stage / "conservative_ranking_abstention_plan.jsonl").open("w") as handle:
            for row in plans:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "scene_name": scene,
            "candidate_count": len(plans),
            "near_duplicate_component_count": len(representatives),
            "abstained_as_nonrepresentative_count": len(plans) - len(representatives),
            "diagnostic_budget_selected_counts": {str(budget): max(1, int(np.ceil(budget * len(representatives)))) for budget in BUDGETS},
            "ground_truth_usage": "none", "proposal_materialization_applied": False, "ap_computed": False,
        }
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output_root / scene)
        summaries.append(summary)
        print(f"[N2 保守排序计划] {ordinal}/{len(selected_scenes)} {scene}", flush=True)
    root = {
        "diagnostic_type": "no-GT pre-registered conservative N2 ranking and abstention plan",
        "decision_constraint": CONTRACT, "scene_count": len(summaries),
        "candidate_count": sum(row["candidate_count"] for row in summaries),
        "near_duplicate_component_count": sum(row["near_duplicate_component_count"] for row in summaries),
        "abstained_as_nonrepresentative_count": sum(row["abstained_as_nonrepresentative_count"] for row in summaries),
        "diagnostic_budgets": BUDGETS,
        "proposal_materialization_applied": False, "ap_computed": False,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
