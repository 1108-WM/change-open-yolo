#!/usr/bin/env python3
"""Build a GT-free top-M class-hypothesis plan for exact native geometry."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(args: argparse.Namespace) -> dict:
    rows = _read_jsonl(args.oof_root / "oof_predictions.jsonl")
    native_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    output = []
    for row in rows:
        if str(row["candidate_source"]) == "native":
            native_groups[(str(row["scene_name"]), str(row["semantic_evidence_node_key"]))].append(row)
        else:
            output.append({
                "scene_name": str(row["scene_name"]), "candidate_source": str(row["candidate_source"]),
                "candidate_id": int(row["candidate_id"]),
                "semantic_evidence_node_key": str(row["semantic_evidence_node_key"]),
                "class_index": int(row["class_index"]), "exact_node_class_rank": 1,
                "exact_node_class_count": 1, "keep_topm": True,
                "competition_score": (
                    float(row["oof_predictions"]["C_joint_yolo_alpha"])
                    if str(row["candidate_source"]) == "track" else float(row["original_score"])
                ),
                "decision_state": "plan_only; no candidate removed or modified",
            })
    node_size_histogram = Counter()
    keep_count_histogram = Counter()
    for (scene, key), group in sorted(native_groups.items()):
        ordered = sorted(
            group,
            key=lambda row: (-float(row["oof_predictions"]["C_joint_yolo_alpha"]), int(row["candidate_id"])),
        )
        node_size_histogram[len(ordered)] += 1
        keep_count = min(args.top_m, len(ordered))
        if args.cumulative_mass is not None and ordered:
            scores = np.asarray([
                max(0.0, float(row["oof_predictions"]["C_joint_yolo_alpha"])) for row in ordered
            ], dtype=np.float64)
            if scores.sum() > 0:
                mass_count = int(np.searchsorted(np.cumsum(scores), args.cumulative_mass * scores.sum(), side="left")) + 1
                keep_count = max(keep_count, mass_count)
            keep_count = min(keep_count, args.max_m, len(ordered))
        keep_count_histogram[keep_count] += 1
        for rank, row in enumerate(ordered, 1):
            output.append({
                "scene_name": scene, "candidate_source": "native",
                "candidate_id": int(row["candidate_id"]), "semantic_evidence_node_key": key,
                "class_index": int(row["class_index"]), "exact_node_class_rank": rank,
                "exact_node_class_count": len(ordered), "exact_node_keep_count": keep_count,
                "keep_topm": rank <= keep_count,
                "competition_score": float(row["oof_predictions"]["C_joint_yolo_alpha"]),
                "decision_state": "plan_only; no candidate removed or modified",
            })
    output.sort(key=lambda row: (row["scene_name"], row["candidate_source"], row["candidate_id"]))
    if len(output) != len(rows):
        raise AssertionError("top-M plan must cover every valid OOF candidate")
    identities = {(row["scene_name"], row["candidate_source"], row["candidate_id"]) for row in output}
    if len(identities) != len(output):
        raise ValueError("duplicate candidate identity in top-M plan")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "candidates.jsonl").open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    kept = [row for row in output if row["keep_topm"]]
    summary = {
        "diagnostic_type": "Z4 GT-free exact-native-geometry top-M class plan",
        "ground_truth_usage": "none", "candidate_membership_modified": False,
        "candidate_scores_modified": False, "candidate_classes_modified": False,
        "top_m": args.top_m, "candidate_count": len(output), "kept_plan_count": len(kept),
        "cumulative_mass": args.cumulative_mass, "max_m": args.max_m,
        "suppressed_plan_count": len(output) - len(kept),
        "native_exact_geometry_node_count": len(native_groups),
        "source_candidate_counts": dict(Counter(row["candidate_source"] for row in output)),
        "source_kept_counts": dict(Counter(row["candidate_source"] for row in kept)),
        "node_size_histogram": {str(key): value for key, value in sorted(node_size_histogram.items())},
        "keep_count_histogram": {str(key): value for key, value in sorted(keep_count_histogram.items())},
        "contracts": {
            "native": (
                "keep fixed top-M existing class hypotheses per exact geometry by joint OOF score"
                if args.cumulative_mass is None else
                "keep at least top-M and enough ranked hypotheses for cumulative normalized OOF score mass, capped by max-M"
            ),
            "track": "keep all", "pair_union": "keep all",
            "top1_hard_replacement": False, "geometry_change": False,
            "selection_basis": "top-5 is the smallest pre-registered M reaching >=97% TP50-class rank coverage",
        },
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--cumulative-mass", type=float)
    parser.add_argument("--max-m", type=int, default=20)
    args = parser.parse_args()
    if args.top_m <= 0:
        raise SystemExit("--top-m must be positive")
    if args.max_m < args.top_m:
        raise SystemExit("--max-m must be >= --top-m")
    if args.cumulative_mass is not None and not 0.0 < args.cumulative_mass <= 1.0:
        raise SystemExit("--cumulative-mass must be in (0,1]")
    for name in ("oof_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
