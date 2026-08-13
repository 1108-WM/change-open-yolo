#!/usr/bin/env python3
"""Merge and audit atomic Z6c nested-gate outer-fold outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("gated_semantic_only", "gated_semantic_plus_dino")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-root-pattern",
        default="docs/diagnostics/z6c_nested_abstain_gate_oof_official100_20260812_fold{}",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("docs/diagnostics/z6c_nested_abstain_gate_oof_official100_20260812"),
    )
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument(
        "--diagnostic-type",
        default="official100 merged nested-cross-fitted Z6c accept/abstain gate",
    )
    parser.add_argument(
        "--acceptance-contract",
        default="accept proposed new class iff nested-OOF predicted AP-quality delta > 0; otherwise keep current",
    )
    args = parser.parse_args()
    args.output_dir = _resolve(args.output_dir)
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    rows = []
    fold_summaries = []
    seen = set()
    state_counts = {system: Counter() for system in args.systems}
    for fold_index in range(5):
        fold_root = _resolve(Path(args.fold_root_pattern.format(fold_index)))
        summary = json.loads((fold_root / "summary.json").read_text())
        if summary["fold_count"] != 1 or summary["folds"][0]["fold_index"] != fold_index:
            raise ValueError(f"invalid summary for fold {fold_index}")
        fold_summaries.extend(summary["folds"])
        for row in _read_jsonl(fold_root / "oof_selections.jsonl"):
            key = (str(row["scene_name"]), int(row["prediction_index"]))
            if key in seen:
                raise ValueError(f"duplicate selection key: {key}")
            seen.add(key)
            rows.append(row)
            for system in args.systems:
                state_counts[system][row["selectors"][system]["gate_state"]] += 1

    rows.sort(key=lambda row: (str(row["scene_name"]), int(row["prediction_index"])))
    scene_count = len({str(row["scene_name"]) for row in rows})
    if scene_count != 100 or len(rows) != 66759:
        raise ValueError(f"unexpected merged coverage: scenes={scene_count}, predictions={len(rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "oof_selections.jsonl"
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    payload = {
        "diagnostic_type": args.diagnostic_type,
        "scene_count": scene_count,
        "prediction_count": len(rows),
        "fold_count": 5,
        "folds": fold_summaries,
        "gate_state_counts": {system: dict(state_counts[system]) for system in args.systems},
        "oof_selections_sha256": digest,
        "acceptance_contract": args.acceptance_contract,
        "class_id_is_feature": False,
        "class_name_is_feature": False,
        "ground_truth_usage": "official_train_supervision_only_with_nested_scene_isolated_oof",
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "inference_plan_written": False,
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
