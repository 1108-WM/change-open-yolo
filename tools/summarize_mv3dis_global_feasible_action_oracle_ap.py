#!/usr/bin/env python3
"""Merge chunked sufficient statistics for the GT-only global oracle AP."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_mv3dis_global_feasible_action_oracle_gt import _average_precision  # noqa: E402


OVERLAPS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.25)


def _key(payload, stem, overlap):
    matches = []
    prefix = f"{stem}_"
    for key in payload.files:
        if key.startswith(prefix):
            value = float(key[len(prefix):].replace("_", "."))
            if np.isclose(value, overlap):
                matches.append(key)
    if len(matches) != 1:
        raise KeyError(f"cannot uniquely locate {stem} statistics for overlap {overlap}")
    return matches[0]


def merge_records(paths):
    values = {}
    for overlap in OVERLAPS:
        true = []
        score = []
        false_negatives = 0
        has_gt = False
        has_pred = False
        for path in paths:
            with np.load(path) as payload:
                true_key = _key(payload, "true", overlap)
                suffix = true_key[len("true_"):]
                true.append(np.asarray(payload[true_key], dtype=np.float64))
                score.append(np.asarray(payload[f"score_{suffix}"], dtype=np.float64))
                false_negatives += int(payload[f"fn_{suffix}"][0])
                has_gt = has_gt or bool(payload[f"has_gt_{suffix}"][0])
                has_pred = has_pred or bool(payload[f"has_pred_{suffix}"][0])
        values[overlap] = _average_precision(true, score, false_negatives, has_gt, has_pred)
    return {
        "ap": float(np.mean([value for overlap, value in values.items() if overlap != 0.25])),
        "ap50": float(values[0.5]),
        "ap25": float(values[0.25]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-record", type=Path, action="append")
    parser.add_argument("--oracle-record", type=Path, action="append")
    parser.add_argument("--score-record", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen_records = args.frozen_record or []
    oracle_records = args.oracle_record or []
    score_records = args.score_record or []
    if score_records and (frozen_records or oracle_records):
        raise SystemExit("score records cannot be mixed with geometry records")
    if not score_records and (not frozen_records or len(frozen_records) != len(oracle_records)):
        raise SystemExit("provide equal non-empty frozen/oracle records, or score records")
    for path in frozen_records + oracle_records + score_records:
        if not path.is_file():
            raise FileNotFoundError(path)
    if score_records:
        result = {
            "diagnostic_type": "GT-only merged fixed-mask ideal score/ranking oracle AP",
            "merge_contract": "concatenate per-scene sufficient statistics before one AP computation; never average chunk AP.",
            "chunk_count": len(score_records),
            "global_feasible_geometry_ideal_score_class_agnostic": merge_records(score_records),
        }
    else:
        frozen = merge_records(frozen_records)
        oracle = merge_records(oracle_records)
        result = {
            "diagnostic_type": "GT-only merged global-feasible geometry oracle AP",
            "merge_contract": "concatenate per-scene sufficient statistics before one AP computation; never average chunk AP.",
            "chunk_count": len(frozen_records),
            "frozen_f2_fixed_score_class_agnostic": frozen,
            "global_feasible_oracle_fixed_score_class_agnostic": oracle,
            "oracle_minus_frozen_f2": {key: oracle[key] - frozen[key] for key in frozen},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
