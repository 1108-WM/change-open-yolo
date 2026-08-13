#!/usr/bin/env python3
"""Sum chunked fixed-mask ranking-ceiling counts before computing one ceiling."""

import argparse
import json
from pathlib import Path

import numpy as np


OVERLAPS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.input]
    totals = {"valid_gt_instance_count": 0, "prediction_count": 0}
    for overlap in OVERLAPS:
        totals[f"match_count_{overlap}"] = 0
    for payload in payloads:
        for key in totals:
            totals[key] += int(payload["totals"][key])
    values = {overlap: totals[f"match_count_{overlap}"] / max(1, totals["valid_gt_instance_count"]) for overlap in OVERLAPS}
    result = {
        "diagnostic_type": "GT-only merged fixed-mask threshold-specific ranking ceiling",
        "merge_contract": "sum per-threshold maximum-match counts and valid GT counts before computing one ceiling; never average chunk ceilings.",
        "chunk_count": len(payloads),
        "totals": totals,
        "ceiling": {
            "ap": float(np.mean([value for overlap, value in values.items() if overlap != 0.25])),
            "ap50": float(values[0.5]), "ap25": float(values[0.25]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
