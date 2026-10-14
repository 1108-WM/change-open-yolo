#!/usr/bin/env python3
"""Evaluate one genuinely new geometry-class append at a time over official100."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fi1_evaluator_delta_core import IDENTITY, run_audit
from tools.fi1_gain_attribution_core import read_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b1-predictions', type=Path, required=True)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--population-root', type=Path, required=True)
    p.add_argument('--attribution-root', type=Path, required=True)
    p.add_argument('--attribution-audit', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()

    def requests(b1, population, bindings):
        fields = (*IDENTITY, 'score', 'source', 'fold_index', 'is_new_geometry', 'candidate_score_source')
        return [{k: r[k] for k in fields} for r in read_rows(bindings.check(args.candidates))]

    run_audit(args, 'append_new', requests)


if __name__ == '__main__':
    main()
