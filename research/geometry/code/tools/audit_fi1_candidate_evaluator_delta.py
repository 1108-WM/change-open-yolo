#!/usr/bin/env python3
"""Explicit existing-member restore/rescore diagnostics over frozen official100."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fi1_evaluator_delta_core import IDENTITY, identity_equal, run_audit
from tools.fi1_gain_attribution_core import read_rows, unique


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b1-predictions', type=Path, required=True)
    p.add_argument('--attribution-root', type=Path, required=True)
    p.add_argument('--attribution-audit', type=Path, required=True)
    p.add_argument('--population-root', type=Path, required=True)
    p.add_argument('--requests', type=Path, required=True,
                   help='Fixed identity/score rows; never select requests by old GT labels.')
    p.add_argument('--action', choices=('restore_existing', 'rescore_existing'), default='restore_existing')
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()

    def requests(b1, population, bindings):
        rows = read_rows(bindings.check(args.requests))
        unique(rows, 'prediction_id')
        originals = {r['prediction_id']: r for r in b1}
        selected = []
        for row in rows:
            original = originals.get(row['prediction_id'])
            if original is None or not identity_equal(original, row) or row['score'] != original['score']:
                raise ValueError('requested member identity/score differs from B1')
            # Use canonical B1 identity, never propagate labels from request files.
            item = {k: original[k] for k in (*IDENTITY, 'score', 'source', 'column_index')}
            item['fold_index'] = population[item['geometry_key']]['fold_index']
            if 'fold_index' in row and row['fold_index'] != item['fold_index']:
                raise ValueError('request fold differs from population')
            if args.action == 'rescore_existing':
                item['new_score'] = row['new_score']
            selected.append(item)
        return selected

    run_audit(args, args.action, requests)


if __name__ == '__main__':
    main()
