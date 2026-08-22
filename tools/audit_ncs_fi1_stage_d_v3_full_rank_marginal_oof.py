#!/usr/bin/env python3
"""Independently audit D-v3 OOF models and the append-only score plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.audit_ncs_fi1_stage_d_v2_rank_marginal_oof as shared
from tools.build_ncs_fi1_stage_d_v3_full_rank_marginal_dataset_gt import FEATURE_NAMES


shared.FEATURE_NAMES = FEATURE_NAMES
shared.VERSION = "ncs_fi1_stage_d_v3_full_rank_marginal_oof_audit_v1"
shared.SCORE_FIELD = "stage_d_v3_append_score"


def run(args: argparse.Namespace) -> dict:
    return shared.run(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-audit-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({
        "audit_valid": result["audit_valid"], "error_count": result["error_count"],
        "advancement_authorized": result["advancement_gate"]["advancement_authorized"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
