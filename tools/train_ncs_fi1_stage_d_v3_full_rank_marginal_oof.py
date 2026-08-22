#!/usr/bin/env python3
"""Train the frozen D-v3 full-prefix continuous marginal-gain OOF model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.train_ncs_fi1_stage_d_v2_rank_marginal_oof as shared
from tools.build_ncs_fi1_stage_d_v3_full_rank_marginal_dataset_gt import FEATURE_NAMES


shared.FEATURE_NAMES = FEATURE_NAMES
shared.VERSION = "ncs_fi1_stage_d_v3_full_rank_marginal_oof_v1"
shared.STAGE_LABEL = "stage-D-v3"
shared.SCORE_FIELD = "stage_d_v3_append_score"
shared.PLAN_FILE = "stage_d_v3_append_score_plan.jsonl"


def run(args: argparse.Namespace) -> dict:
    return shared.run(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({
        "candidate_count": result["candidate_count"],
        "oof_mae": result["overall_metrics"]["oof_prediction"]["mae"],
        "zero_control_mae": result["overall_metrics"]["zero_prediction_control"]["mae"],
        "confident_positive_count": result["confident_positive"]["count"],
        "advancement_authorized_pending_independent_audit": result["advancement_gate"]["advancement_authorized_pending_independent_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
