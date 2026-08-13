import importlib.util
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).parents[1]


def _module():
    path = PROJECT_ROOT / "tools" / "materialize_multiview_fragment_family_competition.py"
    spec = importlib.util.spec_from_file_location("fragment_family_materializer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _track(proposal_id, lineage=None, observations=None, source_ids=None):
    return {
        "proposal_id": proposal_id,
        "track_id": proposal_id,
        "source_track_ids": source_ids or [proposal_id],
        "lineage_proposal_ids": lineage or [proposal_id],
        "observation_ids": observations or [proposal_id],
        "superpoint_ids": [proposal_id],
        "superpoint_count": 1,
        "point_count": 2,
        "points_path": "unused.npz",
        "mean_node_quality": proposal_id / 100.0,
        "unrelated_field": {"preserve": proposal_id},
    }


def _merged(anchor, absorbed):
    row = _track(
        anchor,
        lineage=[anchor, absorbed],
        observations=[anchor, absorbed],
        source_ids=[anchor, absorbed],
    )
    row["mean_node_quality"] = 0.987
    row["unrelated_field"] = {"exact_f1": True}
    return row


def _plan(anchor, absorbed, action, index=0):
    return {
        "scene_name": "scene0000_00",
        "action_index": index,
        "anchor_proposal_id": anchor,
        "absorbed_proposal_id": absorbed,
        "planned_action": action,
        "quality_evidence_reliable": action == "use_merged_candidate",
        "merged_jointly_dominates": action == "use_merged_candidate",
        "decision_contract": (
            "M replaces A+B only with reliable common non-bridge evidence and "
            "independent Pareto dominance over both A and B"
        ),
        "candidate_action_applied": False,
        "score_used_for_decision": False,
        "ground_truth_usage": "none",
    }


def test_selected_family_uses_exact_f1_m_and_removes_b():
    module = _module()
    source = [_track(1), _track(2), _track(3)]
    merged = [_merged(1, 2), _track(3)]
    final, ledger = module.materialize_family_competition(
        source, merged, [_plan(1, 2, module.USE_MERGED_ACTION)]
    )
    assert [row["proposal_id"] for row in final] == [1, 3]
    assert final[0] is merged[0]
    assert final[0]["mean_node_quality"] == 0.987
    assert final[1] is source[2]
    assert ledger[0]["merged_candidate_selected"] is True
    assert ledger[0]["score_recomputed"] is False
    assert ledger[0]["candidate_field_mutation_count"] == 0


def test_fallback_family_keeps_exact_d2b_a_and_b():
    module = _module()
    source = [_track(1), _track(2), _track(3)]
    merged = [_merged(1, 2), _track(3)]
    final, ledger = module.materialize_family_competition(
        source, merged, [_plan(1, 2, module.KEEP_ORIGINAL_PAIR_ACTION)]
    )
    assert final == source
    assert all(row is expected for row, expected in zip(final, source))
    assert ledger[0]["merged_candidate_selected"] is False
    assert ledger[0]["candidate_action_applied"] is False


def test_overlapping_actions_or_unknown_sources_are_rejected():
    module = _module()
    source = [_track(1), _track(2), _track(3)]
    merged = [_merged(1, 2)]
    invalid_plans = (
        [
            _plan(1, 2, module.USE_MERGED_ACTION, 0),
            _plan(2, 3, module.USE_MERGED_ACTION, 1),
        ],
        [_plan(1, 4, module.USE_MERGED_ACTION)],
    )
    for plans in invalid_plans:
        try:
            module.materialize_family_competition(source, merged, plans)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid family plan was accepted")


def test_merged_candidate_must_conserve_lineage_and_observations():
    module = _module()
    source = [_track(1), _track(2)]
    for key, value in (
        ("lineage_proposal_ids", [1]),
        ("observation_ids", [1]),
        ("source_track_ids", [1]),
    ):
        candidate = _merged(1, 2)
        candidate[key] = value
        try:
            module.materialize_family_competition(
                source, [candidate], [_plan(1, 2, module.USE_MERGED_ACTION)]
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"non-conserving merged field was accepted: {key}")


def test_cli_has_no_gt_native_semantic_score_or_threshold_inputs():
    module = _module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("gt", "native", "semantic", "class", "score", "threshold")
    assert not any(any(token in option for token in forbidden) for option in options)


def test_cli_resolves_and_exposes_frozen_sources():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "materialize_multiview_fragment_family_competition.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-track-root" in result.stdout
    assert "--merged-track-root" in result.stdout
