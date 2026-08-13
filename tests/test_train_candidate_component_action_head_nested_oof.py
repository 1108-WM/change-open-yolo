from tools.train_candidate_component_action_head_nested_oof import (
    POLICY_SPECS,
    choose_with_spec,
)


def _action(kind, name):
    return {
        "scene_name": "scene0001_01",
        "relation_component_id": 0,
        "action_kind": kind,
        "action_name": name,
    }


def test_nested_policy_requires_both_probability_and_quantile_gate():
    actions = [
        _action("coexist", "coexist"),
        _action("baseline_only", "baseline_only"),
        _action("track_only_one", "track_only_one:3"),
    ]
    predictions = {
        ("scene0001_01", 0, "baseline_only"): {
            "predicted_positive_probability": 0.9,
            "predicted_q10_utility": -0.01,
            "predicted_q25_utility": 0.02,
        },
        ("scene0001_01", 0, "track_only_one:3"): {
            "predicted_positive_probability": 0.6,
            "predicted_q10_utility": 0.03,
            "predicted_q25_utility": 0.04,
        },
    }
    q10_p70 = next(row for row in POLICY_SPECS if row["name"] == "q10_p70")
    q25_p70 = next(row for row in POLICY_SPECS if row["name"] == "q25_p70")
    assert choose_with_spec(actions, predictions, q10_p70)["action_name"] == "coexist"
    assert choose_with_spec(actions, predictions, q25_p70)["action_name"] == "baseline_only"
