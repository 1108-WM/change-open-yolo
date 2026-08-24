import pytest

from tools.run_dm_sms1_vlm_batch_smoke import (
    candidate_set_insufficient_or_unclear,
    normalize_attribute_view_ranks,
    outside_candidate_class_terms,
    select_batch,
    summarize_batch_records,
    validate_attribute_output,
    validate_attribute_structure,
)
from tools.run_dm_sms1_single_scene_vlm_smoke import _json_object


def test_select_batch_is_scene_balanced_and_pair_only():
    rows = [
        {"scene_name": "b", "geometry_hash": "2", "task_id": "b2", "candidate_hypotheses": [{}, {}]},
        {"scene_name": "a", "geometry_hash": "2", "task_id": "a2", "candidate_hypotheses": [{}, {}]},
        {"scene_name": "a", "geometry_hash": "1", "task_id": "a1", "candidate_hypotheses": [{}, {}]},
        {"scene_name": "a", "geometry_hash": "0", "task_id": "a0", "candidate_hypotheses": [{}]},
    ]
    selected = select_batch(rows, scene_count=2, per_scene=1)
    assert [row["task_id"] for row in selected] == ["a1", "b2"]


def test_select_batch_can_take_a_fresh_rank_slice_per_scene():
    rows = [
        {
            "scene_name": scene,
            "geometry_hash": str(index),
            "task_id": f"{scene}{index}",
            "candidate_hypotheses": [{}, {}],
        }
        for scene in ("a", "b") for index in range(4)
    ]
    selected = select_batch(rows, scene_count=2, per_scene=2, per_scene_offset=1)
    assert [row["task_id"] for row in selected] == ["a1", "a2", "b1", "b2"]


def test_select_batch_rejects_negative_per_scene_offset():
    with pytest.raises(ValueError, match="per_scene_offset"):
        select_batch([], scene_count=1, per_scene=1, per_scene_offset=-1)


def _item(observation="plain", ranks=None):
    return {
        "observation": observation,
        "supporting_view_ranks": [0, 1, 2] if ranks is None else ranks,
        "confidence": 0.8,
        "counterevidence": "none",
    }


def _attribute():
    return {
        "appearance": _item(),
        "material": _item(),
        "shape_structure": _item(),
        "function_cues": _item(),
        "spatial_context": _item(),
        "cross_view_consistency": {
            "observation": "consistent", "confidence": 0.8, "counterevidence": "none",
        },
        "missing_or_unclear_evidence": [],
    }


def test_one_based_view_ranks_are_normalized_to_contract():
    value = _attribute()
    for field in ("appearance", "material", "shape_structure", "function_cues", "spatial_context"):
        value[field]["supporting_view_ranks"] = ["1", "2", "3"]
    normalized, shifted = normalize_attribute_view_ranks(value)
    assert shifted is True
    assert normalized["appearance"]["supporting_view_ranks"] == [0, 1, 2]
    assert value["appearance"]["supporting_view_ranks"] == ["1", "2", "3"]


def test_out_of_range_view_ranks_are_rejected():
    value = _attribute()
    value["appearance"]["supporting_view_ranks"] = [4, 5, 6]
    with pytest.raises(ValueError, match="invalid view rank"):
        validate_attribute_structure(value)


def test_spatial_context_category_word_is_not_target_leakage():
    value = _attribute()
    value["spatial_context"]["observation"] = "on a desk next to a monitor"
    assert validate_attribute_output(value, ["desk"]) == []


def test_explicit_target_category_guess_is_rejected():
    value = _attribute()
    value["function_cues"]["observation"] = "this object is a desk"
    with pytest.raises(ValueError, match="leaked candidate names"):
        validate_attribute_output(value, ["desk"])


def test_parent_category_in_part_relation_is_not_target_leakage():
    value = _attribute()
    value["function_cues"]["observation"] = "part of a chair, possibly a leg"
    assert validate_attribute_output(value, ["chair"]) == []


def test_parent_category_at_clause_start_is_target_leakage():
    value = _attribute()
    value["function_cues"]["observation"] = "chair with wheels and armrests"
    with pytest.raises(ValueError, match="leaked candidate names"):
        validate_attribute_output(value, ["chair"])


def test_explicit_category_outside_finite_candidates_is_flagged():
    value = _attribute()
    value["function_cues"]["observation"] = "this object is a trash can"
    assert outside_candidate_class_terms(
        value, ["bench", "shower floor"], {"bench", "shower floor", "trash can"}
    ) == ["trash can"]


def test_component_after_with_is_not_outside_target_category():
    value = _attribute()
    value["appearance"]["observation"] = "white wall with a power outlet"
    assert outside_candidate_class_terms(value, ["desk"], {"desk", "power outlet"}) == []


def test_explicit_outside_category_at_clause_start_is_flagged():
    value = _attribute()
    value["function_cues"]["observation"] = "power outlet suggests electrical equipment"
    assert outside_candidate_class_terms(value, ["desk"], {"desk", "power outlet"}) == ["power outlet"]


def test_seat_as_shape_part_is_only_a_warning():
    value = _attribute()
    value["shape_structure"]["observation"] = "rounded backrest and seat"
    assert validate_attribute_output(value, ["seat"]) == ["seat"]


def test_missing_spatial_context_is_a_structural_error():
    value = _attribute()
    del value["spatial_context"]
    with pytest.raises(ValueError, match="misses required fields"):
        validate_attribute_structure(value)


def test_chinese_attribute_text_is_rejected_for_deterministic_term_audit():
    value = _attribute()
    value["function_cues"]["observation"] = "可能是垃圾桶"
    with pytest.raises(ValueError, match="must use English"):
        validate_attribute_structure(value)


def test_json_parser_rejects_trailing_explanation_or_second_object():
    assert _json_object('{"ok": true}') == {"ok": True}
    with pytest.raises(ValueError, match="exactly one JSON object"):
        _json_object('{"ok": true} trailing text')
    with pytest.raises(ValueError, match="exactly one JSON object"):
        _json_object('{"ok": true}{"second": true}')


def _candidate_output(supported):
    return {
        "candidate_results": [
            {"class_index": index, "supported": value, "strong_counterevidence": False}
            for index, value in enumerate(supported)
        ]
    }


def test_candidate_set_is_flagged_when_neither_candidate_has_stable_support():
    assert candidate_set_insufficient_or_unclear(
        _candidate_output([True, False]), _candidate_output([False, True])
    ) is True
    assert candidate_set_insufficient_or_unclear(
        _candidate_output([False, True]), _candidate_output([False, True])
    ) is False


def test_resume_summary_is_recomputed_from_complete_records():
    selected = [{"scene_name": "a"}, {"scene_name": "b"}]
    records = [
        {
            "valid": True, "decision": {"class_changed": True},
            "candidate_set_insufficient_or_unclear": False,
            "json_retried": False,
        },
        {
            "valid": False, "candidate_set_insufficient_or_unclear": True,
            "attribute_term_warnings": ["seat"], "json_retried": True,
            "repair_kinds": ["output_structure"],
        },
    ]
    summary = summarize_batch_records(records, selected)
    assert summary["processed_record_count"] == 2
    assert summary["valid_count"] == 1
    assert summary["invalid_count"] == 1
    assert summary["class_change_count"] == 1
    assert summary["output_structure_repair_count"] == 1
