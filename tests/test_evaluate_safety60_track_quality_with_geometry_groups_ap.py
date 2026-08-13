import json

from tools.evaluate_safety60_track_quality_with_geometry_groups_ap import _write_jsonl


def test_score_ledger_writer_preserves_chinese_fields(tmp_path):
    path = tmp_path / "scores.jsonl"
    _write_jsonl(path, [{"candidate_source": "基线候选", "selected_score": 0.7}])
    assert json.loads(path.read_text())["candidate_source"] == "基线候选"
