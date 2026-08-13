import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_gvc_append_only_candidates.py"
    spec = importlib.util.spec_from_file_location("export_gvc_append_only", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gvc_export_scores_with_scene_rank_and_deduplicates_only_same_class(tmp_path):
    module = _module()
    scene = "scene0000_00"
    tracks = [
        {"track_id": 1, "points_path": str(tmp_path / "track1.npz"), "support_view_count": 2},
        {"track_id": 2, "points_path": str(tmp_path / "track2.npz"), "support_view_count": 4},
        {"track_id": 3, "points_path": str(tmp_path / "track3.npz"), "support_view_count": 3},
    ]
    np.savez_compressed(tmp_path / "track1.npz", point_indices=np.asarray([0, 1, 2]))
    np.savez_compressed(tmp_path / "track2.npz", point_indices=np.asarray([0, 1, 2]))
    np.savez_compressed(tmp_path / "track3.npz", point_indices=np.asarray([0, 1, 2]))
    for root, filename, payload in (
        ("tracks", "automatic_tracks.json", {"tracks": tracks}),
        ("semantic", "automatic_track_yoloworld_semantics.json", [
            {"track_id": 1, "voted_class_index": 0, "top_vote": 8.0, "vote_total": 10.0, "vote_margin": 0.5},
            {"track_id": 2, "voted_class_index": 0, "top_vote": 9.0, "vote_total": 10.0, "vote_margin": 0.8},
            {"track_id": 3, "voted_class_index": 1, "top_vote": 9.0, "vote_total": 10.0, "vote_margin": 0.8},
        ]),
        ("gvc", "track_gvc_feature_ledger.json", [
            {"track_id": 1, "gvc_score": 0.4, "native_top_iou": 0.1, "track_inside_top_native_ratio": 0.2},
            {"track_id": 2, "gvc_score": 0.8, "native_top_iou": 0.1, "track_inside_top_native_ratio": 0.2},
            {"track_id": 3, "gvc_score": 0.7, "native_top_iou": 0.1, "track_inside_top_native_ratio": 0.2},
        ]),
    ):
        root_path = tmp_path / root / scene
        root_path.mkdir(parents=True)
        (root_path / filename).write_text(json.dumps(payload))
    args = SimpleNamespace(
        track_root=tmp_path / "tracks", semantic_root=tmp_path / "semantic", gvc_root=tmp_path / "gvc",
        output_root=tmp_path / "output", labels=["chair", "table"], same_class_dedup_iou=0.5,
    )
    args.output_root.mkdir()
    audit = module._export_scene(scene, args)
    payload = json.loads((args.output_root / scene / "backprojection_candidates.json").read_text())
    assert audit["exported_candidate_count"] == 2
    assert {item["candidate_id"] for item in payload["candidates"]} == {2, 3}
    assert payload["append_only_contract"]["native_overlap_filtering"] is False
    assert payload["score_policy"]["weights"] == module.SCORE_WEIGHTS
    assert payload["candidates"][0]["score"] > 0.0
    assert audit["exported_distributions"]["score"]["count"] == 2
    assert audit["exported_distributions"]["class_counts"] == {"chair": 1, "table": 1}
