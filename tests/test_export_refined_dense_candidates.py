import json

import numpy as np

from tools.export_refined_dense_candidates import build_candidate, export_scene_candidates


def test_build_candidate_uses_native_detector_sam_mean(tmp_path):
    points_path = tmp_path / "points.npz"
    np.savez_compressed(points_path, point_indices=np.array([1, 2, 3]))
    candidate = build_candidate(
        "scene0000_00",
        {
            "instance_id": 4,
            "class_name": "chair",
            "point_indices_path": str(points_path),
            "point_count": 3,
            "frame_count": 2,
            "observation_ids": [1, 2],
        },
        {1: {"score": 0.8, "sam_score": 0.5}, 2: {"score": 0.6, "sam_score": 1.0}},
        {"chair": 7},
    )
    assert candidate["class_id"] == 7
    assert candidate["score"] == 0.5
    assert candidate["support_view_count"] == 2


def test_export_scene_candidates_filters_weak_tracks(tmp_path):
    refined_root = tmp_path / "refined"
    scene = refined_root / "scene0000_00"
    scene.mkdir(parents=True)
    points_path = scene / "points.npz"
    np.savez_compressed(points_path, point_indices=np.arange(120))
    observations_path = scene / "observations.jsonl"
    observations_path.write_text(json.dumps({"observation_id": 0, "score": 0.9, "sam_score": 0.8}) + "\n")
    (scene / "refined_instances.json").write_text(
        json.dumps(
            {
                "source_observations": str(observations_path),
                "instances": [
                    {
                        "instance_id": 0,
                        "class_name": "chair",
                        "point_indices_path": str(points_path),
                        "point_count": 120,
                        "frame_count": 2,
                        "observation_ids": [0],
                    },
                    {
                        "instance_id": 1,
                        "class_name": "chair",
                        "point_indices_path": str(points_path),
                        "point_count": 20,
                        "frame_count": 2,
                        "observation_ids": [0],
                    },
                ],
            }
        )
    )
    summary = export_scene_candidates(
        "scene0000_00", refined_root, tmp_path / "out", {"chair": 0}, min_points=100, min_support_views=2
    )
    assert summary == {"scene_name": "scene0000_00", "candidates": 1, "skipped": 1}
    payload = json.loads((tmp_path / "out" / "scene0000_00" / "backprojection_candidates.json").read_text())
    assert payload["candidates"][0]["class_name"] == "chair"
