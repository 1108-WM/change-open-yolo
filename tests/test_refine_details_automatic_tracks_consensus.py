import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "tools" / "refine_details_automatic_tracks_consensus.py"
    spec = importlib.util.spec_from_file_location("details_consensus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_candidate_uses_visible_ratio_and_mask_support():
    module = _load_module()
    rows = [{
        "visible_counts": {1: 10, 2: 2, 3: 10},
        "inside_counts": {1: 4, 2: 2, 3: 2},
    }]
    selected = module.initial_candidate_superpoints(
        rows,
        {1: 20, 2: 100, 3: 20},
        min_visible_ratio=0.10,
        min_mask_support=0.30,
    )
    assert selected == {1}


def test_consensus_requires_repeated_support_and_reports_visibility():
    module = _load_module()
    rows = [
        {"visible_counts": {1: 10, 2: 10}, "inside_counts": {1: 8, 2: 8}},
        {"visible_counts": {1: 10, 2: 10}, "inside_counts": {1: 7, 2: 2}},
        {"visible_counts": {1: 2, 2: 10}, "inside_counts": {1: 0, 2: 8}},
    ]
    kept, diagnostics = module.consensus_superpoints(
        {1, 2},
        rows,
        min_visible_points=3,
        frame_superpoint_coverage=0.50,
        min_support_frames=2,
        min_consensus_rate=0.30,
        mean_superpoint_coverage=0.55,
    )
    assert kept == {1, 2}
    assert diagnostics[1]["visible_frames"] == 2
    assert diagnostics[1]["support_frames"] == 2
    assert diagnostics[2]["visible_frames"] == 3
    assert diagnostics[2]["support_frames"] == 2


def test_consensus_drops_single_view_and_low_coverage_superpoints():
    module = _load_module()
    rows = [
        {"visible_counts": {1: 10, 2: 10}, "inside_counts": {1: 9, 2: 5}},
        {"visible_counts": {1: 10, 2: 10}, "inside_counts": {1: 1, 2: 5}},
    ]
    kept, _ = module.consensus_superpoints(
        {1, 2},
        rows,
        min_visible_points=3,
        frame_superpoint_coverage=0.50,
        min_support_frames=2,
        min_consensus_rate=0.30,
        mean_superpoint_coverage=0.55,
    )
    assert kept == set()


def test_merged_tracklet_counts_one_vote_per_frame_and_uses_mask_union_max():
    module = _load_module()
    rows = [
        {
            "frame_index": 4,
            "visible_counts": {1: 10, 2: 10},
            "inside_counts": {1: 8, 2: 2},
        },
        {
            "frame_index": 4,
            "visible_counts": {1: 10, 2: 10},
            "inside_counts": {1: 3, 2: 7},
        },
        {
            "frame_index": 8,
            "visible_counts": {1: 10, 2: 10},
            "inside_counts": {1: 9, 2: 6},
        },
    ]
    collapsed = module.collapse_tracklet_rows_by_frame(rows)
    assert [row["frame_index"] for row in collapsed] == [4, 8]
    assert collapsed[0]["inside_counts"] == {1: 8, 2: 7}


def test_merged_tracklet_uses_exact_point_union_when_provenance_is_available():
    module = _load_module()
    rows = [
        {
            "frame_index": 4,
            "visible_counts": {1: 10, 2: 10},
            "inside_counts": {1: 2, 2: 1},
            "inside_point_indices_by_superpoint": {1: [1, 2], 2: [5]},
        },
        {
            "frame_index": 4,
            "visible_counts": {1: 10, 2: 10},
            "inside_counts": {1: 2, 2: 2},
            "inside_point_indices_by_superpoint": {1: [2, 3], 2: [6, 7]},
        },
    ]
    collapsed = module.collapse_tracklet_rows_by_frame(rows)
    assert collapsed == [
        {
            "frame_index": 4,
            "visible_counts": {1: 10, 2: 10},
            "inside_counts": {1: 3, 2: 3},
        }
    ]


def test_merge_refinement_cannot_reintroduce_superpoints_outside_current_union():
    module = _load_module()
    observations = {
        1: {
            "frame_index": 0,
            "visible_counts": {1: 10, 2: 10},
            "inside_counts": {1: 9, 2: 9},
        },
        2: {
            "frame_index": 1,
            "visible_counts": {1: 10, 2: 10},
            "inside_counts": {1: 9, 2: 9},
        },
    }
    kept, _, _, initial = module.refine_tracklet_superpoints(
        [1, 2],
        observations,
        {1: 10, 2: 10},
        min_visible_ratio=0.10,
        min_mask_support=0.30,
        min_visible_points=3,
        frame_superpoint_coverage=0.50,
        min_support_frames=2,
        min_consensus_rate=0.30,
        mean_superpoint_coverage=0.55,
        candidate_superpoints={1},
    )
    assert initial == {1}
    assert kept == {1}
