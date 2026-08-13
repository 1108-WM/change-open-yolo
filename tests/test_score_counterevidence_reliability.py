import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "score_counterevidence_reliability.py"
    spec = importlib.util.spec_from_file_location("counter_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counter_score_rewards_independent_low_depth_and_non_boundary_evidence():
    module = _module()
    common = {
        "positive_weight": 0.1, "negative_weight": 0.9, "counterevidence_eligible_view_rate": 0.8,
        "counter_view_max_camera_baseline_m": 1.0, "counter_view_max_view_angle_deg": 30.0,
        "depth_residual_mean_m": 0.005, "uncovered_near_2px_mask_boundary_ratio": 0.0,
        "sp_inside_top_native_ratio": 0.0, "mask_coverage_std": 0.1,
        "mask_coverage_mean": 0.1, "covered_mask_2px_interior_ratio": 0.1,
    }
    weak = {**common, "depth_residual_mean_m": 0.05, "uncovered_near_2px_mask_boundary_ratio": 0.9, "sp_inside_top_native_ratio": 0.9}
    scored = module._score_scene([common, weak], 0.05)
    assert scored[0]["counterevidence_reliability_score"] > scored[1]["counterevidence_reliability_score"]
