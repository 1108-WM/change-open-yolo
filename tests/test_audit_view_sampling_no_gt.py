import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "audit_view_sampling_no_gt.py"
    spec = importlib.util.spec_from_file_location("view_sampling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uniform_sampling_spans_loaded_sequence():
    module = _module()
    assert module.select_loaded_frame_positions(100, 30, "first").tolist() == list(range(30))
    uniform = module.select_loaded_frame_positions(100, 30, "uniform")
    assert len(uniform) == 30
    assert int(uniform[0]) == 0
    assert int(uniform[-1]) == 99
