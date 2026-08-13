import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "annotate_automatic_track_geometry_relations.py"
    spec = importlib.util.spec_from_file_location("automatic_geometry_relations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relation_routes_append_boundary_and_duplicate():
    module = _module()
    sizes = np.asarray([100])
    assert module.relation_from_overlap(100, sizes, np.asarray([10]))["route"] == "新增实例"
    assert module.relation_from_overlap(100, sizes, np.asarray([50]))["route"] == "边界竞争"
    assert module.relation_from_overlap(100, sizes, np.asarray([80]))["route"] == "内部重复或冲突"
