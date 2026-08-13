import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_z6e_selective_vlm_review_manifest.py"
    spec = importlib.util.spec_from_file_location("z6e_manifest", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_candidate_name_requires_exact_registered_option():
    module = _module()
    review = {"candidate_classes": [{"class_index": 7, "class_name": "desk"}]}
    assert module._candidate_name(review, 7) == "desk"
