import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "verify_automatic_sam_evidence_graph_entry.py"
    spec = importlib.util.spec_from_file_location("verify_automatic_sam_evidence_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_output_allows_only_preflight_manifest(tmp_path):
    verifier = _module()
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "entry_preflight_manifest.json").write_text("{}\n")
    verifier._require_empty_output(output_root)
    (output_root / "actual_result.json").write_text("{}\n")
    try:
        verifier._require_empty_output(output_root)
    except SystemExit as error:
        assert "避免覆盖" in str(error)
    else:
        raise AssertionError("真实结果必须阻止入口覆盖")
