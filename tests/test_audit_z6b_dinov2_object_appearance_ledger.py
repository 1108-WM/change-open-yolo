import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "audit_z6b_dinov2_object_appearance_ledger.py"
    spec = importlib.util.spec_from_file_location("audit_z6b_dino", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_close_uses_absolute_tolerance():
    module = _module()
    assert module._close(0.5, 0.50001, 2e-5)
    assert not module._close(0.5, 0.50003, 2e-5)
