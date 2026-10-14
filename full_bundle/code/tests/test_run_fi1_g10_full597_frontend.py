import json
from pathlib import Path

from tools import run_fi1_g10_full597_frontend as runner


def test_source_inventory_covers_frozen_full597():
    manifest = json.loads((runner.ROOT / "资料/训练597场景清单_20260917.json").read_text())
    scenes, inventory = runner.source_inventory(manifest)
    assert len(scenes) == len(inventory) == 597
    assert sum(row["G10_frame_count"] for row in inventory.values()) == 94446
    assert {inventory[scene]["source"] for scene in scenes[:3]} == set(runner.SOURCE_ROOTS)
    assert {name: sum(row["source"] == name for row in inventory.values())
            for name in runner.SOURCE_ROOTS} == {
                "old_official100": 100,
                "old_ncs_train100": 100,
                "new397": 397,
            }


def test_completion_detection_requires_all_four_stage_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "FRONTEND", tmp_path)
    scene = "scene0000_00"
    receipt_root = tmp_path / "arms/D/stage_receipts"
    receipt_root.mkdir(parents=True)
    for stage in ("hierarchy", "tracks", "consensus"):
        (receipt_root / f"{scene}_{stage}.json").write_text("{}\n")
    assert runner.completed_scenes([scene]) == []
    (receipt_root / f"{scene}_merged.json").write_text("{}\n")
    assert runner.completed_scenes([scene]) == [scene]
