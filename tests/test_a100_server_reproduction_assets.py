from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "configs/repro/scannetv2_val_312.txt"
EXPECTED_SHA = "d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083"


def test_frozen_val312_scene_list_identity():
    rows = [row.strip() for row in SCENES.read_text().splitlines() if row.strip()]
    assert len(rows) == 312
    assert len(rows) == len(set(rows))
    assert all(len(row) == 12 and row.startswith("scene") and row[9] == "_" for row in rows)
    assert hashlib.sha256(SCENES.read_bytes()).hexdigest() == EXPECTED_SHA


def test_reproduction_path_template_is_valid_and_requires_new_paths():
    value = json.loads((ROOT / "configs/repro/fi1_dm_sms1_paths.template.json").read_text())
    assert value["scene_list"] == "configs/repro/scannetv2_val_312.txt"
    assert value["run_root"] == "/ABSOLUTE/PATH/NEW_EMPTY_RUN_ROOT"
    assert value["qwen_model_dir"] == "pretrained/checkpoints/Qwen2.5-VL-7B-Instruct"


def test_reproduction_scripts_parse_without_execution():
    scripts = [
        "scripts/download_scannet200_val_stream.sh",
        "scripts/prepare_scannet200_val_stream.sh",
        "scripts/setup_a100_fi1_dm_sms1_env.sh",
        "scripts/download_fi1_dm_sms1_model_assets.sh",
    ]
    for relative in scripts:
        subprocess.run(["bash", "-n", str(ROOT / relative)], check=True)


def test_setup_material_does_not_embed_credentials():
    paths = [
        ROOT / "docs/A100_SERVER_REPRODUCTION_20260829.md",
        ROOT / "scripts/setup_a100_fi1_dm_sms1_env.sh",
        ROOT / "scripts/download_fi1_dm_sms1_model_assets.sh",
    ]
    forbidden = ("ghp_", "github_pat_", "BEGIN OPENSSH PRIVATE KEY", "HF_TOKEN=")
    text = "\n".join(path.read_text() for path in paths)
    assert not any(value in text for value in forbidden)
