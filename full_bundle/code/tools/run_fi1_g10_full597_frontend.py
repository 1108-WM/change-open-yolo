#!/usr/bin/env python3
"""Run the exploratory GD-SAM stride-10 frontend on all 597 train scans.

This is a thin orchestration layer over ``run_fi1_frontend_comparison.py``.
It changes only the frozen scene inventory and D-arm frame sampling.  The
GD-SAM, projection, hierarchy, tracking, consensus and merge implementations
remain unchanged.  No ground truth, AP evaluation or model fitting is used.

The first three scans cover the three prepared-data roots.  They form a real
GPU/storage gate; after they pass, the same process contract automatically
continues over the remaining scans.  Frame caches and stage receipts make a
rerun resumable after interruption.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.fi1_frontend_protocol import digest, freeze_json, sha256_file as sha, write_json
from tools.run_fi1_frontend_comparison import assets, build_scene_manifest


DATA_BASE = Path("/media/jia/软件1/scannet_additional_train_stream")
OUT = DATA_BASE / "fi1_G10_full597_20260919"
FRONTEND = OUT / "frontend"
LOCAL = ROOT / ".local/fi1_g10_full597_frontend_20260919_v1"
SELECTION = LOCAL / "selection"
TRAIN_MANIFEST = ROOT / "资料/训练597场景清单_20260917.json"
PYTHON = ROOT / ".venvs/yoloe_gpu/bin/python"
FRONTEND_RUNNER = ROOT / "tools/run_fi1_frontend_comparison.py"
RESERVE_BYTES = 50 * 1024**3
PILOT_COUNT = 3

SOURCE_ROOTS = {
    "old_official100": Path("/media/jia/软件1/scannet_train_stream/prepared"),
    "old_ncs_train100": Path("/media/jia/软件1/scannet_train_stream/prepared_ncs"),
    "new397": DATA_BASE / "prepared",
}


def read(path: Path):
    return json.loads(Path(path).read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def allocated(path: Path) -> int:
    total = 0
    for base, _, files in os.walk(path, followlinks=False):
        for name in files:
            item = Path(base) / name
            if not item.is_symlink():
                total += item.stat().st_blocks * 512
    return total


def source_inventory(manifest: dict) -> tuple[list[str], dict[str, dict]]:
    old = manifest["old200"]
    new = manifest["new397"]
    train = manifest["roles"]["train"]
    require(len(old) == len(set(old)) == 200, "old200 inventory differs")
    require(len(new) == len(set(new)) == 397, "new397 inventory differs")
    require(train == sorted(old + new) and len(set(train)) == 597,
            "full597 train inventory differs")

    old_set, new_set = set(old), set(new)
    rows: dict[str, dict] = {}
    for scene in train:
        present = [name for name, root in SOURCE_ROOTS.items() if (root / scene).is_dir()]
        require(len(present) == 1, f"scene must occur in exactly one prepared root: {scene} {present}")
        source = present[0]
        require((scene in old_set) == (source != "new397"), f"old/new source mismatch: {scene}")
        require((scene in new_set) == (source == "new397"), f"new source mismatch: {scene}")
        root = SOURCE_ROOTS[source]
        counts = {}
        for folder in ("color", "depth", "poses"):
            path = root / scene / folder
            require(path.is_dir(), f"missing {folder}: {scene}")
            counts[folder] = len(os.listdir(path))
        require(len(set(counts.values())) == 1 and counts["color"] > 0,
                f"RGB/depth/pose count mismatch: {scene} {counts}")
        rows[scene] = {
            "source": source,
            "prepared_root": str(root),
            "original_frame_count": counts["color"],
            "G10_frame_count": (counts["color"] + 9) // 10,
        }
    require({rows[s]["source"] for s in train[:PILOT_COUNT]} == set(SOURCE_ROOTS),
            "pilot3 must cover all prepared roots")
    return train, rows


def freeze_contracts() -> tuple[list[str], dict[str, dict], dict]:
    manifest = read(TRAIN_MANIFEST)
    scenes, inventory = source_inventory(manifest)
    selection = {
        "version": 1,
        "dataset_role": "train597_g10_frontend",
        "scenes": scenes,
        "pilot_scenes": scenes[:PILOT_COUNT],
        "active_arms": ["D"],
        "frontend": "grounded_sam",
        "stride": 10,
        "auto_expand_after_pilot3_gate": True,
        "selection_uses": "frozen train597 scan names only; no labels, losses or AP",
        "ground_truth_usage": "none",
        "new_training": False,
        "AP": False,
        "exploratory_scale_up": True,
        "pilot30_expansion_threshold_passed": False,
    }
    freeze_json(SELECTION / "selection.json", selection)
    scene_text = "\n".join(scenes) + "\n"
    scene_file = SELECTION / "scenes_execution_order.txt"
    if scene_file.exists():
        require(scene_file.read_text() == scene_text, "frozen scene order changed")
    else:
        scene_file.parent.mkdir(parents=True, exist_ok=True)
        scene_file.write_text(scene_text)
    freeze_json(SELECTION / "source_inventory.json", inventory)
    freeze_json(SELECTION / "selection_receipt.json", {
        "complete": True,
        "inputs": {str(TRAIN_MANIFEST): sha(TRAIN_MANIFEST)},
        "outputs": {
            "selection.json": sha(SELECTION / "selection.json"),
            "scenes_execution_order.txt": sha(scene_file),
            "source_inventory.json": sha(SELECTION / "source_inventory.json"),
        },
    })
    return scenes, inventory, selection


def prepare_frontend(scenes: list[str], inventory: dict[str, dict], selection: dict) -> dict:
    freeze_json(FRONTEND / "selection.json", selection)
    total_frames = 0
    invalid_poses = 0
    rows = []
    for index, scene in enumerate(scenes, 1):
        root = Path(inventory[scene]["prepared_root"])
        manifest = build_scene_manifest(scene, root, root)
        g10 = list(range(0, manifest["original_frame_count"], 10))
        require(len(g10) == inventory[scene]["G10_frame_count"], f"G10 count differs: {scene}")
        require(g10 and all(frame % 10 == 0 for frame in g10), f"invalid G10 frames: {scene}")
        manifest["arms"]["D"] = g10
        manifest["derived_variant"] = {
            "name": "grounded_sam_stride10_full597",
            "compatible_arm": "D",
            "stride": 10,
            "new_visual_inference": True,
            "source": inventory[scene]["source"],
        }
        freeze_json(FRONTEND / "manifests" / f"{scene}.json", manifest)
        bad = sum(not frame["pose_valid"] for frame in manifest["frames"])
        total_frames += len(g10)
        invalid_poses += bad
        rows.append({
            "scene": scene,
            "source": inventory[scene]["source"],
            "ready": True,
            "G10_frames": len(g10),
            "original_frame_count": manifest["original_frame_count"],
            "invalid_pose_count": bad,
        })
        if index % 50 == 0 or index == len(scenes):
            print("PREPARED", index, len(scenes), flush=True)

    asset_hashes = assets()
    contract = {
        "selection_sha256": digest(selection),
        "assets": asset_hashes,
        "variant": "grounded_sam_stride10_full597",
        "stride": 10,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha(__file__),
        "train_manifest": str(TRAIN_MANIFEST),
        "train_manifest_sha256": sha(TRAIN_MANIFEST),
        "ground_truth_usage": "none",
        "new_visual_inference": True,
        "AP_evaluated": False,
    }
    freeze_json(FRONTEND / "contract.json", contract)
    write_json(FRONTEND / "preflight.json", {
        "state": "ready_for_export",
        "ready_scene_count": len(rows),
        "required_scene_count": len(scenes),
        "inventory": rows,
        "source_roots": {name: str(path) for name, path in SOURCE_ROOTS.items()},
        "original_frame_count": sum(r["original_frame_count"] for r in rows),
        "G10_frame_count": total_frames,
        "invalid_pose_count": invalid_poses,
        "ground_truth_usage": "none",
        "real_inference_complete": False,
        "AP_evaluated": False,
    })
    return contract


def completed_scenes(scenes: list[str]) -> list[str]:
    return [scene for scene in scenes
            if all((FRONTEND / "arms/D/stage_receipts" / f"{scene}_{stage}.json").is_file()
                   for stage in ("hierarchy", "tracks", "consensus", "merged"))]


def audit_scene(scene: str) -> dict:
    manifest = read(FRONTEND / "manifests" / f"{scene}.json")
    expected = list(range(0, manifest["original_frame_count"], 10))
    require(manifest["arms"]["D"] == expected, f"scene is not strict G10: {scene}")
    files = {}
    for stage in ("hierarchy", "tracks", "consensus", "merged"):
        receipt = FRONTEND / "arms/D/stage_receipts" / f"{scene}_{stage}.json"
        require(receipt.is_file(), f"missing frontend stage: {scene} {stage}")
        files[str(receipt)] = sha(receipt)
    return {
        "complete": True,
        "scene": scene,
        "strict_stride": 10,
        "G10_frames": len(expected),
        "stage_receipts": files,
        "ground_truth_read": False,
        "training_run": False,
        "AP_run": False,
    }


def run_child(label: str, scenes: list[str], state: dict, code: dict) -> None:
    if not scenes:
        return
    for path, expected in code.items():
        require(sha(path) == expected, "source changed during run: " + path)
    require(shutil.disk_usage(DATA_BASE).free >= RESERVE_BYTES, "less than 50 GiB free")
    command = [PYTHON, FRONTEND_RUNNER, "export", "--output", FRONTEND,
               "--scenes", *scenes, "--arms", "D"]
    log_path = LOCAL / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    state.update(status="running", stage=label, stage_scene_count=len(scenes), updated_at=now())
    write_json(LOCAL / "execution.json", state)
    started = time.monotonic()
    print("START", label, len(scenes), flush=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                progress = read(FRONTEND / "progress.json") if (FRONTEND / "progress.json").exists() else {}
                done = completed_scenes(state["scenes"])
                state.update(
                    child_pid=process.pid,
                    completed_scene_count=len(done),
                    completed_scenes=done,
                    current_scene=progress.get("scene"),
                    current_frame_id=progress.get("frame_id"),
                    free_bytes=shutil.disk_usage(DATA_BASE).free,
                    allocated_bytes=allocated(OUT),
                    updated_at=now(),
                )
                write_json(LOCAL / "execution.json", state)
                require(state["free_bytes"] >= RESERVE_BYTES, "disk reserve reached")
                time.sleep(10)
            require(process.returncode == 0, f"{label} failed; see {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    state.setdefault("stage_history", []).append({
        "stage": label,
        "scene_count": len(scenes),
        "seconds": time.monotonic() - started,
        "log": str(log_path),
    })
    print("DONE", label, state["stage_history"][-1]["seconds"], flush=True)


def run(check_only: bool = False) -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    scenes, inventory, selection = freeze_contracts()
    contract = prepare_frontend(scenes, inventory, selection)
    preflight = read(FRONTEND / "preflight.json")
    require(preflight["ready_scene_count"] == 597, "preflight does not cover full597")
    require(preflight["G10_frame_count"] == 94446, "full597 G10 frame count changed")
    if check_only:
        print(json.dumps({
            "complete": True,
            "scene_count": len(scenes),
            "G10_frame_count": preflight["G10_frame_count"],
            "pilot_scenes": scenes[:PILOT_COUNT],
            "source_counts": {name: sum(row["source"] == name for row in inventory.values())
                              for name in SOURCE_ROOTS},
            "output": str(OUT),
            "ground_truth_read": False,
            "AP_evaluated": False,
        }, ensure_ascii=False, indent=2))
        return

    code_paths = [Path(__file__), FRONTEND_RUNNER, ROOT / "tools/fi1_frontend_protocol.py"]
    code = {str(path.resolve()): sha(path) for path in code_paths}
    done = completed_scenes(scenes)
    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": now(),
        "scenes": scenes,
        "total_scene_count": len(scenes),
        "G10_frame_count": preflight["G10_frame_count"],
        "pilot_scenes": scenes[:PILOT_COUNT],
        "completed_scenes": done,
        "completed_scene_count": len(done),
        "code": code,
        "contract_sha256": digest(contract),
        "ground_truth_read": False,
        "training_run": False,
        "AP_run": False,
        "stage_history": [],
    }

    with (LOCAL / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            pilot_missing = [scene for scene in scenes[:PILOT_COUNT] if scene not in done]
            run_child("pilot3_frontend", pilot_missing, state, code)
            pilot_audits = [audit_scene(scene) for scene in scenes[:PILOT_COUNT]]
            used = allocated(OUT)
            pilot_frames = sum(row["G10_frames"] for row in pilot_audits)
            projected_total = int(used / pilot_frames * preflight["G10_frame_count"])
            free = shutil.disk_usage(DATA_BASE).free
            require(free >= RESERVE_BYTES + int(max(0, projected_total - used) * 1.2),
                    "pilot3 passed but projected storage would violate reserve")
            write_json(LOCAL / "pilot_gate.json", {
                "complete": True,
                "audits": pilot_audits,
                "allocated_bytes": used,
                "pilot_G10_frames": pilot_frames,
                "projected_total_bytes": projected_total,
                "free_bytes": free,
                "reserve_bytes": RESERVE_BYTES,
                "auto_expand_scene_count": 597,
                "passed_at": now(),
            })

            done = completed_scenes(scenes)
            remaining = [scene for scene in scenes if scene not in done]
            run_child("remaining594_frontend", remaining, state, code)
            audits = [audit_scene(scene) for scene in scenes]
            require(sum(row["G10_frames"] for row in audits) == preflight["G10_frame_count"],
                    "completed G10 frame coverage differs")
            completion = {
                "complete": True,
                "scene_count": len(audits),
                "G10_frame_count": preflight["G10_frame_count"],
                "allocated_bytes": allocated(OUT),
                "contract_sha256": digest(contract),
                "ground_truth_read": False,
                "training_run": False,
                "AP_run": False,
                "finished_at": now(),
            }
            write_json(OUT / "g10_frontend_complete.json", completion)
            state.update(status="complete", completed_scenes=scenes,
                         completed_scene_count=len(scenes), current_scene=None,
                         current_frame_id=None, finished_at=completion["finished_at"],
                         allocated_bytes=completion["allocated_bytes"])
        except BaseException as exc:
            state.update(status="failed", error=repr(exc), updated_at=now())
            write_json(LOCAL / "execution.json", state)
            raise
        write_json(LOCAL / "execution.json", state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Freeze and audit contracts without GPU inference")
    args = parser.parse_args()
    run(check_only=args.check)


if __name__ == "__main__":
    main()
