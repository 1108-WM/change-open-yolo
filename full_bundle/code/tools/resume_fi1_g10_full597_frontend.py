#!/usr/bin/env python3
"""Resume the frozen full597 G10 frontend without frequent tree-size scans.

The original runner, selection, manifests, model/code asset contract and every
completed frame/stage receipt remain authoritative.  This wrapper only avoids
walking a growing multi-million-file output tree every ten seconds.  Free disk
space and the 50 GiB reserve are still checked every fifteen seconds.
"""
from __future__ import annotations

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

from tools.fi1_frontend_protocol import digest, sha256_file as sha, write_json
from tools.run_fi1_g10_full597_frontend import (
    DATA_BASE, FRONTEND, OUT, PYTHON, FRONTEND_RUNNER, RESERVE_BYTES,
    audit_scene, completed_scenes, allocated,
)

LOCAL = ROOT / ".local/fi1_g10_full597_frontend_20260919_v1"
RESUME_LOCAL = ROOT / ".local/fi1_g10_full597_frontend_resume_20260920_v1"


def read(path: Path):
    return json.loads(Path(path).read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def run() -> None:
    RESUME_LOCAL.mkdir(parents=True, exist_ok=True)
    selection = read(FRONTEND / "selection.json")
    contract = read(FRONTEND / "contract.json")
    preflight = read(FRONTEND / "preflight.json")
    scenes = selection["scenes"]
    require(len(scenes) == len(set(scenes)) == 597, "frozen selection is not full597")
    require(selection["stride"] == 10 and selection["active_arms"] == ["D"],
            "frozen selection is not strict G10/D")
    require(preflight["G10_frame_count"] == 94446 and preflight["ready_scene_count"] == 597,
            "frozen preflight coverage differs")
    require(contract["selection_sha256"] == digest(selection), "selection contract differs")
    require(sha(contract["runner"]) == contract["runner_sha256"], "original runner changed")
    require(contract["assets"] == __import__(
        "tools.run_fi1_frontend_comparison", fromlist=["assets"]
    ).assets(), "frontend code/model asset contract changed")

    done = completed_scenes(scenes)
    remaining = [scene for scene in scenes if scene not in done]
    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": now(),
        "total_scene_count": 597,
        "G10_frame_count": 94446,
        "resumed_completed_scene_count": len(done),
        "completed_scene_count": len(done),
        "remaining_scene_count": len(remaining),
        "current_scene": None,
        "current_frame_id": None,
        "original_contract_sha256": digest(contract),
        "original_runner": contract["runner"],
        "original_runner_sha256": contract["runner_sha256"],
        "resume_runner_sha256": sha(__file__),
        "ground_truth_read": False,
        "training_run": False,
        "AP_run": False,
    }
    write_json(RESUME_LOCAL / "execution.json", state)
    if not remaining:
        state.update(status="complete", finished_at=now(), allocated_bytes=allocated(OUT))
        write_json(RESUME_LOCAL / "execution.json", state)
        return

    command = [PYTHON, FRONTEND_RUNNER, "export", "--output", FRONTEND,
               "--scenes", *remaining, "--arms", "D"]
    log_path = RESUME_LOCAL / "frontend.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    with (RESUME_LOCAL / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        print("RESUME", len(done), "COMPLETE", len(remaining), "REMAINING", flush=True)
        with log_path.open("a") as log:
            process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    progress = read(FRONTEND / "progress.json") if (FRONTEND / "progress.json").exists() else {}
                    free = shutil.disk_usage(DATA_BASE).free
                    require(free >= RESERVE_BYTES, "disk reserve reached")
                    # Stage receipts are only 4*597 fixed paths; this remains O(scene count),
                    # unlike recursively walking all frame-cache files.
                    done = completed_scenes(scenes)
                    state.update(child_pid=process.pid,
                                 completed_scene_count=len(done),
                                 current_scene=progress.get("scene"),
                                 current_frame_id=progress.get("frame_id"),
                                 free_bytes=free, updated_at=now())
                    write_json(RESUME_LOCAL / "execution.json", state)
                    time.sleep(15)
                require(process.returncode == 0, f"frontend resume failed; see {log_path}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

        audits = [audit_scene(scene) for scene in scenes]
        require(sum(row["G10_frames"] for row in audits) == 94446,
                "completed G10 frame coverage differs")
        completion = {
            "complete": True,
            "scene_count": 597,
            "G10_frame_count": 94446,
            "allocated_bytes": allocated(OUT),
            "original_contract_sha256": digest(contract),
            "resume_seconds": time.monotonic() - started,
            "ground_truth_read": False,
            "training_run": False,
            "AP_run": False,
            "finished_at": now(),
        }
        write_json(OUT / "g10_frontend_complete.json", completion)
        state.update(status="complete", completed_scene_count=597,
                     current_scene=None, current_frame_id=None,
                     allocated_bytes=completion["allocated_bytes"],
                     finished_at=completion["finished_at"])
        write_json(RESUME_LOCAL / "execution.json", state)
        print("COMPLETE", 597, "SCENES", flush=True)


if __name__ == "__main__":
    run()
