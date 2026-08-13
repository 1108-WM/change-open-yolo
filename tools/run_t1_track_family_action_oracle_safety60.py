#!/usr/bin/env python3
"""Resume the safety60 T1b local GT-oracle ledger scene by scene.

This is only an orchestration wrapper around
``diagnose_t1_track_family_action_oracle_gt.py``.  It discovers already
published action rows by their frozen T1 action indices, skips them, and runs
the remaining ranges in isolated subprocesses.  It never materializes a
proposal and never computes AP.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = "t1_track_family_action_oracle_gt_gvc_safety60_uniform30_d1_20260806"
ACTION_TYPES = ("keep", "attach", "reassign", "merge")


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _read_scene_list(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"duplicate scene in {path}")
    return scenes


def _ledger_actions(ledger_root: Path, scene: str) -> list[dict]:
    path = ledger_root / scene / "track_family_actions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(_read_jsonl(path))


def _published_rows(diagnostics_root: Path, prefix: str, scene: str) -> dict[int, dict]:
    """Collect existing non-smoke rows and verify duplicate publications agree."""
    rows: dict[int, dict] = {}
    pattern = f"{prefix}*/{scene}/t1_action_oracle_gt.jsonl"
    for path in sorted(diagnostics_root.glob(pattern)):
        if "_smoke_" in path.parts[-3]:
            continue
        for row in _read_jsonl(path):
            if row.get("scene_name") != scene:
                raise ValueError(f"scene mismatch in {path}")
            index = int(row["t1_action_index"])
            previous = rows.get(index)
            if previous is not None and previous != row:
                raise ValueError(
                    f"conflicting published rows for {scene} action {index}: {path}"
                )
            rows[index] = row
    return rows


def _missing_position_ranges(
    actions: list[dict], action_type: str, covered_indices: set[int], chunk_size: int
) -> list[tuple[int, int]]:
    """Return missing [position_start, position_end) ranges after type filtering."""
    typed_global_indices = [
        index for index, action in enumerate(actions) if action["action_type"] == action_type
    ]
    missing_positions = [
        position
        for position, global_index in enumerate(typed_global_indices)
        if global_index not in covered_indices
    ]
    if not missing_positions:
        return []

    contiguous: list[tuple[int, int]] = []
    start = previous = missing_positions[0]
    for position in missing_positions[1:]:
        if position != previous + 1:
            contiguous.append((start, previous + 1))
            start = position
        previous = position
    contiguous.append((start, previous + 1))

    chunks: list[tuple[int, int]] = []
    for start, end in contiguous:
        while start < end:
            chunk_end = min(start + chunk_size, end)
            chunks.append((start, chunk_end))
            start = chunk_end
    return chunks


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(staging, path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=REPO_ROOT / "output/scannet200/scene_splits/gvc_holdout_20260803/gvc_safety60.txt",
    )
    parser.add_argument(
        "--t1-ledger-root",
        type=Path,
        default=REPO_ROOT / "output/t1_track_family_action_ledger_gvc_safety60_uniform30_d1_20260806",
    )
    parser.add_argument(
        "--d1-root",
        type=Path,
        default=REPO_ROOT / "output/details_same_frame_hierarchy_safe_gvc_safety60_uniform30_d1_20260805",
    )
    parser.add_argument(
        "--track-root",
        type=Path,
        default=REPO_ROOT / "output/automatic_mask_tracks_details_siou_hierarchy_safe_gvc_safety60_uniform30_d1_20260805",
    )
    parser.add_argument(
        "--d2b-root",
        type=Path,
        default=REPO_ROOT / "output/details_iterative_proposal_merges_exact_frame_union_gvc_safety60_uniform30_d2b_20260805",
    )
    parser.add_argument("--diagnostics-root", type=Path, default=REPO_ROOT / "docs/diagnostics")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--start-scene-offset", type=int, default=0)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--reassign-chunk-size", type=int, default=1000)
    parser.add_argument("--other-chunk-size", type=int, default=2000)
    parser.add_argument("--action-types", nargs="+", choices=ACTION_TYPES, default=list(ACTION_TYPES))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit(
            "--allow-gt-diagnostics is required: this runner reads GT and only builds the offline T1b ledger"
        )
    if args.start_scene_offset < 0:
        raise SystemExit("--start-scene-offset must be non-negative")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    if args.reassign_chunk_size <= 0 or args.other_chunk_size <= 0:
        raise SystemExit("chunk sizes must be positive")

    for name in ("scene_list", "t1_ledger_root", "d1_root", "track_root", "d2b_root"):
        path = getattr(args, name).resolve()
        setattr(args, name, path)
        if not path.exists():
            raise FileNotFoundError(path)
    args.diagnostics_root = args.diagnostics_root.resolve()
    args.diagnostics_root.mkdir(parents=True, exist_ok=True)
    args.python = args.python.resolve()
    if not args.python.is_file():
        raise FileNotFoundError(args.python)

    scenes = _read_scene_list(args.scene_list)[args.start_scene_offset :]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    all_scenes = _read_scene_list(args.scene_list)
    offsets = {scene: index for index, scene in enumerate(all_scenes)}

    plan: list[dict] = []
    for scene in scenes:
        actions = _ledger_actions(args.t1_ledger_root, scene)
        published = _published_rows(args.diagnostics_root, args.output_prefix, scene)
        for index, row in published.items():
            if index < 0 or index >= len(actions):
                raise ValueError(f"published action index out of range: {scene} {index}")
            if row["action_type"] != actions[index]["action_type"]:
                raise ValueError(f"published action type mismatch: {scene} {index}")
        covered = set(published)
        for action_type in args.action_types:
            chunk_size = (
                args.reassign_chunk_size if action_type == "reassign" else args.other_chunk_size
            )
            for start, end in _missing_position_ranges(actions, action_type, covered, chunk_size):
                plan.append(
                    {
                        "scene": scene,
                        "scene_offset": offsets[scene],
                        "action_type": action_type,
                        "action_offset": start,
                        "action_count": end - start,
                        "action_end_inclusive": end - 1,
                    }
                )

    total_actions = sum(item["action_count"] for item in plan)
    print(
        json.dumps(
            {
                "selected_scene_count": len(scenes),
                "remaining_chunk_count": len(plan),
                "remaining_action_count": total_actions,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        for item in plan:
            print(json.dumps(item, ensure_ascii=False), flush=True)
        return
    if not plan:
        print("[complete] no missing T1b action rows", flush=True)
        return

    lock_path = args.diagnostics_root / f".{args.output_prefix}_runner.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"another T1b runner may be active: {lock_path}") from exc
    os.write(descriptor, f"pid={os.getpid()} time={_timestamp()}\n".encode())
    os.close(descriptor)

    state_path = args.diagnostics_root / f"{args.output_prefix}_runner_state.json"
    completed_actions = 0
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    environment.setdefault("OMP_NUM_THREADS", "8")
    try:
        for chunk_index, item in enumerate(plan, start=1):
            scene = item["scene"]
            action_type = item["action_type"]
            start = item["action_offset"]
            end = item["action_end_inclusive"]
            output_root = args.diagnostics_root / (
                f"{args.output_prefix}_batch_{scene}_{action_type}_{start:06d}_{end:06d}"
            )
            if output_root.exists() and any(output_root.iterdir()):
                raise FileExistsError(
                    f"incomplete or conflicting output root; inspect without deleting: {output_root}"
                )
            command = [
                str(args.python),
                str(REPO_ROOT / "tools/diagnose_t1_track_family_action_oracle_gt.py"),
                "--scene-list", str(args.scene_list),
                "--t1-ledger-root", str(args.t1_ledger_root),
                "--d1-root", str(args.d1_root),
                "--track-root", str(args.track_root),
                "--d2b-root", str(args.d2b_root),
                "--processed-scene-root", str(REPO_ROOT / "data/scannet200"),
                "--dataset-root", str(REPO_ROOT / "data/scannet200"),
                "--config-path", str(REPO_ROOT / "pretrained/config_scannet200.yaml"),
                "--gt-instance-dir", str(REPO_ROOT / "data/scannet200/ground_truth"),
                "--output-root", str(output_root),
                "--scene-offset", str(item["scene_offset"]),
                "--max-scenes", "1",
                "--action-types", action_type,
                "--action-offset", str(start),
                "--max-actions-per-scene", str(item["action_count"]),
                "--allow-gt-diagnostics",
            ]
            state = {
                "status": "running",
                "updated_at": _timestamp(),
                "chunk_index": chunk_index,
                "chunk_count": len(plan),
                "completed_action_count_this_run": completed_actions,
                "total_action_count_this_run": total_actions,
                "current": item,
                "output_root": str(output_root),
                "command": command,
                "proposal_materialization_applied": False,
                "ap_computed": False,
                "ground_truth_usage": "GT-only diagnostic",
            }
            _atomic_write_json(state_path, state)
            print(
                f"[run {chunk_index}/{len(plan)}] {scene} {action_type} "
                f"positions {start}-{end}",
                flush=True,
            )
            subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
            completed_actions += item["action_count"]

        _atomic_write_json(
            state_path,
            {
                "status": "completed",
                "updated_at": _timestamp(),
                "chunk_count": len(plan),
                "completed_action_count_this_run": completed_actions,
                "total_action_count_this_run": total_actions,
                "proposal_materialization_applied": False,
                "ap_computed": False,
                "ground_truth_usage": "GT-only diagnostic",
            },
        )
        print(f"[complete] wrote {completed_actions} missing action rows", flush=True)
    except BaseException as exc:
        _atomic_write_json(
            state_path,
            {
                "status": "failed",
                "updated_at": _timestamp(),
                "completed_action_count_this_run": completed_actions,
                "total_action_count_this_run": total_actions,
                "error": repr(exc),
                "proposal_materialization_applied": False,
                "ap_computed": False,
                "ground_truth_usage": "GT-only diagnostic",
            },
        )
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
