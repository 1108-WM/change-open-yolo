#!/usr/bin/env python3
"""Pre-register no-GT C1d native-folding and track-replacement actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTAINMENT = .99


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _scenes(path: Path) -> list[str]:
    result = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("场景列表为空或含重复")
    return result


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _exact_groups(scene_root: Path, native_count: int) -> dict[int, str]:
    result = {candidate: f"singleton:{candidate}" for candidate in range(native_count)}
    for row in _jsonl(scene_root / "exact_geometry_groups.jsonl"):
        for candidate in row["candidate_ids"]:
            result[int(candidate)] = f"exact:{row['geometry_hash']}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--relation-ledger-root", type=Path, required=True)
    parser.add_argument("--native-fold-audit-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "relation_ledger_root", "native_fold_audit_root", "native_prediction_cache", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖：{args.output_root}")
    scenes = _scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True)
    total = 0
    summaries = []
    for scene in scenes:
        components = _jsonl(args.relation_ledger_root / scene / "relation_components.jsonl")
        relations = _jsonl(args.relation_ledger_root / scene / "track_native_relations.jsonl")
        native_count = int(__import__("numpy").load(args.native_prediction_cache / f"{scene}_pred_scores.npy", mmap_mode="r").shape[0])
        groups = _exact_groups(args.native_fold_audit_root / scene, native_count)
        covered_by_track = {}
        for relation in relations:
            if float(relation["native_inside_track_ratio"]) >= CONTAINMENT:
                covered_by_track.setdefault(int(relation["proposal_id"]), set()).add(int(relation["native_candidate_id"]))
        actions = []
        for component in components:
            component_id = int(component["component_id"])
            tracks = [int(value) for value in component["track_ids"]]
            natives = [int(value) for value in component["native_candidate_ids"]]
            native_groups = sorted({groups[value] for value in natives})
            base = {
                "scene_name": scene, "component_id": component_id,
                "component_track_ids": tracks, "component_native_candidate_ids": natives,
                "component_native_exact_geometry_group_ids": native_groups,
                "ground_truth_usage": "none", "proposal_materialization_applied": False,
                "decision_state": "pre-registered C1d action only; no action is selected or applied",
            }
            if natives:
                variants = [
                    ("coexist", None, natives, tracks, "raw"),
                    ("native_only", None, natives, [], "raw"),
                    ("native_exact_folded_only", None, natives, [], "exact_folded"),
                    ("track_only_all", None, [], tracks, "raw"),
                ]
                variants.extend(("native_plus_one_track", track, natives, [track], "raw") for track in tracks)
                variants.extend(("track_only_one", track, [], [track], "raw") for track in tracks)
                for track in tracks:
                    removed = sorted(set(natives) & covered_by_track.get(track, set()))
                    if removed:
                        variants.append(("replace_native_covered_099_with_track", track, sorted(set(natives) - set(removed)), [track], "raw"))
                removed_all = sorted(set(natives) & set().union(*(covered_by_track.get(track, set()) for track in tracks))) if tracks else []
                if removed_all:
                    variants.append(("replace_native_covered_099_with_all_tracks", None, sorted(set(natives) - set(removed_all)), tracks, "raw"))
            else:
                variants = [("keep_all_tracks", None, [], tracks, "raw"), ("suppress_all_tracks", None, [], [], "raw")]
                variants.extend(("keep_one_track", track, [], [track], "raw") for track in tracks)
            for kind, selected, kept_native, kept_tracks, mode in variants:
                action_name = kind if selected is None else f"{kind}:{selected}"
                actions.append({
                    **base, "action_name": action_name, "action_kind": kind,
                    "selected_track_id": selected, "kept_native_candidate_ids": kept_native,
                    "kept_track_ids": kept_tracks, "native_geometry_mode": mode,
                    "removed_native_candidate_count": len(natives) - len(kept_native),
                })
        scene_root = args.output_root / scene
        scene_root.mkdir()
        _write_jsonl(scene_root / "c1d_expanded_actions.jsonl", actions)
        summary = {"scene_name": scene, "component_count": len(components), "action_count": len(actions), "native_candidate_count": native_count}
        (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        summaries.append(summary)
        total += len(actions)
    root = {
        "diagnostic_type": "no-GT C1d expanded track-native competition action ledger",
        "ground_truth_usage": "none", "proposal_materialization_applied": False,
        "native_exact_geometry_folding_materialized": False,
        "native_inside_track_containment_threshold": CONTAINMENT,
        "scene_count": len(summaries), "component_count": sum(row["component_count"] for row in summaries),
        "action_count": total, "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
