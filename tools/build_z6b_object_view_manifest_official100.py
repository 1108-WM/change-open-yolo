#!/usr/bin/env python3
"""Build the GT-free Z6b fixed top-3 object-view manifest for official100.

The tool joins the already materialized Z2 Alpha-CLIP view selections to the
frozen Z2c geometry-node ledger and audits the prepared RGB-D assets.  It does
not read GT, recompute projections, generate embeddings, invoke an MLLM, or
change any candidate, class, geometry, score, or inference plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = "z6b_object_view_manifest_official100_v1"
EXPECTED_NODE_COUNT = 9708
SOURCES = ("native", "track", "pair_union")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_paths(prepared_root: Path, scene: str, frame_id: str) -> dict[str, str]:
    scene_root = prepared_root / scene
    return {
        "rgb_path": str(scene_root / "color" / f"{frame_id}.jpg"),
        "depth_path": str(scene_root / "depth" / f"{frame_id}.png"),
        "pose_path": str(scene_root / "poses" / f"{frame_id}.txt"),
        "intrinsics_path": str(scene_root / "intrinsics.txt"),
    }


def _view_role(evidence: dict, frame_id: str) -> tuple[int, str]:
    matches = [
        row for row in evidence["distribution"].get("views", [])
        if str(row["frame_id"]) == str(frame_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f'{evidence["semantic_evidence_node_key"]}: frame {frame_id} has {len(matches)} Z1 matches'
        )
    return int(matches[0]["frame_index"]), str(matches[0]["view_role"])


def _validate_top3(views: list[dict], node_key: str) -> None:
    if len(views) > 3:
        raise ValueError(f"{node_key}: more than three registered views")
    ranks = [int(row["view_rank"]) for row in views]
    if ranks != list(range(1, len(views) + 1)):
        raise ValueError(f"{node_key}: non-contiguous view ranks")
    counts = [int(row["visible_point_count"]) for row in views]
    if counts != sorted(counts, reverse=True):
        raise ValueError(f"{node_key}: views are not sorted by visible point count")
    if len({str(row["frame_id"]) for row in views}) != len(views):
        raise ValueError(f"{node_key}: duplicate frame id")


def _load_contracts(args, scenes: set[str]):
    z2c_summary = json.loads((args.unified_ledger_root / "summary.json").read_text())
    if (
        z2c_summary.get("ground_truth_usage") != "none"
        or z2c_summary.get("candidate_mutation") is not False
        or int(z2c_summary.get("node_count", -1)) != int(args.expected_node_count)
    ):
        raise ValueError("Z2c unified semantic ledger contract is invalid")

    nodes = _read_jsonl(args.unified_ledger_root / "nodes.jsonl")
    nodes = [row for row in nodes if str(row["scene_name"]) in scenes]
    node_by_key = {str(row["semantic_evidence_node_key"]): row for row in nodes}
    if len(node_by_key) != len(nodes):
        raise ValueError("duplicate Z2c node key")

    evidence_rows = _read_jsonl(args.z1_root / "semantic_evidence_nodes.jsonl")
    evidence = {
        str(row["semantic_evidence_node_key"]): row
        for row in evidence_rows if str(row["scene_name"]) in scenes
    }
    if set(evidence) != set(node_by_key):
        raise ValueError("Z1 evidence and Z2c nodes do not have an exact join")

    track_binding = {}
    for row in _read_jsonl(args.z1_root / "candidate_bindings.jsonl"):
        if str(row["scene_name"]) not in scenes or str(row["candidate_source"]) != "track":
            continue
        key = (str(row["scene_name"]), int(row["candidate_id"]))
        if key in track_binding:
            raise ValueError(f"duplicate track binding: {key}")
        track_binding[key] = str(row["semantic_evidence_node_key"])
    return z2c_summary, node_by_key, evidence, track_binding


def _load_registered_views(args, scenes: list[str], track_binding: dict) -> dict[str, dict]:
    records = {}
    for scene in scenes:
        track_path = args.track_alpha_root / scene / "automatic_track_alphaclip_semantics.json"
        for row in json.loads(track_path.read_text()):
            key = track_binding.get((scene, int(row["track_id"])))
            if key is None:
                raise ValueError(f'{scene}: missing track binding {row["track_id"]}')
            if key in records:
                raise ValueError(f"duplicate registered view record: {key}")
            records[key] = row

        node_path = args.node_alpha_root / scene / "geometry_node_alphaclip_semantics.json"
        for row in json.loads(node_path.read_text()):
            key = str(row["semantic_evidence_node_key"])
            if key in records:
                raise ValueError(f"duplicate registered view record: {key}")
            records[key] = row
    return records


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    scene_set = set(scenes)
    z2c_summary, node_by_key, evidence, track_binding = _load_contracts(args, scene_set)
    registered = _load_registered_views(args, scenes, track_binding)
    if set(registered) != set(node_by_key):
        missing = sorted(set(node_by_key) - set(registered))[:5]
        extra = sorted(set(registered) - set(node_by_key))[:5]
        raise ValueError(f"registered view/node join mismatch; missing={missing}, extra={extra}")

    rows = []
    source_counts = Counter()
    source_with_views = Counter()
    source_view_counts = Counter()
    view_count_histogram = Counter()
    view_role_counts = Counter()
    missing_assets = Counter()
    scene_node_counts = Counter()
    scene_view_counts = Counter()
    for key in sorted(node_by_key, key=lambda value: (
        str(node_by_key[value]["scene_name"]), int(node_by_key[value]["node_index"])
    )):
        node = node_by_key[key]
        record = registered[key]
        scene = str(node["scene_name"])
        source = str(node["candidate_source"])
        if source not in SOURCES:
            raise ValueError(f"{key}: unknown source {source}")
        if str(record["scene_name"]) != scene:
            raise ValueError(f"{key}: registered view scene mismatch")
        if source != "track":
            if str(record["candidate_source"]) != source:
                raise ValueError(f"{key}: registered view source mismatch")
            if str(record["geometry_hash"]) != str(node["geometry_hash"]):
                raise ValueError(f"{key}: registered view geometry hash mismatch")
        if int(record["point_count"]) != int(node["point_count"]):
            raise ValueError(f"{key}: registered view point count mismatch")

        crop_contract = dict(record.get("crop_contract", {}))
        # The historical track exporter predates the explicit crop_mode field,
        # but this input root is the frozen limited-context (padding=0.50)
        # ledger. Normalize only that missing metadata; never alter the views.
        if source == "track" and "crop_mode" not in crop_contract:
            crop_contract["crop_mode"] = "limited_context"
        if crop_contract.get("crop_mode") != "limited_context" or float(
            crop_contract.get("crop_padding_ratio", -1)
        ) != 0.5:
            raise ValueError(f"{key}: unexpected crop contract")

        views = []
        for rank, view in enumerate(record.get("views", []), 1):
            frame_id = str(view["frame_id"])
            frame_index, role = _view_role(evidence[key], frame_id)
            paths = _asset_paths(args.prepared_dataset_root, scene, frame_id)
            availability = {name.replace("_path", "_exists"): Path(path).is_file() for name, path in paths.items()}
            for name, exists in availability.items():
                if not exists:
                    missing_assets[name] += 1
            bbox = [int(value) for value in view["bbox_xyxy"]]
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"{key}: invalid bbox for frame {frame_id}")
            views.append({
                "view_rank": rank,
                "frame_id": frame_id,
                "frame_index": frame_index,
                "visible_point_count": int(view["visible_points"]),
                "bbox_xyxy": bbox,
                "view_role": role,
                "selection_provenance": "frozen_z2_top3_visible_point_count",
                **paths,
                **availability,
            })
            view_role_counts[role] += 1
            scene_view_counts[scene] += 1
        _validate_top3(views, key)
        source_counts[source] += 1
        source_with_views[source] += int(bool(views))
        source_view_counts[source] += len(views)
        view_count_histogram[len(views)] += 1
        scene_node_counts[scene] += 1
        rows.append({
            "scene_name": scene,
            "node_index": int(node["node_index"]),
            "semantic_evidence_node_key": key,
            "semantic_evidence_node_id": int(node["semantic_evidence_node_id"]),
            "geometry_node_id": int(node["geometry_node_id"]),
            "candidate_source": source,
            "geometry_hash": str(node["geometry_hash"]),
            "point_count": int(node["point_count"]),
            "bound_candidate_count": int(node["bound_candidate_count"]),
            "representative_candidate_id": int(
                record.get("representative_candidate_id", record.get("track_id"))
            ),
            "geometry_reference": {
                "kind": {
                    "native": "native_cache_mask_column",
                    "track": "frozen_d2b_track_points_path",
                    "pair_union": "frozen_pair_union_points_path",
                }[source],
                "candidate_id": int(
                    record.get("representative_candidate_id", record.get("track_id"))
                ),
            },
            "selected_track_semantic_evidence_node_key": node.get(
                "selected_track_semantic_evidence_node_key"
            ),
            "view_count": len(views),
            "view_availability": "available" if views else "no_z2_view_meeting_min_visible_points",
            "crop_contract": crop_contract,
            "views": views,
            "ground_truth_usage": "none",
            "embedding_generated": False,
            "mllm_invoked": False,
            "candidate_mutation": False,
            "class_mutation": False,
            "score_mutation": False,
            "inference_plan_written": False,
        })

    if len(rows) != int(args.expected_node_count) or len(scene_node_counts) != len(scenes):
        raise ValueError("official100 node/scene coverage is incomplete")
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.tmp.{os.getpid()}"
    staging.mkdir(parents=True)
    try:
        with (staging / "object_view_manifest.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        with (staging / "coverage_by_scene.jsonl").open("w") as handle:
            for scene in scenes:
                handle.write(json.dumps({
                    "scene_name": scene,
                    "node_count": int(scene_node_counts[scene]),
                    "selected_view_count": int(scene_view_counts[scene]),
                }, ensure_ascii=False, sort_keys=True) + "\n")
        manifest_path = staging / "object_view_manifest.jsonl"
        coverage_path = staging / "coverage_by_scene.jsonl"
        summary = {
            "version": VERSION,
            "diagnostic_type": "Z6b GT-free frozen top-3 object-view manifest and prepared-asset audit",
            "scene_count": len(scenes),
            "node_count": len(rows),
            "selected_view_count": sum(row["view_count"] for row in rows),
            "node_with_view_count": sum(bool(row["view_count"]) for row in rows),
            "node_without_view_count": sum(not row["view_count"] for row in rows),
            "source_node_counts": dict(source_counts),
            "source_node_with_view_counts": dict(source_with_views),
            "source_node_without_view_counts": {
                source: int(source_counts[source] - source_with_views[source]) for source in SOURCES
            },
            "source_selected_view_counts": dict(source_view_counts),
            "node_view_count_histogram": {
                str(count): int(view_count_histogram[count]) for count in sorted(view_count_histogram)
            },
            "view_role_counts": dict(view_role_counts),
            "missing_asset_counts": dict(missing_assets),
            "all_registered_assets_present": not missing_assets,
            "join_audit": {
                "z2c_node_count": int(z2c_summary["node_count"]),
                "registered_view_record_count": len(registered),
                "joined_node_count": len(rows),
                "duplicate_node_count": 0,
                "valid": True,
            },
            "selection_contract": {
                "top_views": 3,
                "ranking": "descending visible point count, then frozen Z2 frame order",
                "min_visible_points": 20,
                "crop_mode": "limited_context",
                "crop_padding_ratio": 0.5,
                "provenance": "reuse exact Z2/Z2b Alpha-CLIP view selection; no reprojection or reselection",
            },
            "ground_truth_usage": "none",
            "embedding_generated": False,
            "mllm_invoked": False,
            "candidate_mutation": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "score_mutation": False,
            "inference_plan_written": False,
            "safety60_read": bool(args.safety60_transfer),
            "even48_read": False,
            "test60_read": False,
            "output_sha256": {
                "object_view_manifest.jsonl": _sha256(manifest_path),
                "coverage_by_scene.jsonl": _sha256(coverage_path),
            },
            "params": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items()
            },
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list", type=Path,
        default=Path("output/scannet200/scene_splits/official_train100_20260808/official_train100.txt"),
    )
    parser.add_argument(
        "--unified-ledger-root", type=Path,
        default=Path("docs/diagnostics/z2c_unified_semantic_node_ledger_official100_20260811"),
    )
    parser.add_argument(
        "--z1-root", type=Path,
        default=Path("docs/diagnostics/z1_yoloworld_multiview_distribution_official100_20260811_v3_frozen_support_vote"),
    )
    parser.add_argument(
        "--track-alpha-root", type=Path,
        default=Path("docs/diagnostics/z2_alphaclip_track_limited_context_official100_20260811"),
    )
    parser.add_argument(
        "--node-alpha-root", type=Path,
        default=Path("docs/diagnostics/z2b_native_union_alphaclip_limited_context_official100_20260811"),
    )
    parser.add_argument(
        "--prepared-dataset-root", type=Path,
        default=Path("/media/jia/软件1/scannet_train_stream/prepared"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("docs/diagnostics/z6b_object_view_manifest_official100_20260812"),
    )
    parser.add_argument("--expected-node-count", type=int, default=EXPECTED_NODE_COUNT)
    parser.add_argument("--safety60-transfer", action="store_true")
    args = parser.parse_args()
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path):
            setattr(args, name, _resolve(value))
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
