#!/usr/bin/env python3
"""Build a GT-free unified YOLO-World/Alpha-CLIP geometry-node ledger."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _dense(rows: list[dict], class_count: int) -> np.ndarray:
    result = np.zeros(class_count, dtype=np.float32)
    for row in rows:
        index = int(row["class_index"])
        if not 0 <= index < class_count:
            raise ValueError(f"class index outside [0,{class_count}): {index}")
        result[index] = float(row["probability"])
    total = float(result.sum())
    return result / total if total > 0 else result


def _softmax(logits: list[float], class_count: int) -> np.ndarray:
    if len(logits) != class_count:
        raise ValueError(f"expected {class_count} logits, got {len(logits)}")
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max()
    values = np.exp(values)
    return (values / values.sum()).astype(np.float32)


def _alpha_distribution(row: dict, class_count: int) -> tuple[np.ndarray, bool]:
    views = row.get("views", [])
    if int(row.get("alphaclip_class_index", -1)) < 0 or not views:
        return np.zeros(class_count, dtype=np.float32), False
    probabilities = [_softmax(view["clip_logits"], class_count) for view in views]
    result = np.stack(probabilities).mean(axis=0)
    return (result / result.sum()).astype(np.float32), True


def _entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0]
    if not len(positive):
        return 0.0
    return float(-(positive * np.log(positive)).sum() / math.log(len(distribution)))


def _js(left: np.ndarray, right: np.ndarray) -> float:
    middle = 0.5 * (left + right)
    left_mask, right_mask = left > 0, right > 0
    value = 0.5 * float((left[left_mask] * np.log(left[left_mask] / middle[left_mask])).sum())
    value += 0.5 * float((right[right_mask] * np.log(right[right_mask] / middle[right_mask])).sum())
    return value / math.log(2.0)


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(array.mean()),
        "median": float(np.median(array)), "p90": float(np.quantile(array, 0.9)),
    }


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    scene_set = set(scenes)
    bindings = _read_jsonl(args.z1_root / "candidate_bindings.jsonl")
    evidence_rows = _read_jsonl(args.z1_root / "semantic_evidence_nodes.jsonl")
    evidence = {str(row["semantic_evidence_node_key"]): row for row in evidence_rows}
    if len(evidence) != len(evidence_rows):
        raise ValueError("duplicate Z1 semantic evidence node key")

    source_by_key: dict[str, str] = {}
    binding_count = Counter()
    selected_track_key: dict[str, str | None] = {}
    track_binding_key: dict[tuple[str, int], str] = {}
    for row in bindings:
        if str(row["scene_name"]) not in scene_set:
            continue
        key = str(row["semantic_evidence_node_key"])
        source = str(row["candidate_source"])
        previous = source_by_key.setdefault(key, source)
        if previous != source:
            raise ValueError(f"{key}: evidence node is shared across sources")
        binding_count[key] += 1
        selected = row.get("selected_track_semantic_evidence_node_key")
        if key in selected_track_key and selected_track_key[key] != selected:
            raise ValueError(f"{key}: inconsistent selected-track evidence key")
        selected_track_key[key] = str(selected) if selected is not None else None
        if source == "track":
            track_identity = (str(row["scene_name"]), int(row["candidate_id"]))
            if track_identity in track_binding_key:
                raise ValueError(f"duplicate track binding: {track_identity}")
            track_binding_key[track_identity] = key

    expected_keys = {key for key, row in evidence.items() if str(row["scene_name"]) in scene_set}
    if set(source_by_key) != expected_keys:
        raise ValueError("Z1 bindings do not exactly cover evidence nodes")

    track_alpha: dict[str, tuple[np.ndarray, bool, int]] = {}
    for scene in scenes:
        path = args.track_alpha_root / scene / "automatic_track_alphaclip_semantics.json"
        for row in json.loads(path.read_text()):
            track_id = int(row["track_id"])
            key = track_binding_key.get((scene, track_id))
            if key is None:
                raise ValueError(f"{scene}:track:{track_id}: missing Z1 binding")
            if key in track_alpha:
                raise ValueError(f"duplicate track Alpha key: {key}")
            distribution, available = _alpha_distribution(row, args.class_count)
            track_alpha[key] = (distribution, available, len(row.get("views", [])))

    node_alpha: dict[str, tuple[np.ndarray, bool, int]] = {}
    for scene in scenes:
        path = args.node_alpha_root / scene / "geometry_node_alphaclip_semantics.json"
        for row in json.loads(path.read_text()):
            key = str(row["semantic_evidence_node_key"])
            if key in node_alpha:
                raise ValueError(f"duplicate geometry Alpha key: {key}")
            distribution, available = _alpha_distribution(row, args.class_count)
            node_alpha[key] = (distribution, available, len(row.get("views", [])))

    expected_track = {key for key, source in source_by_key.items() if source == "track"}
    expected_node = {key for key, source in source_by_key.items() if source in {"native", "pair_union"}}
    if set(track_alpha) != expected_track or set(node_alpha) != expected_node:
        raise ValueError("Alpha ledgers do not exactly cover their Z1 evidence-node contracts")

    ordered = sorted(expected_keys, key=lambda key: (
        str(evidence[key]["scene_name"]), int(evidence[key]["semantic_evidence_node_id"])
    ))
    arrays = {
        name: np.zeros((len(ordered), args.class_count), dtype=np.float32)
        for name in ("geometry_yolo", "inherited_yolo", "geometry_alpha", "inherited_alpha")
    }
    metadata, diagnostics = [], defaultdict(lambda: defaultdict(list))
    availability_counts = Counter()
    source_counts = Counter()
    for index, key in enumerate(ordered):
        erow = evidence[key]
        source = source_by_key[key]
        source_counts[source] += 1
        geometry_yolo = _dense(erow["distribution"]["distribution"], args.class_count)
        inherited_key = selected_track_key.get(key) if source == "pair_union" else key
        inherited_erow = evidence[inherited_key] if inherited_key is not None else erow
        inherited_rows = (
            inherited_erow["frozen_support_vote"]["distribution"]
            if source in {"track", "pair_union"}
            else inherited_erow["distribution"]["distribution"]
        )
        inherited_yolo = _dense(inherited_rows, args.class_count)
        if source == "track":
            geometry_alpha, geometry_alpha_available, alpha_views = track_alpha[key]
        else:
            geometry_alpha, geometry_alpha_available, alpha_views = node_alpha[key]
        if source == "pair_union":
            inherited_alpha, inherited_alpha_available, inherited_views = track_alpha[inherited_key]
        else:
            inherited_alpha, inherited_alpha_available, inherited_views = (
                geometry_alpha, geometry_alpha_available, alpha_views
            )
        arrays["geometry_yolo"][index] = geometry_yolo
        arrays["inherited_yolo"][index] = inherited_yolo
        arrays["geometry_alpha"][index] = geometry_alpha
        arrays["inherited_alpha"][index] = inherited_alpha
        availability_counts[f"{source}:geometry_alpha"] += int(geometry_alpha_available)
        availability_counts[f"{source}:inherited_alpha"] += int(inherited_alpha_available)
        geometry_agreement = None
        geometry_js = None
        if geometry_alpha_available and geometry_yolo.sum() > 0:
            geometry_agreement = int(np.argmax(geometry_yolo) == np.argmax(geometry_alpha))
            geometry_js = _js(geometry_yolo, geometry_alpha)
            diagnostics[source]["geometry_yolo_alpha_agreement"].append(geometry_agreement)
            diagnostics[source]["geometry_yolo_alpha_js"].append(geometry_js)
        inherited_agreement = None
        inherited_js = None
        if inherited_alpha_available and inherited_yolo.sum() > 0:
            inherited_agreement = int(np.argmax(inherited_yolo) == np.argmax(inherited_alpha))
            inherited_js = _js(inherited_yolo, inherited_alpha)
            diagnostics[source]["inherited_yolo_alpha_agreement"].append(inherited_agreement)
            diagnostics[source]["inherited_yolo_alpha_js"].append(inherited_js)
        own_inherited_js = _js(geometry_alpha, inherited_alpha) if (
            geometry_alpha_available and inherited_alpha_available
        ) else None
        if own_inherited_js is not None:
            diagnostics[source]["geometry_inherited_alpha_js"].append(own_inherited_js)
        metadata.append({
            "node_index": index, "scene_name": str(erow["scene_name"]),
            "semantic_evidence_node_key": key,
            "semantic_evidence_node_id": int(erow["semantic_evidence_node_id"]),
            "geometry_node_id": int(erow["geometry_node_id"]), "candidate_source": source,
            "geometry_hash": str(erow["geometry_hash"]),
            "bound_candidate_count": int(binding_count[key]),
            "selected_track_semantic_evidence_node_key": inherited_key if source == "pair_union" else None,
            "point_count": int(erow["distribution"]["candidate_point_count"]),
            "geometry_yolo_available": bool(geometry_yolo.sum() > 0),
            "inherited_yolo_available": bool(inherited_yolo.sum() > 0),
            "geometry_alpha_available": geometry_alpha_available,
            "inherited_alpha_available": inherited_alpha_available,
            "geometry_alpha_view_count": alpha_views,
            "inherited_alpha_view_count": inherited_views,
            "geometry_yolo_entropy": _entropy(geometry_yolo),
            "inherited_yolo_entropy": _entropy(inherited_yolo),
            "geometry_alpha_entropy": _entropy(geometry_alpha) if geometry_alpha_available else None,
            "inherited_alpha_entropy": _entropy(inherited_alpha) if inherited_alpha_available else None,
            "geometry_yolo_alpha_top1_agreement": geometry_agreement,
            "inherited_yolo_alpha_top1_agreement": inherited_agreement,
            "geometry_yolo_alpha_js": geometry_js,
            "inherited_yolo_alpha_js": inherited_js,
            "geometry_inherited_alpha_js": own_inherited_js,
        })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "semantic_distributions.npz", **arrays)
    with (args.output_dir / "nodes.jsonl").open("w") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    diagnostic_summary = {
        source: {name: _summary(values) for name, values in fields.items()}
        for source, fields in diagnostics.items()
    }
    summary = {
        "diagnostic_type": "Z2 unified GT-free geometry-node semantic ledger",
        "ground_truth_usage": "none", "candidate_mutation": False,
        "class_count": args.class_count, "scene_count": len(scenes),
        "node_count": len(ordered), "source_node_counts": dict(sorted(source_counts.items())),
        "alpha_availability_counts": dict(sorted(availability_counts.items())),
        "distribution_arrays": list(arrays), "diagnostics": diagnostic_summary,
        "contracts": {
            "native": "own geometry YOLO and own limited-context Alpha",
            "track": "all-frame geometry YOLO plus frozen-support YOLO and track limited-context Alpha",
            "pair_union": "own geometry YOLO/Alpha plus selected-track frozen-support YOLO/Alpha",
            "missing_alpha": "zero distribution plus explicit availability=false",
        },
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--z1-root", type=Path, required=True)
    parser.add_argument("--track-alpha-root", type=Path, required=True)
    parser.add_argument("--node-alpha-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-count", type=int, default=198)
    args = parser.parse_args()
    for name in ("scene_list", "z1_root", "track_alpha_root", "node_alpha_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
