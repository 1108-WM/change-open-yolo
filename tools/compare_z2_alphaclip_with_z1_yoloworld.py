#!/usr/bin/env python3
"""Compare Z1 YOLO-World and Z2 Alpha-CLIP track distributions without GT."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLE_NAMES = ("frozen_support", "sampled_support", "independent_review", "all_view")
ALPHA_NAMES = ("object_center", "limited_context")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _dense(rows: list[dict], class_count: int) -> np.ndarray | None:
    if not rows:
        return None
    result = np.zeros(class_count, dtype=np.float64)
    for row in rows:
        class_index = int(row["class_index"])
        if not 0 <= class_index < class_count:
            raise ValueError(f"class index {class_index} is outside 0..{class_count - 1}")
        result[class_index] = float(row["probability"])
    total = float(result.sum())
    return result / total if total > 0 else None


def _softmax(values: list[float]) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _alpha_distribution(record: dict, class_count: int) -> np.ndarray | None:
    views = record.get("views", [])
    if int(record.get("alphaclip_class_index", -1)) < 0 or not views:
        return None
    rows = []
    for view in views:
        logits = view.get("clip_logits", [])
        if len(logits) != class_count:
            raise ValueError(
                f"{record['scene_name']}:{record['track_id']}: Alpha view logits have {len(logits)} classes"
            )
        rows.append(_softmax(logits))
    result = np.stack(rows, axis=0).mean(axis=0)
    return result / result.sum()


def _metrics(distribution: np.ndarray | None, preferred_top1: int | None = None) -> dict:
    if distribution is None:
        return {"valid": False, "top1_class_index": -1, "top1_probability": 0.0, "margin": 0.0, "entropy": 0.0}
    order = np.argsort(-distribution, kind="stable")
    top1 = int(order[0]) if preferred_top1 is None else int(preferred_top1)
    if not 0 <= top1 < len(distribution) or not np.isclose(
        distribution[top1], distribution.max(), rtol=0.0, atol=2e-6
    ):
        top1 = int(order[0])
    second = float(distribution[order[1]]) if len(order) > 1 else 0.0
    positive = distribution[distribution > 0]
    entropy = float(-(positive * np.log(positive)).sum() / math.log(len(distribution)))
    return {
        "valid": True,
        "top1_class_index": top1,
        "top1_probability": float(distribution[top1]),
        "margin": float(distribution[top1] - second),
        "entropy": entropy,
    }


def _js(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def kl(first, second):
        keep = first > 0
        return float(np.sum(first[keep] * np.log(first[keep] / second[keep])))

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _load_z1(root: Path) -> tuple[dict[tuple[str, int], dict], dict[str, dict]]:
    evidence = {}
    with (root / "semantic_evidence_nodes.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                key = str(row["semantic_evidence_node_key"])
                if key in evidence:
                    raise ValueError(f"duplicate Z1 evidence key: {key}")
                evidence[key] = row
    bindings = {}
    with (root / "candidate_bindings.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["candidate_source"]) != "track":
                continue
            key = (str(row["scene_name"]), int(row["candidate_id"]))
            if key in bindings:
                raise ValueError(f"duplicate Z1 track binding: {key}")
            bindings[key] = row
    return bindings, evidence


def _load_alpha(root: Path, scenes: list[str]) -> dict[tuple[str, int], dict]:
    result = {}
    for scene in scenes:
        rows = json.loads((root / scene / "automatic_track_alphaclip_semantics.json").read_text())
        for row in rows:
            key = (str(row["scene_name"]), int(row["track_id"]))
            if key in result:
                raise ValueError(f"duplicate Alpha-CLIP key: {key}")
            result[key] = row
    return result


def _z1_distributions(evidence: dict, class_count: int) -> dict[str, np.ndarray | None]:
    distribution = evidence["distribution"]
    roles = distribution["role_distributions"]
    frozen = evidence.get("frozen_support_vote")
    return {
        "frozen_support": _dense(frozen["distribution"], class_count) if frozen else None,
        "sampled_support": _dense(roles["track_support_view"]["distribution"], class_count),
        "independent_review": _dense(roles["independent_review_view"]["distribution"], class_count),
        "all_view": _dense(distribution["distribution"], class_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--z1_root", type=Path, required=True)
    parser.add_argument("--object_center_root", type=Path, required=True)
    parser.add_argument("--limited_context_root", type=Path, required=True)
    parser.add_argument("--class_count", type=int, default=198)
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "z1_root", "object_center_root", "limited_context_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output directory exists and is non-empty: {args.output_root}")

    scenes = _read_scenes(args.scene_list)
    bindings, evidence = _load_z1(args.z1_root)
    alpha_records = {
        "object_center": _load_alpha(args.object_center_root, scenes),
        "limited_context": _load_alpha(args.limited_context_root, scenes),
    }
    expected_keys = set(bindings)
    for name, records in alpha_records.items():
        if set(records) != expected_keys:
            raise ValueError(
                f"{name}: Alpha key set differs from Z1 tracks; missing={len(expected_keys - set(records))}, "
                f"extra={len(set(records) - expected_keys)}"
            )

    pair_names = []
    sources = list(ROLE_NAMES) + list(ALPHA_NAMES)
    for left_index, left in enumerate(sources):
        for right in sources[left_index + 1 :]:
            pair_names.append((left, right))
    agreement = {f"{left}__{right}": [0, 0] for left, right in pair_names}
    divergences = {f"{left}__{right}": [] for left, right in pair_names}
    source_metrics = {name: {"probability": [], "margin": [], "entropy": []} for name in sources}
    rows = []
    for key in sorted(expected_keys):
        binding = bindings[key]
        erow = evidence[str(binding["semantic_evidence_node_key"])]
        distributions = _z1_distributions(erow, args.class_count)
        for name in ALPHA_NAMES:
            distributions[name] = _alpha_distribution(alpha_records[name][key], args.class_count)
        metrics = {}
        for name, distribution in distributions.items():
            preferred = None
            if name in ALPHA_NAMES:
                preferred = int(alpha_records[name][key]["alphaclip_class_index"])
            metrics[name] = _metrics(distribution, preferred)
            if metrics[name]["valid"]:
                source_metrics[name]["probability"].append(metrics[name]["top1_probability"])
                source_metrics[name]["margin"].append(metrics[name]["margin"])
                source_metrics[name]["entropy"].append(metrics[name]["entropy"])
        pair_row = {}
        for left, right in pair_names:
            name = f"{left}__{right}"
            if distributions[left] is None or distributions[right] is None:
                pair_row[name] = {"valid": False, "top1_agree": False, "js_divergence": None}
                continue
            same = metrics[left]["top1_class_index"] == metrics[right]["top1_class_index"]
            js = _js(distributions[left], distributions[right])
            agreement[name][1] += 1
            agreement[name][0] += int(same)
            divergences[name].append(js)
            pair_row[name] = {"valid": True, "top1_agree": same, "js_divergence": js}
        rows.append({
            "scene_name": key[0],
            "track_id": key[1],
            "semantic_evidence_node_key": str(binding["semantic_evidence_node_key"]),
            "source_metrics": metrics,
            "pair_metrics": pair_row,
        })

    summary = {
        "comparison_contract": "no GT; fixed tracks; complete 198-class distributions",
        "scene_count": len(scenes),
        "track_count": len(rows),
        "source_statistics": {
            name: {metric: _stats(values) for metric, values in metrics.items()}
            for name, metrics in source_metrics.items()
        },
        "pair_statistics": {
            name: {
                "valid_pair_count": counts[1],
                "top1_agreement_count": counts[0],
                "top1_agreement_percent": 100.0 * counts[0] / counts[1] if counts[1] else None,
                "js_divergence": _stats(divergences[name]),
            }
            for name, counts in agreement.items()
        },
    }
    args.output_root.mkdir(parents=True)
    with (args.output_root / "track_comparison.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
