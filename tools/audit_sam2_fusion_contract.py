#!/usr/bin/env python3
"""审计 SAM2 融合链路的无 GT 接口契约。"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _parse_roots(values):
    roots = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"Candidate root must use name=path, got: {value}")
        roots[name.strip()] = Path(path).expanduser()
    return roots


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _issue(rows, severity, category, scene_name, detail):
    rows.append(
        {
            "severity": severity,
            "category": category,
            "scene_name": scene_name,
            "detail": detail,
        }
    )


def _audit_superpoints(scenes, original_root, ibsp_root, rows):
    changed_scenes = 0
    point_counts = {}
    for scene_name in scenes:
        scene_id = scene_name.removeprefix("scene")
        original_path = original_root / scene_name / f"{scene_id}.npy"
        ibsp_path = ibsp_root / scene_name / f"{scene_id}.npy"
        if not original_path.is_file() or not ibsp_path.is_file():
            _issue(rows, "error", "missing_superpoint_array", scene_name, f"{original_path} | {ibsp_path}")
            continue
        original = np.load(original_path, mmap_mode="r")
        ibsp = np.load(ibsp_path, mmap_mode="r")
        point_counts[scene_name] = int(len(original))
        if original.shape != ibsp.shape or original.ndim != 2 or original.shape[1] < 10:
            _issue(rows, "error", "superpoint_shape", scene_name, f"original={original.shape}, ibsp={ibsp.shape}")
            continue
        if not np.array_equal(original[:, :9], ibsp[:, :9]) or not np.array_equal(original[:, 10:], ibsp[:, 10:]):
            _issue(rows, "error", "non_superpoint_column_changed", scene_name, "f30 IBSp changed columns other than 9")
        if np.any(ibsp[:, 9] < 0):
            _issue(rows, "error", "negative_superpoint_id", scene_name, "f30 IBSp contains a negative id")
        changed_scenes += int(np.any(original[:, 9] != ibsp[:, 9]))
    return point_counts, changed_scenes


def _audit_candidate_root(name, root, scenes, point_counts, labels, rows):
    source_kinds = Counter()
    candidate_count = 0
    for scene_name in scenes:
        path = root / scene_name / "backprojection_candidates.json"
        if not path.is_file():
            _issue(rows, "error", "missing_candidate_json", scene_name, f"{name}: {path}")
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            _issue(rows, "error", "invalid_candidate_json", scene_name, f"{name}: {exc}")
            continue
        if payload.get("scene_name") not in (None, scene_name):
            _issue(rows, "error", "candidate_scene_name", scene_name, f"{name}: {payload.get('scene_name')}")
        for candidate in payload.get("candidates", []):
            candidate_count += 1
            source_kinds[str(candidate.get("source_kind", "unknown"))] += 1
            for field in ("candidate_id", "class_id", "class_name", "score", "seed_points_path", "num_seed_points"):
                if field not in candidate:
                    _issue(rows, "error", "missing_candidate_field", scene_name, f"{name}/{candidate.get('candidate_id')}: {field}")
            class_id = candidate.get("class_id")
            if not isinstance(class_id, int) or not 0 <= class_id < len(labels):
                _issue(rows, "error", "invalid_class_id", scene_name, f"{name}/{candidate.get('candidate_id')}: {class_id}")
            elif candidate.get("class_name") != labels[class_id]:
                _issue(
                    rows,
                    "error",
                    "class_name_id_mismatch",
                    scene_name,
                    f"{name}/{candidate.get('candidate_id')}: {candidate.get('class_name')} != {labels[class_id]}",
                )
            score = candidate.get("score")
            if not isinstance(score, (int, float)) or not np.isfinite(score) or not 0.0 <= score <= 1.0:
                _issue(rows, "error", "invalid_candidate_score", scene_name, f"{name}/{candidate.get('candidate_id')}: {score}")
            seed_path = candidate.get("seed_points_path")
            if seed_path is None:
                continue
            seed_path = _resolve(seed_path)
            if not seed_path.is_file():
                _issue(rows, "error", "missing_seed_points", scene_name, f"{name}/{candidate.get('candidate_id')}: {seed_path}")
                continue
            try:
                indices = np.load(seed_path)["point_indices"].astype(np.int64)
            except (KeyError, OSError, ValueError) as exc:
                _issue(rows, "error", "invalid_seed_points", scene_name, f"{name}/{candidate.get('candidate_id')}: {exc}")
                continue
            unique = np.unique(indices)
            point_count = point_counts.get(scene_name)
            if point_count is not None and (np.any(unique < 0) or np.any(unique >= point_count)):
                _issue(rows, "error", "seed_index_out_of_range", scene_name, f"{name}/{candidate.get('candidate_id')}")
            if int(candidate.get("num_seed_points", -1)) != len(unique):
                _issue(
                    rows,
                    "warning",
                    "seed_count_metadata_mismatch",
                    scene_name,
                    f"{name}/{candidate.get('candidate_id')}: metadata={candidate.get('num_seed_points')}, unique={len(unique)}",
                )
    return {"candidates": int(candidate_count), "source_kinds": dict(sorted(source_kinds.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--original_superpoint_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--ibsp_superpoint_root", type=Path, required=True)
    parser.add_argument("--candidate_root", action="append", required=True, help="Repeat name=path for every candidate source.")
    parser.add_argument("--config", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--fusion_eval_script", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    scenes = _read_scenes(args.scene_list)
    config = yaml.safe_load(args.config.read_text())
    labels = config["network2d"]["text_prompts"]
    roots = _parse_roots(args.candidate_root)
    rows = []
    point_counts, changed_scenes = _audit_superpoints(
        scenes, args.original_superpoint_root, args.ibsp_superpoint_root, rows
    )
    root_summaries = {
        name: _audit_candidate_root(name, root, scenes, point_counts, labels, rows)
        for name, root in roots.items()
    }
    script_text = args.fusion_eval_script.read_text()
    if '--processed_scene_root "$SUPERPOINT_ROOT"' not in script_text:
        _issue(rows, "error", "fusion_superpoint_root_not_forwarded", "", str(args.fusion_eval_script))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "issues.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("severity", "category", "scene_name", "detail"))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "gt_usage": "none",
        "scene_count": len(scenes),
        "f30_changed_scene_count": int(changed_scenes),
        "point_count_range": [min(point_counts.values(), default=0), max(point_counts.values(), default=0)],
        "candidate_roots": root_summaries,
        "issue_counts": dict(sorted(Counter(row["severity"] for row in rows).items())),
        "issues_path": str(args.output_dir / "issues.csv"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
