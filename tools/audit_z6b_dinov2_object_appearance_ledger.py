#!/usr/bin/env python3
"""Independently audit the frozen official100 Z6b DINOv2 appearance ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MANIFEST_SHA256 = "503261f316a0e9e642eb09c63c87d1fd10c9be60b3bd71041d3dfa8fb149107b"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _close(left: float, right: float, tolerance: float) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def run(args: argparse.Namespace) -> dict:
    manifest_path = args.manifest_root / "object_view_manifest.jsonl"
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256 and not args.allow_nonofficial_manifest:
        raise ValueError(f"manifest SHA-256 mismatch: {manifest_sha}")
    manifest = _read_jsonl(manifest_path)
    manifest_by_scene = {}
    for row in manifest:
        manifest_by_scene.setdefault(str(row["scene_name"]), []).append(row)

    root_summary = json.loads((args.ledger_root / "summary.json").read_text())
    if (
        root_summary.get("manifest_sha256") != manifest_sha
        or root_summary.get("ground_truth_usage") != "none"
        or root_summary.get("candidate_mutation") is not False
        or root_summary.get("geometry_mutation") is not False
        or root_summary.get("class_mutation") is not False
        or root_summary.get("score_mutation") is not False
        or root_summary.get("inference_plan_written") is not False
    ):
        raise ValueError("root DINOv2 ledger contract is invalid")

    errors = []
    counts = Counter()
    norms = []
    pairwise_means = []
    dispersions = []
    source_with_embedding = Counter()
    for scene, expected_rows in sorted(manifest_by_scene.items()):
        expected_rows = sorted(expected_rows, key=lambda row: int(row["node_index"]))
        scene_root = args.ledger_root / scene
        ledger = _read_jsonl(scene_root / "dinov2_object_appearance_ledger.jsonl")
        with np.load(scene_root / "dinov2_node_embeddings.npz") as payload:
            if set(payload.files) != {"mean_medoid", "per_view"}:
                errors.append(f"{scene}: unexpected embedding arrays {payload.files}")
                continue
            mean_medoid = np.asarray(payload["mean_medoid"], dtype=np.float32)
            per_view = np.asarray(payload["per_view"], dtype=np.float32)
        if len(ledger) != len(expected_rows):
            errors.append(f"{scene}: ledger count {len(ledger)} != manifest {len(expected_rows)}")
            continue
        if mean_medoid.shape != (len(ledger), 768) or per_view.shape != (len(ledger), 3, 384):
            errors.append(
                f"{scene}: shapes mean_medoid={mean_medoid.shape} per_view={per_view.shape}"
            )
            continue
        for local_index, (expected, actual) in enumerate(zip(expected_rows, ledger)):
            key = str(expected["semantic_evidence_node_key"])
            counts["node_count"] += 1
            if (
                str(actual["semantic_evidence_node_key"]) != key
                or int(actual["node_index"]) != int(expected["node_index"])
                or str(actual["geometry_hash"]) != str(expected["geometry_hash"])
                or str(actual["candidate_source"]) != str(expected["candidate_source"])
                or int(actual["view_count"]) != int(expected["view_count"])
            ):
                errors.append(f"{key}: manifest/ledger metadata mismatch")
                continue
            view_count = int(expected["view_count"])
            valid_views = per_view[local_index, :view_count]
            missing_views = per_view[local_index, view_count:]
            has_embedding = actual["embedding_state"] == "available"
            if has_embedding != bool(view_count):
                errors.append(f"{key}: embedding state/view count mismatch")
                continue
            if view_count == 0:
                counts["node_without_embedding_count"] += 1
                if not np.isnan(mean_medoid[local_index]).all() or not np.isnan(per_view[local_index]).all():
                    errors.append(f"{key}: no-view row must be all NaN")
                continue
            counts["node_with_embedding_count"] += 1
            counts["encoded_view_count"] += view_count
            source_with_embedding[str(expected["candidate_source"])] += 1
            if not np.isfinite(valid_views).all() or not np.isnan(missing_views).all():
                errors.append(f"{key}: per-view finite/NaN layout mismatch")
                continue
            view_norms = np.linalg.norm(valid_views, axis=1)
            norms.extend(float(value) for value in view_norms)
            if np.max(np.abs(view_norms - 1.0)) > args.tolerance:
                errors.append(f"{key}: per-view L2 normalization failed")
                continue
            mean = mean_medoid[local_index, :384]
            medoid = mean_medoid[local_index, 384:]
            if not np.isfinite(mean).all() or not np.isfinite(medoid).all():
                errors.append(f"{key}: node embedding is non-finite")
                continue
            if max(abs(float(np.linalg.norm(mean)) - 1.0), abs(float(np.linalg.norm(medoid)) - 1.0)) > args.tolerance:
                errors.append(f"{key}: mean/medoid L2 normalization failed")
                continue
            recomputed_mean = valid_views.mean(axis=0)
            recomputed_mean /= max(float(np.linalg.norm(recomputed_mean)), 1e-12)
            if np.max(np.abs(recomputed_mean - mean)) > args.tolerance:
                errors.append(f"{key}: mean embedding mismatch")
                continue
            normalized_views = valid_views / np.maximum(
                np.linalg.norm(valid_views, axis=1, keepdims=True), 1e-12
            )
            similarity = normalized_views @ normalized_views.T
            if view_count == 1:
                medoid_index = 0
                pairwise = np.asarray([], dtype=np.float32)
            else:
                medoid_index = int(np.argmax((similarity.sum(axis=1) - 1.0) / (view_count - 1)))
                pairwise = similarity[np.triu_indices(view_count, k=1)]
            if int(actual["medoid_view_index"]) != medoid_index:
                errors.append(f"{key}: medoid index mismatch")
                continue
            if np.max(np.abs(valid_views[medoid_index] - medoid)) > args.tolerance:
                errors.append(f"{key}: medoid embedding mismatch")
                continue
            expected_mean = float(pairwise.mean()) if len(pairwise) else 1.0
            expected_min = float(pairwise.min()) if len(pairwise) else 1.0
            expected_std = float(pairwise.std()) if len(pairwise) else 0.0
            expected_dispersion = float(1.0 - expected_mean) if len(pairwise) else 0.0
            for name, expected_value in (
                ("pairwise_cosine_mean", expected_mean),
                ("pairwise_cosine_min", expected_min),
                ("pairwise_cosine_std", expected_std),
                ("dispersion_one_minus_pairwise_mean", expected_dispersion),
            ):
                if not _close(actual[name], expected_value, args.tolerance):
                    errors.append(f"{key}: {name} mismatch")
                    break
            pairwise_means.append(expected_mean)
            dispersions.append(expected_dispersion)

    expected_counts = {
        "node_count": int(root_summary["node_count"]),
        "node_with_embedding_count": int(root_summary["node_with_embedding_count"]),
        "node_without_embedding_count": int(root_summary["node_without_embedding_count"]),
        "encoded_view_count": int(root_summary["encoded_view_count"]),
    }
    actual_counts = {name: int(counts[name]) for name in expected_counts}
    if actual_counts != expected_counts:
        errors.append(f"root counts mismatch: actual={actual_counts} expected={expected_counts}")

    def stats(values: list[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": float(array.mean()) if len(array) else None,
            "median": float(np.median(array)) if len(array) else None,
            "p10": float(np.quantile(array, 0.1)) if len(array) else None,
            "p90": float(np.quantile(array, 0.9)) if len(array) else None,
            "min": float(array.min()) if len(array) else None,
            "max": float(array.max()) if len(array) else None,
        }

    summary = {
        "diagnostic_type": "independent audit of Z6b DINOv2 object appearance ledger",
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors[:100],
        "manifest_sha256": manifest_sha,
        "scene_count": len(manifest_by_scene),
        **actual_counts,
        "source_node_with_embedding_counts": dict(source_with_embedding),
        "per_view_l2_norm": stats(norms),
        "pairwise_cosine_mean": stats(pairwise_means),
        "dispersion_one_minus_pairwise_mean": stats(dispersions),
        "ground_truth_usage": "none",
        "candidate_mutation": False,
        "class_mutation": False,
        "score_mutation": False,
        "inference_plan_written": False,
        "safety60_read": bool(args.safety60_transfer),
        "even48_read": False,
        "test60_read": False,
        "params": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if errors:
        raise RuntimeError(f"Z6b DINOv2 audit failed with {len(errors)} errors")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path(
        "docs/diagnostics/z6b_object_view_manifest_official100_20260812"
    ))
    parser.add_argument("--ledger-root", type=Path, default=Path(
        "docs/diagnostics/z6b_dinov2_object_appearance_official100_20260812"
    ))
    parser.add_argument("--output-path", type=Path, default=Path(
        "docs/diagnostics/z6b_dinov2_object_appearance_official100_20260812/audit_summary.json"
    ))
    parser.add_argument("--tolerance", type=float, default=2e-5)
    parser.add_argument("--allow-nonofficial-manifest", action="store_true",
                        help="permit a separately audited frozen safety60 manifest")
    parser.add_argument("--safety60-transfer", action="store_true")
    args = parser.parse_args()
    for name in ("manifest_root", "ledger_root", "output_path"):
        setattr(args, name, _resolve(getattr(args, name)))
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
