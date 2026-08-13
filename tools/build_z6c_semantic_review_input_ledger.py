#!/usr/bin/env python3
"""Build a no-GT Z6c geometry-node semantic review input ledger.

The ledger freezes the current hybrid class hypotheses and the YOLO/Alpha
top-5 union candidate space, then joins audited DINOv2 consistency statistics.
It is feature preparation only: no routing threshold, training, class change,
MLLM call, score change, inference plan, GT, or AP is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MANIFEST_SHA256 = "503261f316a0e9e642eb09c63c87d1fd10c9be60b3bd71041d3dfa8fb149107b"
SOURCES = ("native", "track", "pair_union")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _top_indices(values: np.ndarray, top_k: int) -> list[int]:
    values = np.asarray(values, dtype=np.float32)
    valid = np.flatnonzero(np.isfinite(values) & (values > 0))
    order = valid[np.argsort(-values[valid], kind="stable")]
    return [int(value) for value in order[:top_k]]


def _margin(values: np.ndarray) -> float:
    top = _top_indices(values, 2)
    if not top:
        return 0.0
    return float(values[top[0]] - (values[top[1]] if len(top) > 1 else 0.0))


def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return 0.0
    values = values / values.sum()
    return float(-(values * np.log(values)).sum() / math.log(198))


def _candidate_rows(
    yolo: np.ndarray, alpha: np.ndarray, prompts: list[str], top_k: int
) -> tuple[list[dict], list[int], list[int]]:
    yolo_top = _top_indices(yolo, top_k)
    alpha_top = _top_indices(alpha, top_k)
    union = sorted(set(yolo_top) | set(alpha_top), key=lambda index: (
        -max(float(yolo[index]), float(alpha[index])), index
    ))
    rows = [{
        "class_index": index,
        "class_name": prompts[index],
        "yolo_probability": float(yolo[index]),
        "alpha_probability": float(alpha[index]),
        "max_model_probability": max(float(yolo[index]), float(alpha[index])),
        "in_yolo_top5": index in yolo_top,
        "in_alpha_top5": index in alpha_top,
        "yolo_rank": yolo_top.index(index) + 1 if index in yolo_top else None,
        "alpha_rank": alpha_top.index(index) + 1 if index in alpha_top else None,
    } for index in union]
    return rows, yolo_top, alpha_top


def _hypothesis_score(row: dict) -> float:
    if str(row["candidate_source"]) == "pair_union":
        return float(row["original_score"])
    return float(row["oof_predictions"]["C_joint_yolo_alpha"])


def run(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(args.config_path.read_text())
    prompts = [str(value) for value in config["network2d"]["text_prompts"]]
    if len(prompts) != 198:
        raise ValueError("expected exactly 198 registered instance prompts")
    manifest_summary = json.loads((args.view_manifest_root / "summary.json").read_text())
    manifest_path = args.view_manifest_root / "object_view_manifest.jsonl"
    manifest_sha = _sha256(manifest_path)
    if (
        (manifest_sha != EXPECTED_MANIFEST_SHA256 and not args.allow_nonofficial_manifest)
        or manifest_summary["output_sha256"]["object_view_manifest.jsonl"] != manifest_sha
    ):
        raise ValueError("frozen Z6b view manifest SHA mismatch")
    dino_audit = json.loads((args.dino_root / "audit_summary.json").read_text())
    if not dino_audit.get("valid") or int(dino_audit.get("error_count", -1)) != 0:
        raise ValueError("audited Z6b DINOv2 ledger is not valid")

    nodes = _read_jsonl(args.unified_ledger_root / "nodes.jsonl")
    node_by_key = {str(row["semantic_evidence_node_key"]): row for row in nodes}
    if len(node_by_key) != int(args.expected_node_count):
        raise ValueError("unexpected unified node count")
    with np.load(args.unified_ledger_root / "semantic_distributions.npz") as payload:
        distributions = {name: np.asarray(payload[name], dtype=np.float32) for name in payload.files}
    if any(values.shape != (int(args.expected_node_count), 198) for values in distributions.values()):
        raise ValueError("unified semantic distribution shape mismatch")

    bindings = _read_jsonl(args.z1_root / "candidate_bindings.jsonl")
    binding_by_candidate = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"])): row
        for row in bindings
    }
    if len(binding_by_candidate) != len(bindings):
        raise ValueError("duplicate candidate binding")
    oof_rows = _read_jsonl(args.oof_root / "oof_predictions.jsonl")
    oof_by_candidate = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"])): row
        for row in oof_rows
    }
    if len(oof_by_candidate) != len(oof_rows):
        raise ValueError("duplicate OOF candidate row")

    current_by_node = defaultdict(list)
    missing_oof = Counter()
    for key, binding in binding_by_candidate.items():
        source = str(binding["candidate_source"])
        oof = oof_by_candidate.get(key)
        if oof is None:
            missing_oof[source] += 1
            continue
        if str(oof["semantic_evidence_node_key"]) != str(binding["semantic_evidence_node_key"]):
            raise ValueError(f"OOF/Z1 node mismatch: {key}")
        class_index = int(oof["class_index"])
        if not 0 <= class_index < 198:
            raise ValueError(f"invalid current class: {key}")
        current_by_node[str(binding["semantic_evidence_node_key"])].append({
            "candidate_source": source,
            "candidate_id": int(binding["candidate_id"]),
            "class_index": class_index,
            "class_name": prompts[class_index],
            "hybrid_score": _hypothesis_score(oof),
            "score_provenance": (
                "frozen_original_pair_union_score" if source == "pair_union"
                else (
                    "full_official100_C_joint_yolo_alpha"
                    if args.safety60_transfer else "scene_isolated_oof_C_joint_yolo_alpha"
                )
            ),
        })

    view_manifest = {
        str(row["semantic_evidence_node_key"]): row for row in _read_jsonl(manifest_path)
    }
    dino_by_key = {}
    dino_scene_counts = Counter()
    for scene_dir in sorted(path for path in args.dino_root.glob("scene*") if path.is_dir()):
        for row in _read_jsonl(scene_dir / "dinov2_object_appearance_ledger.jsonl"):
            key = str(row["semantic_evidence_node_key"])
            if key in dino_by_key:
                raise ValueError(f"duplicate DINO node: {key}")
            dino_by_key[key] = row
            dino_scene_counts[str(row["scene_name"])] += 1
    if set(view_manifest) != set(node_by_key) or set(dino_by_key) != set(node_by_key):
        raise ValueError("node/view/DINO ledgers do not have exact joins")

    rows = []
    source_counts = Counter()
    review_state_counts = Counter()
    current_in_union = 0
    top1_disagreement = 0
    for key in sorted(node_by_key, key=lambda value: int(node_by_key[value]["node_index"])):
        node = node_by_key[key]
        index = int(node["node_index"])
        yolo = np.maximum(distributions["geometry_yolo"][index], distributions["inherited_yolo"][index])
        alpha = np.maximum(distributions["geometry_alpha"][index], distributions["inherited_alpha"][index])
        candidates, yolo_top, alpha_top = _candidate_rows(yolo, alpha, prompts, args.top_k)
        union_set = {row["class_index"] for row in candidates}
        current = sorted(current_by_node.get(key, []), key=lambda row: (
            -float(row["hybrid_score"]), SOURCES.index(str(row["candidate_source"])), int(row["candidate_id"])
        ))
        representative = current[0] if current else None
        representative_in_union = bool(representative and representative["class_index"] in union_set)
        current_in_union += int(representative_in_union)
        yolo_top1 = yolo_top[0] if yolo_top else None
        alpha_top1 = alpha_top[0] if alpha_top else None
        disagreement = yolo_top1 is not None and alpha_top1 is not None and yolo_top1 != alpha_top1
        top1_disagreement += int(disagreement)
        dino = dino_by_key[key]
        if not current:
            review_state = "no_valid_current_hypothesis"
        elif not candidates:
            review_state = "no_registered_candidate_class"
        elif dino["embedding_state"] != "available":
            review_state = "candidate_review_ready_without_dino"
        else:
            review_state = "candidate_review_ready"
        review_state_counts[review_state] += 1
        source_counts[str(node["candidate_source"])] += 1
        rows.append({
            "scene_name": str(node["scene_name"]),
            "node_index": index,
            "semantic_evidence_node_key": key,
            "candidate_source": str(node["candidate_source"]),
            "geometry_hash": str(node["geometry_hash"]),
            "point_count": int(node["point_count"]),
            "bound_candidate_count": int(node["bound_candidate_count"]),
            "current_hypotheses": current,
            "current_hypothesis_count": len(current),
            "representative_current_class_index": (
                int(representative["class_index"]) if representative else None
            ),
            "representative_current_class_name": (
                str(representative["class_name"]) if representative else None
            ),
            "representative_current_score": (
                float(representative["hybrid_score"]) if representative else None
            ),
            "representative_current_in_candidate_union": representative_in_union,
            "candidate_classes": candidates,
            "candidate_union_size": len(candidates),
            "yolo_top5_class_indices": yolo_top,
            "alpha_top5_class_indices": alpha_top,
            "yolo_top1_class_index": yolo_top1,
            "alpha_top1_class_index": alpha_top1,
            "yolo_alpha_top1_disagreement": disagreement,
            "yolo_margin": _margin(yolo),
            "alpha_margin": _margin(alpha),
            "yolo_entropy": _entropy(yolo),
            "alpha_entropy": _entropy(alpha),
            "geometry_yolo_alpha_js": node.get("geometry_yolo_alpha_js"),
            "inherited_yolo_alpha_js": node.get("inherited_yolo_alpha_js"),
            "view_count": int(view_manifest[key]["view_count"]),
            "view_availability": str(view_manifest[key]["view_availability"]),
            "dino_embedding_state": str(dino["embedding_state"]),
            "dino_pairwise_cosine_mean": dino["pairwise_cosine_mean"],
            "dino_pairwise_cosine_min": dino["pairwise_cosine_min"],
            "dino_pairwise_cosine_std": dino["pairwise_cosine_std"],
            "dino_dispersion": dino["dispersion_one_minus_pairwise_mean"],
            "review_input_state": review_state,
            "routing_decision": None,
            "selected_class_index": None,
            "abstain_fallback": "keep representative current class; if absent, keep original candidate-level contract",
            "ground_truth_usage": "none",
            "model_trained": False,
            "mllm_invoked": False,
            "candidate_mutation": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "score_mutation": False,
            "inference_plan_written": False,
        })

    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"output directory exists: {output_dir}")
    stage = output_dir.parent / f".{output_dir.name}.tmp.{os.getpid()}"
    stage.mkdir(parents=True)
    try:
        ledger_path = stage / "semantic_review_inputs.jsonl"
        with ledger_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "diagnostic_type": "Z6c GT-free frozen semantic candidate review input ledger",
            "scene_count": len({row["scene_name"] for row in rows}),
            "node_count": len(rows),
            "source_node_counts": dict(source_counts),
            "review_input_state_counts": dict(review_state_counts),
            "node_with_current_hypothesis_count": sum(bool(row["current_hypotheses"]) for row in rows),
            "node_without_current_hypothesis_count": sum(not row["current_hypotheses"] for row in rows),
            "representative_current_in_candidate_union_count": current_in_union,
            "representative_current_in_candidate_union_fraction": current_in_union / len(rows),
            "yolo_alpha_top1_disagreement_count": top1_disagreement,
            "yolo_alpha_top1_disagreement_fraction": top1_disagreement / len(rows),
            "missing_oof_candidate_counts": dict(missing_oof),
            "top_k_per_model": args.top_k,
            "manifest_sha256": manifest_sha,
            "ledger_sha256": _sha256(ledger_path),
            "contracts": {
                "candidate_space": "elementwise max of geometry-own/inherited distribution, fixed top-5 per model, deterministic union",
                "current_representative": "highest current hybrid score within the exact semantic evidence node; full current hypothesis list retained",
                "dino": "audited node-level appearance consistency only; not a class generator",
                "routing": "not performed in this stage",
                "fallback": "abstain keeps current class contract",
            },
            "ground_truth_usage": "none",
            "model_trained": False,
            "mllm_invoked": False,
            "candidate_mutation": False,
            "geometry_mutation": False,
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
        (stage / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--unified-ledger-root", type=Path, default=Path(
        "docs/diagnostics/z2c_unified_semantic_node_ledger_official100_20260811"
    ))
    parser.add_argument("--z1-root", type=Path, default=Path(
        "docs/diagnostics/z1_yoloworld_multiview_distribution_official100_20260811_v3_frozen_support_vote"
    ))
    parser.add_argument("--oof-root", type=Path, default=Path(
        "docs/diagnostics/z3_semantic_reliability_oof_official100_20260811"
    ))
    parser.add_argument("--view-manifest-root", type=Path, default=Path(
        "docs/diagnostics/z6b_object_view_manifest_official100_20260812"
    ))
    parser.add_argument("--dino-root", type=Path, default=Path(
        "docs/diagnostics/z6b_dinov2_object_appearance_official100_20260812"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "docs/diagnostics/z6c_semantic_review_input_official100_20260812"
    ))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--expected-node-count", type=int, default=9708)
    parser.add_argument("--allow-nonofficial-manifest", action="store_true",
                        help="permit a separately audited frozen safety60 manifest")
    parser.add_argument("--safety60-transfer", action="store_true")
    args = parser.parse_args()
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path):
            setattr(args, name, _resolve(value))
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
