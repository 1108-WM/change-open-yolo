#!/usr/bin/env python3
"""Run one frozen stage of the FI1-D-v3 × legacy DM-SMS-1 val312 package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_ID = "DM-SMS-1-FI1-D-v3-val312-one-shot-20260824"
DUPLICATE_SAFE_PREREGISTRATION = PROJECT_ROOT / "docs/DM_SMS1_FI1_D_V3_VAL312_DUPLICATE_SAFE_PREREGISTRATION_REVISION_20260824.md"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_config(path: Path) -> dict[str, Path]:
    raw = json.loads(path.read_text())
    required = (
        "scene_list", "prepared_root", "legacy_unique_geometry_root",
        "fi1_d_v3_inference_root", "fi1_d_v3_inference_audit_root",
        "fi1_d_v3_ap_result_root", "fi1_d_v3_ap_audit_root", "config_path",
        "asset_provenance", "alpha_clip_source", "alpha_clip_base",
        "alpha_clip_checkpoint", "sam_source", "sam_checkpoint", "qwen_model_dir",
        "ground_truth_root", "run_root", "preregistration_path",
    )
    missing = [name for name in required if not isinstance(raw.get(name), str) or not raw[name]]
    if missing:
        raise ValueError(f"path config is missing non-empty string keys: {missing}")
    return {name: _resolve(raw[name]) for name in required}


def _outputs(run_root: Path) -> dict[str, Path]:
    return {
        "preflight": run_root / "00_preflight",
        "geometry": run_root / "01_unique_geometry",
        "geometry_audit": run_root / "02_unique_geometry_audit",
        "alpha_views": run_root / "03_alpha_view_manifest",
        "alpha_views_audit": run_root / "04_alpha_view_manifest_audit",
        "alpha_embeddings": run_root / "05_alpha_embedding_ledger",
        "alpha_embeddings_audit": run_root / "06_alpha_embedding_ledger_audit",
        "semantic_manifest": run_root / "07_semantic_arbitration_manifest",
        "attribute_manifest": run_root / "08_attribute_extraction_manifest",
        "candidate_manifest": run_root / "09_candidate_evidence_manifest",
        "qwen_smoke": run_root / "10_qwen_smoke",
        "qwen_smoke_audit": run_root / "11_qwen_smoke_audit",
        "qwen_full": run_root / "12_qwen_full",
        "qwen_full_audit": run_root / "13_qwen_full_audit",
        "pair_decisions": run_root / "14_pair_safe_decisions",
        "full_decisions": run_root / "15_full_safe_decisions",
        "prediction_cache": run_root / "16_prediction_cache",
        "prediction_cache_audit": run_root / "17_prediction_cache_audit",
        "ap": run_root / "18_open_vocab_ap",
        "ap_audit": run_root / "19_open_vocab_ap_audit",
    }


def _tool(name: str) -> str:
    return str(PROJECT_ROOT / "tools" / name)


def _commands(stage: str, cfg: dict[str, Path], out: dict[str, Path], authorize_ap: bool) -> list[list[str]]:
    py = sys.executable
    p = lambda value: str(value)
    commands: dict[str, list[list[str]]] = {
        "preflight": [[
            py, _tool("preflight_dm_sms1_fi1_d_v3_val312.py"),
            "--scene-list", p(cfg["scene_list"]),
            "--prepared-root", p(cfg["prepared_root"]),
            "--legacy-unique-geometry-root", p(cfg["legacy_unique_geometry_root"]),
            "--inference-root", p(cfg["fi1_d_v3_inference_root"]),
            "--inference-audit-root", p(cfg["fi1_d_v3_inference_audit_root"]),
            "--dv3-ap-result-root", p(cfg["fi1_d_v3_ap_result_root"]),
            "--dv3-ap-audit-root", p(cfg["fi1_d_v3_ap_audit_root"]),
            "--config-path", p(cfg["config_path"]),
            "--asset-provenance", p(cfg["asset_provenance"]),
            "--alpha-clip-source", p(cfg["alpha_clip_source"]),
            "--alpha-clip-base", p(cfg["alpha_clip_base"]),
            "--alpha-clip-checkpoint", p(cfg["alpha_clip_checkpoint"]),
            "--sam-source", p(cfg["sam_source"]),
            "--sam-checkpoint", p(cfg["sam_checkpoint"]),
            "--qwen-model-dir", p(cfg["qwen_model_dir"]),
            "--preregistration-path", p(cfg["preregistration_path"]),
            "--duplicate-safe-preregistration-path", p(DUPLICATE_SAFE_PREREGISTRATION),
            "--run-root", p(cfg["run_root"]), "--output-root", p(out["preflight"]),
        ]],
        "geometry": [
            [py, _tool("build_dm_sms1_fi1_d_v3_unique_geometry_ledger.py"),
             "--scene-list", p(cfg["scene_list"]),
             "--inference-root", p(cfg["fi1_d_v3_inference_root"]),
             "--inference-audit-root", p(cfg["fi1_d_v3_inference_audit_root"]),
             "--legacy-unique-geometry-root", p(cfg["legacy_unique_geometry_root"]),
             "--output-root", p(out["geometry"]), "--expected-scene-count", "312"],
            [py, _tool("audit_dm_sms1_fi1_d_v3_unique_geometry_ledger.py"),
             "--ledger-root", p(out["geometry"]),
             "--inference-root", p(cfg["fi1_d_v3_inference_root"]),
             "--output-root", p(out["geometry_audit"]),
             "--expected-candidate-count", "39304",
             "--expected-unique-geometry-count", "39250",
             "--expected-duplicate-group-count", "54",
             "--expected-duplicate-scene-count", "50",
             "--expected-different-class-group-count", "33",
             "--expected-different-score-group-count", "53"],
        ],
        "alpha": [
            [py, _tool("build_dm_sms1_alpha_view_manifest.py"),
             "--scene-list", p(cfg["scene_list"]), "--ledger-root", p(out["geometry"]),
             "--prepared-root", p(cfg["prepared_root"]), "--config-path", p(cfg["config_path"]),
             "--output-root", p(out["alpha_views"]), "--preregistration-path", p(cfg["preregistration_path"]),
             "--expected-scene-count", "312", "--max-views", "20"],
            [py, _tool("audit_dm_sms1_alpha_view_manifest.py"),
             "--scene-list", p(cfg["scene_list"]), "--manifest-root", p(out["alpha_views"]),
             "--ledger-root", p(out["geometry"]), "--prepared-root", p(cfg["prepared_root"]),
             "--config-path", p(cfg["config_path"]), "--output-root", p(out["alpha_views_audit"]),
             "--expected-scene-count", "312", "--max-views", "20"],
            [py, _tool("build_dm_sms1_alpha_embedding_ledger.py"),
             "--scene-list", p(cfg["scene_list"]), "--manifest-root", p(out["alpha_views"]),
             "--manifest-audit-root", p(out["alpha_views_audit"]),
             "--output-root", p(out["alpha_embeddings"]), "--config-path", p(cfg["config_path"]),
             "--asset-provenance", p(cfg["asset_provenance"]),
             "--alpha-clip-source", p(cfg["alpha_clip_source"]), "--alpha-clip-base", p(cfg["alpha_clip_base"]),
             "--alpha-clip-checkpoint", p(cfg["alpha_clip_checkpoint"]),
             "--sam-source", p(cfg["sam_source"]), "--sam-checkpoint", p(cfg["sam_checkpoint"]),
             "--sam-model-type", "vit_b", "--sam-batch-size", "8", "--alpha-batch-size", "32",
             "--sms-threshold", "0.0", "--expected-scene-count", "312", "--resume"],
            [py, _tool("audit_dm_sms1_alpha_embedding_ledger.py"),
             "--scene-list", p(cfg["scene_list"]), "--manifest-root", p(out["alpha_views"]),
             "--ledger-root", p(out["alpha_embeddings"]), "--output-root", p(out["alpha_embeddings_audit"]),
             "--asset-provenance", p(cfg["asset_provenance"]), "--sms-threshold", "0.0"],
        ],
        "manifests": [
            [py, _tool("build_dm_sms1_semantic_arbitration_manifest.py"),
             "--ledger-root", p(out["alpha_embeddings"]), "--scene-list", p(cfg["scene_list"]),
             "--output-root", p(out["semantic_manifest"]), "--target-views", "3", "--max-input-views", "20"],
            [py, _tool("audit_dm_sms1_semantic_arbitration_manifest.py"), p(out["semantic_manifest"]),
             "--alpha-ledger-root", p(out["alpha_embeddings"]),
             "--joint-geometry-root", p(out["geometry"]),
             "--expected-candidate-count", "39304", "--expected-unique-geometry-count", "39250"],
            [py, _tool("build_dm_sms1_attribute_extraction_manifest.py"),
             "--input-root", p(out["semantic_manifest"]), "--output-root", p(out["attribute_manifest"])],
            [py, _tool("audit_dm_sms1_attribute_extraction_manifest.py"), p(out["attribute_manifest"]),
             "--semantic-root", p(out["semantic_manifest"]),
             "--expected-candidate-count", "39304", "--expected-unique-geometry-count", "39250"],
            [py, _tool("build_dm_sms1_candidate_evidence_manifest.py"),
             "--attribute-root", p(out["attribute_manifest"]), "--semantic-root", p(out["semantic_manifest"]),
             "--output-root", p(out["candidate_manifest"]), "--config-path", p(cfg["config_path"])],
            [py, _tool("audit_dm_sms1_candidate_evidence_manifest.py"), p(out["candidate_manifest"]),
             "--attribute-root", p(out["attribute_manifest"]),
             "--semantic-root", p(out["semantic_manifest"]),
             "--config-path", p(cfg["config_path"]),
             "--expected-candidate-count", "39304", "--expected-unique-geometry-count", "39250"],
        ],
        "smoke": [
            [py, _tool("run_dm_sms1_vlm_batch_smoke.py"),
             "--attribute-manifest", p(out["attribute_manifest"] / "attribute_extraction_manifest.jsonl"),
             "--candidate-manifest", p(out["candidate_manifest"] / "candidate_evidence_manifest.jsonl"),
             "--model-dir", p(cfg["qwen_model_dir"]), "--output-root", p(out["qwen_smoke"]),
             "--scene-count", "10", "--per-scene", "2", "--attribute-max-tokens", "700",
             "--candidate-max-tokens", "450", "--config-path", p(cfg["config_path"])],
            [py, _tool("audit_dm_sms1_vlm_batch_outputs.py"), p(out["qwen_smoke"]),
             "--candidate-manifest", p(out["candidate_manifest"] / "candidate_evidence_manifest.jsonl"),
             "--attribute-manifest", p(out["attribute_manifest"] / "attribute_extraction_manifest.jsonl"),
             "--output-root", p(out["qwen_smoke_audit"]), "--config-path", p(cfg["config_path"])],
        ],
        "qwen": [
            [py, _tool("run_dm_sms1_vlm_batch_smoke.py"),
             "--attribute-manifest", p(out["attribute_manifest"] / "attribute_extraction_manifest.jsonl"),
             "--candidate-manifest", p(out["candidate_manifest"] / "candidate_evidence_manifest.jsonl"),
             "--model-dir", p(cfg["qwen_model_dir"]), "--output-root", p(out["qwen_full"]),
             "--scene-count", "312", "--per-scene", "100000", "--attribute-max-tokens", "700",
             "--candidate-max-tokens", "450", "--config-path", p(cfg["config_path"]), "--resume"],
            [py, _tool("audit_dm_sms1_vlm_batch_outputs.py"), p(out["qwen_full"]),
             "--candidate-manifest", p(out["candidate_manifest"] / "candidate_evidence_manifest.jsonl"),
             "--attribute-manifest", p(out["attribute_manifest"] / "attribute_extraction_manifest.jsonl"),
             "--output-root", p(out["qwen_full_audit"]), "--config-path", p(cfg["config_path"])],
        ],
        "decisions": [
            [py, _tool("build_dm_sms1_safe_decision_ledger.py"),
             "--batch-outputs", p(out["qwen_full"] / "batch_outputs.jsonl"),
             "--candidate-manifest", p(out["candidate_manifest"] / "candidate_evidence_manifest.jsonl"),
             "--output-root", p(out["pair_decisions"]), "--config-path", p(cfg["config_path"]),
             "--attribute-manifest", p(out["attribute_manifest"] / "attribute_extraction_manifest.jsonl")],
            [py, _tool("audit_dm_sms1_safe_decision_ledger.py"), p(out["pair_decisions"]),
             "--candidate-manifest", p(out["candidate_manifest"] / "candidate_evidence_manifest.jsonl")],
            [py, _tool("build_dm_sms1_full_safe_decision_ledger.py"),
             "--pair-decisions", p(out["pair_decisions"] / "safe_decisions.jsonl"),
             "--candidate-manifest", p(out["candidate_manifest"] / "candidate_evidence_manifest.jsonl"),
             "--output-root", p(out["full_decisions"])],
            [py, _tool("audit_dm_sms1_full_safe_decision_ledger.py"), p(out["full_decisions"]),
             "--candidate-manifest", p(out["candidate_manifest"] / "candidate_evidence_manifest.jsonl")],
        ],
        "cache": [
            [py, _tool("build_dm_sms1_fi1_d_v3_prediction_cache.py"),
             "--scene-list", p(cfg["scene_list"]), "--ledger-root", p(out["geometry"]),
             "--ledger-audit-root", p(out["geometry_audit"]), "--prepared-root", p(cfg["prepared_root"]),
             "--output-root", p(out["prediction_cache"]), "--expected-scene-count", "312"],
            [py, _tool("audit_dm_sms1_fi1_d_v3_prediction_cache.py"),
             "--cache-root", p(out["prediction_cache"]), "--ledger-root", p(out["geometry"]),
             "--decision-root", p(out["full_decisions"]),
             "--expected-candidate-count", "39304", "--expected-unique-geometry-count", "39250",
             "--output-root", p(out["prediction_cache_audit"])],
        ],
        "audit": [[
            py, _tool("audit_dm_sms1_fi1_d_v3_open_vocab_ap.py"),
            "--result-root", p(out["ap"]), "--scene-list", p(cfg["scene_list"]),
            "--cache-root", p(out["prediction_cache"]), "--cache-audit-root", p(out["prediction_cache_audit"]),
            "--decision-root", p(out["full_decisions"]), "--preregistration-path", p(cfg["preregistration_path"]),
            "--duplicate-safe-preregistration-path", p(DUPLICATE_SAFE_PREREGISTRATION),
            "--output-root", p(out["ap_audit"]), "--expected-scene-count", "312",
        ]],
    }
    if stage == "ap":
        if not authorize_ap:
            raise PermissionError("the AP stage additionally requires --authorize-ap")
        return [[
            py, _tool("evaluate_dm_sms1_fi1_d_v3_open_vocab_ap_gt.py"),
            "--scene-list", p(cfg["scene_list"]), "--ground-truth-root", p(cfg["ground_truth_root"]),
            "--cache-root", p(out["prediction_cache"]), "--cache-audit-root", p(out["prediction_cache_audit"]),
            "--decision-root", p(out["full_decisions"]), "--preregistration-path", p(cfg["preregistration_path"]),
            "--duplicate-safe-preregistration-path", p(DUPLICATE_SAFE_PREREGISTRATION),
            "--output-root", p(out["ap"]), "--expected-scene-count", "312",
            "--dataset-name", "ScanNet200-val312", "--authorization-id", AUTHORIZATION_ID,
            "--allow-gt-evaluation",
        ]]
    return commands[stage]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True, help="JSON path configuration")
    parser.add_argument(
        "--stage", required=True,
        choices=("preflight", "geometry", "alpha", "manifests", "smoke", "qwen", "decisions", "cache", "ap", "audit"),
    )
    parser.add_argument("--authorize-ap", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    config_path = _resolve(args.paths)
    cfg = _load_config(config_path)
    out = _outputs(cfg["run_root"])
    commands = _commands(args.stage, cfg, out, args.authorize_ap)
    for index, command in enumerate(commands, 1):
        print(json.dumps({"stage": args.stage, "command_index": index, "argv": command}, ensure_ascii=False))
        if not args.print_only:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
