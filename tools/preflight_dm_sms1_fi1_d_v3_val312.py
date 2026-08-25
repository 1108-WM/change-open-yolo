#!/usr/bin/env python3
"""Read-only preflight for the frozen FI1-D-v3 × DM-SMS-1 val312 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_dm_sms1_vlm_batch_smoke import _verify_model_revision  # noqa: E402


EXPECTED_DV3 = {"ap": 0.526344, "ap50": 0.729213, "ap25": 0.826733}
DUPLICATE_SAFE_PREREGISTRATION = PROJECT_ROOT / "docs/DM_SMS1_FI1_D_V3_VAL312_DUPLICATE_SAFE_PREREGISTRATION_REVISION_20260824.md"
TERMINAL_SAFE_KEEP_PREREGISTRATION = PROJECT_ROOT / "docs/DM_SMS1_FI1_D_V3_VAL312_TERMINAL_SAFE_KEEP_PREREGISTRATION_REVISION_20260825.md"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenes(path: Path) -> list[str]:
    scenes = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if len(scenes) != 312 or len(scenes) != len(set(scenes)):
        raise ValueError("joint evaluation requires exactly 312 unique scenes")
    return scenes


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(args: argparse.Namespace) -> dict:
    if getattr(args, "duplicate_safe_preregistration_path", None) is None:
        args.duplicate_safe_preregistration_path = DUPLICATE_SAFE_PREREGISTRATION
    if getattr(args, "terminal_safe_keep_preregistration_path", None) is None:
        args.terminal_safe_keep_preregistration_path = TERMINAL_SAFE_KEEP_PREREGISTRATION
    path_names = (
        "scene_list", "prepared_root", "legacy_unique_geometry_root", "inference_root",
        "inference_audit_root", "dv3_ap_result_root", "dv3_ap_audit_root", "config_path",
        "asset_provenance", "alpha_clip_source", "alpha_clip_base", "alpha_clip_checkpoint",
        "sam_source", "sam_checkpoint", "qwen_model_dir", "run_root", "output_root",
        "preregistration_path", "duplicate_safe_preregistration_path",
        "terminal_safe_keep_preregistration_path",
    )
    for name in path_names:
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _scenes(args.scene_list)
    required_files = {
        "config": args.config_path,
        "asset_provenance": args.asset_provenance,
        "preregistration": args.preregistration_path,
        "duplicate_safe_preregistration": args.duplicate_safe_preregistration_path,
        "terminal_safe_keep_preregistration": args.terminal_safe_keep_preregistration_path,
        "alpha_clip_base": args.alpha_clip_base,
        "alpha_clip_checkpoint": args.alpha_clip_checkpoint,
        "sam_checkpoint": args.sam_checkpoint,
        "legacy_unique_ledger": args.legacy_unique_geometry_root / "unique_geometry_ledger.jsonl",
        "legacy_unique_summary": args.legacy_unique_geometry_root / "summary.json",
        "inference_summary": args.inference_root / "summary.json",
        "inference_audit": args.inference_audit_root / "summary.json",
        "dv3_ap_summary": args.dv3_ap_result_root / "summary.json",
        "dv3_ap_audit": args.dv3_ap_audit_root / "summary.json",
    }
    missing = [f"{name}: {path}" for name, path in required_files.items() if not path.is_file()]
    for name, path in (
        ("prepared_root", args.prepared_root),
        ("alpha_clip_source", args.alpha_clip_source),
        ("sam_source", args.sam_source),
        ("qwen_model_dir", args.qwen_model_dir),
    ):
        if not path.is_dir():
            missing.append(f"{name}: {path}")
    if missing:
        raise FileNotFoundError("required deployment assets are missing: " + "; ".join(missing))
    if args.run_root.exists():
        if not args.run_root.is_dir() or any(args.run_root.iterdir()):
            raise FileExistsError(f"joint run root is not an empty directory: {args.run_root}")
    if args.output_root.exists():
        raise FileExistsError(f"preflight output already exists: {args.output_root}")

    plan_summary_path = args.inference_root / "complete_plan" / "summary.json"
    plan_summary = json.loads(plan_summary_path.read_text())
    plan_path = args.inference_root / "complete_plan" / str(plan_summary["files"]["plan"])
    if not plan_path.is_file() or _sha256(plan_path) != str(plan_summary["hashes"]["plan"]):
        raise ValueError("FI1-D-v3 complete plan identity is invalid")
    plan_rows = _rows(plan_path)
    plan_keys = [str(row.get("plan_key", "")) for row in plan_rows]
    if (
        int(plan_summary.get("scene_count", -1)) != 312
        or len(plan_rows) != 39304
        or any(not key for key in plan_keys)
        or len(plan_keys) != len(set(plan_keys))
        or {str(row.get("scene_name", "")) for row in plan_rows} != set(scenes)
    ):
        raise ValueError("FI1-D-v3 complete plan coverage or key identity is invalid")
    inference_summary = json.loads(required_files["inference_summary"].read_text())
    inference_audit = json.loads(required_files["inference_audit"].read_text())
    if (
        int(inference_summary.get("scene_count", -1)) != 312
        or inference_summary.get("ground_truth_usage") != "none"
        or inference_summary.get("ap_computed") is not False
        or inference_summary.get("candidate_deletion_count") != 0
        or inference_summary.get("geometry_mutation") is not False
        or inference_summary.get("class_mutation") is not False
        or inference_audit.get("audit_valid") is not True
        or int(inference_audit.get("error_count", -1)) != 0
        or inference_audit.get("advancement_gate", {}).get("advancement_authorized") is not True
    ):
        raise ValueError("FI1-D-v3 frozen inference audit is not valid")

    ap_summary = json.loads(required_files["dv3_ap_summary"].read_text())
    ap_audit = json.loads(required_files["dv3_ap_audit"].read_text())
    for metric, expected in EXPECTED_DV3.items():
        if not math.isclose(float(ap_summary["challenger"][metric]), expected, rel_tol=0.0, abs_tol=5e-7):
            raise ValueError(f"frozen FI1-D-v3 {metric} differs from the authorized val312 result")
    if (
        ap_audit.get("audit_valid") is not True
        or int(ap_audit.get("error_count", -1)) != 0
        or str(ap_summary.get("input_provenance", {}).get("plan_sha256", "")) != _sha256(plan_path)
    ):
        raise ValueError("FI1-D-v3 AP audit or plan provenance is invalid")
    _verify_model_revision(args.qwen_model_dir)

    legacy_summary = json.loads(required_files["legacy_unique_summary"].read_text())
    if (
        int(legacy_summary.get("scene_count", -1)) != 312
        or legacy_summary.get("contract_valid") is not True
        or legacy_summary.get("ground_truth_usage") != "none"
        or legacy_summary.get("ap_computed") is not False
    ):
        raise ValueError("FI1-Legacy unique geometry input contract is invalid")
    missing_prepared = []
    for scene in scenes:
        stem = scene[len("scene"):] if scene.startswith("scene") else scene
        if not (args.prepared_root / scene / f"{stem}.npy").is_file():
            missing_prepared.append(scene)
    if missing_prepared:
        raise FileNotFoundError(f"prepared point clouds are missing: {missing_prepared[:3]}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    result = {
        "version": "dm_sms1_fi1_d_v3_val312_preflight_v1",
        "preflight_valid": True,
        "scene_count": 312,
        "fi1_d_v3_candidate_count": len(plan_rows),
        "fi1_d_v3_metrics": EXPECTED_DV3,
        "fi1_d_v3_plan_sha256": _sha256(plan_path),
        "fi1_d_v3_inference_summary_sha256": _sha256(required_files["inference_summary"]),
        "fi1_d_v3_inference_audit_sha256": _sha256(required_files["inference_audit"]),
        "fi1_d_v3_ap_summary_sha256": _sha256(required_files["dv3_ap_summary"]),
        "fi1_d_v3_ap_audit_sha256": _sha256(required_files["dv3_ap_audit"]),
        "legacy_unique_geometry_sha256": _sha256(required_files["legacy_unique_ledger"]),
        "config_sha256": _sha256(args.config_path),
        "asset_provenance_sha256": _sha256(args.asset_provenance),
        "preregistration_sha256": _sha256(args.preregistration_path),
        "duplicate_safe_preregistration_sha256": _sha256(args.duplicate_safe_preregistration_path),
        "terminal_safe_keep_preregistration_sha256": _sha256(args.terminal_safe_keep_preregistration_path),
        "ground_truth_read": False,
        "ap_computed": False,
        "qwen_inference_run": False,
        "gpu_inference_run": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--legacy-unique-geometry-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--inference-audit-root", type=Path, required=True)
    parser.add_argument("--dv3-ap-result-root", type=Path, required=True)
    parser.add_argument("--dv3-ap-audit-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--asset-provenance", type=Path, required=True)
    parser.add_argument("--alpha-clip-source", type=Path, required=True)
    parser.add_argument("--alpha-clip-base", type=Path, required=True)
    parser.add_argument("--alpha-clip-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-source", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-model-dir", type=Path, required=True)
    parser.add_argument("--preregistration-path", type=Path, required=True)
    parser.add_argument(
        "--duplicate-safe-preregistration-path", type=Path,
        default=DUPLICATE_SAFE_PREREGISTRATION,
    )
    parser.add_argument(
        "--terminal-safe-keep-preregistration-path", type=Path,
        default=TERMINAL_SAFE_KEEP_PREREGISTRATION,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
