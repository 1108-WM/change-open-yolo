#!/usr/bin/env python3
"""Evaluate one frozen candidate-quality reranking on class-agnostic AP.

The experiment fits the preregistered official100 ``D_plus_gvc`` q regressor
once, applies it to the disjoint safety60 candidates, and compares original
scores with predicted IoU scores. Candidate masks, classes, counts, and track
geometry are read-only and shared between score variants.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    NATIVE_SOURCE,
    TRACK_FEATURES,
    TRACK_SOURCE,
    candidate_ledger_path,
    read_jsonl,
    read_scene_list,
)
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    UNIFIED_PREDICTED_CLASS,
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    _merge_scan_matches,
    instance_eval,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    base_sample_weights,
    canonicalize_predictions,
    feature_matrix,
    fit_predict,
    make_model,
    source_balanced_weights,
    target_values,
)


EXPECTED_NATIVE = {
    "ap": 0.47029171282921506,
    "ap50": 0.6349018488674009,
    "ap25": 0.747776289333802,
}
EXPECTED_COMBINED = {
    "ap": 0.48043642810813414,
    "ap50": 0.648657490448241,
    "ap25": 0.7604179649278366,
}
FEATURE_GROUP = "D_plus_gvc"
TARGET = "q"
PROTOCOL_NAME = "official100_v2"
MIN_REGION_SIZE = 100
EXPECTED_SPLIT_SHA256 = "710157cb78b66048a181f5e38701a274f24add6416a2f25627d9e499b0668c72"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_file_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 不是有限数")
    return result


def _flatten_gvc(row: dict) -> dict:
    result = dict(row)
    excluded = row.get("gvc_source_frame_excluded")
    if not isinstance(excluded, dict) or not isinstance(excluded.get("gvc"), dict):
        raise ValueError("缺少 gvc_source_frame_excluded 特征")
    result.update({
        "gvc_excluded_mean": _finite(excluded["gvc"]["mean"], "gvc mean"),
        "gvc_excluded_max": _finite(excluded["gvc"]["max"], "gvc max"),
        "gvc_excluded_variance": _finite(excluded["gvc"]["variance"], "gvc variance"),
        "gvc_excluded_selected_view_count": _finite(
            excluded["selected_view_count"], "selected view count"
        ),
        "gvc_excluded_matched_view_count": _finite(
            excluded["matched_selected_view_count"], "matched view count"
        ),
        "gvc_excluded_zero_support_fraction": _finite(
            excluded["zero_support_selected_view_fraction"], "zero support fraction"
        ),
    })
    return result


def safety_feature_row(row: dict, track: dict | None) -> dict:
    """Map a safety60 ledger row to the exact official100 feature contract."""
    result = _flatten_gvc(row)
    source = result.get("candidate_source")
    required = (
        "scene_name", "candidate_id", "original_source_score", "point_count",
        "point_fraction_of_scene",
    )
    for field in required:
        if field not in result:
            raise ValueError(f"缺少候选字段 {field}")
    if source == NATIVE_SOURCE:
        if track is not None:
            raise ValueError("native 候选不能绑定轨迹记录")
        # These fields are absent for native rows during official100 training.
        for field in TRACK_FEATURES:
            result[field] = None
    elif source == TRACK_SOURCE:
        if track is None:
            raise ValueError("轨迹候选缺少 automatic_tracks.json 记录")
        if int(track["track_id"]) != int(result["candidate_id"]):
            raise ValueError("轨迹编号与候选编号不一致")
        for field in TRACK_FEATURES:
            if field == "source_frame_count":
                if "frame_ids" not in track:
                    raise ValueError("轨迹记录缺少派生 source_frame_count 所需的 frame_ids")
                value = len(track["frame_ids"])
                if int(result["source_frame_count"]) != value:
                    raise ValueError("质量账本 source_frame_count 与轨迹 frame_ids 不一致")
                result[field] = float(value)
                continue
            if field not in track:
                raise ValueError(f"轨迹记录缺少特征 {field}")
            result[field] = _finite(track[field], field)
    else:
        raise ValueError(f"未知候选来源：{source}")
    return result


def _canonical_mask_audit(masks: np.ndarray, point_counts: list[int]) -> dict:
    """Digest and count every mask in bounded-memory column blocks."""
    if masks.ndim != 2 or masks.shape[1] != len(point_counts):
        raise ValueError("native mask 维度与账本数量不一致")
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(masks.shape), "dtype": str(masks.dtype)}).encode())
    actual_counts = np.empty(masks.shape[1], dtype=np.int64)
    block_size = 16
    for start in range(0, masks.shape[1], block_size):
        stop = min(start + block_size, masks.shape[1])
        block = np.asarray(masks[:, start:stop], dtype=bool)
        digest.update(start.to_bytes(8, "little"))
        digest.update(np.packbits(block, axis=None).tobytes())
        actual_counts[start:stop] = np.count_nonzero(block, axis=0)
    expected = np.asarray(point_counts, dtype=np.int64)
    if not np.array_equal(actual_counts, expected):
        bad = np.flatnonzero(actual_counts != expected)[:5].tolist()
        raise ValueError(f"native mask 点数与候选账本不一致：{bad}")
    return {
        "shape": list(masks.shape),
        "dtype": str(masks.dtype),
        "canonical_element_sha256": digest.hexdigest(),
    }


def _load_track_points(track: dict, point_count: int) -> tuple[np.ndarray, str]:
    path = Path(track["points_path"])
    if not path.is_file():
        raise FileNotFoundError(f"轨迹点文件不存在：{path}")
    with np.load(path) as payload:
        raw = np.asarray(payload["point_indices"], dtype=np.int64)
    points = np.unique(raw)
    if len(points) != len(raw) or np.any(points < 0) or np.any(points >= point_count):
        raise ValueError(f"轨迹点编号重复或越界：{path}")
    if len(points) != int(track["point_count"]):
        raise ValueError(f"轨迹点数与记录不一致：{path}")
    return points, hashlib.sha256(points.tobytes()).hexdigest()


def load_safety_scene(
    scene: str,
    native_cache: Path,
    ledger_root: Path,
    track_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict], dict]:
    ledger_path = ledger_root / scene / "c1_gvc_quality_ledger.json"
    track_path = track_root / scene / "automatic_tracks.json"
    rows = json.loads(ledger_path.read_text())
    tracks = json.loads(track_path.read_text()).get("tracks", [])
    tracks = sorted(tracks, key=lambda value: int(value["track_id"]))
    track_by_id = {int(track["track_id"]): track for track in tracks}
    if len(track_by_id) != len(tracks):
        raise ValueError(f"{scene}: 轨迹编号重复")

    native_rows = sorted(
        (row for row in rows if row.get("candidate_source") == NATIVE_SOURCE),
        key=lambda value: int(value["candidate_id"]),
    )
    track_rows = sorted(
        (row for row in rows if row.get("candidate_source") == TRACK_SOURCE),
        key=lambda value: int(value["candidate_id"]),
    )
    if len(native_rows) + len(track_rows) != len(rows):
        raise ValueError(f"{scene}: 账本含未知候选来源")
    if [int(row["candidate_id"]) for row in native_rows] != list(range(len(native_rows))):
        raise ValueError(f"{scene}: native 候选编号不连续")
    track_candidate_ids = [int(row["candidate_id"]) for row in track_rows]
    if len(track_candidate_ids) != len(set(track_candidate_ids)):
        raise ValueError(f"{scene}: 轨迹候选编号重复")
    if set(track_by_id) != set(track_candidate_ids):
        raise ValueError(f"{scene}: 轨迹记录与质量账本不一一对应")

    prefix = native_cache / f"{scene}_pred_"
    masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
    scores = np.load(str(prefix) + "scores.npy", mmap_mode="r")
    classes = np.load(str(prefix) + "classes.npy", mmap_mode="r")
    if masks.shape[1] != len(scores) or len(scores) != len(classes) or len(scores) != len(native_rows):
        raise ValueError(f"{scene}: native 缓存与质量账本数量不一致")
    ledger_scores = np.asarray([row["original_source_score"] for row in native_rows], dtype=np.float64)
    if not np.allclose(np.asarray(scores, dtype=np.float64), ledger_scores, rtol=0.0, atol=1e-7):
        raise ValueError(f"{scene}: native 原始分数与质量账本不一致")
    mask_audit = _canonical_mask_audit(masks, [int(row["point_count"]) for row in native_rows])

    point_count = masks.shape[0]
    track_masks = np.zeros((point_count, len(tracks)), dtype=bool)
    track_digests = []
    for column_index, track in enumerate(tracks):
        track_id = int(track["track_id"])
        points, digest = _load_track_points(track, point_count)
        track_masks[points, column_index] = True
        row = track_rows[column_index]
        if int(row["candidate_id"]) != track_id:
            raise ValueError(f"{scene}: 轨迹列表顺序与账本顺序不一致")
        if int(row["point_count"]) != len(points):
            raise ValueError(f"{scene}: 轨迹 mask 点数与质量账本不一致")
        if not math.isclose(
            float(row["original_source_score"]), float(track["mean_node_quality"]),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(f"{scene}: 轨迹原始分数合同不一致")
        track_digests.append({"track_id": track_id, "point_indices_sha256": digest})

    feature_rows = [safety_feature_row(row, None) for row in native_rows]
    feature_rows.extend(safety_feature_row(row, track_by_id[int(row["candidate_id"])]) for row in track_rows)
    audit = {
        "scene_name": scene,
        "native_count": len(native_rows),
        "track_count": len(track_rows),
        "native_mask": mask_audit,
        "native_score_file_sha256": _sha256(Path(str(prefix) + "scores.npy")),
        "native_class_file_sha256": _sha256(Path(str(prefix) + "classes.npy")),
        "quality_ledger_sha256": _sha256(ledger_path),
        "automatic_tracks_sha256": _sha256(track_path),
        "track_point_digests": track_digests,
    }
    return masks, np.asarray(scores, dtype=np.float64), np.asarray(classes), feature_rows, tracks, audit


def fit_quality_model(
    official_scene_list: Path, records_root: Path, seed: int,
) -> tuple[object, list[str], dict]:
    scenes = read_scene_list(official_scene_list)
    ledger_paths = [candidate_ledger_path(records_root, scene) for scene in scenes]
    rows = [row for path in ledger_paths for row in read_jsonl(path)]
    if not rows:
        raise ValueError("official100 候选质量账本为空")
    matrix, feature_names = feature_matrix(rows, FEATURE_GROUP, PROTOCOL_NAME)
    labels = target_values(rows, TARGET)
    weights = source_balanced_weights(rows, base_sample_weights(rows))
    model = make_model(TARGET, seed, labels, weights, source_only=False)
    # fit_predict keeps the fit contract identical to the OOF implementation.
    fitted = canonicalize_predictions(
        TARGET, fit_predict(model, matrix, labels, weights, matrix[:1])
    )
    if fitted.shape != (1,):
        raise AssertionError("全量模型拟合失败")
    source_counts = {
        source: sum(row["candidate_source"] == source for row in rows)
        for source in (NATIVE_SOURCE, TRACK_SOURCE)
    }
    return model, feature_names, {
        "training_scene_count": len(set(row["scene_name"] for row in rows)),
        "training_candidate_count": len(rows),
        "training_source_counts": source_counts,
        "target": "label_best_gt_iou",
        "prediction": "clipped_to_[0,1]",
        "feature_group": FEATURE_GROUP,
        "feature_names": feature_names,
        "protocol_name": PROTOCOL_NAME,
        "random_seed": seed,
        "sample_weight": "native geometry folding, then equal native/track total mass",
        "model": type(model).__name__,
        "training_ledgers_combined_sha256": _combined_file_digest(ledger_paths),
    }


def _prediction(
    masks: np.ndarray, scores: np.ndarray, candidate_count: int,
) -> dict:
    if masks.shape[1] != candidate_count or len(scores) != candidate_count:
        raise ValueError("预测 mask 与分数数量不一致")
    return {
        "pred_masks": masks,
        "pred_scores": np.asarray(scores, dtype=np.float32),
        "pred_classes": np.full(candidate_count, UNIFIED_PREDICTED_CLASS, dtype=np.int64),
    }


def _uuid_score_map(pred_rows: dict, scores: np.ndarray, point_counts: list[int]) -> dict:
    included = [index for index, count in enumerate(point_counts) if count >= MIN_REGION_SIZE]
    rows = pred_rows["chair"]
    if len(rows) != len(included):
        raise ValueError("评测器接纳的候选数量与点数合同不一致")
    return {row["uuid"]: float(scores[index]) for row, index in zip(rows, included)}


def _set_match_scores(matches: dict, scores_by_uuid: dict) -> None:
    seen = set()
    for match in matches.values():
        for rows in match["pred"].values():
            for row in rows:
                uuid = row["uuid"]
                if uuid not in scores_by_uuid:
                    raise ValueError("候选分数映射缺少预测 UUID")
                row["confidence"] = scores_by_uuid[uuid]
                seen.add(uuid)
        for rows in match["gt"].values():
            for row in rows:
                for prediction in row["matched_pred"]:
                    prediction["confidence"] = scores_by_uuid[prediction["uuid"]]
    if seen != set(scores_by_uuid):
        raise ValueError("候选分数映射含未使用 UUID")


def _evaluate(name: str, matches: dict, output_dir: Path) -> dict:
    ap_scores, _, _, _, _, _ = instance_eval.evaluate_matches(matches)
    averages = instance_eval.compute_averages(ap_scores)
    instance_eval.write_result_file(averages, str(output_dir / f"{name}.csv"))
    chair = averages["classes"]["chair"]
    return {
        "ap": float(chair["ap"]),
        "ap50": float(chair["ap50%"]),
        "ap25": float(chair["ap25%"]),
    }


def assert_baseline(actual: dict, expected: dict, tolerance: float) -> None:
    errors = {key: actual[key] - expected[key] for key in expected}
    if any(abs(value) > tolerance for value in errors.values()):
        raise ValueError(f"冻结 AP 基线复现失败：{errors}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def run(args: argparse.Namespace) -> dict:
    split_sha256 = _sha256(args.split_manifest)
    if split_sha256 != EXPECTED_SPLIT_SHA256:
        raise ValueError(
            f"冻结五折清单 SHA-256 不一致：{split_sha256}"
        )
    training_scenes = read_scene_list(args.official_scene_list)
    safety_scenes = read_scene_list(args.safety_scene_list)
    overlap = sorted(set(training_scenes) & set(safety_scenes))
    if overlap:
        raise ValueError(f"official100 与 safety60 场景重叠：{overlap}")
    if len(training_scenes) != 100 or len(safety_scenes) != 60:
        raise ValueError("本冻结实验要求 official100 和 safety60 恰好为 100/60 个场景")

    model, feature_names, model_contract = fit_quality_model(
        args.official_scene_list, args.records_root, args.random_seed
    )
    _configure_scannet200_instance_eval()
    native_matches, combined_matches = {}, {}
    native_original_uuid_scores, native_q_uuid_scores = {}, {}
    combined_original_uuid_scores, combined_q_uuid_scores = {}, {}
    identity_rows, score_rows = [], []
    safety_ledger_paths, track_paths = [], []

    original_load_ids = instance_eval.util_3d.load_ids

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    try:
        for scene_index, scene in enumerate(safety_scenes, start=1):
            masks, native_scores, native_classes, rows, tracks, audit = load_safety_scene(
                scene, args.native_cache, args.safety_ledger_root, args.track_root
            )
            matrix, names = feature_matrix(rows, FEATURE_GROUP, PROTOCOL_NAME)
            if names != feature_names:
                raise ValueError("safety60 特征顺序与 official100 模型不一致")
            q_scores = canonicalize_predictions(TARGET, model.predict(matrix))
            native_count = masks.shape[1]
            native_q = q_scores[:native_count]
            track_q = q_scores[native_count:]
            track_original = np.asarray(
                [float(track["mean_node_quality"]) for track in tracks], dtype=np.float64
            )
            track_masks = np.zeros((masks.shape[0], len(tracks)), dtype=bool)
            for column_index, track in enumerate(tracks):
                points, _ = _load_track_points(track, masks.shape[0])
                track_masks[points, column_index] = True

            gt_file = str(args.gt_dir / f"{scene}.txt")
            native_gt, native_pred = instance_eval.assign_instances_for_scan(
                _prediction(masks, native_scores, native_count), gt_file
            )
            scene_native_original_map = _uuid_score_map(
                native_pred, native_scores, [int(row["point_count"]) for row in rows[:native_count]]
            )
            scene_native_q_map = _uuid_score_map(
                native_pred, native_q, [int(row["point_count"]) for row in rows[:native_count]]
            )
            native_original_uuid_scores.update(scene_native_original_map)
            native_q_uuid_scores.update(scene_native_q_map)
            native_matches[os.path.abspath(gt_file)] = {
                "gt": copy.deepcopy(native_gt), "pred": copy.deepcopy(native_pred)
            }

            track_gt, track_pred = instance_eval.assign_instances_for_scan(
                _prediction(track_masks, track_original, len(tracks)), gt_file
            )
            track_original_map = _uuid_score_map(
                track_pred, track_original, [int(track["point_count"]) for track in tracks]
            )
            track_q_map = _uuid_score_map(
                track_pred, track_q, [int(track["point_count"]) for track in tracks]
            )
            merged_gt, merged_pred = _merge_scan_matches(
                native_gt, native_pred, track_gt, track_pred
            )
            combined_matches[os.path.abspath(gt_file)] = {"gt": merged_gt, "pred": merged_pred}
            combined_original_uuid_scores.update(scene_native_original_map)
            combined_q_uuid_scores.update(scene_native_q_map)
            combined_original_uuid_scores.update(track_original_map)
            combined_q_uuid_scores.update(track_q_map)

            identity_rows.append(audit)
            safety_ledger_paths.append(args.safety_ledger_root / scene / "c1_gvc_quality_ledger.json")
            track_paths.append(args.track_root / scene / "automatic_tracks.json")
            for index, row in enumerate(rows):
                score_rows.append({
                    "scene_name": scene,
                    "candidate_source": row["candidate_source"],
                    "candidate_id": int(row["candidate_id"]),
                    "original_source_score": float(
                        native_scores[index] if index < native_count else track_original[index - native_count]
                    ),
                    "predicted_iou_score": float(q_scores[index]),
                    "native_class_id": int(native_classes[index]) if index < native_count else None,
                    "geometry_modified": False,
                })
            print(f"[身份与匹配] {scene_index}/60 {scene}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    _set_match_scores(native_matches, native_original_uuid_scores)
    native_original = _evaluate("native_original_score", native_matches, args.output_dir)
    assert_baseline(native_original, EXPECTED_NATIVE, args.baseline_tolerance)
    _set_match_scores(combined_matches, combined_original_uuid_scores)
    combined_original = _evaluate("native_plus_track_original_score", combined_matches, args.output_dir)
    assert_baseline(combined_original, EXPECTED_COMBINED, args.baseline_tolerance)
    print("冻结 AP 基线已精确复现，开始评估预注册预测质量分数。", flush=True)

    _set_match_scores(native_matches, native_q_uuid_scores)
    native_q = _evaluate("native_predicted_quality_score", native_matches, args.output_dir)
    _set_match_scores(combined_matches, combined_q_uuid_scores)
    combined_q = _evaluate("native_plus_track_predicted_quality_score", combined_matches, args.output_dir)

    _write_jsonl(args.output_dir / "candidate_scores.jsonl", score_rows)
    _write_jsonl(args.output_dir / "scene_identity_audit.jsonl", identity_rows)
    variants = {
        "native_original_score": native_original,
        "native_predicted_quality_score": native_q,
        "native_plus_track_original_score": combined_original,
        "native_plus_track_predicted_quality_score": combined_q,
    }
    summary = {
        "diagnostic_type": "一次性 safety60 类别无关 AP 重新排序实验；不是开放词汇正式结果",
        "preregistered_constraint": (
            "固定 mask、类别、候选数量和几何；仅用 official100 全量拟合的 "
            "D_plus_gvc 预测 IoU 替换分数；未扫描混合公式、阈值或候选动作"
        ),
        "training_safety_scene_overlap": overlap,
        "model_contract": model_contract,
        "identity_contract": {
            "candidate_geometry_modified": False,
            "candidate_count_modified": False,
            "candidate_classes_modified": False,
            "only_score_changed": True,
            "native_mask_element_digests_recorded": True,
            "track_point_index_digests_recorded": True,
            "native_count": sum(row["native_count"] for row in identity_rows),
            "track_count": sum(row["track_count"] for row in identity_rows),
        },
        "input_sha256": {
            "official_scene_list": _sha256(args.official_scene_list),
            "safety_scene_list": _sha256(args.safety_scene_list),
            "frozen_split_manifest": split_sha256,
            "safety_quality_ledgers_combined": _combined_file_digest(safety_ledger_paths),
            "automatic_tracks_combined": _combined_file_digest(track_paths),
        },
        "baseline_tolerance": args.baseline_tolerance,
        "baseline_reproduced": True,
        "variants": variants,
        "deltas": {
            "native_predicted_minus_original": {
                key: native_q[key] - native_original[key] for key in native_original
            },
            "combined_predicted_minus_original": {
                key: combined_q[key] - combined_original[key] for key in combined_original
            },
            "track_gain_with_original_scores": {
                key: combined_original[key] - native_original[key] for key in native_original
            },
            "track_gain_with_predicted_scores": {
                key: combined_q[key] - native_q[key] for key in native_q
            },
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-scene-list", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--safety-scene-list", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--safety-ledger-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260808)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-12)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics")
    for name in (
        "official_scene_list", "records_root", "split_manifest", "safety_scene_list",
        "native_cache", "safety_ledger_root", "track_root", "gt_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    required_files = (args.official_scene_list, args.split_manifest, args.safety_scene_list)
    required_dirs = (
        args.records_root, args.native_cache, args.safety_ledger_root,
        args.track_root, args.gt_dir,
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in required_dirs:
        if not path.is_dir():
            raise NotADirectoryError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，拒绝覆盖：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
