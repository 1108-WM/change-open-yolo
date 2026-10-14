"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import hashlib

import json

import math

from pathlib import Path

import numpy as np

from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_FEATURES, TRACK_SOURCE

from tools.fi1_instance_eval_shared import UNIFIED_PREDICTED_CLASS, instance_eval

FEATURE_GROUP = "D_plus_gvc"

PROTOCOL_NAME = "official100_v2"

MIN_REGION_SIZE = 100

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
