"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import hashlib

import json

from pathlib import Path

import numpy as np

from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE, candidate_ledger_path, read_jsonl

from tools.fi1_instance_eval_shared import DIAGNOSTIC, OFFICIAL, _record_from_matches

from tools.fi1_instance_eval_shared import _append, _average_precision

from tools.fi1_instance_eval_shared import instance_eval

from tools.fi1_shared.evaluate_candidate_quality_reranking_class_agnostic_ap import MIN_REGION_SIZE, _load_track_points, _prediction

from tools.fi1_shared.evaluate_official100_geometry_group_ranking_oof_ap import EXPECTED_OOF_SHA256, geometry_groups_and_audit, load_oof_predictions

EPS = 1e-10

ACTION_ZH = {
    "coexist": "共存",
    "baseline_only": "仅基线候选",
    "track_only_one": "仅一条轨迹候选",
}

ACTION_TIE_PRIORITY = {
    "coexist": 0,
    "baseline_only": 1,
    "track_only_one": 2,
}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))

def configure_track_score_context(args, scenes: list[str]) -> dict:
    """配置固定轨迹分数上下文；折外质量只用于评测排序，不进入动作特征。"""
    mode = getattr(args, "track_score_mode", "original")
    if mode == "original":
        args.track_oof_scores = {}
        return {
            "track_score_mode": "original",
            "oof_predictions_path": None,
            "oof_predictions_sha256": None,
        }
    if mode != "oof_quality":
        raise ValueError(f"未知轨迹分数上下文：{mode}")
    path = getattr(args, "oof_predictions", None)
    if path is None or not Path(path).is_file():
        raise ValueError("轨迹折外质量分数上下文必须提供 --oof-predictions")
    digest = _sha256(Path(path))
    if digest != EXPECTED_OOF_SHA256:
        raise ValueError(f"候选质量折外预测 SHA-256 不一致：{digest}")
    scores = load_oof_predictions(Path(path))
    expected_scene_set = set(scenes)
    observed_track_scenes = {
        scene for scene, source, _ in scores if source == TRACK_SOURCE
    }
    if observed_track_scenes != expected_scene_set:
        raise ValueError("轨迹折外质量分数场景集合与当前场景清单不一致")
    args.track_oof_scores = scores
    return {
        "track_score_mode": "oof_quality",
        "oof_predictions_path": str(Path(path).resolve()),
        "oof_predictions_sha256": digest,
    }

def _read_components(root: Path, scene: str) -> list[dict]:
    rows = read_jsonl(root / scene / "relation_components.jsonl")
    rows = sorted(rows, key=lambda row: int(row["relation_component_id"]))
    ids = [int(row["relation_component_id"]) for row in rows]
    if ids != list(range(len(rows))):
        raise ValueError(f"{scene}: 关系组件编号不连续")
    return rows

def build_component_actions(component: dict) -> list[dict]:
    """按冻结三动作合同生成一个关系组件的候选动作。"""
    component_id = int(component["relation_component_id"])
    group_ids = sorted(str(value) for value in component["native_exact_geometry_group_ids"])
    track_ids = sorted(int(value) for value in component["track_ids"])
    if not group_ids or not track_ids:
        raise ValueError(f"组件 {component_id} 必须同时含基线几何组与轨迹候选")
    base = {
        "relation_component_id": component_id,
        "component_native_exact_geometry_group_ids": group_ids,
        "component_track_ids": track_ids,
    }
    rows = [{
        **base,
        "action_name": "coexist",
        "action_name_zh": ACTION_ZH["coexist"],
        "action_kind": "coexist",
        "selected_track_id": None,
        "kept_native_exact_geometry_group_ids": group_ids,
        "kept_track_ids": track_ids,
    }, {
        **base,
        "action_name": "baseline_only",
        "action_name_zh": ACTION_ZH["baseline_only"],
        "action_kind": "baseline_only",
        "selected_track_id": None,
        "kept_native_exact_geometry_group_ids": group_ids,
        "kept_track_ids": [],
    }]
    rows.extend({
        **base,
        "action_name": f"track_only_one:{track_id}",
        "action_name_zh": f"{ACTION_ZH['track_only_one']}：{track_id}",
        "action_kind": "track_only_one",
        "selected_track_id": track_id,
        "kept_native_exact_geometry_group_ids": [],
        "kept_track_ids": [track_id],
    } for track_id in track_ids)
    return rows

def select_candidate_sets(
    all_native_representatives: set[int],
    all_track_ids: set[int],
    component_native_representatives: set[int],
    component_track_ids: set[int],
    action_native_representatives: set[int],
    action_track_ids: set[int],
) -> tuple[set[int], set[int]]:
    """保持组件外候选不变，只替换当前组件的保留集合。"""
    if not action_native_representatives <= component_native_representatives:
        raise ValueError("动作保留了组件外基线候选")
    if not action_track_ids <= component_track_ids:
        raise ValueError("动作保留了组件外轨迹候选")
    kept_native = (
        all_native_representatives - component_native_representatives
    ) | action_native_representatives
    kept_tracks = (all_track_ids - component_track_ids) | action_track_ids
    return kept_native, kept_tracks

def _canonical_native_groups(
    scene: str,
    masks: np.ndarray,
    native_rows: list[dict],
    original_scores: np.ndarray,
) -> tuple[dict[str, dict], dict]:
    groups, audit = geometry_groups_and_audit(
        masks,
        [int(row["point_count"]) for row in native_rows],
        [int(row["native_exact_geometry_group_size"]) for row in native_rows],
    )
    result = {}
    for index, members in enumerate(groups):
        group_id = f"{scene}:native_geometry:{index:04d}"
        representative = min(
            members, key=lambda candidate_id: (-float(original_scores[candidate_id]), candidate_id)
        )
        result[group_id] = {
            "member_candidate_ids": [int(value) for value in members],
            "representative_candidate_id": int(representative),
        }
    return result, audit

def _scene_inputs(scene: str, args) -> dict:
    ledger_rows = read_jsonl(candidate_ledger_path(args.records_root, scene))
    native_rows = sorted(
        (row for row in ledger_rows if row["candidate_source"] == NATIVE_SOURCE),
        key=lambda row: int(row["candidate_id"]),
    )
    track_rows = sorted(
        (row for row in ledger_rows if row["candidate_source"] == TRACK_SOURCE),
        key=lambda row: int(row["candidate_id"]),
    )
    if len(native_rows) + len(track_rows) != len(ledger_rows):
        raise ValueError(f"{scene}: 候选账本含未知来源")
    if [int(row["candidate_id"]) for row in native_rows] != list(range(len(native_rows))):
        raise ValueError(f"{scene}: 基线候选编号不连续")

    cache_root = args.records_root / scene / "native_cache"
    masks = np.load(cache_root / f"{scene}_pred_masks.npy", mmap_mode="r")
    original_scores = np.asarray(
        np.load(cache_root / f"{scene}_pred_scores.npy", mmap_mode="r"), dtype=np.float64
    )
    if masks.shape[1] != len(native_rows) or len(original_scores) != len(native_rows):
        raise ValueError(f"{scene}: 基线缓存与候选账本数量不一致")
    if not np.allclose(
        original_scores,
        np.asarray([float(row["original_source_score"]) for row in native_rows]),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(f"{scene}: 基线原始分数与候选账本不一致")
    groups, group_audit = _canonical_native_groups(
        scene, masks, native_rows, original_scores
    )

    track_path = (
        args.records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
    )
    tracks = json.loads(track_path.read_text()).get("tracks", [])
    track_by_id = {int(row["track_id"]): row for row in tracks}
    if len(track_by_id) != len(tracks):
        raise ValueError(f"{scene}: 轨迹编号重复")
    expected_track_ids = {int(row["candidate_id"]) for row in track_rows}
    if set(track_by_id) != expected_track_ids:
        raise ValueError(f"{scene}: 轨迹文件与候选账本不一一对应")

    components = _read_components(args.relation_feature_ledger_root, scene)
    relation_rows = read_jsonl(
        args.relation_feature_ledger_root / scene / "relation_features.jsonl"
    )
    relations_by_component: dict[int, list[dict]] = {}
    for row in relation_rows:
        relations_by_component.setdefault(int(row["relation_component_id"]), []).append(row)
    if set(relations_by_component) != {int(row["relation_component_id"]) for row in components}:
        raise ValueError(f"{scene}: 关系记录与关系组件不一致")

    controlled_groups: dict[str, int] = {}
    controlled_tracks: dict[int, int] = {}
    for component in components:
        component_id = int(component["relation_component_id"])
        for group_id in component["native_exact_geometry_group_ids"]:
            if group_id not in groups:
                raise ValueError(f"{scene}: 关系组件引用未知基线几何组 {group_id}")
            previous = controlled_groups.setdefault(str(group_id), component_id)
            if previous != component_id:
                raise ValueError(f"{scene}: 基线几何组跨组件重复")
        for track_id in component["track_ids"]:
            track_id = int(track_id)
            if track_id not in track_by_id:
                raise ValueError(f"{scene}: 关系组件引用未知轨迹 {track_id}")
            previous = controlled_tracks.setdefault(track_id, component_id)
            if previous != component_id:
                raise ValueError(f"{scene}: 轨迹跨组件重复")
        relation_group_ids = {
            str(row["native_exact_geometry_group_id"])
            for row in relations_by_component[component_id]
        }
        relation_track_ids = {
            int(row["track_id"]) for row in relations_by_component[component_id]
        }
        if relation_group_ids != set(component["native_exact_geometry_group_ids"]):
            raise ValueError(f"{scene}: 组件基线几何组未被关系记录完整覆盖")
        if relation_track_ids != {int(value) for value in component["track_ids"]}:
            raise ValueError(f"{scene}: 组件轨迹未被关系记录完整覆盖")
        for row in relations_by_component[component_id]:
            expected_members = groups[str(row["native_exact_geometry_group_id"])][
                "member_candidate_ids"
            ]
            if [int(value) for value in row["native_member_candidate_ids"]] != expected_members:
                raise ValueError(f"{scene}: 关系记录的完全相同掩码组成员不一致")

    representative_ids = sorted(
        group["representative_candidate_id"] for group in groups.values()
    )
    track_ids = sorted(track_by_id)
    track_masks = np.zeros((masks.shape[0], len(track_ids)), dtype=bool)
    track_scores = np.zeros(len(track_ids), dtype=np.float64)
    track_point_counts = []
    for column, track_id in enumerate(track_ids):
        points, _ = _load_track_points(track_by_id[track_id], masks.shape[0])
        track_masks[points, column] = True
        if getattr(args, "track_score_mode", "original") == "oof_quality":
            key = (scene, TRACK_SOURCE, track_id)
            if key not in args.track_oof_scores:
                raise ValueError(f"{scene}: 缺少轨迹候选折外质量分数 {track_id}")
            track_scores[column] = float(args.track_oof_scores[key]["q"])
        else:
            track_scores[column] = float(track_by_id[track_id]["mean_node_quality"])
        track_point_counts.append(len(points))

    combined_masks = np.concatenate(
        [np.asarray(masks[:, representative_ids], dtype=bool), track_masks], axis=1
    )
    combined_scores = np.concatenate([original_scores[representative_ids], track_scores])
    gt, pred = instance_eval.assign_instances_for_scan(
        _prediction(combined_masks, combined_scores, len(combined_scores)),
        str(args.gt_dir / f"{scene}.txt"),
    )
    native_valid = [
        candidate_id for candidate_id in representative_ids
        if int(native_rows[candidate_id]["point_count"]) >= MIN_REGION_SIZE
    ]
    track_valid = [
        track_id for track_id, count in zip(track_ids, track_point_counts)
        if count >= MIN_REGION_SIZE
    ]
    expected_keys = [*(('native', value) for value in native_valid),
                     *(('track', value) for value in track_valid)]
    if len(pred["chair"]) != len(expected_keys):
        raise ValueError(f"{scene}: 评测器接纳候选数量与点数合同不一致")
    uuid_by_candidate = {
        key: row["uuid"] for key, row in zip(expected_keys, pred["chair"])
    }
    return {
        "native_rows": native_rows,
        "groups": groups,
        "group_audit": group_audit,
        "track_by_id": track_by_id,
        "components": components,
        "relations_by_component": relations_by_component,
        "all_native_representatives": set(representative_ids),
        "all_track_ids": set(track_ids),
        "controlled_group_ids": set(controlled_groups),
        "controlled_track_ids": set(controlled_tracks),
        "gt": gt,
        "pred": pred,
        "uuid_by_candidate": uuid_by_candidate,
        "input_sha256": {
            "candidate_labels": _sha256(candidate_ledger_path(args.records_root, scene)),
            "native_masks": _sha256(cache_root / f"{scene}_pred_masks.npy"),
            "native_scores": _sha256(cache_root / f"{scene}_pred_scores.npy"),
            "filtered_tracks": _sha256(track_path),
            "relation_features": _sha256(
                args.relation_feature_ledger_root / scene / "relation_features.jsonl"
            ),
            "relation_components": _sha256(
                args.relation_feature_ledger_root / scene / "relation_components.jsonl"
            ),
            "ground_truth": _sha256(args.gt_dir / f"{scene}.txt"),
        },
    }

def _scene_records(cache: dict, kept_native: set[int], kept_tracks: set[int]) -> dict:
    uuid_by_candidate = cache["uuid_by_candidate"]
    kept_uuid = {
        uuid_by_candidate[("native", candidate_id)]
        for candidate_id in kept_native if ("native", candidate_id) in uuid_by_candidate
    }
    kept_uuid.update(
        uuid_by_candidate[("track", track_id)]
        for track_id in kept_tracks if ("track", track_id) in uuid_by_candidate
    )
    pred = {
        "chair": [row for row in cache["pred"]["chair"] if row["uuid"] in kept_uuid]
    }
    gt = {
        "chair": [
            dict(row, matched_pred=[
                match for match in row["matched_pred"] if match["uuid"] in kept_uuid
            ])
            for row in cache["gt"]["chair"]
        ]
    }
    return {
        str(int(round(threshold * 100))): _record_from_matches(gt, pred, threshold)
        for threshold in DIAGNOSTIC
    }

def global_metrics(fixed_records: dict[str, dict], trial_records: dict) -> dict:
    """将固定场景记录与一个试验场景合并，计算全局 AP 和匹配覆盖。"""
    threshold_metrics = {}
    for threshold in DIAGNOSTIC:
        tag = str(int(round(threshold * 100)))
        total = {
            "true": list(fixed_records[tag]["true"]),
            "score": list(fixed_records[tag]["score"]),
            "fn": int(fixed_records[tag]["fn"]),
            "has_gt": bool(fixed_records[tag]["has_gt"]),
            "has_pred": bool(fixed_records[tag]["has_pred"]),
        }
        _append(total, trial_records[tag])
        ap = _average_precision(
            total["true"], total["score"], total["fn"],
            total["has_gt"], total["has_pred"],
        )
        true_values = np.concatenate(total["true"]) if total["true"] else np.empty(0)
        true_positive_count = int(np.rint(float(true_values.sum())))
        false_negative_count = int(total["fn"])
        threshold_metrics[tag] = {
            "iou_threshold": float(threshold),
            "ap": float(ap),
            "true_positive_count": true_positive_count,
            "false_negative_count": false_negative_count,
            "matched_gt_coverage": float(
                true_positive_count / max(1, true_positive_count + false_negative_count)
            ),
        }
    official_tags = [str(int(round(value * 100))) for value in OFFICIAL]
    return {
        "official_ap": float(np.mean([
            threshold_metrics[tag]["ap"] for tag in official_tags
        ])),
        "threshold_metrics": threshold_metrics,
    }

def _action_candidate_sets(cache: dict, component: dict, action: dict) -> tuple[set[int], set[int]]:
    component_groups = [str(value) for value in component["native_exact_geometry_group_ids"]]
    component_native = {
        int(cache["groups"][group_id]["representative_candidate_id"])
        for group_id in component_groups
    }
    action_native = {
        int(cache["groups"][group_id]["representative_candidate_id"])
        for group_id in action["kept_native_exact_geometry_group_ids"]
    }
    component_tracks = {int(value) for value in component["track_ids"]}
    action_tracks = {int(value) for value in action["kept_track_ids"]}
    return select_candidate_sets(
        cache["all_native_representatives"], cache["all_track_ids"],
        component_native, component_tracks, action_native, action_tracks,
    )
