#!/usr/bin/env python3
"""构建 official100 组件动作的离线真实标签效用账本。

每次只改变一个基线候选—轨迹候选关系组件，其他组件固定为“共存”。
动作空间固定为：共存、仅保留基线候选、仅保留一条指定轨迹候选。
账本记录全局类别无关 AP 及各 IoU 阈值相对共存状态的变化，但不训练模型、
不选择阈值、不生成推理动作，也不修改任何候选文件。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    NATIVE_SOURCE,
    TRACK_SOURCE,
    candidate_ledger_path,
    read_jsonl,
    read_scene_list,
)
from tools.construct_c1c_global_feasible_ap_oracle_gt import (  # noqa: E402
    DIAGNOSTIC,
    OFFICIAL,
    _record_from_matches,
)
from tools.diagnose_d2b_native_track_ranking_oracle_gt import (  # noqa: E402
    _append,
    _average_precision,
    _empty_record,
)
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    UNIFIED_PREDICTED_CLASS,
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    MIN_REGION_SIZE,
    _load_track_points,
    _prediction,
)
from tools.evaluate_official100_geometry_group_ranking_oof_ap import (  # noqa: E402
    EXPECTED_OOF_SHA256,
    geometry_groups_and_audit,
    load_oof_predictions,
)


EPS = 1e-10
EXPECTED_SCENE_LIST_SHA256 = "dfa9017e206190eb2973b247c78e4bf1b2d9c01bb8468a15775c30335e44fb68"
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


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def _decision(delta: float) -> str:
    if delta > EPS:
        return "strict_positive"
    if delta < -EPS:
        return "harmful"
    return "neutral"


def relation_label_context(rows: list[dict]) -> dict:
    labels = [row.get("labels", {}) for row in rows]
    reliable = [bool(label.get("reliable_pair")) for label in labels]
    if reliable and all(reliable):
        state = "all_relations_reliable"
    elif any(reliable):
        state = "partially_reliable"
    else:
        state = "no_reliable_relation"
    return {
        "relation_count": len(rows),
        "reliable_relation_count": sum(reliable),
        "reliable_relation_fraction": float(sum(reliable) / max(1, len(rows))),
        "relation_label_reliability_state": state,
        "target_state_counts": dict(sorted(Counter(
            str(label.get("target_state", "missing")) for label in labels
        ).items())),
        "relative_quality_state_counts": dict(sorted(Counter(
            str(label.get("relative_quality_state", "missing")) for label in labels
        ).items())),
    }


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


def _fixed_records_without_scene(
    baseline_records: dict[str, dict], excluded_scene: str
) -> dict[str, dict]:
    result = {str(int(round(value * 100))): _empty_record() for value in DIAGNOSTIC}
    for scene, records in baseline_records.items():
        if scene == excluded_scene:
            continue
        for tag, values in records.items():
            _append(result[tag], values)
    return result


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


def _metric_delta(metrics: dict, baseline: dict) -> dict:
    result = {
        "official_ap": metrics["official_ap"] - baseline["official_ap"],
        "threshold_metrics": {},
    }
    for tag, row in metrics["threshold_metrics"].items():
        base = baseline["threshold_metrics"][tag]
        result["threshold_metrics"][tag] = {
            "ap": row["ap"] - base["ap"],
            "true_positive_count": row["true_positive_count"] - base["true_positive_count"],
            "false_negative_count": row["false_negative_count"] - base["false_negative_count"],
            "matched_gt_coverage": row["matched_gt_coverage"] - base["matched_gt_coverage"],
        }
    return result


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


def run(args: argparse.Namespace) -> dict:
    all_scenes = read_scene_list(args.scene_list)
    if len(all_scenes) != args.expected_scene_count:
        raise ValueError(
            f"固定协议要求 {args.expected_scene_count} 个场景，实际为 {len(all_scenes)}"
        )
    scene_list_sha = _sha256(args.scene_list)
    if args.expected_scene_count == 100 and scene_list_sha != EXPECTED_SCENE_LIST_SHA256:
        raise ValueError(f"official100 场景清单 SHA-256 不一致：{scene_list_sha}")
    scenes = all_scenes if args.max_scenes is None else all_scenes[:args.max_scenes]
    if not scenes:
        raise ValueError("烟测场景数量必须为正数")

    relation_summary_path = args.relation_feature_ledger_root / "summary.json"
    relation_summary = json.loads(relation_summary_path.read_text())
    if int(relation_summary["scene_count"]) != args.expected_scene_count:
        raise ValueError("关系特征账本场景数与冻结协议不一致")
    if relation_summary.get("feature_ground_truth_usage") != "none":
        raise ValueError("关系特征账本的特征字段不得使用真实标签")
    score_context = configure_track_score_context(args, all_scenes)

    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(
        original_load_ids(path)
    )
    caches, baseline_records = {}, {}
    try:
        for index, scene in enumerate(scenes, start=1):
            cache = _scene_inputs(scene, args)
            caches[scene] = cache
            baseline_records[scene] = _scene_records(
                cache, cache["all_native_representatives"], cache["all_track_ids"]
            )
            print(f"[固定共存上下文] {index}/{len(scenes)} {scene}", flush=True)

        empty_fixed = {str(int(round(value * 100))): _empty_record() for value in DIAGNOSTIC}
        all_baseline = {tag: _empty_record() for tag in empty_fixed}
        for records in baseline_records.values():
            for tag, values in records.items():
                _append(all_baseline[tag], values)
        # global_metrics 需要一个试验场景；这里把全部基线作为固定部分并追加空记录。
        empty_trial = {
            tag: (np.empty(0), np.empty(0), 0, False, False) for tag in empty_fixed
        }
        baseline_metrics = global_metrics(all_baseline, empty_trial)

        staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        scene_summaries = []
        action_kind_counts: Counter[str] = Counter()
        decision_counts: Counter[str] = Counter()
        best_action_kind_counts: Counter[str] = Counter()
        positive_component_scenes = set()
        component_count = 0
        action_count = 0
        try:
            for scene_index, scene in enumerate(scenes, start=1):
                cache = caches[scene]
                fixed = _fixed_records_without_scene(baseline_records, scene)
                scene_rows = []
                scene_component_rows = []
                for component in cache["components"]:
                    component_id = int(component["relation_component_id"])
                    context = relation_label_context(
                        cache["relations_by_component"][component_id]
                    )
                    evaluated = []
                    for action in build_component_actions(component):
                        kept_native, kept_tracks = _action_candidate_sets(
                            cache, component, action
                        )
                        trial_records = _scene_records(cache, kept_native, kept_tracks)
                        metrics = global_metrics(fixed, trial_records)
                        delta = _metric_delta(metrics, baseline_metrics)
                        action_kind = str(action["action_kind"])
                        row = {
                            "scene_name": scene,
                            **action,
                            "component_relation_label_context": context,
                            "component_kept_native_representative_candidate_ids": sorted(
                                int(cache["groups"][group_id]["representative_candidate_id"])
                                for group_id in action["kept_native_exact_geometry_group_ids"]
                            ),
                            "component_kept_track_ids": sorted(
                                int(value) for value in action["kept_track_ids"]
                            ),
                            "global_candidate_counts_after_action": {
                                "native_representative": len(kept_native),
                                "track": len(kept_tracks),
                            },
                            "candidate_count_change_vs_coexist": {
                                "native_representative": len(kept_native) - len(cache["all_native_representatives"]),
                                "track": len(kept_tracks) - len(cache["all_track_ids"]),
                            },
                            "labels": {
                                "ground_truth_usage": "official_train_offline_action_utility_only",
                                "global_metrics_after_action": metrics,
                                "delta_vs_fixed_coexist": delta,
                                "decision_vs_coexist": _decision(delta["official_ap"]),
                            },
                            "contracts": {
                                "other_components_fixed_to_coexist": True,
                                "global_ap_non_additive_warning": True,
                                "component_action_selected_for_inference": False,
                                "candidate_geometry_modified": False,
                                "candidate_files_modified": False,
                            },
                        }
                        evaluated.append(row)
                        action_kind_counts[action_kind] += 1
                        decision_counts[row["labels"]["decision_vs_coexist"]] += 1
                        action_count += 1
                    ordered = sorted(
                        evaluated,
                        key=lambda row: (
                            -float(row["labels"]["global_metrics_after_action"]["official_ap"]),
                            ACTION_TIE_PRIORITY[str(row["action_kind"])],
                            row["action_name"],
                        ),
                    )
                    best = ordered[0]
                    second = ordered[1] if len(ordered) > 1 else None
                    best_gain = float(best["labels"]["delta_vs_fixed_coexist"]["official_ap"])
                    if best_gain > EPS:
                        positive_component_scenes.add(scene)
                    best_action_kind_counts[str(best["action_kind"])] += 1
                    component_summary = {
                        "scene_name": scene,
                        "relation_component_id": component_id,
                        "relation_count": int(component["relation_count"]),
                        "native_geometry_group_count": int(component["native_geometry_group_count"]),
                        "track_count": int(component["track_count"]),
                        "relation_label_context": context,
                        "best_action_name": best["action_name"],
                        "best_action_name_zh": best["action_name_zh"],
                        "best_action_kind": best["action_kind"],
                        "best_official_ap_gain_vs_coexist": best_gain,
                        "best_minus_second_official_ap_margin": (
                            float(best["labels"]["global_metrics_after_action"]["official_ap"])
                            - float(second["labels"]["global_metrics_after_action"]["official_ap"])
                            if second is not None else None
                        ),
                        "coexist_is_best_or_tied": best_gain <= EPS,
                        "positive_non_coexist_action_exists": best_gain > EPS,
                        "action_count": len(evaluated),
                    }
                    for row in evaluated:
                        row["component_summary"] = component_summary
                        scene_rows.append(row)
                    scene_component_rows.append(component_summary)
                    component_count += 1

                scene_root = staging / scene
                scene_root.mkdir()
                _write_jsonl(scene_root / "component_action_utilities.jsonl", scene_rows)
                _write_jsonl(scene_root / "component_summaries.jsonl", scene_component_rows)
                scene_summary = {
                    "scene_name": scene,
                    "component_count": len(scene_component_rows),
                    "action_count": len(scene_rows),
                    "positive_non_coexist_component_count": sum(
                        row["positive_non_coexist_action_exists"] for row in scene_component_rows
                    ),
                    "native_candidate_count": len(cache["native_rows"]),
                    "native_exact_geometry_group_count": len(cache["groups"]),
                    "native_representative_count_in_fixed_context": len(cache["all_native_representatives"]),
                    "track_count_in_fixed_context": len(cache["all_track_ids"]),
                    "controlled_native_geometry_group_count": len(cache["controlled_group_ids"]),
                    "controlled_track_count": len(cache["controlled_track_ids"]),
                    "uncontrolled_track_count_kept_in_all_actions": len(
                        cache["all_track_ids"] - cache["controlled_track_ids"]
                    ),
                    "input_sha256": cache["input_sha256"],
                }
                (scene_root / "summary.json").write_text(
                    json.dumps(scene_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                )
                scene_summaries.append(scene_summary)
                print(
                    f"[动作效用] {scene_index}/{len(scenes)} {scene}: "
                    f"组件={len(scene_component_rows)}, 动作={len(scene_rows)}",
                    flush=True,
                )

            payload = {
                "version": "official100_component_action_utility_ledger_v1",
                "diagnostic_type": "official train GT-only fixed-coexist single-component global AP utility ledger",
                "action_space_zh": ["共存", "仅基线候选", "仅一条轨迹候选"],
                "score_context": score_context,
                "scene_count": len(scenes),
                "expected_full_scene_count": args.expected_scene_count,
                "is_smoke_subset": len(scenes) != args.expected_scene_count,
                "component_count": component_count,
                "action_count": action_count,
                "fixed_coexist_global_metrics": baseline_metrics,
                "action_kind_counts": dict(sorted(action_kind_counts.items())),
                "decision_counts_vs_coexist": dict(sorted(decision_counts.items())),
                "best_action_kind_counts": dict(sorted(best_action_kind_counts.items())),
                "scene_count_with_positive_non_coexist_component": len(positive_component_scenes),
                "official_ap_thresholds": list(OFFICIAL),
                "diagnostic_iou_thresholds": list(DIAGNOSTIC),
                "ground_truth_usage": "official_train_offline_action_utility_labels_only",
                "feature_ground_truth_usage": "none; relation features are joined by component identity",
                "global_ap_non_additive_warning": (
                    "每条效用只表示其他组件固定共存时的单组件反事实；多个正动作的效用不可直接相加"
                ),
                "selection_head_trained": False,
                "action_threshold_selected": False,
                "inference_action_generated": False,
                "candidate_files_modified": False,
                "safety60_evaluated": False,
                "input_provenance": {
                    "scene_list_path": str(args.scene_list.resolve()),
                    "scene_list_sha256": scene_list_sha,
                    "relation_feature_ledger_summary_path": str(relation_summary_path.resolve()),
                    "relation_feature_ledger_summary_sha256": _sha256(relation_summary_path),
                },
                "scene_summaries": scene_summaries,
                "params": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                    if key != "track_oof_scores"
                },
            }
            (staging / "summary.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            os.replace(staging, args.output_root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
        caches.clear()
        gc.collect()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--track-score-mode", choices=("original", "oof_quality"), default="original"
    )
    parser.add_argument("--oof-predictions", type=Path)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics")
    for name in (
        "scene_list", "records_root", "relation_feature_ledger_root", "gt_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.oof_predictions is not None:
        args.oof_predictions = _resolve(args.oof_predictions)
    if args.expected_scene_count < 1:
        raise ValueError("--expected-scene-count 必须为正数")
    if args.max_scenes is not None and args.max_scenes < 1:
        raise ValueError("--max-scenes 必须为正数")
    for path in (args.scene_list,):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.records_root, args.relation_feature_ledger_root, args.gt_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，拒绝覆盖：{args.output_root}")
    summary = run(args)
    print(json.dumps({
        key: summary[key] for key in (
            "scene_count", "component_count", "action_count",
            "decision_counts_vs_coexist", "best_action_kind_counts",
            "scene_count_with_positive_non_coexist_component",
        )
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
