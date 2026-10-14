#!/usr/bin/env python3
"""在 official100 上评估第一版完全相同几何掩码组感知折外排序。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
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
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    _merge_scan_matches,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    _evaluate,
    _load_track_points,
    _prediction,
    _set_match_scores,
    _sha256,
    _uuid_score_map,
)


EXPECTED_OOF_SHA256 = "a25f737e44da7d1409bbdc37f52c90a943d1ce4ab21901a0de1e974db2ca40ef"
EXPECTED_SPLIT_SHA256 = "710157cb78b66048a181f5e38701a274f24add6416a2f25627d9e499b0668c72"
TAIL_OFFSET = 2.0


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_oof_predictions(path: Path) -> dict[tuple[str, str, int], dict]:
    predictions = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            str(row["scene_name"]),
            str(row["candidate_source"]),
            int(row["candidate_id"]),
        )
        if key in predictions:
            raise ValueError(f"折外预测候选身份重复：{key}")
        value = float(row["predictions"]["D_plus_gvc"]["q"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"折外预测质量越界：{key}={value}")
        predictions[key] = {
            "q": value,
            "label_best_gt_iou": float(row["label_best_gt_iou"]),
            "label_valid_iou25": int(row["label_valid_iou25"]),
            "label_valid_iou50": int(row["label_valid_iou50"]),
        }
    if not predictions:
        raise ValueError("折外预测文件为空")
    return predictions


def geometry_groups_and_audit(
    masks: np.ndarray,
    point_counts: list[int],
    expected_group_sizes: list[int] | None,
) -> tuple[list[list[int]], dict]:
    """用有界内存计算每个基线候选的完全相同掩码分组。"""
    if masks.ndim != 2 or masks.shape[1] != len(point_counts):
        raise ValueError("基线掩码数量与候选账本不一致")
    if expected_group_sizes is not None and len(expected_group_sizes) != masks.shape[1]:
        raise ValueError("完全相同掩码组大小字段数量不一致")
    digests = [None] * masks.shape[1]
    actual_counts = np.empty(masks.shape[1], dtype=np.int64)
    whole_digest = hashlib.sha256()
    whole_digest.update(json.dumps({"shape": list(masks.shape), "dtype": str(masks.dtype)}).encode())
    for start in range(0, masks.shape[1], 16):
        stop = min(start + 16, masks.shape[1])
        block = np.asarray(masks[:, start:stop], dtype=bool)
        packed = np.packbits(block, axis=0)
        whole_digest.update(start.to_bytes(8, "little"))
        whole_digest.update(np.packbits(block, axis=None).tobytes())
        actual_counts[start:stop] = np.count_nonzero(block, axis=0)
        for column in range(stop - start):
            digests[start + column] = hashlib.sha256(
                np.ascontiguousarray(packed[:, column]).tobytes()
            ).hexdigest()
    if not np.array_equal(actual_counts, np.asarray(point_counts, dtype=np.int64)):
        bad = np.flatnonzero(actual_counts != np.asarray(point_counts, dtype=np.int64))[:5]
        raise ValueError(f"基线掩码点数与账本不一致：{bad.tolist()}")
    grouped = defaultdict(list)
    for candidate_id, digest in enumerate(digests):
        grouped[digest].append(candidate_id)
    groups = sorted(grouped.values(), key=lambda members: members[0])
    actual_group_sizes = np.empty(masks.shape[1], dtype=np.int64)
    for members in groups:
        actual_group_sizes[members] = len(members)
    if expected_group_sizes is not None and not np.array_equal(
        actual_group_sizes, np.asarray(expected_group_sizes, dtype=np.int64)
    ):
        bad = np.flatnonzero(
            actual_group_sizes != np.asarray(expected_group_sizes, dtype=np.int64)
        )[:5]
        raise ValueError(f"完全相同掩码组大小与训练账本不一致：{bad.tolist()}")
    return groups, {
        "candidate_count": masks.shape[1],
        "geometry_group_count": len(groups),
        "singleton_group_count": sum(len(members) == 1 for members in groups),
        "maximum_group_size": max(len(members) for members in groups),
        "canonical_mask_element_sha256": whole_digest.hexdigest(),
    }


def build_group_aware_scores(
    original_scores: np.ndarray,
    oof_quality_scores: np.ndarray,
    groups: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """固定每组代表，并构造原始分数与质量分数两个无调参版本。"""
    original_scores = np.asarray(original_scores, dtype=np.float64)
    oof_quality_scores = np.asarray(oof_quality_scores, dtype=np.float64)
    if original_scores.shape != oof_quality_scores.shape:
        raise ValueError("原始分数与折外质量分数数量不一致")
    if np.any(original_scores < 0.0) or np.any(original_scores > 1.0):
        raise ValueError("原始分数必须位于 [0,1]")
    original_variant = original_scores - TAIL_OFFSET
    quality_variant = original_scores - TAIL_OFFSET
    group_rows = []
    covered = []
    for group_index, members in enumerate(groups):
        members = sorted(int(value) for value in members)
        if not members:
            raise ValueError("完全相同掩码组不能为空")
        covered.extend(members)
        representative = min(
            members, key=lambda candidate_id: (-original_scores[candidate_id], candidate_id)
        )
        group_quality = float(np.median(oof_quality_scores[members]))
        original_variant[representative] = original_scores[representative]
        quality_variant[representative] = group_quality
        group_rows.append({
            "geometry_group_index": group_index,
            "member_candidate_ids": members,
            "group_size": len(members),
            "representative_candidate_id": representative,
            "representative_original_score": float(original_scores[representative]),
            "group_median_oof_quality": group_quality,
            "member_oof_quality_min": float(np.min(oof_quality_scores[members])),
            "member_oof_quality_max": float(np.max(oof_quality_scores[members])),
        })
    if sorted(covered) != list(range(len(original_scores))):
        raise ValueError("完全相同掩码组未恰好覆盖全部基线候选")
    return original_variant, quality_variant, group_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def run(args: argparse.Namespace) -> dict:
    split_sha = _sha256(args.split_manifest)
    oof_sha = _sha256(args.oof_predictions)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"冻结五折清单 SHA-256 不一致：{split_sha}")
    if oof_sha != EXPECTED_OOF_SHA256:
        raise ValueError(f"候选质量折外预测 SHA-256 不一致：{oof_sha}")
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("本实验固定要求 official100 的 100 个场景")
    oof = load_oof_predictions(args.oof_predictions)

    _configure_scannet200_instance_eval()
    baseline_matches, combined_matches = {}, {}
    score_maps = {
        "baseline_all_original": {},
        "baseline_group_original": {},
        "baseline_group_quality": {},
        "combined_all_original": {},
        "combined_group_original": {},
        "combined_baseline_quality_track_original": {},
        "combined_baseline_original_track_quality": {},
        "combined_group_quality": {},
    }
    identity_rows, group_rows_all, score_rows = [], [], []
    observed_oof_keys = set()

    original_load_ids = instance_eval.util_3d.load_ids

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    try:
        for scene_index, scene in enumerate(scenes, start=1):
            ledger_rows = read_jsonl(candidate_ledger_path(args.records_root, scene))
            baseline_rows = sorted(
                (row for row in ledger_rows if row["candidate_source"] == NATIVE_SOURCE),
                key=lambda row: int(row["candidate_id"]),
            )
            track_rows = sorted(
                (row for row in ledger_rows if row["candidate_source"] == TRACK_SOURCE),
                key=lambda row: int(row["candidate_id"]),
            )
            if len(baseline_rows) + len(track_rows) != len(ledger_rows):
                raise ValueError(f"{scene}: 候选账本含未知来源")
            if [int(row["candidate_id"]) for row in baseline_rows] != list(range(len(baseline_rows))):
                raise ValueError(f"{scene}: 基线候选编号不连续")

            cache = args.records_root / scene / "native_cache"
            masks = np.load(cache / f"{scene}_pred_masks.npy", mmap_mode="r")
            original_scores = np.asarray(
                np.load(cache / f"{scene}_pred_scores.npy", mmap_mode="r"), dtype=np.float64
            )
            classes = np.load(cache / f"{scene}_pred_classes.npy", mmap_mode="r")
            if masks.shape[1] != len(original_scores) or len(classes) != len(original_scores):
                raise ValueError(f"{scene}: 基线缓存维度不一致")
            ledger_original = np.asarray(
                [float(row["original_source_score"]) for row in baseline_rows], dtype=np.float64
            )
            if not np.allclose(original_scores, ledger_original, rtol=0.0, atol=1e-7):
                raise ValueError(f"{scene}: 基线缓存分数与训练账本不一致")

            groups, audit = geometry_groups_and_audit(
                masks,
                [int(row["point_count"]) for row in baseline_rows],
                [int(row["native_exact_geometry_group_size"]) for row in baseline_rows],
            )
            baseline_q = np.empty(len(baseline_rows), dtype=np.float64)
            for row in baseline_rows:
                key = (scene, NATIVE_SOURCE, int(row["candidate_id"]))
                if key not in oof:
                    raise ValueError(f"折外预测缺少基线候选：{key}")
                baseline_q[int(row["candidate_id"])] = oof[key]["q"]
                if not math.isclose(
                    float(row["label_best_gt_iou"]), oof[key]["label_best_gt_iou"],
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    raise ValueError(f"折外预测标签与候选账本不一致：{key}")
                observed_oof_keys.add(key)
            group_original, group_quality, scene_group_rows = build_group_aware_scores(
                original_scores, baseline_q, groups
            )
            for row in scene_group_rows:
                row["scene_name"] = scene
                labels = {
                    float(baseline_rows[candidate_id]["label_best_gt_iou"])
                    for candidate_id in row["member_candidate_ids"]
                }
                if len(labels) != 1:
                    raise ValueError(f"{scene}: 完全相同掩码组的 IoU 标签不一致")
                row["label_best_gt_iou"] = labels.pop()
            group_rows_all.extend(scene_group_rows)

            track_payload = json.loads(
                (args.records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json").read_text()
            )
            tracks = sorted(track_payload.get("tracks", []), key=lambda row: int(row["track_id"]))
            track_by_id = {int(track["track_id"]): track for track in tracks}
            track_ids = [int(row["candidate_id"]) for row in track_rows]
            if set(track_by_id) != set(track_ids) or len(track_by_id) != len(tracks):
                raise ValueError(f"{scene}: 轨迹候选与训练账本不一一对应")
            track_original = np.asarray(
                [float(track_by_id[track_id]["mean_node_quality"]) for track_id in track_ids],
                dtype=np.float64,
            )
            track_q = np.empty(len(track_rows), dtype=np.float64)
            for index, row in enumerate(track_rows):
                key = (scene, TRACK_SOURCE, int(row["candidate_id"]))
                if key not in oof:
                    raise ValueError(f"折外预测缺少轨迹候选：{key}")
                track_q[index] = oof[key]["q"]
                if not math.isclose(
                    float(row["original_source_score"]), track_original[index],
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    raise ValueError(f"{scene}: 轨迹原始分数合同不一致")
                observed_oof_keys.add(key)

            track_masks = np.zeros((masks.shape[0], len(tracks)), dtype=bool)
            for column_index, track_id in enumerate(track_ids):
                points, _ = _load_track_points(track_by_id[track_id], masks.shape[0])
                track_masks[points, column_index] = True

            gt_file = str(args.gt_dir / f"{scene}.txt")
            baseline_gt, baseline_pred = instance_eval.assign_instances_for_scan(
                _prediction(masks, original_scores, len(original_scores)), gt_file
            )
            point_counts = [int(row["point_count"]) for row in baseline_rows]
            scene_baseline_all = _uuid_score_map(baseline_pred, original_scores, point_counts)
            scene_baseline_group_original = _uuid_score_map(
                baseline_pred, group_original, point_counts
            )
            scene_baseline_group_quality = _uuid_score_map(
                baseline_pred, group_quality, point_counts
            )
            score_maps["baseline_all_original"].update(scene_baseline_all)
            score_maps["baseline_group_original"].update(scene_baseline_group_original)
            score_maps["baseline_group_quality"].update(scene_baseline_group_quality)
            baseline_matches[os.path.abspath(gt_file)] = {
                "gt": copy.deepcopy(baseline_gt), "pred": copy.deepcopy(baseline_pred)
            }

            track_gt, track_pred = instance_eval.assign_instances_for_scan(
                _prediction(track_masks, track_original, len(track_original)), gt_file
            )
            track_point_counts = [int(row["point_count"]) for row in track_rows]
            scene_track_original = _uuid_score_map(track_pred, track_original, track_point_counts)
            scene_track_quality = _uuid_score_map(track_pred, track_q, track_point_counts)
            merged_gt, merged_pred = _merge_scan_matches(
                baseline_gt, baseline_pred, track_gt, track_pred
            )
            combined_matches[os.path.abspath(gt_file)] = {"gt": merged_gt, "pred": merged_pred}
            score_maps["combined_all_original"].update(scene_baseline_all)
            score_maps["combined_all_original"].update(scene_track_original)
            score_maps["combined_group_original"].update(scene_baseline_group_original)
            score_maps["combined_group_original"].update(scene_track_original)
            score_maps["combined_baseline_quality_track_original"].update(
                scene_baseline_group_quality
            )
            score_maps["combined_baseline_quality_track_original"].update(scene_track_original)
            score_maps["combined_baseline_original_track_quality"].update(
                scene_baseline_group_original
            )
            score_maps["combined_baseline_original_track_quality"].update(scene_track_quality)
            score_maps["combined_group_quality"].update(scene_baseline_group_quality)
            score_maps["combined_group_quality"].update(scene_track_quality)

            representative_ids = {
                row["representative_candidate_id"] for row in scene_group_rows
            }
            for candidate_id, row in enumerate(baseline_rows):
                score_rows.append({
                    "scene_name": scene,
                    "candidate_source": "基线候选",
                    "candidate_id": candidate_id,
                    "is_geometry_group_representative": candidate_id in representative_ids,
                    "original_score": float(original_scores[candidate_id]),
                    "oof_quality": float(baseline_q[candidate_id]),
                    "group_original_variant_score": float(group_original[candidate_id]),
                    "group_quality_variant_score": float(group_quality[candidate_id]),
                })
            for index, row in enumerate(track_rows):
                score_rows.append({
                    "scene_name": scene,
                    "candidate_source": "轨迹候选",
                    "candidate_id": int(row["candidate_id"]),
                    "is_geometry_group_representative": True,
                    "original_score": float(track_original[index]),
                    "oof_quality": float(track_q[index]),
                    "group_original_variant_score": float(track_original[index]),
                    "group_quality_variant_score": float(track_q[index]),
                })
            audit.update({
                "scene_name": scene,
                "track_count": len(track_rows),
                "baseline_score_file_sha256": _sha256(cache / f"{scene}_pred_scores.npy"),
                "baseline_class_file_sha256": _sha256(cache / f"{scene}_pred_classes.npy"),
            })
            identity_rows.append(audit)
            print(f"[分组与匹配] {scene_index}/100 {scene}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    if observed_oof_keys != set(oof):
        missing = list(set(oof) - observed_oof_keys)[:5]
        extra = list(observed_oof_keys - set(oof))[:5]
        raise ValueError(f"折外预测身份未完整守恒：未使用={missing}，额外={extra}")

    results = {}
    for name in (
        "baseline_all_original", "baseline_group_original", "baseline_group_quality",
    ):
        _set_match_scores(baseline_matches, score_maps[name])
        results[name] = _evaluate(name, baseline_matches, args.output_dir)
    for name in (
        "combined_all_original",
        "combined_group_original",
        "combined_baseline_quality_track_original",
        "combined_baseline_original_track_quality",
        "combined_group_quality",
    ):
        _set_match_scores(combined_matches, score_maps[name])
        results[name] = _evaluate(name, combined_matches, args.output_dir)

    _write_jsonl(args.output_dir / "geometry_groups.jsonl", group_rows_all)
    _write_jsonl(args.output_dir / "candidate_scores.jsonl", score_rows)
    _write_jsonl(args.output_dir / "scene_identity_audit.jsonl", identity_rows)
    group_sizes = np.asarray([row["group_size"] for row in group_rows_all], dtype=np.int64)
    summary = {
        "实验类型": "official100 场景隔离折外类别无关 AP；第一版完全相同几何掩码组感知排序",
        "实验约束": (
            "不读取 safety60；不删除候选、不改变掩码/类别/数量/几何；每组代表由原始分数最高且编号最小固定决定；"
            "非代表仅进入固定分数尾部；质量分数只使用冻结 D_plus_gvc 折外预测的组内中位数"
        ),
        "场景数": len(scenes),
        "基线候选数": sum(row["candidate_count"] for row in identity_rows),
        "轨迹候选数": sum(row["track_count"] for row in identity_rows),
        "完全相同几何掩码组数": len(group_rows_all),
        "完全相同几何掩码组大小": {
            "最小": int(group_sizes.min()),
            "中位数": float(np.median(group_sizes)),
            "百分之九十": float(np.quantile(group_sizes, 0.9)),
            "最大": int(group_sizes.max()),
        },
        "输入摘要": {
            "五折清单_SHA256": split_sha,
            "候选质量折外预测_SHA256": oof_sha,
            "场景清单_SHA256": _sha256(args.scene_list),
        },
        "候选是否删除": False,
        "候选几何是否修改": False,
        "候选类别是否修改": False,
        "是否选择阈值或权重": False,
        "AP结果": results,
        "变化": {
            "基线_组代表原始分数_减_全部原始分数": {
                key: results["baseline_group_original"][key] - results["baseline_all_original"][key]
                for key in results["baseline_all_original"]
            },
            "基线_组质量分数_减_组代表原始分数": {
                key: results["baseline_group_quality"][key] - results["baseline_group_original"][key]
                for key in results["baseline_group_original"]
            },
            "组合_组代表原始分数_减_全部原始分数": {
                key: results["combined_group_original"][key] - results["combined_all_original"][key]
                for key in results["combined_all_original"]
            },
            "组合_组质量分数_减_组代表原始分数": {
                key: results["combined_group_quality"][key] - results["combined_group_original"][key]
                for key in results["combined_group_original"]
            },
            "组合_仅基线使用组质量_减_双方原始分数": {
                key: results["combined_baseline_quality_track_original"][key]
                - results["combined_group_original"][key]
                for key in results["combined_group_original"]
            },
            "组合_仅轨迹使用质量_减_双方原始分数": {
                key: results["combined_baseline_original_track_quality"][key]
                - results["combined_group_original"][key]
                for key in results["combined_group_original"]
            },
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "records_root", "oof_predictions", "gt_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    for path in (args.scene_list, args.split_manifest, args.oof_predictions):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.records_root, args.gt_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，拒绝覆盖：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
