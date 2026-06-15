import argparse
import json
import os
import os.path as osp
import sys
from collections import Counter, defaultdict

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from evaluate.replica.eval_semantic_instance import CLASS_LABELS, ID_TO_LABEL, PRED_ID_TO_ID, VALID_CLASS_IDS
from utils.backprojection_fusion import _load_seed_indices, load_backprojection_candidates


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _load_gt_instances(gt_path, min_region_size=100):
    gt_ids = np.loadtxt(gt_path, dtype=np.int64)
    instances = []
    valid_ids = set(int(item) for item in VALID_CLASS_IDS)
    for instance_id in np.unique(gt_ids):
        if instance_id == 0:
            continue
        label_id = int(instance_id // 1000)
        if label_id not in valid_ids:
            continue
        count = int((gt_ids == instance_id).sum())
        if count < min_region_size:
            continue
        instances.append(
            {
                "instance_id": int(instance_id),
                "label_id": label_id,
                "label_name": ID_TO_LABEL[label_id],
                "vert_count": count,
            }
        )
    counts = {item["instance_id"]: item["vert_count"] for item in instances}
    meta = {item["instance_id"]: item for item in instances}
    return gt_ids, instances, counts, meta


def _best_gt_for_indices(indices, gt_ids, gt_counts, gt_meta, class_id=None):
    if indices is None or len(indices) == 0:
        return {
            "best_iou": 0.0,
            "best_class_aware_iou": 0.0,
            "best_gt": None,
            "best_class_aware_gt": None,
        }
    candidate_size = int(len(indices))
    hit_ids, hit_counts = np.unique(gt_ids[indices], return_counts=True)

    best_iou = 0.0
    best_gt = None
    best_class_iou = 0.0
    best_class_gt = None
    target_label_id = None
    if class_id is not None:
        try:
            target_label_id = int(PRED_ID_TO_ID[int(class_id)])
        except Exception:
            target_label_id = None

    for instance_id, intersection in zip(hit_ids, hit_counts):
        instance_id = int(instance_id)
        if instance_id not in gt_counts:
            continue
        union = candidate_size + gt_counts[instance_id] - int(intersection)
        iou = float(intersection / max(1, union))
        gt = gt_meta[instance_id]
        if iou > best_iou:
            best_iou = iou
            best_gt = gt
        if target_label_id is not None and gt["label_id"] == target_label_id and iou > best_class_iou:
            best_class_iou = iou
            best_class_gt = gt

    return {
        "best_iou": best_iou,
        "best_class_aware_iou": best_class_iou,
        "best_gt": best_gt,
        "best_class_aware_gt": best_class_gt,
    }


def _summarize_records(records, num_gt):
    if not records:
        return {
            "num_candidates": 0,
            "mean_best_iou": 0.0,
            "recall_candidate_at_25": 0.0,
            "recall_candidate_at_50": 0.0,
            "class_aware_candidate_at_25": 0.0,
            "class_aware_candidate_at_50": 0.0,
            "gt_covered_at_25": 0,
            "gt_covered_at_50": 0,
            "gt_recall_at_25": 0.0,
            "gt_recall_at_50": 0.0,
        }
    best_ious = np.asarray([item["best_iou"] for item in records], dtype=np.float32)
    class_ious = np.asarray([item.get("best_class_aware_iou", 0.0) for item in records], dtype=np.float32)
    covered_25 = {
        item["best_gt"]["instance_id"]
        for item in records
        if item["best_gt"] is not None and item["best_iou"] >= 0.25
    }
    covered_50 = {
        item["best_gt"]["instance_id"]
        for item in records
        if item["best_gt"] is not None and item["best_iou"] >= 0.50
    }
    return {
        "num_candidates": len(records),
        "mean_best_iou": float(best_ious.mean()),
        "median_best_iou": float(np.median(best_ious)),
        "max_best_iou": float(best_ious.max()),
        "recall_candidate_at_25": float((best_ious >= 0.25).mean()),
        "recall_candidate_at_50": float((best_ious >= 0.50).mean()),
        "class_aware_candidate_at_25": float((class_ious >= 0.25).mean()),
        "class_aware_candidate_at_50": float((class_ious >= 0.50).mean()),
        "gt_covered_at_25": len(covered_25),
        "gt_covered_at_50": len(covered_50),
        "gt_recall_at_25": float(len(covered_25) / max(1, num_gt)),
        "gt_recall_at_50": float(len(covered_50) / max(1, num_gt)),
    }


def _load_mask_records(mask_path, scene_name, gt_ids, gt_counts, gt_meta):
    path = osp.join(mask_path, f"{scene_name}.pt")
    if not osp.exists(path):
        return []
    payload = torch.load(path, map_location="cpu")
    masks = _to_numpy(payload[0]).astype(bool)
    records = []
    for mask_id in range(masks.shape[1]):
        indices = np.flatnonzero(masks[:, mask_id])
        match = _best_gt_for_indices(indices, gt_ids, gt_counts, gt_meta)
        records.append(
            {
                "candidate_id": int(mask_id),
                "candidate_type": "mask3d",
                "num_points": int(len(indices)),
                **match,
            }
        )
    return records


def _load_bpr_records(candidates_by_scene, scene_name, num_points, gt_ids, gt_counts, gt_meta):
    records = []
    for candidate in candidates_by_scene.get(scene_name, []):
        indices = _load_seed_indices(candidate, num_points)
        if indices is None:
            continue
        match = _best_gt_for_indices(
            indices,
            gt_ids,
            gt_counts,
            gt_meta,
            class_id=candidate.get("class_id"),
        )
        records.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "candidate_type": "bpr",
                "class_id": candidate.get("class_id"),
                "class_name": candidate.get("class_name"),
                "score": candidate.get("score"),
                "fusion_score": candidate.get("fusion_score"),
                "support_view_count": candidate.get("support_view_count", 1),
                "num_points": int(len(indices)),
                "source_json": candidate.get("_source_json"),
                **match,
            }
        )
    return records


def _class_breakdown(records):
    by_class = defaultdict(list)
    for item in records:
        class_name = item.get("class_name")
        if class_name is None and item.get("best_gt") is not None:
            class_name = item["best_gt"]["label_name"]
        by_class[class_name or "unknown"].append(item)
    return {name: _summarize_records(items, num_gt=1) for name, items in sorted(by_class.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica"])
    parser.add_argument("--gt_dir", default="./data/replica/ground_truth")
    parser.add_argument("--scene_names", default=None, help="Comma-separated scene names; default: all GT txt files")
    parser.add_argument("--mask3d_path", default="./output/replica/replica_masks")
    parser.add_argument("--bpr_candidates", default=None, help="Backprojection candidate directory or JSON")
    parser.add_argument("--sam_candidates", default=None, help="Optional SAM-refined candidate directory or JSON")
    parser.add_argument("--min_region_size", default=100, type=int)
    parser.add_argument("--output_json", default="./output/oracle_diagnostics/replica_oracle_diagnostics.json")
    args = parser.parse_args()

    if args.scene_names:
        scene_names = [item.strip() for item in args.scene_names.split(",") if item.strip()]
    else:
        scene_names = sorted(osp.splitext(name)[0] for name in os.listdir(args.gt_dir) if name.endswith(".txt"))

    sources = {}
    if args.mask3d_path:
        sources["mask3d"] = {"type": "mask3d", "path": args.mask3d_path}
    if args.bpr_candidates:
        candidates, summary = load_backprojection_candidates(args.bpr_candidates)
        sources["bpr"] = {"type": "bpr", "path": args.bpr_candidates, "candidates": candidates, "summary": summary}
    if args.sam_candidates:
        candidates, summary = load_backprojection_candidates(args.sam_candidates)
        sources["sam"] = {"type": "bpr", "path": args.sam_candidates, "candidates": candidates, "summary": summary}

    report = {
        "dataset_name": args.dataset_name,
        "scene_names": scene_names,
        "sources": {name: {k: v for k, v in src.items() if k != "candidates"} for name, src in sources.items()},
        "scenes": {},
        "summary": {},
    }
    all_records_by_source = defaultdict(list)
    total_gt = 0

    for scene_name in scene_names:
        gt_path = osp.join(args.gt_dir, f"{scene_name}.txt")
        gt_ids, gt_instances, gt_counts, gt_meta = _load_gt_instances(gt_path, args.min_region_size)
        total_gt += len(gt_instances)
        scene_report = {
            "num_points": int(len(gt_ids)),
            "num_gt_instances": len(gt_instances),
            "gt_class_counts": dict(Counter(item["label_name"] for item in gt_instances)),
            "sources": {},
        }
        for source_name, source in sources.items():
            if source["type"] == "mask3d":
                records = _load_mask_records(source["path"], scene_name, gt_ids, gt_counts, gt_meta)
            else:
                records = _load_bpr_records(source["candidates"], scene_name, len(gt_ids), gt_ids, gt_counts, gt_meta)
            all_records_by_source[source_name].extend(records)
            scene_report["sources"][source_name] = {
                "summary": _summarize_records(records, len(gt_instances)),
                "records": records,
            }
        report["scenes"][scene_name] = scene_report

    for source_name, records in all_records_by_source.items():
        report["summary"][source_name] = {
            "overall": _summarize_records(records, total_gt),
            "class_breakdown": _class_breakdown(records),
        }

    output_dir = osp.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved oracle diagnostics to {args.output_json}")
    print(f"Total GT instances: {total_gt}")
    for source_name, source_summary in report["summary"].items():
        s = source_summary["overall"]
        print(
            f"{source_name}: n={s['num_candidates']} "
            f"meanIoU={s['mean_best_iou']:.3f} "
            f"cand@25={s['recall_candidate_at_25']:.3f} "
            f"cand@50={s['recall_candidate_at_50']:.3f} "
            f"class@25={s['class_aware_candidate_at_25']:.3f} "
            f"class@50={s['class_aware_candidate_at_50']:.3f} "
            f"gt@25={s['gt_recall_at_25']:.3f} "
            f"gt@50={s['gt_recall_at_50']:.3f}"
        )


if __name__ == "__main__":
    main()
