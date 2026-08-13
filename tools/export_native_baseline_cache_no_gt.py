#!/usr/bin/env python3
"""以冻结强基线配置导出 native 三维预测缓存，不读取 GT 或运行 AP。

输出严格复用 Mask3D + YOLO-World 投票、SAM-fused、BPR 和 native 分数，
只保存评测格式的 mask/class/score 数组，作为独立场景的几何关系输入；不调用
官方评测，不读取任何 ground truth。
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import OpenYolo3D
from utils.backprojection_fusion import append_backprojection_proposals, load_backprojection_candidates


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _save(scene_name, prediction, output_dir):
    masks, classes, scores = (_as_numpy(item) for item in prediction[:3])
    keep = np.asarray(scores >= 0.0, dtype=bool)
    np.save(output_dir / f"{scene_name}_pred_masks.npy", np.asarray(masks[:, keep], dtype=bool))
    np.save(output_dir / f"{scene_name}_pred_classes.npy", np.asarray(classes[keep], dtype=np.int64))
    np.save(output_dir / f"{scene_name}_pred_scores.npy", np.asarray(scores[keep], dtype=np.float32))
    return int(keep.sum())


def _scene_prediction(openyolo3d, scene_name, args, candidates_by_scene, depth_scale):
    processed_file = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    prediction = openyolo3d.predict(
        path_2_scene_data=str(args.dataset_root / scene_name), depth_scale=depth_scale,
        datatype="mesh", processed_scene=str(processed_file), path_to_3d_masks=str(args.mask_root),
        is_gt=False, path_to_2d_preds=str(args.bboxes_2d_root), save_2d_preds=False, reuse_2d_preds=True,
    )[scene_name]
    if args.mask3d_yoloworld_only:
        return prediction[:3]
    points_xyz, _ = openyolo3d.world2cam.load_ply(openyolo3d.world2cam.mesh)
    superpoints = np.asarray(np.load(processed_file, mmap_mode="r")[:, 9], dtype=np.int64)
    fused = append_backprojection_proposals(
        scene_name, prediction[0], prediction[1], prediction[2], candidates_by_scene,
        points_xyz=points_xyz[:, :3], point_segments=superpoints,
        min_score=0.50, min_seed_points=80, max_existing_iou=0.30,
        max_seed_in_existing_mask_ratio=0.70, max_proposal_iou=0.50,
        max_candidates=15, score_scale=2.00, use_candidate_fusion_score=False,
        blocked_classes={"rug"}, source_score_scales={"sam_fused": 1.2, "bpr": 1.0},
        source_priorities={"sam_fused": 2.0, "bpr": 1.0},
        source_max_candidates={"sam_fused": 12, "bpr": 3},
        superpoint_refine=True, superpoint_min_coverage=0.30,
        superpoint_max_expansion_ratio=3.0, superpoint_min_view_siou=0.60,
    )
    return fused[:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--mask_root", type=Path, default=Path("output/scannet200/scannet200_masks"))
    parser.add_argument("--bboxes_2d_root", type=Path, default=Path("output/scannet200/bboxes_2d"))
    parser.add_argument("--sam_fused_root", type=Path, default=Path("output/sam_fused_proposals_scannet200_s5_m30_prefilter"))
    parser.add_argument("--bpr_root", type=Path, default=Path("output/backprojection_candidates_scannet200_mv_m20"))
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_scene_count", type=int, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mask3d_yoloworld_only", action="store_true", help="仅导出 Mask3D+YOLO-World，不加载 SAM-fused 或 BPR；只用于无 GT 源归因对照。")
    parser.add_argument("--without_sam_fused", action="store_true", help="保留 BPR，仅移除 SAM-fused；只用于无 GT 的来源消融。")
    parser.add_argument("--without_bpr", action="store_true", help="保留 SAM-fused，仅移除 BPR；只用于无 GT 的来源消融。")
    args = parser.parse_args()
    # 与冻结 native 入口一致：仅允许读取项目已固定的旧版 YOLO-World 缓存，
    # 不重算、不改写二维检测。
    os.environ.setdefault("OPENYOLO3D_ALLOW_LEGACY_2D_CACHE", "1")
    for name in ("scene_list", "mask_root", "bboxes_2d_root", "sam_fused_root", "bpr_root", "dataset_root", "processed_scene_root", "config_path", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    if len(scenes) != args.expected_scene_count or len(scenes) != len(set(scenes)):
        raise SystemExit("场景列表数量不符或含重复场景。")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    missing = [scene for scene in scenes if not (args.mask_root / f"{scene}.pt").is_file() or not (args.bboxes_2d_root / f"{scene}.pt").is_file()]
    if missing:
        raise SystemExit(f"冻结基线输入不完整：{missing[:5]}")
    with args.config_path.open() as handle:
        config = yaml.safe_load(handle)
    depth_scale = float(config["openyolo3d"]["depth_scale"])
    source_modes = sum(bool(value) for value in (
        args.mask3d_yoloworld_only, args.without_sam_fused, args.without_bpr,
    ))
    if source_modes > 1:
        raise SystemExit("--mask3d_yoloworld_only、--without_sam_fused 与 --without_bpr 最多只能指定一个。")
    if args.mask3d_yoloworld_only:
        candidates, candidate_summary = {}, {"files": [], "loaded": 0, "mode": "mask3d_yoloworld_only"}
    elif args.without_sam_fused:
        candidates, candidate_summary = load_backprojection_candidates(str(args.bpr_root))
        candidate_summary["mode"] = "mask3d_yoloworld_plus_bpr_without_sam_fused"
    elif args.without_bpr:
        candidates, candidate_summary = load_backprojection_candidates(str(args.sam_fused_root))
        candidate_summary["mode"] = "mask3d_yoloworld_plus_sam_fused_without_bpr"
    else:
        candidates, candidate_summary = load_backprojection_candidates(f"{args.sam_fused_root},{args.bpr_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    openyolo3d = OpenYolo3D(str(args.config_path))
    exported = 0
    for index, scene_name in enumerate(scenes, start=1):
        paths = [args.output_dir / f"{scene_name}_pred_{suffix}.npy" for suffix in ("masks", "classes", "scores")]
        if args.resume and all(path.is_file() for path in paths):
            continue
        prediction = _scene_prediction(openyolo3d, scene_name, args, candidates, depth_scale)
        count = _save(scene_name, prediction, args.output_dir)
        exported += 1
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {count} 条 native 候选", flush=True)
        for attr in ("world2cam", "mesh_projections", "preds_3d", "preds_2d", "predicted_masks", "predicated_scores", "predicated_classes"):
            if hasattr(openyolo3d, attr):
                setattr(openyolo3d, attr, None)
        del prediction
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    candidate_inputs = {
        key: value for key, value in candidate_summary.items() if key != "files"
    }
    candidate_inputs["file_count"] = len(candidate_summary.get("files", []))
    payload = {
        "gt_usage": "none；不读取 GT，不运行 AP。",
        "decision_state": (
            "仅导出 Mask3D+YOLO-World，用于无 GT 的 SAM/BPR 源归因对照。"
            if args.mask3d_yoloworld_only else
            "仅导出 Mask3D+YOLO-World+BPR，移除 SAM-fused，用于无 GT 的来源消融。"
            if args.without_sam_fused else
            "仅导出 Mask3D+YOLO-World+SAM-fused，移除 BPR，用于无 GT 的来源消融。"
            if args.without_bpr else
            "仅导出冻结强基线的 native 预测缓存，未接入任何 CER/新候选逻辑。"
        ),
        "scene_count": len(scenes), "exported_this_run": exported,
        # ``load_backprojection_candidates`` 还会返回数千个具体文件路径；这些
        # 路径对可追溯性没有额外帮助，却会使运行日志异常庞大，因此只记录
        # 来源与文件数量。候选内容本身从未被修改。
        "candidate_inputs": candidate_inputs,
        "params": vars(args),
    }
    (args.output_dir / "native_cache_no_gt_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
