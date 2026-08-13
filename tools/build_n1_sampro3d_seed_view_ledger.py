#!/usr/bin/env python3
"""N1a paper-reference SAMPro3D seed/view ledger, without running SAM.

It keeps uniform30, frozen D1/D2b, and original ScanNet superpoints intact.
Each raw superpoint absent from frozen D2b is a no-GT 3D seed.  The ledger
records its most cross-view-visible point and all eligible RGB-D views, while
leaving any later SAM re-prompt, hypothesis retention, candidate family, and
AP oracle to separate stages.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CONTRACT = "N1a no-GT paper-reference seed/view ledger; it does not invoke SAM, create masks, modify proposals, score candidates, read native predictions, or compute AP."


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _scenes(path):
    rows = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or duplicated")
    return rows


def seed_view_rows(superpoints, occupied_ids, projections, visibility, scale_yx, min_visible_points):
    """Pure deterministic raw-superpoint seed/view construction for regression tests."""
    rows = []
    for superpoint_id in sorted(map(int, np.unique(superpoints))):
        if superpoint_id in occupied_ids:
            continue
        points = np.flatnonzero(superpoints == superpoint_id).astype(np.int64)
        visible = visibility[:, points]
        counts = visible.sum(axis=0)
        anchor = int(points[np.argmax(counts)])
        views = []
        for frame_index in np.flatnonzero(visibility[:, anchor]):
            visible_count = int(visible[frame_index].sum())
            if visible_count < min_visible_points:
                continue
            xy_depth = projections[frame_index, anchor]
            views.append({"frame_index": int(frame_index), "seed_xy": [float(xy_depth[0] / scale_yx[1]), float(xy_depth[1] / scale_yx[0])], "visible_seed_superpoint_point_count": visible_count})
        rows.append({"seed_superpoint_id": superpoint_id, "seed_point_index": anchor, "seed_superpoint_point_count": int(len(points)), "visible_view_count": len(views), "views": sorted(views, key=lambda r: (-r["visible_seed_superpoint_point_count"], r["frame_index"]))})
    return rows


def _scene(scene, args):
    from utils import WORLD_2_CAM
    processed = np.load(args.processed_scene_root / scene / f"{scene.replace('scene', '')}.npy", mmap_mode="r")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    tracks = json.loads((args.d2b_root / scene / "automatic_tracks.json").read_text())["tracks"]
    occupied = {int(item) for track in tracks for item in track["superpoint_ids"]}
    world = WORLD_2_CAM(str(args.dataset_root / scene), args.depth_scale, args.config)
    proj_raw, vis_raw = world.get_mesh_projections()
    projections = proj_raw.detach().cpu().numpy().astype(np.float64, copy=False)
    visibility = vis_raw.detach().cpu().numpy().astype(bool, copy=False)
    rows = seed_view_rows(superpoints, occupied, projections, visibility, (world.depth_resolution[0] / world.image_resolution[0], world.depth_resolution[1] / world.image_resolution[1]), args.min_visible_points)
    for row in rows:
        row.update({"scene_name": scene, "seed_origin": "raw_superpoint_absent_from_frozen_d2b", "ground_truth_usage": "none", "proposal_materialization_applied": False, "ap_computed": False, "decision_constraint": CONTRACT})
        for view in row["views"]:
            view["frame_id"] = Path(world.color_paths[view["frame_index"]]).stem
    return rows, {"scene_name": scene, "raw_superpoint_count": int(len(np.unique(superpoints))), "frozen_d2b_occupied_superpoint_count": len(occupied), "unclaimed_seed_count": len(rows), "seed_with_eligible_view_count": sum(bool(row["views"]) for row in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-list", type=Path, required=True); p.add_argument("--d2b-root", type=Path, required=True)
    p.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200")); p.add_argument("--dataset-root", type=Path, default=Path("data/scannet200")); p.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml")); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--min-visible-points", type=int, default=20); p.add_argument("--max-scenes", type=int)
    args = p.parse_args()
    for name in ("scene_list", "d2b_root", "processed_scene_root", "dataset_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()): raise SystemExit(f"output root is non-empty: {args.output_root}")
    if args.min_visible_points <= 0: raise SystemExit("--min-visible-points must be positive")
    args.config = yaml.safe_load(args.config_path.read_text()); args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _scenes(args.scene_list)[:args.max_scenes]
    args.output_root.mkdir(parents=True); summaries=[]
    try:
        for index, scene in enumerate(scenes, 1):
            rows, summary = _scene(scene, args); staging=args.output_root/f".{scene}.tmp.{os.getpid()}"; staging.mkdir()
            with (staging/"seed_view_ledger.jsonl").open("w") as h:
                for row in rows: h.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")
            (staging/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n"); os.replace(staging,args.output_root/scene); summaries.append(summary); print(f"[seed-view] {index}/{len(scenes)} {scene}: {len(rows)}",flush=True)
        payload={"diagnostic_type":"N1a SAMPro3D paper-reference no-GT raw-superpoint seed/view ledger","decision_constraint":CONTRACT,"ground_truth_usage":"none","proposal_materialization_applied":False,"ap_computed":False,"params":{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items() if k!='config'},"scene_summaries":summaries}
        (args.output_root/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    except Exception:
        for tmp in args.output_root.glob(".*.tmp.*"): shutil.rmtree(tmp)
        raise

if __name__ == "__main__": main()
