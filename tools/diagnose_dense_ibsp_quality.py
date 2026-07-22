#!/usr/bin/env python3
"""Create non-GT visual and structural diagnostics for dense IBSp experiments."""

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _label_colors(labels):
    labels = labels.astype(np.uint64)
    colors = np.empty((*labels.shape, 3), dtype=np.uint8)
    colors[..., 0] = ((labels * 97 + 61) & 255).astype(np.uint8)
    colors[..., 1] = ((labels * 57 + 131) & 255).astype(np.uint8)
    colors[..., 2] = ((labels * 173 + 19) & 255).astype(np.uint8)
    colors[labels == 0] = 0
    return colors


def _overlay_labels(image, labels, alpha=0.55):
    image = image[..., :3].astype(np.float32)
    if labels.shape != image.shape[:2]:
        raise ValueError(f"Image/label-map shape mismatch: {image.shape} vs {labels.shape}")
    colors = _label_colors(labels).astype(np.float32)
    foreground = labels > 0
    output = image.copy()
    output[foreground] = (1.0 - alpha) * image[foreground] + alpha * colors[foreground]
    return np.clip(output, 0, 255).astype(np.uint8)


def _scene_rows(dense_scenes, ibsp_scenes, refined_root):
    ibsp_by_scene = {item["scene_name"]: item for item in ibsp_scenes}
    rows = []
    instance_payloads = {}
    for dense in dense_scenes:
        scene_name = dense["scene_name"]
        refined_path = Path(refined_root) / scene_name / "refined_instances.json"
        refined = _load_json(refined_path)
        instance_payloads[scene_name] = refined
        diagnostics = refined["diagnostics"]
        instances = refined["instances"]
        boundary = ibsp_by_scene[scene_name].get("boundary", {})
        rows.append(
            {
                "scene_name": scene_name,
                "frame_count": int(dense["frame_count"]),
                "observation_count": int(dense["observation_count"]),
                "mean_masks_per_frame": float(dense["mean_masks_per_frame"]),
                "mean_label_coverage": float(dense["mean_label_coverage"]),
                "observed_edges": int(boundary.get("edges_with_boundary_observation", 0)),
                "conflict_edges": int(boundary.get("edges_with_conflict", 0)),
                "pruned_edges": int(boundary.get("pruned_edges", 0)),
                "pruned_edge_ratio": float(boundary.get("pruned_edge_ratio", 0.0)),
                "reliable_observations": int(diagnostics["reliable_observation_count"]),
                "raw_tracks": int(diagnostics["raw_track_count"]),
                "duplicate_tracks_removed": int(diagnostics["duplicate_track_count"]),
                "ambiguous_superpoints_removed": int(diagnostics["ambiguous_superpoint_count"]),
                "output_instances": int(diagnostics["output_instance_count"]),
                "multi_view_instances": int(sum(item["frame_count"] >= 2 for item in instances)),
                "single_frame_instances": int(sum(item["frame_count"] <= 1 for item in instances)),
                "largest_instance_points": int(max([item["point_count"] for item in instances] or [0])),
            }
        )
    return rows, instance_payloads


def select_review_scenes(rows, low_coverage_count, high_instance_count):
    low_coverage = sorted(rows, key=lambda row: (row["mean_label_coverage"], row["scene_name"]))[:low_coverage_count]
    high_instances = sorted(
        rows, key=lambda row: (-row["output_instances"], -row["single_frame_instances"], row["scene_name"])
    )[:high_instance_count]
    selected = {}
    for row in low_coverage:
        selected.setdefault(row["scene_name"], {"row": row, "reasons": []})["reasons"].append("low_coverage")
    for row in high_instances:
        selected.setdefault(row["scene_name"], {"row": row, "reasons": []})["reasons"].append("high_instance_count")
    return [selected[name] for name in sorted(selected)]


def _best_frame(scene_summary):
    frames = scene_summary.get("frames", [])
    if not frames:
        return None
    return max(frames, key=lambda item: (float(item["label_coverage"]), int(item["kept_masks"])))


def _find_color_path(data_root, scene_name, frame_id):
    color_dir = Path(data_root) / scene_name / "color"
    candidates = sorted(color_dir.glob(f"{frame_id}.*"))
    return candidates[0] if candidates else None


def run(args):
    dense_root = Path(args.dense_root)
    dense_payload = _load_json(dense_root / "dense_frame_instance_observations_summary.json")
    ibsp_payload = _load_json(Path(args.ibsp_root) / "geometric_superpoints_summary.json")
    rows, instances = _scene_rows(dense_payload["scenes"], ibsp_payload["scenes"], args.refined_root)
    selected = select_review_scenes(rows, args.low_coverage_count, args.high_instance_count)
    dense_by_scene = {item["scene_name"]: item for item in dense_payload["scenes"]}

    diagnostic_dir = Path(args.output_diagnostics)
    visual_dir = Path(args.output_visuals)
    visual_index = []
    review_rows = []
    for item in selected:
        scene_name = item["row"]["scene_name"]
        frame = _best_frame(dense_by_scene[scene_name])
        overlay_path = ""
        if frame is not None:
            frame_id = str(frame["frame_id"])
            label_path = dense_root / scene_name / "frame_label_maps" / f"{frame_id}.png"
            color_path = _find_color_path(args.data_root, scene_name, frame_id)
            if label_path.is_file() and color_path is not None:
                labels = imageio.imread(label_path)
                image = imageio.imread(color_path)
                overlay = _overlay_labels(image, labels, alpha=args.overlay_alpha)
                output_path = visual_dir / scene_name / f"frame{frame_id}_label_overlay.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(output_path, overlay)
                overlay_path = str(output_path)
        instance_rows = [
            {
                "instance_id": int(value["instance_id"]),
                "class_name": value["class_name"],
                "frame_count": int(value["frame_count"]),
                "superpoint_count": int(value["superpoint_count"]),
                "point_count": int(value["point_count"]),
                "score": float(value["score"]),
            }
            for value in instances[scene_name]["instances"]
        ]
        review_rows.append({
            **item["row"],
            "review_reasons": ";".join(item["reasons"]),
            "representative_frame_id": "" if frame is None else str(frame["frame_id"]),
            "representative_frame_coverage": 0.0 if frame is None else float(frame["label_coverage"]),
            "overlay_path": overlay_path,
            "instances": instance_rows,
        })
        visual_index.append({"scene_name": scene_name, "overlay_path": overlay_path, "review_reasons": item["reasons"]})

    fields = [key for key in rows[0].keys()] + ["review_reasons", "representative_frame_id", "representative_frame_coverage", "overlay_path"]
    _write_csv(diagnostic_dir / "scene_quality.csv", rows, list(rows[0].keys()))
    _write_csv(diagnostic_dir / "review_scenes.csv", review_rows, fields)
    _write_json(diagnostic_dir / "review_scenes.json", review_rows)
    _write_json(visual_dir / "visual_review_index.json", visual_index)
    return {"scene_rows": rows, "review_scenes": review_rows}


def main():
    parser = argparse.ArgumentParser(description="Diagnose dense observation, IBSp, and superpoint-track quality without GT.")
    parser.add_argument("--dense_root", required=True)
    parser.add_argument("--ibsp_root", required=True)
    parser.add_argument("--refined_root", required=True)
    parser.add_argument("--data_root", default="./data/scannet200")
    parser.add_argument("--output_diagnostics", required=True)
    parser.add_argument("--output_visuals", required=True)
    parser.add_argument("--low_coverage_count", default=6, type=int)
    parser.add_argument("--high_instance_count", default=6, type=int)
    parser.add_argument("--overlay_alpha", default=0.55, type=float)
    args = parser.parse_args()
    result = run(args)
    print(f"Reviewed {len(result['review_scenes'])} scenes from {len(result['scene_rows'])} scenes.")


if __name__ == "__main__":
    main()
