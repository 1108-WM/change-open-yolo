import argparse
import json
import os
import os.path as osp
import shutil


def _parse_classes(value):
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _iter_candidate_jsons(input_dir):
    for root, _, files in os.walk(input_dir):
        for filename in sorted(files):
            if filename == "backprojection_candidates.json":
                yield osp.join(root, filename)


def _copy_or_link_seed_files(candidates, output_scene_dir):
    seed_dir = osp.join(output_scene_dir, "seed_points")
    os.makedirs(seed_dir, exist_ok=True)
    for candidate_id, candidate in enumerate(candidates):
        src = candidate.get("seed_points_path")
        if not src or not osp.exists(src):
            continue
        dst = osp.join(seed_dir, f"candidate{candidate_id:04d}_points.npz")
        shutil.copy2(src, dst)
        candidate["candidate_id"] = int(candidate_id)
        candidate["seed_points_path"] = dst


def filter_dataset(input_dir, output_dir, allowed_classes=None, blocked_classes=None):
    allowed_classes = _parse_classes(allowed_classes) if allowed_classes else None
    blocked_classes = _parse_classes(blocked_classes) if blocked_classes else None
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "allowed_classes": sorted(allowed_classes) if allowed_classes else None,
        "blocked_classes": sorted(blocked_classes) if blocked_classes else None,
        "scenes": [],
    }
    for json_path in _iter_candidate_jsons(input_dir):
        with open(json_path) as f:
            payload = json.load(f)
        candidates = []
        for candidate in payload.get("candidates", []):
            class_name = candidate.get("class_name")
            if allowed_classes is not None and class_name not in allowed_classes:
                continue
            if blocked_classes is not None and class_name in blocked_classes:
                continue
            candidates.append(dict(candidate))

        scene_name = payload.get("scene_name") or osp.basename(osp.dirname(json_path))
        output_scene_dir = osp.join(output_dir, scene_name)
        os.makedirs(output_scene_dir, exist_ok=True)
        _copy_or_link_seed_files(candidates, output_scene_dir)

        output_payload = dict(payload)
        output_payload["num_candidates"] = len(candidates)
        output_payload["candidates"] = candidates
        output_payload["class_filter"] = {
            "allowed_classes": sorted(allowed_classes) if allowed_classes else None,
            "blocked_classes": sorted(blocked_classes) if blocked_classes else None,
            "source_json": json_path,
        }
        output_json = osp.join(output_scene_dir, "backprojection_candidates.json")
        with open(output_json, "w") as f:
            json.dump(output_payload, f, indent=2)
        summary["scenes"].append(
            {
                "scene_name": scene_name,
                "source_json": json_path,
                "output_json": output_json,
                "input_candidates": len(payload.get("candidates", [])),
                "output_candidates": len(candidates),
            }
        )

    summary["input_candidates"] = sum(item["input_candidates"] for item in summary["scenes"])
    summary["output_candidates"] = sum(item["output_candidates"] for item in summary["scenes"])
    summary_path = osp.join(output_dir, "class_filter_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved class-filtered candidates to {output_dir}")
    print(f"Kept {summary['output_candidates']} / {summary['input_candidates']} candidates")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allowed_classes", default=None)
    parser.add_argument("--blocked_classes", default=None)
    args = parser.parse_args()
    filter_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        allowed_classes=args.allowed_classes,
        blocked_classes=args.blocked_classes,
    )


if __name__ == "__main__":
    main()
