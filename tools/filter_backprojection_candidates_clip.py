import argparse
import json
import os
import os.path as osp
import sys

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.clip_object_rescore import load_clip_object_features
from utils.backprojection_fusion import _candidate_instance_key


def _parse_class_filter(value):
    if value is None:
        return None
    parsed = {item.strip() for item in str(value).split(",") if item.strip()}
    return parsed or None


def _iter_candidate_jsons(path):
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename == "backprojection_candidates.json":
                yield osp.join(root, filename)


def _feature_key(record):
    return _candidate_instance_key(record)


def filter_candidates(
    candidates_dir,
    clip_features,
    output_dir,
    min_detector_prob=0.03,
    keep_missing=False,
    blocked_classes=None,
):
    blocked_classes = _parse_class_filter(blocked_classes)
    features_by_scene, feature_summary = load_clip_object_features(clip_features)
    feature_lookup = {}
    for records in features_by_scene.values():
        for record in records:
            key = _feature_key(record)
            if key is not None:
                feature_lookup[key] = record

    os.makedirs(output_dir, exist_ok=True)
    total_input = 0
    total_kept = 0
    total_filtered = 0
    per_scene = {}

    for json_path in _iter_candidate_jsons(candidates_dir):
        with open(json_path) as f:
            payload = json.load(f)
        scene_name = str(payload.get("scene_name"))
        kept = []
        filtered = []
        for candidate in payload.get("candidates", []):
            total_input += 1
            candidate_id = int(candidate.get("candidate_id", -1))
            class_name = candidate.get("class_name")
            if blocked_classes is not None and class_name in blocked_classes:
                filtered.append(
                    {
                        "candidate_id": candidate_id,
                        "class_name": class_name,
                        "reason": "class_blocked",
                    }
                )
                total_filtered += 1
                continue

            feature = feature_lookup.get(_candidate_instance_key(candidate, scene_name=scene_name))
            if feature is None:
                feature = feature_lookup.get((scene_name, candidate_id))
            if feature is None:
                if keep_missing:
                    kept.append(candidate)
                    total_kept += 1
                else:
                    filtered.append(
                        {
                            "candidate_id": candidate_id,
                            "class_name": class_name,
                            "reason": "missing_clip_feature",
                        }
                    )
                    total_filtered += 1
                continue

            class_id = int(candidate.get("class_id", -1))
            probs = feature.get("clip_probs", [])
            detector_prob = float(probs[class_id]) if 0 <= class_id < len(probs) else 0.0
            if detector_prob < min_detector_prob:
                filtered.append(
                    {
                        "candidate_id": candidate_id,
                        "class_name": class_name,
                        "reason": "low_detector_clip_prob",
                        "detector_clip_prob": detector_prob,
                        "clip_top1": feature.get("clip_topk", [{}])[0],
                    }
                )
                total_filtered += 1
                continue

            record = dict(candidate)
            record["clip_detector_prob"] = detector_prob
            record["clip_topk"] = feature.get("clip_topk", [])
            kept.append(record)
            total_kept += 1

        scene_dir = osp.join(output_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        output_json = osp.join(scene_dir, "backprojection_candidates.json")
        output_payload = dict(payload)
        output_payload["candidates"] = kept
        output_payload["num_candidates"] = len(kept)
        output_payload["clip_filter"] = {
            "source_json": json_path,
            "clip_features": clip_features,
            "min_detector_prob": min_detector_prob,
            "keep_missing": keep_missing,
            "blocked_classes": sorted(blocked_classes) if blocked_classes is not None else None,
            "filtered": filtered,
        }
        with open(output_json, "w") as f:
            json.dump(output_payload, f, indent=2)
        per_scene[scene_name] = {"input": len(payload.get("candidates", [])), "kept": len(kept), "filtered": len(filtered)}

    summary_path = osp.join(output_dir, "clip_filter_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "candidates_dir": candidates_dir,
                "clip_features": clip_features,
                "feature_summary": feature_summary,
                "min_detector_prob": min_detector_prob,
                "keep_missing": keep_missing,
                "blocked_classes": sorted(blocked_classes) if blocked_classes is not None else None,
                "total_input": total_input,
                "total_kept": total_kept,
                "total_filtered": total_filtered,
                "per_scene": per_scene,
            },
            f,
            indent=2,
        )
    print(f"Kept {total_kept}/{total_input} candidates; filtered {total_filtered}.")
    print(f"Saved filtered candidates to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_dir", default="./output/backprojection_candidates_replica_mv_m20")
    parser.add_argument("--clip_features", default="./output/clip_object_features_replica_mv_m20")
    parser.add_argument("--output_dir", default="./output/backprojection_candidates_replica_mv_m20_clip_filter")
    parser.add_argument("--min_detector_prob", default=0.03, type=float)
    parser.add_argument("--keep_missing", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--blocked_classes", default=None)
    args = parser.parse_args()

    filter_candidates(
        candidates_dir=args.candidates_dir,
        clip_features=args.clip_features,
        output_dir=args.output_dir,
        min_detector_prob=args.min_detector_prob,
        keep_missing=args.keep_missing,
        blocked_classes=args.blocked_classes,
    )


if __name__ == "__main__":
    main()
