#!/usr/bin/env python3
"""Evaluate a frozen native baseline and native plus semantic track candidates.

Track geometry, YOLO-World semantic votes, and track scores must already be
frozen before this GT-reading evaluation starts.  Ground truth is used only by
the official ScanNet200 evaluator and never changes masks, classes, scores, or
candidate selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _load_native(cache_root: Path, scene: str) -> dict[str, np.ndarray]:
    prefix = cache_root / f"{scene}_pred_"
    masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
    classes = np.asarray(np.load(str(prefix) + "classes.npy"), dtype=np.int64)
    scores = np.asarray(np.load(str(prefix) + "scores.npy"), dtype=np.float32)
    if masks.ndim != 2 or masks.shape[1] != len(classes) or len(classes) != len(scores):
        raise ValueError(f"{scene}: native prediction dimensions disagree")
    return {"pred_masks": masks, "pred_classes": classes, "pred_scores": scores}


def _load_track_prediction(
    track_root: Path,
    semantic_root: Path,
    scene: str,
    point_count: int,
    score_field: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    payload = json.loads((track_root / scene / "automatic_tracks.json").read_text())
    tracks = payload.get("tracks", [])
    semantic_rows = json.loads(
        (semantic_root / scene / "automatic_track_yoloworld_semantics.json").read_text()
    )
    semantics = {int(row["track_id"]): row for row in semantic_rows}
    if len(semantics) != len(semantic_rows):
        raise ValueError(f"{scene}: duplicate semantic track IDs")

    valid = []
    skipped_missing_semantics = 0
    skipped_invalid_class = 0
    skipped_empty = 0
    for track in tracks:
        track_id = int(track["track_id"])
        semantic = semantics.get(track_id)
        if semantic is None:
            skipped_missing_semantics += 1
            continue
        class_index = int(semantic.get("voted_class_index", -1))
        if class_index not in instance_eval.PRED_ID_TO_ID or int(
            instance_eval.PRED_ID_TO_ID[class_index]
        ) < 0:
            skipped_invalid_class += 1
            continue
        points = np.unique(
            np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64)
        )
        points = points[(points >= 0) & (points < point_count)]
        if len(points) == 0:
            skipped_empty += 1
            continue
        valid.append(
            (points, class_index, max(0.0, float(track.get(score_field, 0.0))))
        )

    masks = np.zeros((point_count, len(valid)), dtype=bool)
    classes = np.zeros(len(valid), dtype=np.int64)
    scores = np.zeros(len(valid), dtype=np.float32)
    for index, (points, class_index, score) in enumerate(valid):
        masks[points, index] = True
        classes[index] = class_index
        scores[index] = score
    return (
        {"pred_masks": masks, "pred_classes": classes, "pred_scores": scores},
        {
            "source_track_count": len(tracks),
            "evaluated_track_count": len(valid),
            "skipped_missing_semantics": skipped_missing_semantics,
            "skipped_invalid_class": skipped_invalid_class,
            "skipped_empty": skipped_empty,
        },
    )


class PredictionMapping(Mapping):
    def __init__(
        self,
        scenes: list[str],
        native_root: Path,
        track_root: Path | None = None,
        semantic_root: Path | None = None,
        score_field: str = "mean_node_quality",
    ) -> None:
        self.scenes = scenes
        self.native_root = native_root
        self.track_root = track_root
        self.semantic_root = semantic_root
        self.score_field = score_field
        self.scene_audits: list[dict] = []

    def __len__(self) -> int:
        return len(self.scenes)

    def __iter__(self):
        return iter(self.scenes)

    def __getitem__(self, scene: str):
        if scene not in self.scenes:
            raise KeyError(scene)
        return self._prediction(scene)

    def items(self):
        self.scene_audits.clear()
        for index, scene in enumerate(self.scenes, start=1):
            prediction = self._prediction(scene)
            print(f"[prediction] {index}/{len(self.scenes)} {scene}", flush=True)
            yield scene, prediction

    def _prediction(self, scene: str) -> dict[str, np.ndarray]:
        native = _load_native(self.native_root, scene)
        native_count = int(native["pred_masks"].shape[1])
        if self.track_root is None:
            self.scene_audits.append(
                {"scene_name": scene, "native_candidate_count": native_count}
            )
            return native
        tracks, audit = _load_track_prediction(
            self.track_root,
            self.semantic_root,
            scene,
            int(native["pred_masks"].shape[0]),
            self.score_field,
        )
        prediction = {
            "pred_masks": np.concatenate([native["pred_masks"], tracks["pred_masks"]], axis=1),
            "pred_classes": np.concatenate([native["pred_classes"], tracks["pred_classes"]]),
            "pred_scores": np.concatenate([native["pred_scores"], tracks["pred_scores"]]),
        }
        self.scene_audits.append(
            {"scene_name": scene, "native_candidate_count": native_count, **audit}
        )
        return prediction


def _evaluate(predictions: PredictionMapping, gt_root: Path, output_file: Path) -> dict:
    averages, _, _, _ = instance_eval.evaluate(
        predictions,
        str(gt_root),
        str(output_file),
        dataset="scannet200",
    )
    return {
        "ap": float(averages["all_ap"]),
        "ap50": float(averages["all_ap_50%"]),
        "ap25": float(averages["all_ap_25%"]),
        "head_ap": float(averages["head_ap"]),
        "common_ap": float(averages["common_ap"]),
        "tail_ap": float(averages["tail_ap"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--track-score-field", default="mean_node_quality")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_evaluation:
        raise SystemExit("pass --allow-gt-evaluation for the final GT-reading AP step")
    for name in (
        "scene_list", "native_prediction_cache", "track_root", "semantic_root",
        "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise SystemExit("--max-scenes must be positive")
        scenes = scenes[: args.max_scenes]

    baseline_predictions = PredictionMapping(scenes, args.native_prediction_cache)
    baseline = _evaluate(
        baseline_predictions, args.gt_instance_dir, args.output_dir / "native_open_vocab.csv"
    )
    gc.collect()

    combined_predictions = PredictionMapping(
        scenes,
        args.native_prediction_cache,
        args.track_root,
        args.semantic_root,
        args.track_score_field,
    )
    combined = _evaluate(
        combined_predictions,
        args.gt_instance_dir,
        args.output_dir / "native_plus_f2_tracks_open_vocab.csv",
    )
    payload = {
        "diagnostic_type": "official ScanNet200 open-vocabulary instance AP on frozen predictions",
        "scene_count": len(scenes),
        "ground_truth_usage": "evaluation_only",
        "candidate_contract": (
            "native unchanged; append F2 strict-mutual-duplicate-filtered tracks with "
            "frozen YOLO-World votes and original mean_node_quality"
        ),
        "official_ap_thresholds": [round(value, 2) for value in np.arange(0.50, 0.95, 0.05)],
        "ap25_threshold": 0.25,
        "native": baseline,
        "native_plus_f2_tracks": combined,
        "delta": {key: combined[key] - baseline[key] for key in baseline},
        "track_audit": {
            "source_track_count": sum(row["source_track_count"] for row in combined_predictions.scene_audits),
            "evaluated_track_count": sum(row["evaluated_track_count"] for row in combined_predictions.scene_audits),
            "skipped_missing_semantics": sum(row["skipped_missing_semantics"] for row in combined_predictions.scene_audits),
            "skipped_invalid_class": sum(row["skipped_invalid_class"] for row in combined_predictions.scene_audits),
            "skipped_empty": sum(row["skipped_empty"] for row in combined_predictions.scene_audits),
        },
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
