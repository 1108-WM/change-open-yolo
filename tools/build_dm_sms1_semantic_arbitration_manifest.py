#!/usr/bin/env python3
"""Build the first no-GT input ledger for the second innovation point.

This stage only materializes frozen geometry, finite class hypotheses and three
deterministically complementary views.  It deliberately does not infer a new
class, delete a proposal, change a score, read ground truth or run AP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=65536)
def camera_center(pose_path: str | Path) -> np.ndarray:
    """Return the camera centre from a ScanNet 4x4 camera-to-world pose."""
    matrix = np.asarray(np.loadtxt(pose_path), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid pose matrix: {pose_path}")
    return matrix[:3, 3]


def _view_key(view: Mapping[str, object]) -> tuple[float, int, str]:
    return (
        -float(view.get("visible_ratio", 0.0)),
        int(view.get("frame_index", 0)),
        str(view.get("frame_id", "")),
    )


def select_complementary_views(
    views: Sequence[Mapping[str, object]],
    target_count: int = 3,
    max_input_views: int = 20,
    centers: Mapping[str, np.ndarray] | None = None,
) -> list[int]:
    """Select up to three high-coverage but spatially diverse views.

    The first view is the highest-visible view.  Each later view maximizes its
    minimum normalized camera-centre distance to already selected views, with
    visibility and frame id used only as deterministic tie breakers.  This is
    an input-selection rule, not a semantic decision.
    """
    if target_count <= 0 or max_input_views <= 0:
        raise ValueError("view counts must be positive")
    candidates = [
        (index, view) for index, view in enumerate(views[:max_input_views])
        if _finite(view.get("visible_ratio", 0.0))
        and float(view.get("visible_ratio", 0.0)) > 0.0
        and bool(view.get("sam_mask_valid", False))
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda item: _view_key(item[1]))
    selected: list[int] = [candidates[0][0]]
    if target_count == 1 or len(candidates) == 1:
        return selected

    def distance(index_a: int, index_b: int) -> float:
        if centers is None:
            # Deterministic frame-index proxy when pose centres are unavailable.
            a = float(views[index_a].get("frame_index", index_a))
            b = float(views[index_b].get("frame_index", index_b))
            return abs(a - b)
        frame_a = str(views[index_a].get("frame_id"))
        frame_b = str(views[index_b].get("frame_id"))
        return float(np.linalg.norm(centers[frame_a] - centers[frame_b]))

    while len(selected) < min(target_count, len(candidates)):
        remaining = [index for index, _ in candidates if index not in selected]
        scored = []
        for index in remaining:
            min_distance = min(distance(index, prior) for prior in selected)
            scored.append((min_distance, float(views[index].get("visible_ratio", 0.0)),
                           -int(views[index].get("frame_index", index)),
                           str(views[index].get("frame_id", "")), index))
        # max distance, then visibility, then lower frame index/id.
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        selected.append(scored[0][-1])
    return selected


def finite_candidate_classes(
    canonical_class_index: int,
    alpha_class_index: int | None,
    class_count: int = 198,
) -> list[dict]:
    """Create a finite, provenance-labelled hypothesis set without choosing."""
    if class_count <= 0:
        raise ValueError("class_count must be positive")
    result: list[dict] = []
    seen: set[int] = set()
    if 0 <= int(canonical_class_index) < class_count:
        result.append({"class_index": int(canonical_class_index), "sources": ["frozen_control"]})
        seen.add(int(canonical_class_index))
    if alpha_class_index is not None and 0 <= int(alpha_class_index) < class_count:
        value = int(alpha_class_index)
        if value in seen:
            result[0]["sources"].append("alpha_main")
        else:
            result.append({"class_index": value, "sources": ["alpha_main"]})
            seen.add(value)
    if not result:
        raise ValueError("geometry has no valid class hypothesis")
    return result


def _asset(path: str | Path, kind: str) -> str:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return str(resolved)


def _read_records(ledger_root: Path, scenes: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for scene in scenes:
        path = ledger_root / "scenes" / scene / "records.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def _view_paths(view: Mapping[str, object]) -> tuple[str, str, str, str]:
    """Recover sensor paths omitted by the compact Stage D records."""
    rgb_path = Path(str(view["rgb_path"]))
    scene_root = rgb_path.parent.parent
    frame_id = str(view["frame_id"])
    depth = str(view.get("depth_path") or (scene_root / "depth" / f"{frame_id}.png"))
    pose = str(view.get("pose_path") or (scene_root / "poses" / f"{frame_id}.txt"))
    intrinsics = str(view.get("intrinsics_path") or (scene_root / "intrinsics.txt"))
    return str(rgb_path), depth, pose, intrinsics


def build_row(row: Mapping[str, object], target_count: int, max_input_views: int) -> dict:
    views = list(row.get("views", []))
    if not views:
        raise ValueError(f"{row.get('geometry_key')}: no Stage D views")
    centers: dict[str, np.ndarray] = {}
    for view in views[:max_input_views]:
        frame_id = str(view["frame_id"])
        _rgb_path, _depth_path, pose_path, _intrinsics_path = _view_paths(view)
        pose_path = _asset(pose_path, "pose")
        centers[frame_id] = camera_center(pose_path)
    selected_indices = select_complementary_views(
        views, target_count=target_count, max_input_views=max_input_views, centers=centers,
    )
    selected = []
    for rank, index in enumerate(selected_indices):
        view = views[index]
        rgb_path, depth_path, pose_path, intrinsics_path = _view_paths(view)
        selected.append({
            "selection_rank": rank,
            "source_view_index": int(index),
            "frame_id": str(view["frame_id"]),
            "frame_index": int(view["frame_index"]),
            "visible_ratio": float(view["visible_ratio"]),
            "visible_point_count": int(view["visible_point_count"]),
            "rgb_path": _asset(rgb_path, "RGB"),
            "depth_path": _asset(depth_path, "depth"),
            "pose_path": _asset(pose_path, "pose"),
            "intrinsics_path": _asset(intrinsics_path, "intrinsics"),
            "sam_box_prompt_xyxy": list(view["sam_box_prompt_xyxy"]),
            "sam_mask_sha256": str(view["sam_mask_sha256"]),
            "sam_mask_valid": bool(view["sam_mask_valid"]),
            "sam_mask_area": int(view["sam_mask_area"]),
            "view_selection_reason": "highest_visible_then_farthest_camera_center",
        })
    canonical = int(row["canonical_frozen_class_index"])
    alpha = int(row["alpha_class_index"]) if row.get("alpha_class_index") is not None else None
    return {
        "scene_name": str(row["scene_name"]),
        "geometry_key": str(row["geometry_key"]),
        "geometry_hash": str(row["geometry_hash"]),
        "point_count": int(row["point_count"]),
        "canonical_candidate_source": str(row["canonical_candidate_source"]),
        "canonical_candidate_id": int(row["canonical_candidate_id"]),
        "canonical_frozen_class_index": canonical,
        "canonical_frozen_score": float(row["canonical_frozen_score"]),
        "alpha_class_index": alpha,
        "alpha_top_similarity": float(row["alpha_top_similarity"]),
        "sms_keep": bool(row["sms_keep"]),
        "finite_class_hypotheses": finite_candidate_classes(canonical, alpha),
        "selected_views": selected,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "class_decision_made": False,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
    }


def run(args: argparse.Namespace) -> dict:
    args.ledger_root = _resolve(args.ledger_root)
    args.scene_list = _resolve(args.scene_list)
    args.output_root = _resolve(args.output_root)
    scenes = [line.strip() for line in args.scene_list.read_text().splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)) or not scenes:
        raise ValueError("scene list must be nonempty and unique")
    input_summary = json.loads((args.ledger_root / "summary.json").read_text())
    if input_summary.get("ground_truth_read") is not False or input_summary.get("ap_computed") is not False:
        raise ValueError("Stage D input is not no-GT")
    rows = _read_records(args.ledger_root, scenes)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"output root is non-empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    built = [build_row(row, args.target_views, args.max_input_views) for row in rows]
    identities = [(row["scene_name"], row["geometry_hash"]) for row in built]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate geometry identity")
    with (args.output_root / "semantic_arbitration_manifest.jsonl").open("w") as handle:
        for row in built:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "version": "dm_sms1_semantic_arbitration_manifest_v1",
        "scene_count": len(scenes),
        "geometry_count": len(built),
        "selected_view_count": sum(len(row["selected_views"]) for row in built),
        "view_target_shortfall_geometry_count": sum(
            len(row["selected_views"]) < args.target_views for row in built
        ),
        "candidate_hypothesis_count": sum(len(row["finite_class_hypotheses"]) for row in built),
        "target_views": args.target_views,
        "max_input_views": args.max_input_views,
        "view_selection_contract": "first highest visible ratio; then farthest camera centre, visible ratio and frame tie-breaks",
        "class_hypothesis_contract": "frozen control plus Alpha-CLIP top class; no class selected",
        "geometry_contract": "exact Stage D geometry, mask and membership frozen",
        "mutation_contract": {
            "candidate_mutation": False,
            "geometry_mutation": False,
            "score_mutation": False,
            "proposal_deletion": False,
        },
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
        "class_decision_made": False,
        "input_provenance": {
            "scene_list": str(args.scene_list),
            "scene_list_sha256": _sha256(args.scene_list),
            "stage_d_ledger": str(args.ledger_root),
            "stage_d_summary_sha256": _sha256(args.ledger_root / "summary.json"),
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, default=Path(
        "output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"
    ))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-views", type=int, default=3)
    parser.add_argument("--max-input-views", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
