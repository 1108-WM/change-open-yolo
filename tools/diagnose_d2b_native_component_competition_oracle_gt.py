#!/usr/bin/env python3
"""GT-only C1c global-feasible D2b/native component-competition oracle.

For every official IoU threshold, maximum one-to-one matching is solved over
the immutable native masks and strict-filtered D2b tracks.  Only candidates
inside the frozen no-GT relation components may be suppressed: matched
candidates are kept, unmatched component candidates abstain.  Native masks
with no relation to any track remain frozen and present.  This gives a global
feasible component action diagnostic, not an inference rule or mAP result.
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "tools"):
    if str(item) not in sys.path: sys.path.insert(0, str(item))
from build_track_native_competition_ledger import _native_cache_contract, _read_scenes
from diagnose_n1_sampro3d_candidate_space_oracle_gt import _load_gt
from diagnose_n2_medoid_candidate_oracle_gt import maximum_matching, native_edges, track_edges
from diagnose_d2b_native_track_ranking_oracle_gt import _average_precision, _append, _empty_record, _scene_ap_records
from diagnose_gvc_class_agnostic_ap import UNIFIED_PREDICTED_CLASS, _class_agnostic_gt_ids, _configure_scannet200_instance_eval, instance_eval

CONTRACT = ("GT-only C1c global component competition: GT selects threshold-specific "
            "one-to-one matches and suppresses only unmatched component candidates. "
            "It must not become a selector, score, materialized candidate set, or inference rule.")
AP25_THRESHOLD = 0.25
OFFICIAL_AP_THRESHOLDS = tuple(sorted(
    round(float(value), 2) for value in instance_eval.opt["overlaps"] if float(value) >= .50
))
EXTRA_DIAGNOSTIC_THRESHOLDS = (0.95,)
DIAGNOSTIC_THRESHOLDS = (AP25_THRESHOLD, *OFFICIAL_AP_THRESHOLDS, *EXTRA_DIAGNOSTIC_THRESHOLDS)

def _resolve(path):
    path = Path(path); return path if path.is_absolute() else ROOT / path

def _component_members(root, scene):
    rows = [json.loads(line) for line in (root / scene / "relation_components.jsonl").read_text().splitlines() if line]
    tracks, natives = set(), set()
    for row in rows:
        tracks.update(int(x) for x in row["track_ids"])
        natives.update(int(x) for x in row["native_candidate_ids"])
    return tracks, natives

def _load_tracks(root, scene):
    rows = json.loads((root / scene / "automatic_tracks.json").read_text())["tracks"]
    result = {int(row["track_id"]): row for row in rows}
    if len(result) != len(rows): raise ValueError(f"{scene}: duplicate track ID")
    return result

def _track_points(row, count):
    points = np.unique(np.asarray(np.load(row["points_path"])["point_indices"], dtype=np.int64))
    points = points[(points >= 0) & (points < count)]
    if len(points) != int(row["point_count"]): raise ValueError("track point count differs")
    return points

def _prediction(native_masks, native_scores, tracks, kept_native, kept_tracks):
    native_ids = sorted(kept_native)
    track_ids = sorted(kept_tracks)
    masks = np.zeros((native_masks.shape[0], len(track_ids)), dtype=bool)
    scores = np.zeros(len(track_ids), dtype=np.float32)
    for col, track_id in enumerate(track_ids):
        masks[_track_points(tracks[track_id], native_masks.shape[0]), col] = True
        scores[col] = float(tracks[track_id]["mean_node_quality"])
    return {"pred_masks": np.concatenate([native_masks[:, native_ids], masks], axis=1),
            "pred_scores": np.concatenate([native_scores[native_ids], scores]),
            "pred_classes": np.full(len(native_ids) + len(track_ids), UNIFIED_PREDICTED_CLASS, dtype=np.int64)}

def _record(prediction, gt_file, threshold):
    gt, pred = instance_eval.assign_instances_for_scan(prediction, str(gt_file))
    return _scene_ap_records(gt, pred, threshold)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--relation-ledger-root", type=Path, required=True)
    parser.add_argument("--filtered-d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics: raise SystemExit("--allow-gt-diagnostics is required")
    for name in ("scene_list","relation_ledger_root","filtered_d2b_track_root","native_prediction_cache","gt_instance_dir","output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    all_scenes = _read_scenes(args.scene_list); scenes = all_scenes if args.max_scenes is None else all_scenes[:args.max_scenes]
    if not scenes: raise SystemExit("--max-scenes must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()): raise SystemExit("output root is non-empty")
    contract = _native_cache_contract(args.native_prediction_cache, len(all_scenes))
    args.output_root.mkdir(parents=True, exist_ok=True)
    coverage, records = {}, {str(int(t*100)): {"native": _empty_record(), "component_oracle": _empty_record()} for t in DIAGNOSTIC_THRESHOLDS}
    _configure_scannet200_instance_eval(); original = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda filename: _class_agnostic_gt_ids(original(filename))
    try:
      for ordinal, scene in enumerate(scenes, 1):
        gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene}.txt", 100)
        edges_native = native_edges(args.native_prediction_cache, scene, gt_ids, gt_sizes)
        edges_track = track_edges(args.filtered_d2b_track_root, scene, gt_ids, gt_sizes)
        edges = {**edges_native, **edges_track}
        component_tracks, component_natives = _component_members(args.relation_ledger_root, scene)
        tracks = _load_tracks(args.filtered_d2b_track_root, scene)
        if component_tracks != set(tracks): raise ValueError(f"{scene}: ledger tracks differ from frozen tracks")
        native_masks = np.load(args.native_prediction_cache / f"{scene}_pred_masks.npy", mmap_mode="r")
        native_scores = np.asarray(np.load(args.native_prediction_cache / f"{scene}_pred_scores.npy", mmap_mode="r"), dtype=np.float32)
        if component_natives - set(range(native_masks.shape[1])): raise ValueError(f"{scene}: invalid native ID")
        gt_file = args.gt_instance_dir / f"{scene}.txt"
        native_prediction = _prediction(native_masks, native_scores, tracks, set(range(native_masks.shape[1])), set())
        rows = []
        for threshold in DIAGNOSTIC_THRESHOLDS:
            tag = str(int(round(threshold * 100)))
            native_count, native_selected = maximum_matching(edges_native, threshold)
            all_count, selected = maximum_matching(edges, threshold)
            kept_tracks = {int(key[4:]) for key in selected if key.startswith("d2b_")}
            kept_component_native = {int(key[7:]) for key in selected if key.startswith("native_") and int(key[7:]) in component_natives}
            kept_native = (set(range(native_masks.shape[1])) - component_natives) | kept_component_native
            prediction = _prediction(native_masks, native_scores, tracks, kept_native, kept_tracks)
            _append(records[tag]["native"], _record(native_prediction, gt_file, threshold))
            _append(records[tag]["component_oracle"], _record(prediction, gt_file, threshold))
            coverage[tag] = coverage.get(tag, {"native":0,"oracle":0,"valid_gt":0,"kept_tracks":0,"kept_component_native":0})
            coverage[tag]["native"] += native_count; coverage[tag]["oracle"] += all_count; coverage[tag]["valid_gt"] += len(gt_sizes)
            coverage[tag]["kept_tracks"] += len(kept_tracks); coverage[tag]["kept_component_native"] += len(kept_component_native)
            rows.append({"scene_name":scene,"threshold":threshold,"native_maximum_matching":native_count,"global_component_oracle_maximum_matching":all_count,"increment_vs_native":all_count-native_count,"kept_track_count":len(kept_tracks),"kept_component_native_count":len(kept_component_native),"ground_truth_usage":"offline_diagnostic_only","proposal_materialization_applied":False})
            del prediction
        (args.output_root / f"{scene}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n")
        del native_prediction, native_masks, tracks; gc.collect(); print(f"[C1c component oracle] {ordinal}/{len(scenes)} {scene}", flush=True)
    finally: instance_eval.util_3d.load_ids = original
    metrics = {}
    for threshold in DIAGNOSTIC_THRESHOLDS:
        tag=str(int(round(threshold*100))); native_record=records[tag]["native"]; oracle_record=records[tag]["component_oracle"]
        native_ap=_average_precision(native_record["true"], native_record["score"], native_record["fn"], native_record["has_gt"], native_record["has_pred"])
        oracle_ap=_average_precision(oracle_record["true"], oracle_record["score"], oracle_record["fn"], oracle_record["has_gt"], oracle_record["has_pred"])
        metrics[tag]={"threshold":threshold,"native_frozen_ap":native_ap,"component_oracle_fixed_score_ap":oracle_ap,"fixed_score_gain_vs_native":oracle_ap-native_ap,**coverage[tag],"coverage_increment_vs_native":coverage[tag]["oracle"]-coverage[tag]["native"]}
    official=[str(int(round(t*100))) for t in OFFICIAL_AP_THRESHOLDS]
    summary={"diagnostic_type":"GT-only C1c global-feasible component competition oracle","decision_constraint":CONTRACT,"native_cache_contract":contract,"official_ap_thresholds":list(OFFICIAL_AP_THRESHOLDS),"ap25_threshold":AP25_THRESHOLD,"extra_diagnostic_thresholds":list(EXTRA_DIAGNOSTIC_THRESHOLDS),"scene_count":len(scenes),"threshold_metrics":metrics,"aggregate":{"native_ap":float(np.mean([metrics[t]["native_frozen_ap"] for t in official])),"component_oracle_fixed_score_ap":float(np.mean([metrics[t]["component_oracle_fixed_score_ap"] for t in official])),"fixed_score_gain_vs_native":float(np.mean([metrics[t]["fixed_score_gain_vs_native"] for t in official])),"coverage_increment_like_ap":float(np.mean([metrics[t]["coverage_increment_vs_native"]/max(1,metrics[t]["valid_gt"]) for t in official]))},"proposal_materialization_applied":False,"params":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    (args.output_root / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True))
if __name__ == "__main__": main()
