#!/usr/bin/env python3
"""Create a no-GT, non-materializing N2 family cleanup plan.

Only empty geometry and strict exact geometry equality are eligible for a
fallback plan.  Containment, overlap, low agreement, and all quality values
remain unresolved competition states.  This file cannot mutate proposals.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "N2 no-GT cleanup plan only; no proposal/hypothesis/score/native/GT/AP mutation."

def _resolve(path):
    path=Path(path); return path if path.is_absolute() else ROOT/path

def _scenes(path):
    rows=[x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if not rows or len(rows)!=len(set(rows)): raise ValueError("scene list is empty or duplicated")
    return rows

def cleanup_plan(rows):
    """Pure deterministic planning; equivalent union geometry shares one token."""
    by_fingerprint=defaultdict(list)
    for row in rows:
        if row["family_union_superpoint_ids"]:
            by_fingerprint[row["exact_duplicate_fingerprint"]].append(row)
    decisions={}
    for row in rows:
        key=row["candidate_family_key"]
        if not row["family_union_superpoint_ids"]:
            decisions[key]={"plan_state":"empty_family_no_candidate", "fallback_kind":"none", "canonical_family_key":None, "fallback_d2b_track_id":None}
        elif row["exact_mutual_duplicate"]:
            decisions[key]={"plan_state":"exact_d2b_duplicate_fallback", "fallback_kind":"frozen_d2b", "canonical_family_key":None, "fallback_d2b_track_id":int(row["best_d2b_track_id"])}
    for members in by_fingerprint.values():
        if len(members) < 2:
            continue
        active=[r for r in members if r["candidate_family_key"] not in decisions]
        if not active: continue
        canonical=min(active,key=lambda r:(int(r["seed_superpoint_id"]),r["candidate_family_key"]))
        for row in active:
            key=row["candidate_family_key"]
            decisions[key]={"plan_state":"exact_geometry_canonical" if row is canonical else "exact_geometry_duplicate_fallback", "fallback_kind":"canonical_family", "canonical_family_key":canonical["candidate_family_key"], "fallback_d2b_track_id":None}
    for row in rows:
        key=row["candidate_family_key"]
        decisions.setdefault(key,{"plan_state":"hold_for_later_competition", "fallback_kind":"none", "canonical_family_key":None, "fallback_d2b_track_id":None})
    output=[]
    for row in sorted(rows,key=lambda r:r["candidate_family_key"]):
        decision=decisions[row["candidate_family_key"]]
        output.append({"scene_name":row["scene_name"],"candidate_family_key":row["candidate_family_key"],"seed_superpoint_id":int(row["seed_superpoint_id"]),"family_union_superpoint_count":int(row["family_union_superpoint_count"]),"reliable_core_superpoint_count":int(row["reliable_core_superpoint_count"]),"unknown_boundary_superpoint_count":int(row["unknown_boundary_superpoint_count"]),"exact_duplicate_fingerprint":row["exact_duplicate_fingerprint"],"best_d2b_track_id":row["best_d2b_track_id"],"family_contained_by_d2b":bool(row["family_contained_by_d2b"]),"d2b_contained_by_family":bool(row["d2b_contained_by_family"]),**decision,"ground_truth_usage":"none","proposal_materialization_applied":False,"ap_computed":False,"decision_constraint":CONTRACT})
    return output

def _write_jsonl(path,rows):
    with path.open("w") as h:
        for row in rows: h.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-list",type=Path,required=True);p.add_argument("--family-ledger-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--max-scenes",type=int)
    a=p.parse_args()
    for name in ("scene_list","family_ledger_root","output_root"): setattr(a,name,_resolve(getattr(a,name)))
    if a.output_root.exists() and any(a.output_root.iterdir()): raise SystemExit(f"output root is non-empty: {a.output_root}")
    scenes=_scenes(a.scene_list)[:a.max_scenes];a.output_root.mkdir(parents=True,exist_ok=True);summaries=[]
    for number,scene in enumerate(scenes,1):
        rows=[json.loads(x) for x in (a.family_ledger_root/scene/"candidate_family_quality_ledger.jsonl").read_text().splitlines() if x]
        plan=cleanup_plan(rows);stage=a.output_root/f".{scene}.tmp.{os.getpid()}";stage.mkdir();_write_jsonl(stage/"candidate_family_cleanup_plan.jsonl",plan)
        counts=defaultdict(int)
        for row in plan: counts[row["plan_state"]]+=1
        summary={"scene_name":scene,"family_count":len(plan),"plan_state_counts":dict(sorted(counts.items())),"ground_truth_usage":"none","proposal_materialization_applied":False,"ap_computed":False}
        (stage/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n");os.replace(stage,a.output_root/scene);summaries.append(summary);print(f"[N2 cleanup plan] {number}/{len(scenes)} {scene}: {len(plan)}",flush=True)
    totals=defaultdict(int)
    for row in summaries:
        for key,value in row["plan_state_counts"].items(): totals[key]+=value
    root={"diagnostic_type":"N2 no-GT conservative candidate-family cleanup plan","decision_constraint":CONTRACT,"scene_count":len(summaries),"family_count":sum(x["family_count"] for x in summaries),"plan_state_counts":dict(sorted(totals.items())),"proposal_materialization_applied":False,"ap_computed":False,"scene_summaries":summaries,"params":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (a.output_root/"summary.json").write_text(json.dumps(root,ensure_ascii=False,indent=2,sort_keys=True)+"\n");print(json.dumps({k:v for k,v in root.items() if k!="scene_summaries"},ensure_ascii=False,indent=2,sort_keys=True))
if __name__=="__main__": main()
