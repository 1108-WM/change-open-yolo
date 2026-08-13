#!/usr/bin/env python3
"""Convert frozen N1 seed/view rows into bounded, resumable SAM prompt plans."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _resolve(p):
    p=Path(p); return p if p.is_absolute() else ROOT/p

def _scenes(p): return [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]

def make_plan(rows, max_seeds, max_views):
    eligible=[r for r in rows if r.get("views")]
    eligible.sort(key=lambda r: (-len(r["views"]), -int(r["seed_superpoint_point_count"]), int(r["seed_superpoint_id"])))
    plan=[]
    for row in eligible[:max_seeds]:
        views=row["views"][:max_views]
        plan.append({"track_id":int(row["seed_superpoint_id"]),"seed_superpoint_id":int(row["seed_superpoint_id"]),"seed_point_index":int(row["seed_point_index"]),"seed_origin":row["seed_origin"],"prompt_point_index":int(row["seed_point_index"]),"prompt_superpoint_id":int(row["seed_superpoint_id"]),"common_core_superpoint_ids":[int(row["seed_superpoint_id"])],"prompt_frames":[{"frame_id":v["frame_id"],"frame_index":int(v["frame_index"]),"prompt_xy":v["seed_xy"],"visible_common_core_point_count":int(v["visible_seed_superpoint_point_count"])} for v in views]})
    return plan

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--scene-list',type=Path,required=True); p.add_argument('--seed-ledger-root',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--max-seeds-per-scene',type=int,required=True); p.add_argument('--max-views-per-seed',type=int,required=True); p.add_argument('--seed-offset',type=int,default=0); p.add_argument('--max-scenes',type=int)
    a=p.parse_args();
    for n in ('scene_list','seed_ledger_root','output_root'): setattr(a,n,_resolve(getattr(a,n)))
    if a.output_root.exists() and any(a.output_root.iterdir()): raise SystemExit('output root is non-empty')
    if min(a.max_seeds_per_scene,a.max_views_per_seed)<=0 or a.seed_offset<0: raise SystemExit('budgets must be positive and seed offset non-negative')
    scenes=_scenes(a.scene_list)[:a.max_scenes]; a.output_root.mkdir(parents=True); summaries=[]
    for scene in scenes:
        rows=[json.loads(x) for x in (a.seed_ledger_root/scene/'seed_view_ledger.jsonl').read_text().splitlines() if x]; plan=make_plan(rows,a.max_seeds_per_scene+a.seed_offset,a.max_views_per_seed)[a.seed_offset:]; stage=a.output_root/f'.{scene}.tmp.{os.getpid()}'; stage.mkdir(); (stage/'core_prompt_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n'); summary={'scene_name':scene,'eligible_seed_count':sum(bool(r['views']) for r in rows),'seed_offset':a.seed_offset,'planned_seed_count':len(plan),'prompt_request_count':sum(len(r['prompt_frames']) for r in plan),'ground_truth_usage':'none','proposal_materialization_applied':False}; (stage/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); os.replace(stage,a.output_root/scene); summaries.append(summary)
    (a.output_root/'core_prompt_plan_summary.json').write_text(json.dumps({'paper_reference':True,'sam_executed':False,'params':vars(a),'scenes':summaries},default=str,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__': main()
