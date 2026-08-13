#!/usr/bin/env python3
"""No-GT N2a fixed family-formation variants over frozen N1 observations."""
import argparse,json,os
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CONTRACT="N2a no-GT family-formation variant ledger; no final proposal/materialization/AP/GT/native/semantic use."
def _resolve(p): p=Path(p);return p if p.is_absolute() else ROOT/p
def _scenes(p):
 r=[x.strip() for x in Path(p).read_text().splitlines() if x.strip()]
 if not r or len(r)!=len(set(r)):raise ValueError("scene list is empty or duplicated")
 return r
def _j(a,b):
 a,b=set(a),set(b);return len(a&b)/max(1,len(a|b))
def form_variants(rows):
 """One mask per view for all multi-view variants; same-frame masks stay alternatives."""
 eligible=[r for r in rows if r["candidate_superpoint_ids"] and r["seed_retained_after_lift"]]
 by_view=defaultdict(list)
 for r in eligible:by_view[int(r["frame_index"])].append(r)
 for v in by_view:by_view[v].sort(key=lambda r:(-float(r["sam_predicted_iou"]),r["candidate_id"]))
 if not eligible:return {"eligible_member_count":0,"variant_rows":[]}
 def medoid_score(r):
  vals=[]
  for v,other in by_view.items():
   if v!=int(r["frame_index"]):vals.append(max((_j(r["candidate_superpoint_ids"],x["candidate_superpoint_ids"]) for x in other),default=0.0))
  return float(np.mean(vals)) if vals else 0.0
 medoid=min(eligible,key=lambda r:(-medoid_score(r),-float(r["sam_predicted_iou"]),r["candidate_id"]))
 selected={}
 for v,members in by_view.items():
  selected[v]=min(members,key=lambda r:(-_j(r["candidate_superpoint_ids"],medoid["candidate_superpoint_ids"]),-float(r["sam_predicted_iou"]),r["candidate_id"]))
 support=Counter()
 for r in selected.values():support.update(set(map(int,r["candidate_superpoint_ids"])))
 union=sorted(support);consensus=sorted(sp for sp,n in support.items() if n>=2)
 top=min(eligible,key=lambda r:(-float(r["sam_predicted_iou"]),r["candidate_id"]))
 common={"eligible_member_count":len(eligible),"eligible_view_count":len(by_view),"selected_one_member_per_view":{str(v):r["candidate_id"] for v,r in sorted(selected.items())},"medoid_candidate_id":medoid["candidate_id"],"medoid_cross_view_mean_jaccard":medoid_score(medoid)}
 return {"eligible_member_count":len(eligible),"variant_rows":[
  {**common,"variant_name":"sam_top_single_observation_control","superpoint_ids":sorted(map(int,top["candidate_superpoint_ids"])),"source_candidate_ids":[top["candidate_id"]]},
  {**common,"variant_name":"cross_view_consistency_medoid","superpoint_ids":sorted(map(int,medoid["candidate_superpoint_ids"])),"source_candidate_ids":[medoid["candidate_id"]]},
  {**common,"variant_name":"two_view_superpoint_consensus","superpoint_ids":consensus,"source_candidate_ids":[r["candidate_id"] for _,r in sorted(selected.items())]},
  {**common,"variant_name":"compatible_view_union_high_recall_diagnostic","superpoint_ids":union,"source_candidate_ids":[r["candidate_id"] for _,r in sorted(selected.items())]},]}
def _write(p,rows):
 with p.open("w") as h:
  for r in rows:h.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scene-list",type=Path,required=True);p.add_argument("--candidate-ledger-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--max-scenes",type=int);a=p.parse_args()
 for n in("scene_list","candidate_ledger_root","output_root"):setattr(a,n,_resolve(getattr(a,n)))
 if a.output_root.exists() and any(a.output_root.iterdir()):raise SystemExit(f"output root is non-empty: {a.output_root}")
 scenes=_scenes(a.scene_list)[:a.max_scenes];a.output_root.mkdir(parents=True,exist_ok=True);summaries=[]
 for i,scene in enumerate(scenes,1):
  raw=[json.loads(x) for x in (a.candidate_ledger_root/scene/"observation_candidate_ledger.jsonl").read_text().splitlines() if x]
  groups=defaultdict(list)
  for r in raw:
   if r.get("ground_truth_usage")!="none" or r.get("proposal_materialization_applied"):raise ValueError("candidate input is not frozen no-GT")
   groups[r["candidate_family_key"]].append(r)
  out=[];empty=0
  for key,members in sorted(groups.items()):
   formed=form_variants(members);empty+=int(not formed["eligible_member_count"])
   for row in formed["variant_rows"]:out.append({"scene_name":scene,"candidate_family_key":key,"seed_superpoint_id":int(members[0]["seed_superpoint_id"]),**row,"superpoint_count":len(row["superpoint_ids"]),"ground_truth_usage":"none","proposal_materialization_applied":False,"ap_computed":False,"decision_constraint":CONTRACT})
  stage=a.output_root/f".{scene}.tmp.{os.getpid()}";stage.mkdir();_write(stage/"family_formation_variant_ledger.jsonl",out);s={"scene_name":scene,"family_count":len(groups),"family_without_eligible_member_count":empty,"variant_row_count":len(out),"ground_truth_usage":"none","proposal_materialization_applied":False,"ap_computed":False};(stage/"summary.json").write_text(json.dumps(s,ensure_ascii=False,indent=2,sort_keys=True)+"\n");os.replace(stage,a.output_root/scene);summaries.append(s);print(f"[N2a variants] {i}/{len(scenes)} {scene}: {len(out)}",flush=True)
 root={"diagnostic_type":"N2a no-GT fixed family formation variants","decision_constraint":CONTRACT,"scene_count":len(summaries),"family_count":sum(x["family_count"] for x in summaries),"family_without_eligible_member_count":sum(x["family_without_eligible_member_count"] for x in summaries),"variant_row_count":sum(x["variant_row_count"] for x in summaries),"proposal_materialization_applied":False,"ap_computed":False,"scene_summaries":summaries,"params":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
 (a.output_root/"summary.json").write_text(json.dumps(root,ensure_ascii=False,indent=2,sort_keys=True)+"\n");print(json.dumps({k:v for k,v in root.items() if k!="scene_summaries"},ensure_ascii=False,indent=2,sort_keys=True))
if __name__=="__main__":main()
