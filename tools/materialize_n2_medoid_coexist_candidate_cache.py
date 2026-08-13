#!/usr/bin/env python3
"""Materialize one no-GT geometry cache candidate per D2b-disjoint N2 medoid family."""
import argparse,json,os
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CONTRACT="N2 medoid coexist geometry-cache materialization only; frozen D2b/native remain unchanged and no GT is read."
def R(p):p=Path(p);return p if p.is_absolute() else ROOT/p
def S(p):
 r=[x.strip() for x in Path(p).read_text().splitlines() if x.strip()]
 if not r or len(r)!=len(set(r)):raise ValueError("invalid scene list")
 return r
def main():
 p=argparse.ArgumentParser();p.add_argument("--scene-list",type=Path,required=True);p.add_argument("--variant-ledger-root",type=Path,required=True);p.add_argument("--competition-plan-root",type=Path,required=True);p.add_argument("--processed-scene-root",type=Path,default=Path("data/scannet200"));p.add_argument("--output-root",type=Path,required=True);a=p.parse_args()
 for n in("scene_list","variant_ledger_root","competition_plan_root","processed_scene_root","output_root"):setattr(a,n,R(getattr(a,n)))
 if a.output_root.exists() and any(a.output_root.iterdir()):raise SystemExit("output root non-empty")
 a.output_root.mkdir(parents=True,exist_ok=True);summ=[]
 for i,s in enumerate(S(a.scene_list),1):
  variants=[json.loads(x) for x in (a.variant_ledger_root/s/"family_formation_variant_ledger.jsonl").read_text().splitlines() if x];med={r["candidate_family_key"]:r for r in variants if r["variant_name"]=="cross_view_consistency_medoid"}
  plan=[json.loads(x) for x in (a.competition_plan_root/s/"n2_medoid_d2b_competition_plan.jsonl").read_text().splitlines() if x];keep=[r["n2_family_key"] for r in plan if r["plan_state"]=="coexist_no_d2b_overlap"]
  grouped=defaultdict(list)
  for k in keep:
   if k not in med:raise ValueError(f"missing medoid {k}")
   grouped[tuple(med[k]["superpoint_ids"])].append(k)
  processed=np.load(a.processed_scene_root/s/f"{s.replace('scene','')}.npy",mmap_mode="r");sp=np.asarray(processed[:,9],dtype=np.int64);bysp={int(x):np.flatnonzero(sp==x).astype(np.int64) for x in np.unique(sp)}
  stage=a.output_root/f".{s}.tmp.{os.getpid()}";points=stage/"points";points.mkdir(parents=True);out=[]
  for idx,(geometry,families) in enumerate(sorted(grouped.items(),key=lambda x:(x[0],x[1]))):
   canonical=min(families);ids=np.unique(np.concatenate([bysp[x] for x in geometry])).astype(np.int64) if geometry else np.empty(0,dtype=np.int64);path=points/f"candidate{idx:06d}_points.npz";np.savez_compressed(path,point_indices=ids)
   out.append({"candidate_id":idx,"scene_name":s,"canonical_family_key":canonical,"duplicate_family_keys":sorted(families),"superpoint_ids":list(geometry),"point_count":len(ids),"points_path":str(a.output_root/s/"points"/path.name),"formation_variant":"cross_view_consistency_medoid","ground_truth_usage":"none","proposal_materialization_applied":True,"decision_constraint":CONTRACT})
  with (stage/"n2_medoid_candidates.json").open("w") as h:json.dump({"scene_name":s,"candidates":out},h,ensure_ascii=False,indent=2)
  z={"scene_name":s,"coexist_family_count":len(keep),"exact_geometry_candidate_count":len(out),"deduplicated_family_count":len(keep)-len(out),"ground_truth_usage":"none","proposal_materialization_applied":True};(stage/"summary.json").write_text(json.dumps(z,ensure_ascii=False,indent=2)+"\n");os.replace(stage,a.output_root/s);summ.append(z);print(f"[N2 cache] {i} {s}: {len(out)}",flush=True)
 root={"diagnostic_type":"N2 medoid D2b-disjoint geometry candidate cache","decision_constraint":CONTRACT,"scene_count":len(summ),"coexist_family_count":sum(x["coexist_family_count"] for x in summ),"exact_geometry_candidate_count":sum(x["exact_geometry_candidate_count"] for x in summ),"deduplicated_family_count":sum(x["deduplicated_family_count"] for x in summ),"proposal_materialization_applied":True,"ap_computed":False};(a.output_root/"summary.json").write_text(json.dumps(root,ensure_ascii=False,indent=2)+"\n");print(json.dumps(root,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
