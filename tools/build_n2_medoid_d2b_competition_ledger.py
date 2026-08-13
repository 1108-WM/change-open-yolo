#!/usr/bin/env python3
"""No-GT pair ledger between frozen N2a medoids and frozen D2b tracks."""
import argparse,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
V="cross_view_consistency_medoid"; CONTRACT="No-GT N2-medoid/D2b relation ledger only; no candidate/track mutation or materialization."
def R(p):p=Path(p);return p if p.is_absolute() else ROOT/p
def S(p):return [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]
def rel(a,b):
 a,b=set(a),set(b);i=len(a&b);return {"intersection_superpoint_count":i,"iou":i/max(1,len(a|b)),"n2_covered_ratio":i/max(1,len(a)),"d2b_covered_ratio":i/max(1,len(b))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--scene-list",type=Path,required=True);p.add_argument("--variant-ledger-root",type=Path,required=True);p.add_argument("--d2b-track-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);a=p.parse_args()
 for n in("scene_list","variant_ledger_root","d2b_track_root","output_root"):setattr(a,n,R(getattr(a,n)))
 if a.output_root.exists() and any(a.output_root.iterdir()):raise SystemExit("output root is non-empty")
 a.output_root.mkdir(parents=True,exist_ok=True);summ=[]
 for idx,s in enumerate(S(a.scene_list),1):
  med=[json.loads(x) for x in (a.variant_ledger_root/s/"family_formation_variant_ledger.jsonl").read_text().splitlines() if x and json.loads(x)["variant_name"]==V]
  tracks=json.loads((a.d2b_track_root/s/"automatic_tracks.json").read_text())["tracks"];rows=[]
  for m in med:
   for t in tracks:
    q=rel(m["superpoint_ids"],t["superpoint_ids"]);state="nonoverlap" if not q["intersection_superpoint_count"] else ("exact_mutual_duplicate" if q["n2_covered_ratio"]>.99 and q["d2b_covered_ratio"]>.99 else ("n2_contained_by_d2b" if q["n2_covered_ratio"]>.99 else ("d2b_contained_by_n2" if q["d2b_covered_ratio"]>.99 else "partial_overlap")))
    rows.append({"scene_name":s,"n2_family_key":m["candidate_family_key"],"d2b_track_id":int(t["track_id"]),"relation_state":state,**q,"ground_truth_usage":"none","proposal_materialization_applied":False,"decision_constraint":CONTRACT})
  st=a.output_root/f".{s}.tmp.{os.getpid()}";st.mkdir();
  with (st/"n2_medoid_d2b_relation_ledger.jsonl").open("w") as h:
   for x in rows:h.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
  c={};
  for x in rows:c[x["relation_state"]]=c.get(x["relation_state"],0)+1
  z={"scene_name":s,"medoid_family_count":len(med),"d2b_track_count":len(tracks),"pair_count":len(rows),"relation_counts":c,"ground_truth_usage":"none","proposal_materialization_applied":False};(st/"summary.json").write_text(json.dumps(z,ensure_ascii=False,indent=2)+"\n");os.replace(st,a.output_root/s);summ.append(z);print(f"[N2-D2b] {idx} {s}",flush=True)
 total={};
 for z in summ:
  for k,v in z["relation_counts"].items():total[k]=total.get(k,0)+v
 root={"diagnostic_type":"no-GT N2 medoid to frozen D2b pair ledger","decision_constraint":CONTRACT,"scene_count":len(summ),"pair_count":sum(x["pair_count"] for x in summ),"relation_counts":total,"proposal_materialization_applied":False,"ap_computed":False};(a.output_root/"summary.json").write_text(json.dumps(root,ensure_ascii=False,indent=2)+"\n");print(json.dumps(root,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
