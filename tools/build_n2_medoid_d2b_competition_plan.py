#!/usr/bin/env python3
"""Make a conservative no-GT N2-medoid versus frozen-D2b competition plan."""
import argparse,json,os
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTRACT="No-GT N2/D2b competition plan only; no proposal materialization, deletion, or score mutation."
def R(p):p=Path(p);return p if p.is_absolute() else ROOT/p
def S(p):
 r=[x.strip() for x in Path(p).read_text().splitlines() if x.strip()]
 if not r or len(r)!=len(set(r)):raise ValueError("scene list invalid")
 return r
def decide(rows):
 by=defaultdict(list)
 for r in rows:by[r["n2_family_key"]].append(r)
 out=[]
 for family,pairs in sorted(by.items()):
  c=Counter(x["relation_state"] for x in pairs)
  if c["exact_mutual_duplicate"]:
   state="fallback_exact_d2b_duplicate";reason="strict mutual geometry duplicate"
  elif c["partial_overlap"] or c["n2_contained_by_d2b"] or c["d2b_contained_by_n2"]:
   state="fallback_d2b_unresolved_overlap";reason="containment/partial overlap is not proof of replacement"
  else:
   state="coexist_no_d2b_overlap";reason="no superpoint overlap with any frozen D2b track"
  out.append({"scene_name":pairs[0]["scene_name"],"n2_family_key":family,"plan_state":state,"reason":reason,"pair_relation_counts":dict(sorted(c.items())),"ground_truth_usage":"none","proposal_materialization_applied":False,"ap_computed":False,"decision_constraint":CONTRACT})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--scene-list",type=Path,required=True);p.add_argument("--relation-ledger-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);a=p.parse_args()
 for n in("scene_list","relation_ledger_root","output_root"):setattr(a,n,R(getattr(a,n)))
 if a.output_root.exists() and any(a.output_root.iterdir()):raise SystemExit("output root non-empty")
 a.output_root.mkdir(parents=True,exist_ok=True);ss=[]
 for i,s in enumerate(S(a.scene_list),1):
  rows=[json.loads(x) for x in (a.relation_ledger_root/s/"n2_medoid_d2b_relation_ledger.jsonl").read_text().splitlines() if x];plan=decide(rows);st=a.output_root/f".{s}.tmp.{os.getpid()}";st.mkdir()
  with (st/"n2_medoid_d2b_competition_plan.jsonl").open("w") as h:
   for x in plan:h.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
  z={"scene_name":s,"family_count":len(plan),"plan_state_counts":dict(sorted(Counter(x["plan_state"] for x in plan).items())),"ground_truth_usage":"none","proposal_materialization_applied":False,"ap_computed":False};(st/"summary.json").write_text(json.dumps(z,ensure_ascii=False,indent=2)+"\n");os.replace(st,a.output_root/s);ss.append(z);print(f"[N2-D2b plan] {i}/{len(S(a.scene_list))} {s}",flush=True)
 c=Counter()
 for z in ss:c.update(z["plan_state_counts"])
 root={"diagnostic_type":"conservative no-GT N2 medoid/D2b competition plan","decision_constraint":CONTRACT,"scene_count":len(ss),"family_count":sum(x["family_count"] for x in ss),"plan_state_counts":dict(sorted(c.items())),"proposal_materialization_applied":False,"ap_computed":False};(a.output_root/"summary.json").write_text(json.dumps(root,ensure_ascii=False,indent=2)+"\n");print(json.dumps(root,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
