from pathlib import Path
import sys,csv,numpy as np
root=Path(__file__).resolve().parents[4];sys.path.insert(0,str(root/'research/geometry/code/tools'))
import run_fi1_query_transfer_eval as run
from verify_fi1_nt_region_identity import vector
c=run.read(run.OUT/'contract.json');rows=list(csv.DictReader((run.OUT/'evaluation/opportunities.csv').open()));lookup={(r['scene'],r['candidate_id'],r['mode']):r for r in rows};n=0
for s in c['scenes']:
 gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True);keep=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[keep],sizes[keep]
 for e in run.examples(s):
  own=set(map(int,e['parent']));halo=set(map(int,e['halo']));visible={int(p) for p,g in zip(e['halo'],e['visible']) if g>=2};before=vector(own,gt,ids,sizes);i=int(before.argmax()) if len(ids) and before.max()>0 else None
  if i is None:continue
  target=int(ids[i]);hits=sum(gt[p]==target for p in own)
  if hits<20 or hits/len(own)<.6:continue
  add={p for p in visible-own if gt[p]==target};delete={p for p in visible&own if gt[p]>=1000 and gt[p]!=target};feasible=(own-delete)|add
  upper={p for p in halo if (gt[p]==target if p in visible else p in own)}
  if len(feasible)<100:feasible=own.copy()
  if len(upper)<100:upper=own.copy()
  fi=vector(feasible,gt,ids,sizes)[i];ui=vector(upper,gt,ids,sizes)[i]
  for m in run.MODES:
   r=lookup[s,e['candidate_id'],m];assert int(r['addition_opportunity'])==len(add);assert int(r['deletion_opportunity'])==len(delete);assert abs(float(r['known_feasible_iou'])-fi)<1e-12;assert abs(float(r['generous_feasible_iou'])-ui)<1e-12;n+=1
run.seal(run.OUT/'opportunities_verified.json',dict(complete=True,rows=n,independent_set_constructed=True,no_GT_rule_feedback=True,code={str(Path(__file__).resolve()):run.diag.sha(Path(__file__).resolve())}))
print('verified opportunities',n)
