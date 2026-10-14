import csv,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import sys
ROOT=Path('/home/jia/Wm/wm_open-yolo/OpenYOLO3D');sys.path.insert(0,str(ROOT))
from tools.run_fi1_local_edit_ablation import original
P=ROOT/'.local/fi1_local_edit_ablation_development200_20261012_v1';S=original.OUT
O=ROOT/'.local/fi1_local_edit_ablation_independent_analysis_20261012';O.mkdir(exist_ok=True)
c=json.load(open(P/'contract.json'));checks=Counter()
for name in ('code','frozen_SL_models','GT_inputs'):
 for f,h in c[name].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h;checks[name+'_hashes']+=1
for stage in ('models','pilot','predictions','conditions','labels'):
 for f,h in json.load(open(P/(stage+'_frozen.json')))['outputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h;checks[stage+'_hashes']+=1
target_csv={(r['scene'],r['arm'],r['mode'],int(r['gt_id'])):r for r in csv.DictReader(open(P/'evaluation/targets.csv'))}
candidate_csv={(r['scene'],r['candidate_id'],r['arm']):r for r in csv.DictReader(open(P/'evaluation/candidates.csv'))}
metrics=defaultdict(Counter);pool=defaultdict(Counter);pool_targets=defaultdict(list);per_candidate=[];scope_stats=defaultdict(Counter);strata_stats=[];gate=Counter();best_cases=[]
for si,s in enumerate(c['scenes']):
 gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True);ok=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[ok],sizes[ok];where={int(g):i for i,g in enumerate(ids)}
 idx=np.searchsorted(ids,gt);safe=np.minimum(idx,len(ids)-1);bins=np.where((idx<len(ids))&(ids[safe]==gt),idx,len(ids))
 def iou(p):
  if len(p)<100:return np.zeros(len(ids))
  hit=np.bincount(bins[p],minlength=len(ids)+1)[:len(ids)];return hit/(len(p)+sizes-hit)
 _,pp=original.audit.prior.old.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
 for p in pp.values():baseline=np.maximum(baseline,iou(p))
 best={};plans=json.load(open(S/'plan'/s/'candidates.json'));meta={r['candidate_id']:r for r in json.load(open(P/'labels'/s/'meta.json'))}
 with np.load(S/'features'/s/'data.npz') as f,np.load(P/'predictions'/s/'predictions.npz') as pr,np.load(P/'labels'/s/'data.npz') as lab,np.load(S/'predictions'/s/'predictions.npz') as old:
  for row in plans:
   cid=row['candidate_id'];parent=f[cid+'_parent'];halo=f[cid+'_halo'];present=np.isin(halo,parent);seen=f[cid+'_gd10_selected6_groups']>=2;y=lab[cid+'_y'];m=meta[cid];gid=m['target'];known=gid is not None;before=iou(parent);k=where[gid] if known else None
   quotas=np.array([.25,.25,.20,.20,.10]);ns=np.array([len(lab[cid+'_stratum'+str(i)]) for i in range(5)]);active=ns>0
   if ns[0]>0:
    share=float(quotas[0]/quotas[active].sum());factor=share/(ns[0]/len(halo));strata_stats.append(dict(scene=s,candidate=cid,delete_points=int(ns[0]),halo_points=len(halo),sampling_inflation=factor,expected_repeats=4096*share/ns[0],good=m['good']))
   cp=pr[cid+'_C_candidate_probability'].astype(np.float32);allow=cp[0]>=.7
   gate['allowed']+=allow;gate['rejected']+=not allow;gate['true_edit_allowed']+=allow and m['scope']==0;gate['true_keep_allowed']+=allow and m['scope']==1
   scenarios={}
   for arm in ('A','A0','B','C','C_ungated'):
    prob=pr[cid+'_'+('C' if arm=='C_ungated' else arm)+'_probability'].astype(np.float32)
    state=present.copy();state[present&seen&(prob[:,1]>=.8)]=False;state[~present&seen&(prob[:,0]>=.8)]=True
    if arm=='C' and not allow:state=present.copy()
    p=halo[state] if state.sum()>=100 else parent
    if arm!='C_ungated':
     assert np.array_equal(p,pr[cid+'_'+arm+'_points']);checks['point_replays']+=1
     r=candidate_csv[s,cid,arm];assert int(r['added'])==len(np.setdiff1d(p,parent));assert int(r['deleted'])==len(np.setdiff1d(parent,p))
     if known:assert abs(iou(p)[k]-float(r['repaired_iou']))<1e-12;checks['parent_iou_replays']+=1
     if arm=='A':assert np.array_equal(p,old[cid+'_gd10_selected6_points']);checks['A_reuse']+=1
    add=np.setdiff1d(p,parent);delete=np.setdiff1d(parent,p);d=np.setdiff1d(parent,delete);d=d if len(d)>=100 else parent
    scenarios[arm+'::full']=p;scenarios[arm+'::add_only']=np.union1d(parent,add);scenarios[arm+'::delete_only']=d
    t=metrics[arm];t['actions']+=bool(len(add)+len(delete));t['added']+=len(add);t['deleted']+=len(delete)
    if known:
     aa=int((gt[add]==gid).sum());ao=int(((gt[add]>=1000)&(gt[add]!=gid)).sum());dd=int(((gt[delete]>=1000)&(gt[delete]!=gid)).sum());dt=int((gt[delete]==gid).sum());delta=float(iou(p)[k]-before[k])
     t.update(correct_add=aa,wrong_add=ao,correct_delete=dd,wrong_delete=dt,parent_gain02=delta>=.02-1e-12,parent_loss02=delta<-.02-1e-12,good_loss02=before[k]>=.9 and delta<-.02-1e-12,good_loss10=before[k]>=.9 and delta<-.1-1e-12)
     sc=scope_stats[arm+'::'+m['scope_name']];sc.update(actions=bool(len(add)+len(delete)),correct_delete=dd,wrong_delete=dt,correct_add=aa,wrong_add=ao)
     per_candidate.append(dict(scene=s,candidate=cid,arm=arm,source=row['source'],scope=m['scope_name'],parent_iou=float(before[k]),iou=float(iou(p)[k]),delta=delta,correct_delete=dd,wrong_delete=dt,correct_add=aa,wrong_add=ao,p_edit=float(cp[0]),gate_allowed=bool(allow)))
     if arm in ('B','C_ungated'):
      # Only diagnostic: a perfect pointwise verifier keeps the correct edits.
      correct_d=delete[(gt[delete]>=1000)&(gt[delete]!=gid)];ideal_d=np.setdiff1d(parent,correct_d);ideal_d=ideal_d if len(ideal_d)>=100 else parent
      scenarios[arm+'::oracle_censor_delete']=ideal_d
      scenarios[arm+'::oracle_censor_both']=np.union1d(ideal_d,add[gt[add]==gid])
    else:t['unknown_identity_edits']+=len(add)+len(delete)
   # Perfectly predicting C's existing target label, unlike knowing actual edit utility.
   scenarios['C_oracle_scope::full']=scenarios['C_ungated::full'] if m['scope']==0 else parent
   scenarios['C_oracle_scope::delete_only']=scenarios['C_ungated::delete_only'] if m['scope']==0 else parent
   for name,p in scenarios.items():
    value=iou(p);best[name]=np.maximum(best.get(name,baseline.copy()),value)
   if known:
    raw=scenarios['C_ungated::full'];a=iou(raw)[k]-before[k]
    gate['rejected_positive_parent']+=not allow and a>=.02-1e-12
    gate['rejected_negative_parent']+=not allow and a<-.02-1e-12
    gate['allowed_positive_parent']+=allow and a>=.02-1e-12
    gate['allowed_negative_parent']+=allow and a<-.02-1e-12
 for name,values in best.items():
  gains=values-baseline>=.02-1e-12;valuable=gains.copy()
  for t in (.75,.9,.95):valuable|=(baseline<=t)&(values>t)
  pool[name]['gain02']+=int(gains.sum());pool[name]['value']+=int(valuable.sum())
  for t in (.5,.75,.9,.95):pool[name]['new_cross_'+str(t)]+=int(((baseline<=t)&(values>t)).sum())
  for i in np.flatnonzero(valuable):pool_targets[name].append(dict(scene=s,target=int(ids[i]),before=float(baseline[i]),after=float(values[i])))
  arm,mode=name.split('::')
  if arm in ('A','A0','B','C') and mode in ('full','add_only','delete_only'):
   for i,gid in enumerate(ids):
    reported=target_csv[s,arm,mode,int(gid)];assert abs(values[i]-float(reported['expanded_NT_best']))<1e-12;assert abs(baseline[i]-float(reported['original_NT_best']))<1e-12;checks['pool_replays']+=1
 if (si+1)%25==0:print('audited',si+1,flush=True)

pilot=[]
for a in ('B','C'):
 for fold in range(5):
  d=json.load(open(P/f'pilot/{a}/fold{fold}/fit.json'));mm=d['metrics'];assert all(c['folds'][s]!=fold for s,cid in d['keys'])
  pilot.append(dict(arm=a,fold=fold,delete_recall=mm['correct_delete']/mm['delete_opportunity'],known_delete_precision=mm['correct_delete']/(mm['correct_delete']+mm['wrong_delete']),wrong_delete=mm['wrong_delete']))
res=dict(checks=dict(checks),metrics={k:dict(v) for k,v in metrics.items()},pools={k:dict(v) for k,v in pool.items()},pool_targets=dict(pool_targets),gate=dict(gate),scope_stats={k:dict(v) for k,v in scope_stats.items()},pilot=pilot,strata_summary=dict(delete_candidates=len(strata_stats),inflation_quantiles=np.quantile([r['sampling_inflation'] for r in strata_stats],[0,.25,.5,.75,1]).tolist(),delete_points_le10=sum(r['delete_points']<=10 for r in strata_stats),max_expected_repeat=max(r['expected_repeats'] for r in strata_stats)),scope='Posthoc audit only; original predictions/gates unchanged; C ungated/oracle masks are diagnostics, not deployment outputs.')
res=json.loads(json.dumps(res,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
(O/'audit.json').write_text(json.dumps(res,indent=2))
for name,rr in [('candidate_diagnosis.csv',per_candidate),('sampling_diagnosis.csv',strata_stats)]:
 with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
print(json.dumps({k:v for k,v in res.items() if k not in ('pool_targets','scope_stats','pilot')},indent=2))
