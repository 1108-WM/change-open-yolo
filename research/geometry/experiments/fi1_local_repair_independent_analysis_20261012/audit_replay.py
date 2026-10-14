import json,csv,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

ROOT=Path('/home/jia/Wm/wm_open-yolo/OpenYOLO3D')
P=ROOT/'.local/fi1_candidate_local_repair_learning_development200_20261012_v1'
OUT=ROOT/'.local/fi1_local_repair_independent_analysis_20261012'
OUT.mkdir(exist_ok=True)
c=json.load(open(P/'contract.json'))
rows=list(csv.DictReader(open(P/'evaluation/candidates.csv')))
lookup={(r['scene'],r['candidate_id'],r['arm']):r for r in rows}
targetrows=list(csv.DictReader(open(P/'evaluation/targets.csv')))
baseline={(r['scene'],int(r['gt_id'])):float(r['original_NT_best']) for r in targetrows}
arms=c['arms']; totals={a:Counter() for a in arms}; pools=defaultdict(dict); detail=[]; calibration=defaultdict(Counter); labels_count=Counter(); unknown=[]; checks=Counter(); examples=[]

def score(mask,gid,gt,n):
 hit=np.count_nonzero(gt[mask]==gid)
 return hit/(len(mask)+n-hit) if len(mask)>=100 else 0.

def record(s,gid,arm,kind,q):
 key=(arm,kind); key2=(s,gid)
 pools[key][key2]=max(pools[key].get(key2,baseline[key2]),q)

for si,s in enumerate(c['scenes']):
 gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64)
 plans=json.load(open(P/'plan'/s/'candidates.json'))
 for bank in ('features','predictions'):
  f=P/bank/s/('data.npz' if bank=='features' else 'predictions.npz')
  seal=json.load(open(P/(bank+'_frozen.json')))['outputs']
  assert hashlib.sha256(f.read_bytes()).hexdigest()==seal[str(f)]
  checks['artifact_hashes']+=1
 with np.load(P/'features'/s/'data.npz') as z,np.load(P/'predictions'/s/'predictions.npz') as pred,np.load(P/'labels'/s/'labels.npz') as lab:
  for r in plans:
   cid=r['candidate_id']; parent=z[cid+'_parent'];halo=z[cid+'_halo']; member=np.isin(halo,parent);y=lab[cid+'_y'];labels_count['parents']+=1
   ref=lookup[s,cid,arms[0]];known=bool(ref['parent_target']);gid=int(ref['parent_target']) if known else None
   if known:
    hit=np.count_nonzero(gt[parent]==gid); n=np.count_nonzero(gt==gid);purity=hit/len(parent)
    labels_count['known']+=1;labels_count['known_purity_lt_half']+=purity<.5;labels_count['known_iou_lt_point1']+=float(ref['parent_iou'])<.1
    ideal=halo[gt[halo]==gid]
    if len(ideal)<100: ideal=np.union1d(ideal,np.setdiff1d(parent,ideal)[:100-len(ideal)])
    for a in arms:record(s,gid,a,'full_domain_ideal',score(ideal,gid,gt,n))
   for a in arms:
    rep=lookup[s,cid,a];p=pred[cid+'_'+a+'_probability'].astype(np.float32);seen=z[cid+'_'+a+'_groups']>=2
    add=(~member)&seen&(p[:,0]>=.8);delete=member&seen&(p[:,1]>=.8)
    reconstructed=halo[(member&~delete)|add]
    if len(reconstructed)<100: reconstructed=parent;add[:]=False;delete[:]=False
    actual=pred[cid+'_'+a+'_points'];assert np.array_equal(actual,reconstructed);checks['action_replay']+=1
    t=totals[a];t['actions']+=bool(add.any() or delete.any());t['add_only_actions']+=bool(add.any() and not delete.any());t['delete_only_actions']+=bool(delete.any() and not add.any());t['both_actions']+=bool(delete.any() and add.any())
    t['added']+=int(add.sum());t['deleted']+=int(delete.sum())
    for kind,mask,correct,confidence in [('add',add,y==0,p[:,0]),('delete',delete,y==1,p[:,1])]:
     for lo,hi in [(.8,.9),(.9,.95),(.95,1.001)]:
      m=mask&(confidence>=lo)&(confidence<hi);cc=calibration[(a,kind,lo)];cc['total']+=int(m.sum());cc['correct']+=int((m&correct).sum());cc['unknown']+=int((m&(y==2)).sum());cc['known_wrong']+=int((m&(y!=2)&~correct).sum())
    t['all_domain_points']+=len(halo);t['visible_two_groups']+=int(seen.sum());t['supervision_needed_add']+=int(((~member)&seen&(y==0)).sum());t['supervision_needed_delete']+=int((member&seen&(y==1)).sum())
    if not known:
     if add.any() or delete.any():unknown.append(dict(scene=s,candidate=cid,arm=a,source=r['source'],added=int(add.sum()),deleted=int(delete.sum()),parent=len(parent),parent_iou=float(rep['parent_iou']),max_target_probability=float(p[:,0].max())))
     continue
    assert abs(score(actual,gid,gt,n)-float(rep['repaired_iou']))<1e-12;checks['parent_iou']+=1
    ideal_obs=halo[(member&~seen)|(seen&(gt[halo]==gid))]
    if len(ideal_obs)<100:ideal_obs=np.union1d(ideal_obs,np.setdiff1d(parent,ideal_obs)[:100-len(ideal_obs)])
    add_only=np.union1d(parent,halo[add]);delete_only=halo[member&~delete]
    if len(delete_only)<100:delete_only=parent
    ideal_add=np.union1d(parent,halo[seen&(gt[halo]==gid)])
    ideal_del=halo[member&(~seen|(gt[halo]==gid))]
    if len(ideal_del)<100:ideal_del=np.union1d(ideal_del,np.setdiff1d(parent,ideal_del)[:100-len(ideal_del)])
    # Independent count formula for the combined visible oracle, including minimum size.
    exact_hit=int(((member&~seen&(gt[halo]==gid))|(seen&(gt[halo]==gid))).sum())
    fixed_wrong=int((member&~seen&(gt[halo]!=gid)).sum())
    formula=exact_hit/(n+max(fixed_wrong,100-exact_hit))
    assert abs(formula-score(ideal_obs,gid,gt,n))<1e-12
    checks['oracle_count_formula']+=1
    scenarios={'visible_ideal':ideal_obs,'visible_ideal_add_only':ideal_add,'visible_ideal_delete_only':ideal_del,'pred_add_only':add_only,'pred_delete_only':delete_only,'actual':actual}
    for kind,mask in scenarios.items():record(s,gid,a,kind,score(mask,gid,gt,n))
    needed_add=(~member)&seen&(y==0);needed_del=member&seen&(y==1)
    t['correct_add']+=int((add&(y==0)).sum());t['correct_delete']+=int((delete&(y==1)).sum());t['false_add']+=int((add&(y==1)).sum());t['false_delete']+=int((delete&(y==0)).sum())
    detail.append(dict(scene=s,candidate=cid,arm=a,source=r['source'],target=gid,parent_iou=float(rep['parent_iou']),actual_iou=score(actual,gid,gt,n),baseline=baseline[s,gid],ideal_iou=score(ideal_obs,gid,gt,n),needed_add=int(needed_add.sum()),needed_delete=int(needed_del.sum()),correct_add=int((add&(y==0)).sum()),correct_delete=int((delete&(y==1)).sum()),parent_points=len(parent),halo_points=len(halo)))
    if (float(rep['parent_delta'])<-.1) or rep['joint_value']=='True':
     x=z[cid+'_'+a+'_x'].astype(np.float32)
     examples.append(dict(scene=s,candidate=cid,arm=a,value=rep['joint_value']=='True',parent_points=len(parent),added=int(add.sum()),deleted=int(delete.sum()),added_probability_quantiles=np.quantile(p[add,0],[0,.5,1]).tolist() if add.any() else [],added_groups_hist=dict(zip(*[v.tolist() for v in np.unique(z[cid+'_'+a+'_groups'][add],return_counts=True)])),added_mask_support=float(x[add,60].mean()) if add.any() else None))
 if (si+1)%25==0:print('audited',si+1,flush=True)

pool_result={}
for (a,kind),v in pools.items():
 gain=[(s,g) for (s,g),x in v.items() if x>=baseline[s,g]+.02-1e-12]
 value=[(s,g) for (s,g),x in v.items() if x>=baseline[s,g]+.02-1e-12 or any(baseline[s,g]<=t<x for t in (.75,.9,.95))]
 crosses={str(t):sum(baseline[s,g]<=t<x for (s,g),x in v.items()) for t in (.5,.75,.9,.95)}
 fixed_gain=sum(s in c['fixed10'] for s,g in gain)
 pool_result[a+'::'+kind]=dict(gain02=len(gain),value=len(value),crosses=crosses,fixed10_gain02=fixed_gain,targets=gain)
result=dict(checks=dict(checks),labels=dict(labels_count),totals={a:dict(v) for a,v in totals.items()},pools=pool_result,calibration={str(k):dict(v) for k,v in calibration.items()},unknown_edits=sorted(unknown,key=lambda r:r['deleted'],reverse=True),examples=examples,scope='Posthoc diagnostic only; no models, predictions, thresholds or gates changed. Oracle scenarios use GT and are not deployable outputs. All pool scenario scores refer to original parent matched identity.')
(OUT/'audit.json').write_text(json.dumps(result,indent=2))
with (OUT/'candidate_diagnosis.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
print(json.dumps({k:v for k,v in result.items() if k not in ('unknown_edits','examples','pools')},indent=2))
print('POOLS',json.dumps({k:{n:x for n,x in v.items() if n!='targets'} for k,v in pool_result.items()},indent=2))
print('UNKNOWN_TOP',json.dumps(result['unknown_edits'][:8]))
