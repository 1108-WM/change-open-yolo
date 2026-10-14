"""Read-only posthoc audit of frozen results; never selects a deployment rule."""
from pathlib import Path
import sys,json,hashlib
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools import run_fi1_region_affiliation as run
R=run.OUT;OUT=Path(__file__).resolve().parent
c=json.loads((R/'contract.json').read_text())
d=pd.read_csv(R/'evaluation/regions.csv')
p=pd.read_csv(R/'evaluation/candidates.csv')
t=pd.read_csv(R/'evaluation/targets.csv')
d=d.merge(p[['scene','candidate_id','arm','identity_scope']],on=['scene','candidate_id','arm'],validate='many_to_one')
answer={'posthoc_only':True,'new_training':False,'new_rules':False,'groups':{},'calibration':{},'evidence':{},'witnesses':[]}
for (split,arm),g in d.groupby(['split','arm']):
    pos=g[g.safe];q={'regions':len(g),'safe':len(pos),'positive_parents':len(pos[['scene','candidate_id']].drop_duplicates()),'positive_scenes':pos.scene.nunique(),'PR_AP':average_precision_score(g.safe,g.safe_probability),'high_confidence':{}}
    for threshold in [.5,.9,.98]:
        v=g[g.safe_probability>=threshold]
        q['high_confidence'][str(threshold)]={'regions':len(v),'safe':int(v.safe.sum()),'scopes':v.identity_scope.value_counts().to_dict(),'categories':v.category.value_counts().to_dict(),'point_fraction_sums':{k:float((v[k]*v.points).sum()) for k in ['target_fraction','other_fraction','unknown_fraction']}}
    q['positive_score_quantiles']=pos.safe_probability.quantile([0,.25,.5,.75,1]).to_dict()
    q['pure_target_as_unknown']=int(((g.target_fraction==1)&(g.unknown_probability>.9)).sum())
    q['unknown_as_target']=int(((g.unknown_fraction==1)&(g.target_probability>.9)).sum())
    answer['groups'][split+'_'+arm]=q
    official=json.loads((R/'summary.json').read_text())['splits'][split]['classification'][arm]
    assert np.isclose(q['PR_AP'],official['safe_average_precision'])
for arm in ['A','C']:
    grid=json.loads((R/'calibration'/f'{arm}.json').read_text())['grid']
    best=max(grid,key=lambda x:x['metrics']['precision'])
    answer['calibration'][arm]={'rules':len(grid),'max_precision':best,'distinct_metric_outcomes':len({json.dumps(v['metrics'],sort_keys=True) for v in grid}),'unknown_range':[min(v['metrics']['unknown_fraction'] for v in grid),max(v['metrics']['unknown_fraction'] for v in grid)],'all_support':all(v['metrics']['actions']>=3 and v['metrics']['action_scenes']>=2 for v in grid)}
cov=[]
for s in c['scenes']:
    with np.load(R/'features'/s/'data.npz') as z:
        for m in json.loads((R/'features'/s/'meta.json').read_text()):
            cid=m['candidate_id'];v=z[cid+'_views'];ids=z[cid+'_region_ids']
            usable=(v[:,:,0]>.5)&(v[:,:,1]>0)&(v[:,:,5]>.5)&(v[:,:,6]<.5)
            # Parent and a distinct unambiguous neighbor match; not a correctness label.
            competing=usable&(v[:,:,7]>.5)&(v[:,:,8]<.5)&(v[:,:,15]<.5)
            for j,rid in enumerate(ids):cov.append({'scene':s,'candidate_id':cid,'region':int(rid),'usable':int(usable[j].sum()),'competing':int(competing[j].sum())})
v=pd.DataFrame(cov);v.to_csv(OUT/'evidence_by_region.csv',index=False)
d=d.merge(v,on=['scene','candidate_id','region'],validate='many_to_one')
for (split,arm),g in d.groupby(['split','arm']):
    if arm!='C':continue
    q={}
    for name,sel in [('all',g),('safe',g[g.safe]),('unsafe',g[~g.safe])]:
        q[name]={'regions':len(sel),'usable_two':int((sel.usable>=2).sum()),'different_neighbor_two':int((sel.competing>=2).sum())}
    for label,sel in [('usable_two',g[g.usable>=2]),('insufficient',g[g.usable<2])]:
        q[label]={'regions':len(sel),'safe':int(sel.safe.sum()),'PR_AP':float(average_precision_score(sel.safe,sel.safe_probability)) if sel.safe.any() else None}
    answer['evidence'][split]=q
# Reconstruct pure-region opportunity witnesses for calibration and holdout only.
op=t[(t.arm=='A')&(t.split!='train')&(t.GT_pure_union_best-t.NT_best>=.02-1e-12)]
for s,gg in op.groupby('scene'):
    gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64)
    with np.load(run.SOURCE/'features'/s/'data.npz') as z,np.load(run.REGIONS/'regions'/s/'regions.npz') as bank:
        for target in gg.itertuples():
            for m in p[(p.scene==s)&(p.arm=='A')&(p.target==target.target)].itertuples():
                cid=m.candidate_id;parent=z[cid+'_parent'];halo=z[cid+'_halo'];blocks=run.prior.old.unpack(bank,cid);regionids=bank[cid+'_budget96']
                rows=d[(d.scene==s)&(d.candidate_id==cid)&(d.arm=='A')].set_index('region')
                pure=[blocks[int(i)] for i in regionids if rows.loc[int(i),'safe']]
                union=np.unique(np.concatenate(pure)) if pure else np.array([],int)
                q=np.setdiff1d(parent,halo[union]);hit=int((gt[q]==target.target).sum());iou=hit/(len(q)+int((gt==target.target).sum())-hit)
                if iou-target.NT_best<.02-1e-12:continue
                w={'scene':s,'split':target.split,'target':int(target.target),'candidate':cid,'source':m.source,'parent_iou':m.parent_iou,'NT_best':target.NT_best,'ideal_iou':iou,'safe_regions':len(pure),'union_points':len(union),'arms':{}}
                for arm in ['A','C']:
                    ss=d[(d.scene==s)&(d.candidate_id==cid)&(d.arm==arm)&d.safe]
                    eligible=ss[(ss.safe_probability>=.5)&(ss.target_probability<=.1)&(ss.unknown_probability<=.1)]
                    usableids=ss.loc[ss.usable>=2,'region'].astype(int)
                    useful=np.unique(np.concatenate([blocks[i] for i in usableids])) if len(usableids) else np.array([],int)
                    w['arms'][arm]={'safe_score_max':float(ss.safe_probability.max()),'safe_score_median':float(ss.safe_probability.median()),'pure_pass_loose_region_gate':len(eligible),'pure_usable_two':int((ss.usable>=2).sum()),'pure_distinct_competitor_two':int((ss.competing>=2).sum()),'safe_union_points_with_usable_evidence':len(useful)}
                answer['witnesses'].append(w)
assert len(op)==5
# Verify core source/snapshot/model/summary identities without rewriting old seals.
checked={}
for pth in [ROOT/'tools/fi1_region_affiliation_core.py',ROOT/'tools/run_fi1_region_affiliation.py',ROOT/'tools/evaluate_fi1_region_affiliation.py',ROOT/'tools/verify_fi1_region_affiliation.py']:
    h=hashlib.sha256(pth.read_bytes()).hexdigest();assert h==c['code'][str(pth)]
    assert h==hashlib.sha256((R/'code_snapshot'/pth.relative_to(ROOT)).read_bytes()).hexdigest();checked[str(pth)]=h
answer['core_source_hashes_verified']=checked
for name in ['models','rules','evaluation','predictions','probabilities']:
    manifest=json.loads((R/(name+'_frozen.json')).read_text());n=0
    for path,h in manifest['outputs'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h;n+=1
    answer[name+'_hashes_verified']=n
point_replays=0
for s in c['scenes']:
    with np.load(run.SOURCE/'features'/s/'data.npz') as original,np.load(R/'predictions'/s/'data.npz') as prediction:
        for row in json.loads((R/'features'/s/'meta.json').read_text()):
            cid=row['candidate_id']
            for arm in ['A','C']:
                assert np.array_equal(original[cid+'_parent'],prediction[cid+'_'+arm+'_points'])
                point_replays+=1
answer['independently_replayed_noop_pointsets']=point_replays
assert run.core.split_scenes(list(c['folds']),c['folds'])==c['splits']
physical=[{s.split('_')[0] for s in c['splits'][k]} for k in ['train','calibration','holdout']]
assert not (physical[0]&physical[1] or physical[0]&physical[2] or physical[1]&physical[2])
answer['physical_splits_verified']=True
def convert(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x))
(OUT/'audit.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2,default=convert)+'\n')
print(json.dumps(answer,ensure_ascii=False,indent=2,default=convert))
