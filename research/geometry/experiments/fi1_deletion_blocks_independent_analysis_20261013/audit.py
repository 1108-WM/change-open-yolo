"""Independent read-only replay and posthoc diagnostics. Never produces deployment outputs."""
import sys,json,csv,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools import run_fi1_deletion_blocks as run
OUT=Path(__file__).resolve().parent; EXP=run.OUT
def read(p):return json.loads(Path(p).read_text())
def records(p):
    with open(p) as f:return list(csv.DictReader(f))
def stats(v):
    a=np.asarray(v,float)
    return dict(n=len(a),sum=float(a.sum()),q=np.quantile(a,[0,.5,.9,.99,1]).tolist()) if len(a) else dict(n=0)
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def ious(p,gt,ids,sizes):
    if len(p)<100:return np.zeros(len(ids))
    vals,cnt=np.unique(gt[p],return_counts=True);hits=dict(zip(vals,cnt));a=np.array([hits.get(g,0) for g in ids]);return a/(len(p)+sizes-a)

c=read(EXP/'contract.json');result={'posthoc_only':True,'hashes':{},'folds':{},'training':{},'blocks':{},'replay':{},'point_scores':{},'witness':[]}
for category,mapping in [('code',c['code']),('SL',c['frozen_SL_models']),('amendment',read(EXP/'execution_amendment_v2.json')['code'])]+[(s,read(EXP/(s+'_frozen.json'))['outputs']) for s in ('labels','generators','training_blocks','verifiers','predictions','evaluation')]:
    for p,h in mapping.items():assert digest(p)==h,(category,p)
    result['hashes'][category]=len(mapping)
print('hashes verified',result['hashes'],flush=True)
for fold in range(5):
    train={s for s in c['scenes'] if c['folds'][s]!=fold};hold=set(c['scenes'])-train
    for inner in [i for i in range(5) if i!=fold]+[None]:
        h=read(EXP/'generators'/f'outer{fold}'/('full' if inner is None else f'inner{inner}')/'history.json')
        expected={s for s in train if inner is None or c['folds'][s]!=inner}
        assert set(h['train_scenes'])==expected and set(h['predict_scenes'])=={s for s in c['scenes'] if c['folds'][s]==(fold if inner is None else inner)}
        assert {s[:9] for s in expected}.isdisjoint({s[:9] for s in hold}|{s[:9] for s in h['predict_scenes']})
    meta=read(EXP/'training_blocks'/f'outer{fold}'/'meta.json');ds=meta['details'];assert all(s in train for s,_,_ in meta['keys'])
    z=np.load(EXP/'training_blocks'/f'outer{fold}'/'data.npz');x,y=z['x'],z['y']
    result['training'][fold]=dict(blocks=len(y),parents=len({(s,k) for s,k,_ in meta['keys']}),scenes=len({s for s,k,_ in meta['keys']}),size=stats([d['points'] for d in ds]),positive_size=stats([d['points'] for d,t in zip(ds,y) if t[0]==1]),unknown_parent_blocks=sum(not d['stable'] for d in ds),true_delta=stats([d['delta'] for d in ds if d['stable']]),positive_delta=stats([d['delta'] for d,t in zip(ds,y) if t[0]==1]),models={})
    for kind in ('summary','relation'):
        h=read(EXP/'verifiers'/f'outer{fold}'/kind/'history.json');assert set(h['train_scenes'])==train
        result['training'][fold]['models'][kind]=dict(params=h['model_parameters'],positive_cost=h['positive_cost'],loss_first=h['epochs'][0]['loss'],loss_last=h['epochs'][-1]['loss'],optimizer_steps=len(h['epochs'])*int(np.ceil(len(y)/128)))
    result['folds'][fold]=dict(train=len(train),heldout=len(hold))

br=records(EXP/'evaluation/blocks.csv');cr=records(EXP/'evaluation/candidates.csv');tr=records(EXP/'evaluation/targets.csv')
reported={(r['scene'],r['candidate_id'],r['arm']):r for r in cr};targets={(r['scene'],r['arm'],int(r['gt_id'])):r for r in tr}
for source in ('B','G'):
    rr=[r for r in br if r['source']==source];stable=[r for r in rr if r['stable']=='True']
    item=dict(size=stats([int(r['points']) for r in rr]),stable_size=stats([int(r['points']) for r in stable]),pure_size=stats([int(r['points']) for r in stable if r['known_other_only']=='True']),kinds={})
    for kind in ('summary','relation'):
        selected=[r for r in rr if r[kind+'_selected']=='True'];ss=[r for r in selected if r['stable']=='True']
        true=np.array([float(r['delta']) for r in stable]);pred=np.array([float(r[kind+'_expected_delta']) for r in stable]);tp=sum(int(r['pollution_deleted']) for r in ss);fp=sum(int(r['target_deleted']) for r in ss)
        item['kinds'][kind]=dict(selected=len(selected),selected_stable=len(ss),selected_unstable=len(selected)-len(ss),stable_safe=sum(r['known_other_only']=='True' and float(r['delta'])>0 for r in ss),stable_other=tp,stable_target=fp,stable_unknown=sum(int(r['unknown_deleted']) for r in ss),delta_MAE=float(np.abs(pred-true).mean()),zero_baseline_delta_MAE=float(np.abs(true).mean()),selected_predicted_negative_delta=sum(float(r[kind+'_expected_delta'])<0 for r in selected),selected_actual_negative_delta=sum(float(r['delta'])<0 for r in ss),probability_bins={})
        for lo,hi in [(0,.1),(.1,.5),(.5,.9),(.9,.99),(.99,1.0001)]:
            rows=[r for r in rr if lo<=float(r[kind+'_safe_probability'])<hi];sr=[r for r in rows if r['stable']=='True'];pure=sum(r['known_other_only']=='True' and float(r['delta'])>0 for r in sr)
            item['kinds'][kind]['probability_bins'][str(lo)]=dict(blocks=len(rows),stable=len(sr),safe=pure,target=sum(int(r['target_deleted']) for r in sr),other=sum(int(r['pollution_deleted']) for r in sr))
    result['blocks'][source]=item
scores=defaultdict(list);counts=Counter();aggregate={a:Counter() for a in run.ARMS};pool=Counter();oracle=Counter();witness=read(EXP/'block_opportunity_diagnosis.json')['witnesses'];wkeys={(w['scene'],w['candidate_id']) for w in witness}
for si,s in enumerate(c['scenes']):
    gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);assert digest(c['jobs'][s]['GT'])==c['GT_inputs'][c['jobs'][s]['GT']]
    vals,sizes=np.unique(gt,return_counts=True);mask=(sizes>=100)&np.isin(vals//1000,c['class_ids']);ids=vals[mask];sizes=sizes[mask]
    _,nt=run.previous.original.audit.prior.old.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
    for p in nt.values():baseline=np.maximum(baseline,ious(p,gt,ids,sizes))
    best={a:baseline.copy() for a in run.ARMS};obest={a:baseline.copy() for a in ['visible_known_other','B_proposed_known_other','G_proposed_known_other','B_pure_blocks','G_pure_blocks']};fold=c['folds'][s]
    metadata={r['candidate_id']:r for r in read(EXP/'labels'/s/'meta.json')}
    with np.load(run.SOURCE/'features'/s/'data.npz') as ft,np.load(EXP/'generators'/f'outer{fold}'/'full'/'proposals'/s/'data.npz') as prop,np.load(EXP/'predictions'/s/'predictions.npz') as pred,np.load(EXP/'labels'/s/'data.npz') as labels:
        for r in read(run.SOURCE/'plan'/s/'candidates.json'):
            cid=r['candidate_id'];p=ft[cid+'_parent'];halo=ft[cid+'_halo'];present=np.isin(halo,p);vis=ft[cid+'_'+run.OBS+'_groups']>=2;y=labels[cid+'_y'];meta=metadata[cid]
            if meta['stable']:
                q=np.setdiff1d(p,halo[present&vis&(y==1)])
                if len(q)<100:q=p
                obest['visible_known_other']=np.maximum(obest['visible_known_other'],ious(q,gt,ids,sizes))
            for source in ('B','G'):
                stored=prop[cid+'_'+source+'_probability'];prob=stored.astype(float);idx=prop[cid+'_'+source+'_block_indices'];offs=prop[cid+'_'+source+'_block_offsets'];bs=[idx[a:b] for a,b in zip(offs[:-1],offs[1:])]
                actual=present&vis&(stored[:,1]>=.8)
                assert np.array_equal(np.sort(idx),np.flatnonzero(actual))
                counts[source+'_fp16_threshold_extra']+=int((actual&(prob[:,1]<.8)).sum())
                counts[source+'_fp16_threshold_extra_stable_target']+=int((actual&(prob[:,1]<.8)&(y==0)).sum())
                counts['proposal_membership_replays']+=1
                if meta['stable']:
                    q=np.setdiff1d(p,halo[actual&(y==1)])
                    if len(q)<100:q=p
                    obest[source+'_proposed_known_other']=np.maximum(obest[source+'_proposed_known_other'],ious(q,gt,ids,sizes))
                    pure=[b for b in bs if np.all(y[b]==1)]
                    q=np.setdiff1d(p,halo[np.concatenate(pure)]) if pure else p
                    if len(q)<100:q=p
                    obest[source+'_pure_blocks']=np.maximum(obest[source+'_pure_blocks'],ious(q,gt,ids,sizes))
                for kind in ('raw','summary','relation'):
                    arm=source+'_'+kind;selected=np.ones(len(bs),bool) if kind=='raw' else pred[cid+'_'+arm+'_block_prediction'][:,0].astype(float)>=.9
                    chosen=[b for b,yes in zip(bs,selected) if yes];delete=halo[np.concatenate(chosen)] if chosen else np.empty(0,int);expected=np.setdiff1d(p,delete)
                    if len(expected)<100:expected=p
                    assert np.array_equal(expected,pred[cid+'_'+arm+'_points']);counts['action_rule_replays']+=1
                if meta['stable']:
                    for name,mask in [('visible_pollution',present&vis&(y==1)),('visible_target',present&vis&(y==0))]:
                        # Keep all pollution probabilities, a deterministic subset of target scores.
                        values=prob[mask,1];scores[source+'_'+name].append(values if name=='visible_pollution' else values[::10])
                        counts[source+'_'+name]+=int(mask.sum());counts[source+'_'+name+'_above08']+=int((values>=.8).sum())
                if (s,cid) in wkeys:
                    pure=[b for b in bs if meta['stable'] and np.all((gt[halo[b]]>=1000)&(gt[halo[b]]!=meta['target']))]
                    result['witness'].append(dict(scene=s,candidate=cid,source=source,blocks=len(bs),points=len(idx),pure_blocks=len(pure),pure_points=sum(len(b) for b in pure),other_score=stats(prob[present&vis&(y==1),1])))
            for arm in run.ARMS:
                repaired=pred[cid+'_'+arm+'_points'];assert np.isin(repaired,p).all();deleted=np.setdiff1d(p,repaired);row=reported[s,cid,arm];assert len(deleted)==int(row['deleted'])
                after=ious(repaired,gt,ids,sizes);best[arm]=np.maximum(best[arm],after);counts['pointsets']+=1
                if row['target']:
                    target=int(row['target']);lost=int((gt[deleted]==target).sum());other=int(((gt[deleted]>=1000)&(gt[deleted]!=target)).sum());assert lost==int(row['deleted_target']) and other==int(row['deleted_other'])
                    k=int(np.flatnonzero(ids==target)[0]);assert abs(after[k]-float(row['repaired_iou']))<1e-12
                    aggregate[arm].update(deleted_target=lost,deleted_other=other)
            counts['parents']+=1
    for arm in run.ARMS:
        for k,g in enumerate(ids):assert abs(best[arm][k]-float(targets[s,arm,int(g)]['expanded_NT_best']))<1e-12;counts['pool_targets']+=1
        pool[arm]+=int((best[arm]-baseline>=.02-1e-12).sum())
    for arm in obest:oracle[arm]+=int((obest[arm]-baseline>=.02-1e-12).sum())
    if si%20==0:print('replayed',si+1,flush=True)
for key,v in scores.items():result['point_scores'][key]=stats(np.concatenate(v))
summary=read(EXP/'summary.json')
for arm,a in aggregate.items():
    for k,v in a.items():assert v==summary['arms'][arm][k]
    assert pool[arm]==summary['arms'][arm]['pool_gain02']
result['replay']=dict(counts=counts,totals=aggregate,pool_gain02=pool,stable_known_other_posthoc_oracles=oracle)
(OUT/'audit.json').write_text(json.dumps(result,indent=2))
print('complete',dict(counts),flush=True)
