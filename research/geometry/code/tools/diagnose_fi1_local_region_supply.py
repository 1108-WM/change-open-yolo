#!/usr/bin/env python3
"""Post-seal, original outer-training-only sample and feasible capacity census."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_local_region_supply as run
from tools.evaluate_fi1_deletion_blocks import iou
CATEGORIES=('target_only','known_pollution_only','target_neighbor_mixed','annotation_unknown','identity_unknown')


def labels(blocks,gt,parent,halo,meta):
    n=len(blocks);counts=np.zeros((n,3),np.int32);category=np.full(n,4,np.uint8);safe=np.zeros(n,bool);delta=np.full(n,np.nan)
    if not meta['stable']:
        counts[:,2]=[len(b) for b in blocks];return counts,category,safe,delta
    gid=meta['target'];total=int((gt==gid).sum());hit=int((gt[parent]==gid).sum());before=hit/(len(parent)+total-hit)
    for k,b in enumerate(blocks):
        raw=gt[halo[b]];target=int((raw==gid).sum());other=int(((raw>=1000)&(raw!=gid)).sum());unknown=len(b)-target-other;counts[k]=target,other,unknown
        category[k]=0 if target==len(b) else 1 if other==len(b) else 2 if target and other else 3
        after=(hit-target)/(len(parent)-len(b)+total-hit+target) if len(parent)-len(b)>=100 else before
        delta[k]=after-before;safe[k]=category[k]==1 and delta[k]>1e-12
    return counts,category,safe,delta


def arm_stats(blocks,gt,parent,halo,meta,annotation=None):
    counts,category,safe,delta=labels(blocks,gt,parent,halo,meta) if annotation is None else annotation
    sizes=np.asarray([len(b) for b in blocks],np.int64);bins=run.core.size_bins(sizes)
    stats=Counter(regions=len(blocks),safe_actions=int(safe.sum()),safe_parents=int(safe.any()),mixed_parents=int((category==2).any()),unknown_parents=int((category==4).any()),target_parents=int((category==0).any()),parents_with_regions=int(bool(blocks)),safe64=int((safe&(sizes>=64)).sum()),safe256=int((safe&(sizes>=256)).sum()))
    for k,name in enumerate(CATEGORIES):stats[name]=int((category==k).sum())
    for k,name in enumerate(run.core.SIZE_NAMES):
        stats['size_'+name]=int((bins==k).sum());stats['safe_size_'+name]=int((safe&(bins==k)).sum())
    coverage=run.core.union(blocks);stats['covered_unique_points']=len(coverage)
    if meta['stable']:
        raw=gt[halo[coverage]];stats['covered_pollution']=int(((raw>=1000)&(raw!=meta['target'])).sum());stats['covered_target']=int((raw==meta['target']).sum());stats['covered_unknown']=int((raw<1000).sum())
    stats['single_block_target_gain02']=int(np.any(delta>=.02-1e-12))
    return stats,(counts,category,safe,delta),coverage


def feasible_scores(blocks,safe,coverage,gt,parent,halo,meta,ids,sizes,before):
    if not meta['stable']:return before.copy(),before.copy(),before.copy(),dict(pure_removed=0,correct_point_removed=0)
    gid=meta['target'];correct=coverage[(gt[halo[coverage]]>=1000)&(gt[halo[coverage]]!=gid)]
    pointmask=run.core.remove(parent,halo,correct);point_score=iou(pointmask,gt,ids,sizes)
    pure=run.core.union([b for b,yes in zip(blocks,safe) if yes]);q=run.core.remove(parent,halo,pure);union_score=iou(q,gt,ids,sizes)
    single=before.copy()
    # Evaluate every valid target for each safe whole-region removal, without
    # silently assigning pool value only to the parent target.
    phit=np.array([int((gt[parent]==g).sum()) for g in ids]);index={int(g):k for k,g in enumerate(ids)}
    for block,yes in zip(blocks,safe):
        if not yes:continue
        values,counts=np.unique(gt[halo[block]],return_counts=True);hit=phit.copy()
        for value,count in zip(values,counts):
            if int(value) in index:hit[index[int(value)]]-=int(count)
        single=np.maximum(single,hit/(len(parent)-len(block)+sizes-hit))
    return point_score,union_score,single,dict(pure_removed=len(parent)-len(q),correct_point_removed=len(parent)-len(pointmask))


def diagnose_scene(s):
    c=run.read(run.OUT/'contract.json');fold_of_scene=c['folds'][s];guard=run.Guard(c,'diagnose',(fold_of_scene+1)%5);sys.addaudithook(guard.hook);d=run.OUT/'diagnosis'/s
    if (d/'receipt.json').exists():return run.read(d/'receipt.json')
    job=c['jobs'][s];assert run.sha(job['GT'])==c['GT_inputs'][job['GT']];gt=np.loadtxt(job['GT'],dtype=np.int64);ids,sizes=run.old.previous.original.audit.prior.valid_targets(gt,c['class_ids'])
    _,nt=run.old.previous.original.audit.prior.old.load_nt(job);baseline=np.zeros(len(ids))
    for p in nt.values():baseline=np.maximum(baseline,iou(p,gt,ids,sizes))
    original_meta={r['candidate_id']:r for r in run.read(run.SOURCE/'labels'/s/'meta.json')};stable_meta={r['candidate_id']:r for r in run.read(run.PREVIOUS/'labels'/s/'meta.json')}
    totals={f:{a:Counter() for a in run.ARMS} for f in range(5) if f!=fold_of_scene};best={f:{a:{mode:baseline.copy() for mode in ('point','union','single','either')} for a in run.ARMS} for f in totals}
    visible=baseline.copy();candidates=[];arrays={'gt_ids':ids,'gt_sizes':sizes,'NT_best':baseline};sizes_output=[]
    features=np.load(run.SOURCE/'features'/s/'data.npz');bank=np.load(run.OUT/'regions'/s/'regions.npz');region_meta={r['candidate_id']:r for r in run.read(run.OUT/'regions'/s/'meta.json')}
    for row in run.read(run.SOURCE/'plan'/s/'candidates.json'):
        cid=row['candidate_id'];parent=features[cid+'_parent'];halo=features[cid+'_halo'];present=np.isin(halo,parent);seen=features[cid+'_'+run.OBS+'_groups']>=2
        _,meta=run.old.core.stable_supervision(gt,parent,halo,original_meta[cid],c['class_ids']);assert meta['stable']==stable_meta[cid]['stable']
        before=iou(parent,gt,ids,sizes);blocks=run.old.unpack(bank,cid);annotation=labels(blocks,gt,parent,halo,meta)
        counts,category,safe,delta=annotation;arrays.update({cid+'_counts':counts,cid+'_category':category,cid+'_safe':safe,cid+'_delta':delta})
        if meta['stable']:
            pollution=np.flatnonzero(present&seen&(gt[halo]>=1000)&(gt[halo]!=meta['target']));visible=np.maximum(visible,iou(run.core.remove(parent,halo,pollution),gt,ids,sizes))
        else:pollution=np.empty(0,np.int32)
        geometry={}
        for arm in run.ARMS[2:]:
            indices=np.arange(len(blocks)) if arm=='geometry_raw' else bank[cid+'_budget'+arm.removeprefix('geometry')]
            selected=[blocks[i] for i in indices];ann=tuple(a[indices] for a in annotation);stats,ann,coverage=arm_stats(selected,gt,parent,halo,meta,ann)
            point,whole,single,extra=feasible_scores(selected,ann[2],coverage,gt,parent,halo,meta,ids,sizes,before);geometry[arm]=(stats,point,whole,single,extra)
            for i,b in zip(indices,selected):sizes_output.append(dict(scene=s,candidate_id=cid,source=row['source'],arm=arm,region=int(i),points=len(b),category=CATEGORIES[int(category[i])],safe=bool(safe[i]),delta=None if not meta['stable'] else float(delta[i])))
        for fold in totals:
            guard.fold=fold;gpath=run.PREVIOUS/'generators'/f'outer{fold}'/f'inner{fold_of_scene}'
            h=run.read(gpath/'history.json');assert s in h['predict_scenes'] and all(c['folds'][q] not in (fold,fold_of_scene) for q in h['train_scenes'])
            with np.load(gpath/'proposals'/s/'data.npz') as g:
                cached=run.old.unpack(g,cid,'_G');rebuilt=run.core.probability_blocks(parent,halo,g[cid+'_G_probability'],features[cid+'_'+run.OBS+'_groups'],features[cid+'_neighbors'])
                groups=dict(geometry)
                for arm,bb in (('G_cached',cached),('G_float32',rebuilt)):
                    stats,ann,coverage=arm_stats(bb,gt,parent,halo,meta);point,whole,single,extra=feasible_scores(bb,ann[2],coverage,gt,parent,halo,meta,ids,sizes,before);groups[arm]=(stats,point,whole,single,extra)
                    for i,b in enumerate(bb):sizes_output.append(dict(scene=s,fold=fold,candidate_id=cid,source=row['source'],arm=arm,region=i,points=len(b),category=CATEGORIES[int(ann[1][i])],safe=bool(ann[2][i]),delta=None if not meta['stable'] else float(ann[3][i])))
                for arm,(stats,point,whole,single,extra) in groups.items():
                    totals[fold][arm].update(stats)
                    for mode,score in (('point',point),('union',whole),('single',single),('either',np.maximum(whole,single))):best[fold][arm][mode]=np.maximum(best[fold][arm][mode],score)
                    gid=meta['target'];k=None if gid is None else int(np.flatnonzero(ids==gid)[0]);record=dict(scene=s,fold=fold,candidate_id=cid,source=row['source'],arm=arm,stable=meta['stable'],target=gid,visible_pollution=len(pollution),**dict(stats),**extra)
                    if k is not None:record.update(parent_iou=float(before[k]),NT_best=float(baseline[k]),whole_iou=float(whole[k]),single_iou=float(single[k]),pool_delta=float(max(whole[k],single[k])-baseline[k]))
                    candidates.append(record)
    features.close();bank.close();arrays['visible_known_other_best']=visible
    fold_results={}
    for fold in totals:
        arm_results={}
        for arm in run.ARMS:
            m=dict(totals[fold][arm]);modes=best[fold][arm]
            for mode in modes:
                arrays['fold'+str(fold)+'_'+arm+'_'+mode]=modes[mode];m[mode+'_pool_gain02']=int((modes[mode]-baseline>=.02-1e-12).sum())
            for t in (.5,.75,.78,.9,.95,.98):
                m['either_new_cross_'+str(t)]=int(((baseline<=t)&(modes['either']>t)).sum())
            m.update(safe_scene=int(m['safe_actions']>0),mixed_scene=int(m['target_neighbor_mixed']>0),unknown_scene=int(m['identity_unknown']>0),pool_gain_scene=int(m['either_pool_gain02']>0))
            arm_results[arm]=m
        fold_results[str(fold)]=arm_results
    d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'data.npz',**arrays);run.seal(d/'stats.json',dict(scene=s,diagnosed_as_training_outer=(fold_of_scene+1)%5,folds=fold_results,visible_pool_gain02=int((visible-baseline>=.02-1e-12).sum())))
    for name,rows in (('candidates.csv',candidates),('regions.csv',sizes_output)):
        with (d/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    oo={str(d/n):run.sha(d/n) for n in ('data.npz','stats.json','candidates.csv','regions.csv')};receipt=dict(complete=True,scene=s,outputs=oo);run.seal(d/'receipt.json',receipt);return receipt


def gate(m):
    req=run.core.SUPPLY_GATE;mapping={'min_regions':'regions','min_safe_actions':'safe_actions','min_safe_parents':'safe_parents','min_safe_scenes':'safe_scene','min_safe64':'safe64','min_safe256':'safe256','min_mixed_scenes':'mixed_scene','min_unknown_regions':'identity_unknown','min_unknown_parents':'unknown_parents','min_unknown_scenes':'unknown_scene','min_pool_gain02_targets':'either_pool_gain02','min_pool_gain02_scenes':'pool_gain_scene'}
    return {name:m.get(mapping[name],0)>=threshold for name,threshold in req.items()}


def diagnose():
    c=run.read(run.OUT/'contract.json');assert run.read(run.OUT/'regions_frozen.json')['complete'];outputs={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(diagnose_scene,s) for s in c['scenes']])):
            r=f.result();outputs.update(r['outputs'])
            if (i+1)%10==0:run.stage('outer_training_region_census',completed=i+1,total=200)
    fold_stats={f:{a:Counter() for a in run.ARMS} for f in range(5)};visible={f:0 for f in range(5)};all_candidates=[];poolrows=[];unioncapacity={a:{m:set() for m in ('point','union','single','either')} for a in run.ARMS[2:]};unionvisible=set();raw_rows=[]
    for s in c['scenes']:
        data=run.read(run.OUT/'diagnosis'/s/'stats.json');scene_fold=c['folds'][s]
        with np.load(run.OUT/'diagnosis'/s/'data.npz') as z:
            ids=z['gt_ids'];baseline=z['NT_best'];v=z['visible_known_other_best'];unionvisible.update((s,int(ids[k])) for k in np.flatnonzero(v-baseline>=.02-1e-12))
            for fold,arms in data['folds'].items():
                fold=int(fold);assert fold!=scene_fold;visible[fold]+=data['visible_pool_gain02']
                for a,counts in arms.items():
                    fold_stats[fold][a].update(counts)
                    for mode in ('point','union','single','either'):
                        scores=z['fold'+str(fold)+'_'+a+'_'+mode]
                        for k,gid in enumerate(ids):poolrows.append(dict(scene=s,fold=fold,arm=a,mode=mode,target=int(gid),NT_best=float(baseline[k]),diagnostic_best=float(scores[k]),delta=float(scores[k]-baseline[k])))
                        if a.startswith('geometry'):unioncapacity[a][mode].update((s,int(ids[k])) for k in np.flatnonzero(scores-baseline>=.02-1e-12))
        with (run.OUT/'diagnosis'/s/'candidates.csv').open() as f:all_candidates.extend(csv.DictReader(f))
        with (run.OUT/'diagnosis'/s/'regions.csv').open() as f:raw_rows.extend(csv.DictReader(f))
    # Summarize unique regional sample scales per fold; five sets overlap.
    sizes={}
    for fold in range(5):
        sizes[str(fold)]={}
        for arm in run.ARMS:
            rows=[r for r in raw_rows if r['arm']==arm and c['folds'][r['scene']]!=fold and (arm.startswith('geometry') or int(r['fold'])==fold)]
            positive=[int(r['points']) for r in rows if r['safe']=='True'];all_sizes=[int(r['points']) for r in rows]
            sizes[str(fold)][arm]=dict(safe_size_quantiles=np.quantile(positive,[0,.5,.9,1]).tolist() if positive else [],all_size_quantiles=np.quantile(all_sizes,[0,.5,.9,1]).tolist() if all_sizes else [])
    summary=dict(complete=True,training_scope_only=True,formal_OOF=False,new_training=False,parents=1599,scenes_union=200,folds={},union200_geometry_capacity={a:{m:len(v) for m,v in modes.items()} for a,modes in unioncapacity.items()},union200_visible_capacity=len(unionvisible),sizes=sizes,main_budget=96,no_new_actions=True,SL_modified=False,AP_run=False)
    for fold in range(5):
        assert sum(c['folds'][s]!=fold for s in c['scenes'])==160
        summary['folds'][str(fold)]=dict(train_scenes=160,parents=sum(1 for r in all_candidates if int(r['fold'])==fold and r['arm']=='geometry96'),visible_capacity=visible[fold],arms={a:dict(v) for a,v in fold_stats[fold].items()})
        main=summary['folds'][str(fold)]['arms']['geometry96'];g=gate(main);summary['folds'][str(fold)]['main_supply_gate']=g;summary['folds'][str(fold)]['main_supply_passed']=all(g.values())
    summary['all_training_supply_passed']=all(f['main_supply_passed'] for f in summary['folds'].values())
    d=run.OUT/'evaluation';d.mkdir(exist_ok=True)
    for name,rows in (('candidates.csv',all_candidates),('pool_targets.csv',poolrows)):
        with (d/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    run.seal(run.OUT/'summary.json',summary);outputs.update({str(p):run.sha(p) for p in (d/'candidates.csv',d/'pool_targets.csv',run.OUT/'summary.json')});run.seal(run.OUT/'diagnosis_frozen.json',dict(complete=True,outputs=outputs,training_scope_only=True));run.stage('supply_diagnosis_complete',supply_passed=summary['all_training_supply_passed'])


if __name__=='__main__':diagnose()
