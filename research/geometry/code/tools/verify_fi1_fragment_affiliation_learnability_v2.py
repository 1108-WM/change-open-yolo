#!/usr/bin/env python3
"""Verify feature/model identities, OOF replay, and full union geometry value."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,time,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from tools import run_fi1_fragment_affiliation_learnability as run
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
from tools.fi1_r3_core import gt_iou

def main():
    out=run.OUT;c=run.diag.read(out/'contract.json');sf=run.diag.read(out/'features_frozen.json');pf=run.diag.read(out/'predictions_frozen.json');results=run.diag.read(out/'results.json');source=run.diag.read(run.SOURCE/'contract.json')
    checks=Counter();bound=dict(c['source_files']);bound.update(c['code']);bound.update(sf['outputs']);bound.update(pf['outputs']);bound.update(source['inputs']);bound[c['model_checkpoint']]=c['model_sha256']
    run.diag.parent.diag.prior.src.ctl.base.verify_files(bound);checks['bound_files']=len(bound)
    for p,h in c['code'].items():assert sha(out/'code_snapshot'/Path(p).relative_to(ROOT))==h
    sys.addaudithook(run.diag.Guard(source,True).hook)
    items,labels,lookup=run._load_labels(c);known=labels>=0
    with np.load(out/'features.npz') as z:S=z['summary'];V=z['visual']
    assert len(items)==39405 and (labels==1).sum()==2377 and (labels==0).sum()==15129 and (labels<0).sum()==21899
    folds=np.array([c['folds'][s] for s,_,_ in items]);pred={name:np.load(out/(name+'_oof.npy')) for name in ('summary','summary_plus_visual')}
    for name,X in (('summary',S),('summary_plus_visual',np.c_[S,V])):
        for fold in range(5):
            test=folds==fold;train=(folds!=fold)&known;m=joblib.load(out/'models'/name/f'fold{fold}.joblib');replayed=m.predict_proba(X[test])[:,1];assert np.array_equal(replayed,pred[name][test]);checks['scores_replayed']+=int(test.sum())
            r=results['results'][name]['folds'][str(fold)]
            trainscenes={items[i][0].split('_')[0] for i in np.flatnonzero(train)};testscenes={items[i][0].split('_')[0] for i in np.flatnonzero(test)};assert not trainscenes&testscenes
            assert sorted(trainscenes)==sorted({s.split('_')[0] for s in r['train_scenes']})
            for split,idx in (('training',train),('heldout',test&known)):
                pp=m.predict_proba(X[idx])[:,1];y=labels[idx];assert np.isclose(r[split]['roc_auc'],roc_auc_score(y,pp),atol=1e-14);assert np.isclose(r[split]['pair_average_precision'],average_precision_score(y,pp),atol=1e-14);checks['fold_metric_reconstructions']+=1
        y=labels[known];pp=pred[name][known];r=results['results'][name]['pooled'];assert np.isclose(r['roc_auc'],roc_auc_score(y,pp),atol=1e-14);assert np.isclose(r['pair_average_precision'],average_precision_score(y,pp),atol=1e-14)
    # Per-scene independent action composition and union quality, after sealed OOF.
    diagnostics={name:{str(th):Counter() for th in (.5,.9)} for name in pred};newtargets={name:{str(th):{t:set() for t in c['thresholds']} for th in (.5,.9)} for name in pred};scene_outputs={};counterexample_scores=[];oracle_scores=[];union_rows=[]
    disjoint=np.zeros(len(items),bool);no_joint=np.zeros(len(items),bool);no_views=np.zeros(len(items),bool)
    oldoracle=run.diag.read(run.SOURCE/'summary.json')['opportunity_rows'];oldbad=run.diag.read(run.SOURCE/'identity_counterexamples.json')['rows']
    for scene in c['scenes']:
        print('verify_geometry',scene,flush=True);ii=np.array([i for i,(s,l,r) in enumerate(items) if s==scene],int);nt,pts=run.diag.load_nt(source['jobs'][scene]);keys=list(pts)
        raw=np.load(source['jobs'][scene]['paths']['processed'],mmap_mode='r')[:,:9]
        with np.load(out/'scenes'/scene/'features.npz') as z:assert np.array_equal(z['summary'],S[ii]) and np.array_equal(z['visual'],V[ii])
        meta=run.diag.read(out/'scenes'/scene/'region_views.json');rows_by_key={(r['left'],r['right']):r for r in run.diag.read(run.SOURCE/'features'/scene/'pairs.json')}
        with np.load(out/'scenes'/scene/'region_embeddings.npz') as z:emb=z['embeddings'].astype(np.float32)
        assert len(meta)==len(ii)
        for j,index in enumerate(ii):
            s,l,r=items[index];assert (meta[j]['scene'],meta[j]['left'],meta[j]['right'])==(s,l,r)
            feature=rows_by_key[(l,r)];disjoint[index]=feature['shared_points']==0;no_joint[index]=not feature['views']['joint_groups'];no_views[index]=not meta[j]['views']
            common=np.intersect1d(pts[l],pts[r],assume_unique=True);a=np.setdiff1d(pts[l],common,assume_unique=True);b=np.setdiff1d(pts[r],common,assume_unique=True)
            ownS,ownV=run._row_feature(raw,pts,rows_by_key[(l,r)]) if j==0 else (None,None)
            if ownS is not None:assert np.array_equal(ownS,S[index]) and np.array_equal(ownV,V[index,:len(ownV)]);checks['base_feature_replays']+=1
            groups=[v['group'] for v in meta[j]['views']];assert len(groups)==len(set(groups))<=3
            witness=feature['views']['witnesses'];ranked=sorted(witness,key=lambda w:(-min(w['visible_a']/len(a),w['visible_b']/len(b)),-min(w['visible_a'],w['visible_b']),w['frame']));selected=[];used=set()
            for w in ranked:
                if w['group'] in used:continue
                selected.append(w);used.add(w['group'])
                if len(selected)==3:break
            assert [(v['frame'],v['group'],v['visible_left'],v['visible_right']) for v in meta[j]['views']]==[(w['frame'],w['group'],w['visible_a'],w['visible_b']) for w in selected]
            checks['GT_free_view_selections']+=1
            for vi,v in enumerate(meta[j]['views']):
                close=np.dot(emb[j,vi,0],emb[j,vi,1]);assert np.isclose(close,v['dino_cosine'],atol=.0015);assert np.isfinite(v['sensor_depth']).all();checks['view_embedding_cosines']+=1
            vv=V[index,-61:];assert np.isclose(vv[0],len(groups)/3)
            if groups:
                cos=np.array([r['dino_cosine'] for r in meta[j]['views']]);assert np.allclose(vv[1:5],[cos.mean(),cos.min(),cos.max(),cos.std()],atol=1e-7)
                sensor=np.array([r['sensor_depth'] for r in meta[j]['views']]);assert np.allclose(vv[13:37],sensor.mean(0),atol=1e-7) and np.allclose(vv[37:],sensor.std(0),atol=1e-7)
            else:assert not emb[j].any() and not vv.any()
            checks['region_pair_identities']+=1
        gt=np.loadtxt(source['jobs'][scene]['GT'],dtype=np.int64);assert sha(source['jobs'][scene]['GT'])==source['GT_inputs'][source['jobs'][scene]['GT']]
        base,ids=gt_iou([pts[k] for k in keys],gt,source['class_ids']);bestNT=base.max(0,initial=0);keypos={k:i for i,k in enumerate(keys)};per_scene={}
        for name,pp in pred.items():
            per_scene[name]={}
            for threshold in (.5,.9):
                accepted=ii[pp[ii]>=threshold];stat=diagnostics[name][str(threshold)];unique={};parentbest={};parents={}
                for index in accepted:
                    s,l,r=items[index];p=np.union1d(pts[l],pts[r]);token=hashlib.sha256(p.astype('<i8').tobytes()).hexdigest();unique.setdefault(token,p);parentbest.setdefault(token,[]).append(np.maximum(base[keypos[l]],base[keypos[r]]))
                    parents.setdefault(token,(l,r))
                    stat['accepted_pairs']+=1;stat['known_same']+=int(labels[index]==1);stat['known_different']+=int(labels[index]==0);stat['unknown']+=int(labels[index]<0);stat['NT_exact_duplicate_pairs']+=int(any(np.array_equal(p,pts[k]) for k in keys))
                    for parent,extra in ((pts[l],np.setdiff1d(pts[r],pts[l],assume_unique=True)),(pts[r],np.setdiff1d(pts[l],pts[r],assume_unique=True))):
                        gg,nn=np.unique(gt[parent],return_counts=True);gid=int(gg[nn.argmax()]);reliable=gid in set(ids.tolist()) and nn.max()>=20 and nn.max()/len(parent)>=.75
                        if reliable:
                            target=int(np.sum(gt[extra]==gid));pollution=len(extra)-target;stat['reliable_anchor_directions']+=1;stat['added_target_points']+=target;stat['added_pollution_points']+=pollution;stat['target_majority_directions']+=target>pollution
                uni=list(unique.values());iou,_=gt_iou(uni,gt,source['class_ids']);best=iou.max(0,initial=0);stat['unique_unions']+=len(unique)
                for j,(token,p) in enumerate(unique.items()):
                    bestid=int(ids[iou[j].argmax()]);hit=int(np.sum(gt[p]==bestid));gsize=int(np.sum(gt==bestid));l,r=parents[token];direction=[]
                    for par,extra in ((pts[l],np.setdiff1d(pts[r],pts[l],assume_unique=True)),(pts[r],np.setdiff1d(pts[l],pts[r],assume_unique=True))):
                        target=int(np.sum(gt[extra]==bestid));direction.append(dict(added_target_points=target,added_pollution_points=len(extra)-target,parent_target_points=int(np.sum(gt[par]==bestid))))
                    union_rows.append(dict(model=name,threshold=threshold,scene=scene,points_sha256=token,point_count=len(p),parents=[l,r],diagnostic_best_gt=bestid,iou=float(iou[j].max()),purity=hit/len(p),coverage=hit/gsize,directions_to_union_best_gt=direction))
                for t in c['thresholds']:
                    newly=ids[(bestNT<=t)&(best>t)];newtargets[name][str(threshold)][t].update((scene,int(g)) for g in newly);stat[f'qualified_union_{t:.2f}']+=int((iou.max(1,initial=0)>t).sum());stat[f'redundant_targets_{t:.2f}']+=int(((bestNT>t)&(best>t)).sum())
                    for j,token in enumerate(unique):
                        pb=np.max(parentbest[token],axis=0);stat[f'parent_improved_targets_{t:.2f}']+=int(((pb<=t)&(iou[j]>t)).sum());stat[f'parent_lost_targets_{t:.2f}']+=int(((pb>t)&(iou[j]<=t)).sum())
                per_scene[name][str(threshold)]=dict(accepted_pairs=len(accepted),unique_unions=len(unique),new_targets={f'{t:.2f}':ids[(bestNT<=t)&(best>t)].tolist() for t in c['thresholds']})
        scene_outputs[scene]=per_scene
        local={(items[i][1],items[i][2]):i for i in ii}
        for op in oldoracle:
            if op['scene']!=scene:continue
            index=local.get((op['left'],op['right']),local.get((op['right'],op['left'])))
            oracle_scores.append(dict(scene=scene,gt_id=op['gt_id'],retrieved=index is not None,pair_iou=op['pair_iou'],identity_label=lookup[items[index]]['identity_label'] if index is not None else None,scores={name:float(pp[index]) if index is not None else None for name,pp in pred.items()}))
        for ex in oldbad:
            if ex['scene']!=scene:continue
            index=local.get((ex['left_geometry'],ex['right_geometry']),local.get((ex['right_geometry'],ex['left_geometry'])))
            counterexample_scores.append(dict(scene=scene,left_class=ex['left_class'],right_class=ex['right_class'],scores={name:float(pp[index]) for name,pp in pred.items()}))
    for name in pred:
        for th in (.5,.9):
            stat=diagnostics[name][str(th)];published=results['results'][name]['actions'][str(th)]
            for key,pub in (('accepted_pairs','accepted'),('known_same','same'),('known_different','different'),('unknown','unknown'),('reliable_anchor_directions','reliable_anchor_directions'),('added_target_points','added_target_points'),('added_pollution_points','added_pollution_points'),('target_majority_directions','target_majority_directions')):assert stat[key]==published[pub],(name,th,key)
            stat['new_over_complete_NT']={f'{t:.2f}':len(v) for t,v in newtargets[name][str(th)].items()};assert stat['new_over_complete_NT']==published['new_over_complete_NT']
    frozen={str(out/q):sha(out/q) for q in ('contract.json','features_frozen.json','predictions_frozen.json','results.json','decision.json','report.md')}
    subset={}
    for subsetname,mask in (('disjoint',disjoint),('overlapping',~disjoint),('no_joint_mask_support',no_joint),('no_selected_visual_views',no_views)):
        index=mask&known;subset[subsetname]=dict(all_pairs=int(mask.sum()),unknown=int((mask&~known).sum()),model_metrics={})
        for name,pp in pred.items():
            subset[subsetname]['model_metrics'][name]=run.metrics(labels[index],pp[index]) if len(np.unique(labels[index]))>1 else None
    freeze_json(out/'hypothetical_union_quality.json',dict(complete=True,rows=union_rows,new_pointsets_saved=False,GT_chosen_identity_diagnostic_only=True))
    freeze_json(out/'geometry_supplement.json',dict(complete=True,diagnostics={n:{t:dict(v) for t,v in q.items()} for n,q in diagnostics.items()},subset_metrics=subset,per_scene=scene_outputs,oracle60_scores=oracle_scores,previous13_counterexample_scores=counterexample_scores,original_NT_retained_in_diagnostic=True,no_new_output=True))
    freeze_json(out/'independent_verified.json',dict(complete=True,checks=dict(checks),outputs=frozen,source_sha256=sha(__file__),fold_boundaries_verified=True,unknown_not_negative=True,own_semantic_or_instance_AP=False,SL_modified=False,production_modified=False,limitations='Fivefold development identities already exposed. Appearance and boundary augmentation jointly changed; no attribution to DINO alone. Dense fullframe tokens can include neighboring context. Geometry union checks hypothetical only.'))
    write_json(out/'execution.json',dict(status='complete',stage='independently_verified',updated=time.time()))
    print('verified',dict(checks),flush=True)

if __name__=='__main__':
    with threadpool_limits(4):main()
