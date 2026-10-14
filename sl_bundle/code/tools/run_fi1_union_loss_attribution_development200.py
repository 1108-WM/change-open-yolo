#!/usr/bin/env python3
"""Extend the existing frozen U four-arm replay to the preregistered development200.

Reuses original pool/features/scorer/selection/diagnostic/evaluator functions,
old90 predictions and matches, and the tested per-class matching projection.
The separate100 training reserve never participates in this run.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from tools import run_fi1_union_loss_attribution_train90 as legacy
from tools.fi1_core_ablation import source_pool,features_and_members,learned_selection
from tools.fi1_r2_core import geometry_features
from tools.fi1_r3_core import pair_overlap,independent_context
from tools.fi1_union_loss_attribution import ARMS,THRESHOLDS,key,factorial_selections,member_diagnostics
from tools.fi1_union_budget_sensitivity import conditional_effects
from tools.fi1_union_development_replay import Guard,project_multiclass
from tools.fi1_frontend_protocol import sha256_file as sha,freeze_json,write_json as write

SELECT=ROOT/'.local/fi1_union_representative_train300_20261007_v2'
PARENT=legacy.OUT
OLD=legacy.OLD
OUT=ROOT/'.local/fi1_union_loss_attribution_development200_20261007_v1'
base=legacy.base;read,rows=base.read,base.rows


def prepare(out):
    assert not out.exists() and sklearn.__version__=='1.3.2' and shutil.disk_usage(ROOT).free>=50*1024**3
    m=read(SELECT/'manifest.json');a=read(SELECT/'acceptance.json');p=read(PARENT/'contract.json');pa=read(PARENT/'acceptance.json')
    assert a['complete'] and pa['complete'] and len(m['cohorts']['development200'])==200
    scenes=m['cohorts']['development200'];old=m['cohorts']['historical90'];new=m['cohorts']['new_development110']
    assert set(old)==set(p['scenes']) and not ({s[:9] for s in scenes}&{s[:9] for s in m['cohorts']['reserve100']})
    inputs={str(SELECT/n):sha(SELECT/n) for n in ('contract.json','acceptance.json','manifest.json','final_verification.json','representative_population.json')}
    inputs.update({str(PARENT/n):sha(PARENT/n) for n in ('contract.json','acceptance.json','predictions_frozen.json','final_verification.json')})
    inputs[str(base.MODEL)]=base.MODEL_HASH;labels={}
    for scene in scenes:
        r=read(SELECT/'inventory'/(scene+'.json'));inputs[str(SELECT/'inventory'/(scene+'.json'))]=sha(SELECT/'inventory'/(scene+'.json'))
        inputs.update(r['input_hashes'])
        if scene in old:
            pr=read(PARENT/'scenes'/scene/'receipt.json');inputs.update(pr['outputs'])
            inputs[str(PARENT/'scenes'/scene/'receipt.json')]=sha(PARENT/'scenes'/scene/'receipt.json')
            for n in ('nt_geometry.json','nt_members.jsonl','nt_features.npy','nt_scores.npy'):
                q=OLD/'scenes'/scene/n;inputs[str(q)]=p['inputs'][str(q)]
            er=PARENT/'evaluation'/scene/'receipt.json';r=read(er);inputs[str(er)]=sha(er)
            labels.update(r['outputs'])
    for group in ('train30','train60','combined90'):
        for arm in ARMS:
            q=PARENT/'evaluation'/(group+'_'+arm+'_AP.npy');labels[str(q)]=pa['artifacts'][str(q.relative_to(ROOT))]
    q=PARENT/'evaluation/summary.json';labels[str(q)]=pa['artifacts'][str(q.relative_to(ROOT))]
    code=base.code_paths()
    for name in ('tools/fi1_union_loss_attribution.py','tools/run_fi1_union_loss_attribution_train90.py',
                 'tools/fi1_class_agnostic_ablation.py','tools/fi1_union_budget_sensitivity.py',
                 'tools/fi1_union_development_replay.py','tools/run_fi1_union_loss_attribution_development200.py',
                 'tests/test_fi1_union_development_replay.py','tests/test_fi1_union_loss_attribution.py',
                 'tests/test_fi1_class_agnostic_ablation.py'):
        code[str(ROOT/name)]=sha(ROOT/name)
    assert all(h==p['code'][q] for q,h in code.items() if q in p['code'])
    population={r['scene']:r for r in read(SELECT/'representative_population.json')}
    groups={**p['batches'],'historical90':old,'new110':new,'development200':scenes}
    for origin in sorted({population[s]['origin'] for s in scenes}):groups['origin_'+origin]=[s for s in scenes if population[s]['origin']==origin]
    c=dict(version=out.name,scenes=scenes,old90=old,new110=new,reserve100=m['cohorts']['reserve100'],reporting_groups=groups,
        bindings={s:m['bindings'][s] for s in scenes},inputs=inputs,label_inputs=labels,code=code,
        B_model=str(base.MODEL),B_model_sha256=base.MODEL_HASH,Dv3_training=p['Dv3_training'],B_training=p['B_training'],
        budget=800,arms=list(ARMS),definitions=p['definitions'],member_identity=p['member_identity'],tie_rule=p['tie_rule'],
        authorization='User requested execution of the expanded representative development200 replay, reusing previous code.',
        GT_policy='Seal all800 scene-arm predictions before any raw or certified cached development labels; exact development200 GT whitelist; training reserve100 labels/predictions/assets excluded; all heldout and final test prohibited.',
        scientific_reuse='Original source_pool,features_and_members,learned_selection,factorial_selections,member_diagnostics and original evaluator AST. Old90 assets exact reuse; new110 only recompute inference. One original assignment of all arm-selected members per new scene, tested original per-class projection for subset/rescore.',
        scope='development diagnostic: historical90 GT-exposed; all200 B-seen and some Dv3-seen; nonuniform strata/old90 sampling; no population-unbiased AP or end-to-end generalization claim',
        no_new_rule=True,new_training=False,new_visual_inference=False,new_scoring_model=False,
        frozen_B26_inference_for_new110=True,reserve100_read=False,heldout_read=False,final_test_read=False,production_modified=False,
        no_statistical_significance_claim=True)
    sys.addaudithook(Guard(c).hook);base.verify_files(inputs);base.verify_files(code)
    out.mkdir(parents=True);freeze_json(out/'contract.json',c)
    for q in code:
        dst=out/'code_snapshot'/Path(q).relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dst)
    freeze_json(out/'code_audit.json',dict(complete=True,original_scorer_and_evaluator_unchanged=True,old90_exact_reuse=True,
        new110_actual_pools_and_own_semantics=True,C_context_not_recomputed_after_restriction=True,
        B_D_new_scores_identical=True,reserve100_excluded=True,all_predictions_before_GT=True))
    write(out/'execution.json',dict(status='prepared',stage='tests',scenes=200))
    print('Prepared frozen development200 replay; reserve100 excluded, no GT read.',flush=True)


def check_features(pool,points,evidence,point_count,members,x):
    keys=[g['geometry_key'] for g in pool];iou,cover=pair_overlap([points[k] for k in keys],point_count)
    assert np.allclose(independent_context(members,keys,iou,cover),x[:,20:],atol=1e-14,rtol=0)
    own={g['geometry_key']:geometry_features(g,evidence[g['geometry_key']]) for g in pool}
    if members:assert np.array_equal(x[:,:20],np.asarray([own[m['geometry_key']][m['class_index']] for m in members]))
    expected=set()
    for g in pool:
        aliases=base.original_aliases(g);classes={a['class_index'] for a in aliases if 0<=a['class_index']<198}
        if any(a['source']!='native' for a in aliases):classes|=set(np.flatnonzero(own[g['geometry_key']][:,0]>0).tolist())
        if g['point_count']:expected|={(g['geometry_key'],cl) for cl in classes}
    assert expected=={key(m) for m in members}


def write_rows(path,members):
    with path.open('w') as f:
        for m in members:f.write(json.dumps(m,sort_keys=True)+'\n')


def predict(out):
    c=read(out/'contract.json');sys.addaudithook(Guard(c).hook);base.verify_files(c['inputs']);base.verify_files(c['code'])
    assert read(out/'tests_verified.json')['passed'];model=joblib.load(base.MODEL);outputs={};ntrows=nturows=0
    for n,scene in enumerate(c['scenes'],1):
        d=out/'scenes'/scene;spec=c['bindings'][scene]
        write(out/'execution.json',dict(status='running',stage='predictions',scene=scene,completed=n-1,total=200))
        if (d/'receipt.json').exists():
            r=read(d/'receipt.json');base.verify_files(r['outputs'])
        elif scene in c['old90']:
            d.mkdir(parents=True);source=PARENT/'scenes'/scene;pr=read(source/'receipt.json');base.verify_files(pr['outputs'])
            for q in source.iterdir():
                if q.is_file() and q.name!='receipt.json':(d/q.name).symlink_to(q)
            for name in ('nt_geometry.json','nt_members.jsonl','nt_features.npy','nt_scores.npy'):(d/name).symlink_to(OLD/'scenes'/scene/name)
            r={**{k:v for k,v in pr.items() if k!='outputs'},'old90_predictions_exact_reused':True,
               'outputs':{str(q):sha(q) for q in d.iterdir() if q.is_file()}}
            freeze_json(d/'receipt.json',r)
        else:
            gs,points,evidence=base.load_geometry(spec);nt=source_pool(gs);ntu=source_pool(gs,('native','track','pair_union'))
            def infer(pool):
                keys=[g['geometry_key'] for g in pool]
                members,x=features_and_members(pool,{k:points[k] for k in keys},{k:evidence[k] for k in keys},spec['point_count'])
                normal,scores=learned_selection(members,x,model)
                check_features(pool,points,evidence,spec['point_count'],members,x)
                return members,x,scores,normal
            m0,x0,s0,normal0=infer(nt);m1,x1,s1,normal1=infer(ntu)
            selected=factorial_selections(m0,s0,m1,s1,budget=800);assert selected['A']==normal0 and selected['D']==normal1
            p0={key(m):i for i,m in enumerate(m0)};p1={key(m):i for i,m in enumerate(m1)};ix=np.asarray([p1[key(m)] for m in m0],dtype=int)
            stable=[i for i in range(20) if i not in (14,15,16,17)]
            assert np.array_equal(x0[:,stable],x1[ix][:,stable])
            for arm in ARMS:
                universe=m0 if arm in ('A','C') else m1;independent=[]
                for m in universe:
                    k=key(m);old_score=arm=='A' or arm=='B' and k in p0
                    score=s0[p0[k]] if old_score else s1[p1[k]]
                    identity=m0[p0[k]] if arm in ('A','C') or arm=='B' and k in p0 else m
                    independent.append({**identity,'score':float(score)})
                independent.sort(key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))
                assert selected[arm]==independent[:800]
            diff=s1[ix]-s0
            score_stats=dict(common_scores_changed=int((diff!=0).sum()),common_scores_increased=int((diff>0).sum()),
                common_scores_decreased=int((diff<0).sum()),max_abs_change=float(np.abs(diff).max(initial=0)),
                common_source_features_changed=int(np.any(x0[:,14:18]!=x1[ix,14:18],axis=1).sum()),
                common_context_features_changed=int(np.any(x0[:,20:]!=x1[ix,20:],axis=1).sum()))
            d.mkdir(parents=True)
            for tag,pool,members,x,scores in (('nt',nt,m0,x0,s0),('ntu',ntu,m1,x1,s1)):
                write(d/(tag+'_geometry.json'),pool);write_rows(d/(tag+'_members.jsonl'),members)
                np.save(d/(tag+'_features.npy'),x,allow_pickle=False);np.save(d/(tag+'_scores.npy'),scores,allow_pickle=False)
            for arm in ARMS:write(d/(arm+'_selected.json'),selected[arm])
            counts={a:dict(selected=len(selected[a]),unique_geometry=len({m['geometry_key'] for m in selected[a]}),
                original_members=sum(key(m) in p0 for m in selected[a]),added_members=sum(key(m) not in p0 for m in selected[a])) for a in ARMS}
            r=dict(complete=True,scene=scene,GT_read=False,NT_members=len(m0),NTU_members=len(m1),added_members=len(m1)-len(m0),
                NT_geometry=len(nt),NTU_geometry=len(ntu),score_changes=score_stats,selected_counts=counts,
                independent_member_feature_context_score_sorting=True,A_exact_actual_NT=True,D_exact_normal_NTU=True,
                outputs={str(q):sha(q) for q in d.iterdir() if q.is_file()})
            freeze_json(d/'receipt.json',r)
        outputs.update(r['outputs']);outputs[str(d/'receipt.json')]=sha(d/'receipt.json');ntrows+=r['NT_members'];nturows+=r['NTU_members']
        if n%10==0:print(f'Sealed development predictions {n}/200 (old90 reuse/new110 actual-pool inference)',flush=True)
    base.verify_files(c['inputs']);base.verify_files(c['code'])
    freeze_json(out/'predictions_frozen.json',dict(complete=True,scene_arm_count=800,outputs=outputs,GT_read=False,
        old90_exact_reused=True,new110_inference_checked=True,NT_score_rows_checked=ntrows,NTU_feature_context_rows_checked=nturows,
        reserve100_read=False,contract_sha256=sha(out/'contract.json')))
    print('All800 scene-arm outputs sealed before development labels.',flush=True)


def sum_dict(values):
    if isinstance(values[0],dict):return {k:sum_dict([v[k] for v in values]) for k in values[0]}
    return sum(values)


def diagnosis(pool,points,gt,ev,selected,m0,ntkeys):
    keys=[g['geometry_key'] for g in pool];sizes=np.asarray([g['point_count'] for g in pool])
    ids,counts=np.unique(gt,return_counts=True);good=(ids>=1000)&(counts>=100)&np.isin(ids//1000,ev.VALID_CLASS_IDS)
    ids,counts=ids[good],counts[good];classes=np.asarray([list(ev.VALID_CLASS_IDS).index(int(g)//1000) for g in ids],dtype=int)
    iou,inter,psizes=base.geometry_overlaps([points[k] for k in keys],gt,ids,counts);assert np.array_equal(psizes,sizes)
    census={}
    for i,k in enumerate(keys):
        gs,ns=np.unique(gt[points[k]],return_counts=True);cc=dict(zip(map(int,gs),map(int,ns)));census[k]=cc
        assert np.array_equal(inter[i],np.asarray([cc.get(int(g),0) for g in ids]))
    arms={};covered={};checks=len(keys)*len(ids);originalkeys={key(m) for m in m0}
    for a in ARMS:arms[a],covered[a]=member_diagnostics(keys,sizes,iou,inter,classes,selected[a],m0,ntkeys)
    comparisons={};lookup={k:i for i,k in enumerate(keys)}
    for name,before,after in [('B_minus_A','A','B'),('C_minus_A','A','C'),('D_minus_C','C','D'),('D_minus_B','B','D'),('D_minus_A','A','D')]:
        mb,ma={key(m) for m in selected[before]},{key(m) for m in selected[after]};entered,left=ma-mb,mb-ma
        def describe(mkeys,t):
            correct=wrong=0
            for k,cl in mkeys:
                row=iou[lookup[k]];ok=bool((row[classes==cl]>t).any());correct+=ok;wrong+=not ok and bool((row>t).any())
            return dict(members=len(mkeys),correct_range=int(correct),wrong_class_good_range=int(wrong))
        comparisons[name]=dict(entered_members=len(entered),left_members=len(left),entered_new_members=len(entered-originalkeys),
            left_original_members=len(left&originalkeys),entered_at75=describe(entered,.75),left_at75=describe(left,.75),
            coverage={str(t):dict(gained_GT=len(covered[after][str(t)]-covered[before][str(t)]),lost_GT=len(covered[before][str(t)]-covered[after][str(t)])) for t in THRESHOLDS})
    d=dict(GT=len(ids),arms=arms,comparisons=comparisons,
        pool_coverage=dict(NT={str(t):int((iou[[k in ntkeys for k in keys]].max(axis=0,initial=0)>t).sum()) for t in THRESHOLDS},
                           NTU={str(t):int((iou.max(axis=0,initial=0)>t).sum()) for t in THRESHOLDS}),
        warning='range/class/redundancy descriptions are not evaluator FP counts or additive AP attribution')
    cache=dict(geometry_keys=np.asarray(keys),gt_ids=ids,gt_classes=classes,gt_sizes=counts,
               point_counts=sizes,iou=iou,intersections=inter)
    return d,cache,census,checks


def evaluate(out):
    c=read(out/'contract.json');guard=Guard(c,raw_GT_allowed=True);sys.addaudithook(guard.hook)
    seal=read(out/'predictions_frozen.json');assert seal['complete'] and seal['scene_arm_count']==800 and not seal['GT_read']
    base.verify_files(seal['outputs']);base.verify_files(c['code']);guard.gt_enabled=True;base.verify_files(c['label_inputs'])
    ev=base.evaluator();target=out/'evaluation';target.mkdir(exist_ok=True);matches={a:{} for a in ARMS};descs={};GT=checks=0
    for n,scene in enumerate(c['scenes'],1):
        source=out/'scenes'/scene;d=target/scene;spec=c['bindings'][scene]
        write(out/'execution.json',dict(status='running',stage='matches_and_diagnosis',scene=scene,completed=n-1,total=200))
        if (d/'receipt.json').exists():r=read(d/'receipt.json');base.verify_files(r['outputs'])
        elif scene in c['old90']:
            d.mkdir(parents=True);parentdir=PARENT/'evaluation'/scene;er=read(parentdir/'receipt.json')
            for name in ('diagnosis.json',*(a+'_matches.json' for a in ARMS)):(d/name).symlink_to(parentdir/name)
            r={**{k:v for k,v in er.items() if k!='outputs'},'old90_matches_exact_reused':True,
               'outputs':{str(q):sha(q) for q in d.iterdir() if q.is_file()}}
            freeze_json(d/'receipt.json',r)
        else:
            assert sha(spec['gt_path'])==spec['gt_sha256'];gt=np.loadtxt(spec['gt_path'],dtype=np.int64);assert len(gt)==spec['point_count']
            _,points,_=base.load_geometry(spec);pool=read(source/'ntu_geometry.json');sizes={g['geometry_key']:g['point_count'] for g in pool}
            m0=rows(source/'nt_members.jsonl');ntkeys={g['geometry_key'] for g in read(source/'nt_geometry.json')}
            selected={a:read(source/(a+'_selected.json')) for a in ARMS}
            desc,cache,census,localchecks=diagnosis(pool,points,gt,ev,selected,m0,ntkeys)
            unique={key(m):m for a in ARMS for m in selected[a]};universe=[unique[k] for k in sorted(unique)]
            masks=np.zeros((len(gt),len(universe)),dtype=bool)
            for i,m in enumerate(universe):masks[points[m['geometry_key']],i]=True
            gm,pm=ev.assign_instances_for_scan(dict(pred_masks=masks,
                pred_classes=np.asarray([m['class_index'] for m in universe],dtype=np.int64),
                pred_scores=np.asarray([m['score'] for m in universe],dtype=np.float64)),spec['gt_path']);del masks
            # Original union assignment independently checked against raw point-label census.
            for ci,label in enumerate(ev.CLASS_LABELS):
                rr=[m for m in universe if m['class_index']==ci and sizes[m['geometry_key']]>=100]
                assert len(rr)==len(pm[label])
                for m,p in zip(rr,pm[label]):
                    cc=census[m['geometry_key']];assert p['confidence']==m['score'] and p['vert_count']==sizes[m['geometry_key']]
                    assert p['void_intersection']==sum(v for gid,v in cc.items() if gid//1000 not in set(ev.VALID_CLASS_IDS))
                    assert {g['instance_id']:g['intersection'] for g in p['matched_gt']}=={g['instance_id']:cc[g['instance_id']] for g in gm[label] if cc.get(g['instance_id'],0)}
                    localchecks+=len(gm[label])
                    for g in p['matched_gt']:g.pop('matched_pred',None)
            d.mkdir();write(d/'diagnosis.json',desc);np.savez_compressed(d/'overlap.npz',**cache)
            for a in ARMS:
                projected=project_multiclass(dict(gt=gm,pred=pm),universe,selected[a],sizes,ev.CLASS_LABELS,ev.VALID_CLASS_IDS)
                write(d/(a+'_matches.json'),base.serializable_matches(projected))
            r=dict(complete=True,GT=desc['GT'],GT_sha256=spec['gt_sha256'],intersection_checks=localchecks,
                original_assignment_once_and_tested_projection=True,raw_GT_read=True,
                outputs={str(q):sha(q) for q in d.iterdir() if q.is_file()})
            freeze_json(d/'receipt.json',r)
        GT+=r['GT'];checks+=r['intersection_checks'];descs[scene]=read(d/'diagnosis.json')
        for a in ARMS:matches[a][scene]=read(d/(a+'_matches.json'))
        if n%10==0:print(f'Original matches and diagnostics {n}/200',flush=True)
    parent_summary=read(PARENT/'evaluation/summary.json');AP={};arrays={};effects={};aggs={};classes_report={}
    for group,scans in c['reporting_groups'].items():
        AP[group]={};classes_report[group]={}
        for a in ARMS:
            write(out/'execution.json',dict(status='running',stage='pooled_AP',group=group,arm=a))
            x,*_=ev.evaluate_matches({s:matches[a][s] for s in scans});AP[group][a]=base.metrics(ev,x)
            if group in ('train30','train60','historical90'):
                pg='combined90' if group=='historical90' else group
                assert np.array_equal(x,np.load(PARENT/'evaluation'/(pg+'_'+a+'_AP.npy'),allow_pickle=False),equal_nan=True)
                assert AP[group][a]==parent_summary['AP'][pg][a]
            q=target/(group+'_'+a+'_AP.npy');np.save(q,x,allow_pickle=False);arrays[str(q)]=sha(q)
            classes_report[group][a]=[dict(class_index=i,label=ev.CLASS_LABELS[i],GT=sum(sum(g['instance_id']>=1000 and g['vert_count']>=100 for g in matches[a][s]['gt'][ev.CLASS_LABELS[i]]) for s in scans),
                ap=float(np.mean(x[0,i,:9])) if np.isfinite(x[0,i,:9]).all() else None) for i in range(198)]
        effects[group]=conditional_effects(AP[group])
        prs=[read(out/'scenes'/s/'receipt.json') for s in scans]
        aggs[group]=dict(GT=sum(descs[s]['GT'] for s in scans),arms=sum_dict([descs[s]['arms'] for s in scans]),
            comparisons=sum_dict([descs[s]['comparisons'] for s in scans]),pool_coverage=sum_dict([descs[s]['pool_coverage'] for s in scans]),
            selected_counts=sum_dict([r['selected_counts'] for r in prs]),
            score_changes={k:(max if k=='max_abs_change' else sum)(r['score_changes'][k] for r in prs) for k in prs[0]['score_changes']},
            prebudget_counts={k:sum(r[k] for r in prs) for k in ('NT_members','NTU_members','added_members','NT_geometry','NTU_geometry')})
        print(json.dumps(dict(group=group,AP=AP[group],effects_pp=effects[group])),flush=True)
    base.verify_files(c['inputs']);base.verify_files(c['code']);base.verify_files(seal['outputs'])
    eroutputs={str(q):sha(q) for d in target.iterdir() if d.is_dir() for q in d.iterdir() if q.is_file()}
    freeze_json(target/'summary.json',dict(complete=True,AP_units='fraction',AP=AP,differences_pp=effects,AP_arrays=arrays,
        diagnostics=aggs,per_class_AP_and_GT=classes_report,evaluation_outputs=eroutputs,GT=GT,scenes=200,predictions=800,
        old90_AP_arrays_exact=True,independent_GT_intersections=checks,raw_GT_scans_read=sorted(set(c['new110'])&{Path(p).stem for p in guard.gt_opened}),
        scope=c['scope'],no_new_rule=True,new_training=False,new_visual_inference=False,reserve100_read=False,
        heldout_read=False,final_test_read=False,production_modified=False,no_statistical_significance_claim=True))
    freeze_json(out/'completion_verified.json',dict(complete=True,scenes=200,predictions=800,GT=GT,
        all_predictions_before_development_labels=True,old90_AP_arrays_exact=True,reserve100_read=False,
        tests=read(out/'tests_verified.json'),summary_sha256=sha(target/'summary.json'),independent_GT_intersections=checks))
    write(out/'execution.json',dict(status='complete',stage='evaluated'))


def verify(out):
    c=read(out/'contract.json');guard=Guard(c);sys.addaudithook(guard.hook);guard.gt_enabled=True
    s=read(out/'evaluation/summary.json');seal=read(out/'predictions_frozen.json');done=read(out/'completion_verified.json')
    assert s['complete'] and done['complete'] and done['summary_sha256']==sha(out/'evaluation/summary.json')
    for fs in (c['inputs'],c['code'],c['label_inputs'],seal['outputs'],s['AP_arrays'],s['evaluation_outputs']):base.verify_files(fs)
    ev=base.evaluator();uids={a:set() for a in ARMS};NT=NTU=GT=checks=0;scalar=effect=0
    for scene in c['scenes']:
        d=out/'scenes'/scene;r=read(d/'receipt.json');NT+=r['NT_members'];NTU+=r['NTU_members']
        er=read(out/'evaluation'/scene/'receipt.json');GT+=er['GT'];checks+=er['intersection_checks']
        m0=rows(d/'nt_members.jsonl');m1=rows(d/'ntu_members.jsonl');s0=np.load(d/'nt_scores.npy',allow_pickle=False);s1=np.load(d/'ntu_scores.npy',allow_pickle=False)
        old={key(m):(m,float(sc)) for m,sc in zip(m0,s0)};new={key(m):(m,float(sc)) for m,sc in zip(m1,s1)}
        assert set(old)<=set(new)
        sizes={g['geometry_key']:g['point_count'] for g in read(d/'ntu_geometry.json')}
        diag=read(out/'evaluation'/scene/'diagnosis.json')
        for a in ARMS:
            expected=[]
            for identity,(m,score) in (old.items() if a in ('A','C') else new.items()):
                if a=='B' and identity in old:m,score=old[identity]
                elif a=='C':score=new[identity][1]
                expected.append({**m,'score':score})
            expected.sort(key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))
            selected=read(d/(a+'_selected.json'));assert selected==expected[:800]
            match=read(out/'evaluation'/scene/(a+'_matches.json'));bylabel={label:[] for label in ev.CLASS_LABELS}
            for m in selected:
                if sizes[m['geometry_key']]>=100:bylabel[ev.CLASS_LABELS[m['class_index']]].append(m)
            current=set();pred_ids=[]
            for label in ev.CLASS_LABELS:
                assert len(bylabel[label])==len(match['pred'][label])
                for m,p in zip(bylabel[label],match['pred'][label]):
                    assert p['confidence']==m['score'] and p['vert_count']==sizes[m['geometry_key']]
                    current.add(p['uuid']);pred_ids.append(p['pred_id'])
                    for g in p['matched_gt']:
                        gg=next(x for x in match['gt'][label] if x['instance_id']==g['instance_id'])
                        witness=next(x for x in gg['matched_pred'] if x['uuid']==p['uuid'])
                        assert witness['confidence']==p['confidence'] and witness['intersection']==g['intersection'] and witness['pred_id']==p['pred_id']
            assert sorted(pred_ids)==list(range(len(pred_ids))) and len(current)==len(pred_ids) and not current&uids[a];uids[a]|=current
            if scene in c['old90']:
                assert selected==read(PARENT/'scenes'/scene/(a+'_selected.json'))
                assert match==read(PARENT/'evaluation'/scene/(a+'_matches.json'))
            dsc=diag['arms'][a]['all'];assert dsc['members']==len(selected)
            assert len(current)==len(selected)-dsc['small_ignored']
            for t in ('0.5','0.75','0.9'):
                q=dsc['thresholds'][t];assert sum(q[k] for k in ('correct_range','wrong_class_good_range','no_range_at_threshold'))==len(current)
                assert q['redundant_correct_members']==q['correct_range']-q['unique_correct_GT']
    assert NT==seal['NT_score_rows_checked'] and NTU==seal['NTU_feature_context_rows_checked'] and GT==s['GT'] and checks==s['independent_GT_intersections']
    for group,scans in c['reporting_groups'].items():
        for a in ARMS:
            x=np.load(out/'evaluation'/(group+'_'+a+'_AP.npy'),allow_pickle=False);assert x.shape==(1,198,10)
            calc=dict(ap=float(np.nanmean(x[0,:,:9])),ap50=float(np.nanmean(x[0,:,0])),ap25=float(np.nanmean(x[0,:,9])))
            for metric in calc:assert abs(calc[metric]-s['AP'][group][a][metric])<1e-14;scalar+=1
            if group in ('train30','train60','historical90'):
                pg='combined90' if group=='historical90' else group
                assert np.array_equal(x,np.load(PARENT/'evaluation'/(pg+'_'+a+'_AP.npy'),allow_pickle=False),equal_nan=True)
        assert conditional_effects(s['AP'][group])==s['differences_pp'][group];effect+=18
        assert s['diagnostics'][group]['GT']==sum(read(out/'evaluation'/scene/'receipt.json')['GT'] for scene in scans)
    assert not guard.gt_opened
    freeze_json(out/'final_verification.json',dict(complete=True,scenes=200,predictions=800,GT=GT,
        independent_all_prebudget_sorting=True,NT_members=NT,NTU_members=NTU,
        AP_scalar_checks=scalar,effect_scalar_checks=effect,old90_AP_arrays_exact=12,
        old90_matches_and_predictions_exact=True,all_matching_crosslinks_and_ignored_members_checked=True,
        no_UUID_reuse_across_scans=True,all_hashes_verified=True,raw_GT_checker_read=False,
        reserve100_read=False,heldout_read=False,final_test_read=False,production_modified=False,
        summary_sha256=sha(out/'evaluation/summary.json')))
    write(out/'execution.json',dict(status='complete',stage='verified'))
    print('Independent development200 selections,original matches,AP effects and old90 controls verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','predict','evaluate','verify'));p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args();out=a.output.resolve();assert out.is_relative_to(ROOT/'.local')
    with threadpool_limits(limits=1):
        try:globals()[a.command](out)
        except Exception as exc:
            if out.exists():write(out/'execution.json',dict(status='failed',stage=a.command,error=repr(exc)))
            raise
