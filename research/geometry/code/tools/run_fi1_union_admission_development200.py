#!/usr/bin/env python3
"""Fixed first-round U admission, reusing certified development200 and B26.

Stages are restartable only with matching frozen code/input identities. New
features precede labels; each fold model reads only its training-fold labels.
All OOF selections precede AP. Reserve100 is NEVER evaluated by this runner.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools import run_fi1_union_loss_attribution_development200 as previous
from tools.fi1_union_development_replay import Guard as ReplayGuard, project_multiclass
from tools.fi1_union_admission_learning import (
    ARMS, LEARNED, PARAMS, SEED, scene_folds, admitted_pool, anchor_targets,
    sample_weights, admission_keys, semantic_contrast, boundary_features, region_view_features)
from tools.fi1_core_ablation import source_pool, features_and_members, learned_selection
from tools.fi1_split_grow_core import pose_groups, point_indices
from tools.fi1_frontend_protocol import sha256_file as sha, freeze_json, write_json as write
from tools.fi1_union_loss_attribution import key, member_diagnostics

base = previous.base
read, rows = base.read, base.rows
PRIOR = previous.OUT
INV = ROOT/'.local/fi1_g10_full597_closure_20260921_v1/inventory.json'
OUT = ROOT/'.local/fi1_union_admission_development200_20261007_v2'


class AdmissionGuard(ReplayGuard):
    def __init__(self, contract, phase, training=()):
        super().__init__(contract, raw_GT_allowed=phase in ('labels','evaluate'))
        self.phase=phase; self.gt_enabled=phase in ('labels','evaluate')
        self.training=set(training); self.opened=set()

    def check(self, path):
        p=Path(path).resolve(); ps=str(p)
        if self.phase=='fit' and '/labels/' in ps and p.suffix=='.json':
            if p.stem not in self.training:
                raise PermissionError('held-out fold labels forbidden during fitting')
        if self.phase in ('features','prepare','fit','predict'):
            if '/evaluation/' in ps or ps.endswith('overlap.npz') or ps.endswith('diagnosis.json'):
                raise PermissionError('AP/diagnostic labels cannot enter features or selection')
        super().check(p)
        if ps.startswith('/media/jia/') or ps.startswith(str(ROOT/'.local')):
            self.opened.add(ps)


def stage(out, label, **more):
    write(out/'execution.json',dict(status='running',stage=label,updated=time.time(),**more))


def verified_read(path, expected=None):
    if expected is not None:
        assert sha(path)==expected, ('input identity',str(path))
    return read(path)


def code_files():
    result=base.code_paths()
    for name in (
        'tools/run_fi1_union_admission_development200.py','tools/fi1_union_admission_learning.py',
        'tools/fi1_union_development_replay.py','tools/run_fi1_union_loss_attribution_development200.py',
        'tools/run_fi1_union_loss_attribution_train90.py','tools/fi1_union_loss_attribution.py',
        'tools/fi1_union_budget_sensitivity.py','tools/fi1_class_agnostic_ablation.py',
        'tools/fi1_split_grow_core.py','tools/train_fi1_action_contribution.py',
        'tools/fi1_action_training_data.py','tools/fi1_evaluator_delta_core.py',
        'tools/fi1_gain_attribution_core.py','tools/fi1_scene_reference.py',
        'tests/test_fi1_union_admission_learning.py','tests/test_fi1_union_development_replay.py'):
        result[str(ROOT/name)]=sha(ROOT/name)
    return result


def prepare(out):
    assert not out.exists(), 'use a fresh experiment directory'
    assert sklearn.__version__=='1.3.2'
    assert shutil.disk_usage(ROOT).free >= 50*1024**3
    parent=read(PRIOR/'contract.json'); done=read(PRIOR/'acceptance.json')
    assert done['complete'] and not done['reserve100_read'] and not done['resume_required']
    scenes=parent['scenes'];folds=scene_folds(scenes)
    assert len(scenes)==200 and len({s[:9] for s in scenes})==200
    assert not {s[:9] for s in scenes}&{s[:9] for s in parent['reserve100']}
    guard=AdmissionGuard(parent,'prepare');sys.addaudithook(guard.hook)
    inv=read(INV); jobs={}; inputs={str(INV):sha(INV)}
    for name in ('contract.json','acceptance.json','predictions_frozen.json','final_verification.json'):
        inputs[str(PRIOR/name)]=sha(PRIOR/name)
    for s in scenes:
        j=inv['jobs'][s];assert j['group']=='train'
        b=Path(j['folder'])/'backend/D'/s
        rec=read(b/'receipt.json')
        expected={**rec['inputs'],**{str(b/k):v for k,v in rec['outputs'].items()}}
        paths=dict(unions=b/'champion'/s/'pair_union_append_candidates.jsonl',
            relations=b/'relations'/s/'relation_features_no_gt.jsonl',
            C=b/'dv3/stage_c/stage_c_v2_refined_union_plan.jsonl',
            D=b/'dv3/stage_d/stage_d_v3_append_score_plan.jsonl',
            tracks=b/'filtered_tracks'/s/'automatic_tracks.json',
            observations=b/'automatic_raw_frames'/s/'automatic_observations.jsonl',
            manifest=Path(j['front'])/'manifests'/(s+'.json'),
            processed=Path(j['data'])/s/(s[5:]+'.npy'),receipt=b/'receipt.json')
        for name,p in paths.items():
            h=sha(p)
            if str(p) in expected:assert h==expected[str(p)],p
            inputs[str(p)]=h
        for name in ('geometry','points','evidence'):
            p=parent['bindings'][s][name]
            assert sha(p)==parent['inputs'][p],p
            inputs[p]=parent['inputs'][p]
        # Old prediction hashes are inspected without reading labels.
        for p,h in read(PRIOR/'scenes'/s/'receipt.json')['outputs'].items():
            inputs[p]=h
        inputs[str(PRIOR/'scenes'/s/'receipt.json')]=sha(PRIOR/'scenes'/s/'receipt.json')
        jobs[s]=dict(paths={k:str(v) for k,v in paths.items()},front=j['front'])
    inputs[parent['B_model']]=parent['B_model_sha256'];base.verify_files(inputs)
    contract=dict(version=out.name,scenes=scenes,bindings=parent['bindings'],jobs=jobs,
        reserve100=parent['reserve100'],folds=folds,reporting_groups=parent['reporting_groups'],
        inputs=inputs,code=code_files(),B_model=parent['B_model'],B_model_sha256=parent['B_model_sha256'],
        arms=list(ARMS),learned=list(LEARNED),model_params=PARAMS,seed=SEED,
        existing_gate='stage_a_U_quality > max(stage_a_N,stage_a_T) AND corrected_D_gain > 0',
        labels='Two parent anchors per U; best positive same-parent GT, tie by GT id. Reliable gain labels require parent IoU>=.5; compare U against max(N,T) on that SAME GT. Signed gain=.5*(delta_IoU+delta fraction of strict .50:.05:.90 thresholds). Absolute union IoU anchored to parent fitted on all rows, zero if no parent overlap.',
        operating_point='Admit if ANY direction predicts signed_gain>0 AND anchored_union_IoU>.5. No threshold/capacity search. Source name, class ID, GT size and scene ID are not features.',
        feature_design='Existing: full stored relation/C quality statistics, D/champion predictions, parent sizes and own distribution contrasts. Added: directional added-region boundary contrast against parent internal variation; physically visible point mask support in 30 deterministic uniform cached frames, excluding all parent-track pose groups; joint and separation retained independently. Exact pointset mask aliases are deduplicated within each frame before computing separation.',
        views=dict(count=30,selection='uniform sorted cached observed raw frame IDs, before labels',
                   pose_translation=.1,pose_rotation_degrees=15.,minimum_visible_points=3,minimum_visible_fraction=.1),
        fold_policy='5 balanced physical-scene SHA folds, no AP-based selection. Each fit opens only 160 training-scene label files. All200 features frozen before new labels, all OOF gate and B26 selections sealed before AP. Historical200 labels previously exposed; not a fresh blind development set.',
        scope='First-round development diagnostic only; B26 saw597 and some upstream heads saw these scenes. OOF isolates the NEW admission learner, not the complete system. Reserve100 remains unopened; no automatic production replacement.',
        reuse=['source_pool','features_and_members','learned_selection','check_features','project_multiclass','original evaluator AST','pose_groups','action-head HGB parameters with prespecified larger leaves'],
        budget=800,new_geometry=False,new_visual_inference=False,B26_retrained=False,
        new_admission_models=True,reserve100_read=False,heldout_read=False,final_test_read=False,
        production_modified=False,authorization='User authorized starting the proposed U repair and requested maximal reuse of existing code.')
    out.mkdir(parents=True);freeze_json(out/'contract.json',contract)
    for p in contract['code']:
        dst=out/'code_snapshot'/Path(p).relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    write(out/'execution.json',dict(status='prepared',stage='features'))
    print('Prepared five-arm admission experiment; 5x40 physical-scene folds; reserve excluded.',flush=True)


def feature_scene(out, c, s, existing_only=False):
    """Reuse exact existing193 extraction; optionally skip unused region/view work."""
    d=out/'features'/s
    if (d/'receipt.json').exists():
        r=read(d/'receipt.json');base.verify_files(r['inputs']);base.verify_files(r['outputs']);return r
    spec=c['bindings'][s];job=c['jobs'][s];paths=job['paths'];inputs={}
    def bind(p, expected=None):
        p=str(p);h=sha(p)
        if expected is not None:assert h==expected,('feature asset changed',p)
        inputs[p]=h;return Path(p)
    for p in paths.values():bind(p,c['inputs'][p])
    for name in ('geometry','points','evidence'):bind(spec[name],c['inputs'][spec[name]])
    geometry,points,evidence=base.load_geometry(spec)
    pool=source_pool(geometry,('native','track','pair_union'))
    aliases={}
    for g in pool:
        for a in g['legacy_members']:
            k=(a['candidate_source'],int(a['candidate_id']))
            assert k not in aliases or aliases[k]==g['geometry_key'];aliases[k]=g['geometry_key']
    unions=rows(paths['unions']);C={int(r['union_candidate_id']):r for r in rows(paths['C'])}
    D={int(r['union_candidate_id']):r for r in rows(paths['D']) if r['candidate_variant']=='original'}
    relations={(int(r['track_id']),r['native_exact_geometry_group_id']):r for r in rows(paths['relations'])}
    sample=[];groups={};frames=[]
    if not existing_only:
        tracks={int(t['track_id']):t for t in read(paths['tracks'])['tracks']}
        observations=rows(paths['observations']);byframe=defaultdict(list)
        for o in observations:byframe[int(o['frame_id'])].append(o)
        manifest=read(paths['manifest']);assert manifest['scene_name']==s
        assert sha(paths['processed'])==manifest['files']['processed']['sha256']
        raw=np.load(paths['processed'],mmap_mode='r',allow_pickle=False)
        assert len(raw)==spec['point_count'];raw9=np.asarray(raw[:,:9]).copy();del raw
        entries={int(f['frame_id']):f for f in manifest['frames']}
        available=sorted(byframe)
        sample=[available[i] for i in np.unique(np.linspace(0,len(available)-1,min(30,len(available)),dtype=int))] if available else []
        # Pose grouping includes ALL source frames, not only the 30 sample views.
        poses={}
        for f in available:
            e=entries[f]['pose'];p=bind(e['path'],e['sha256']);pose=np.loadtxt(p)
            assert np.isfinite(pose).all(),('observation with invalid pose',s,f)
            poses[f]=pose
        groups=pose_groups(poses);frames=[]
        for f in sample:
            bank=Path(job['front'])/'frame_cache/grounded_sam'/s/f'frame{f:06d}'
            receipt=read(bind(bank/'receipt.json'))
            vis=np.load(bind(bank/'visibility.npy',receipt['files']['visibility.npy']),allow_pickle=False)
            assert vis.dtype==np.bool_ and vis.shape==(len(raw9),)
            masks=[]
            for o in byframe[f]:
                p=Path(o['point_indices_path']);assert p.parent==bank
                with np.load(bind(p,receipt['files'][p.name]),allow_pickle=False) as z:
                    ids=point_indices(z['point_indices'],len(raw9)).copy()
                assert vis[ids].all();masks.append(ids)
            frames.append(dict(frame=f,group=groups[f],visible=vis,masks=masks))
    records=[];gate={};parent_checks=0
    for u in unions:
        uid=int(u['candidate_id']);uk=aliases[('pair_union',uid)];tk=aliases[('track',int(u['track_id']))]
        nks={aliases[('native',int(n))] for n in u['native_member_candidate_ids']};assert len(nks)==1
        nk=next(iter(nks));U,N,T=points[uk],points[nk],points[tk]
        assert np.array_equal(np.union1d(N,T),U),('not exact union',s,uid)
        assert C[uid]['original_union_point_count']==len(U)
        assert C[uid]['original_union_sha256']==hashlib_sha(U)
        parent_checks+=1
        relation=relations[(int(u['track_id']),u['native_exact_geometry_group_id'])]
        assert relation['contracts']['feature_ground_truth_usage']=='none'
        cq=C[uid]['quality_features'];dq=D[uid]
        fixed={**{'relation__'+k:float(v) for k,v in relation['features'].items()},
               **{'C__'+k:float(v) for k,v in cq.items()}}
        for k in ('corrected_oof_temporary_refined_quality','quality_lower_confidence_bound','confident_removal_count','connectivity_removal_count'):
            fixed['C__'+k]=float(C[uid][k])
        for k in ('corrected_rank_marginal_iou_gain','conservative_gain','candidate_reference_score','stage_d_v3_append_score'):
            fixed['D__'+k]=float(dq[k])
        fixed.update(champion_probability=float(u['threshold_cross_probability']),support_relation_count=float(u['support_relation_count']))
        gate[uk]=bool(cq['stage_a_union_quality']>max(cq['stage_a_native_quality'],cq['stage_a_track_quality']) and dq['corrected_rank_marginal_iou_gain']>0)
        excluded=({groups[int(f)] for f in tracks[int(u['track_id'])]['frame_ids'] if int(f) in groups}
                  if not existing_only else set())
        for direction,pk,ok in [('native',nk,tk),('track',tk,nk)]:
            parent=points[pk];extra=np.setdiff1d(U,parent,assume_unique=True)
            own=dict(fixed,parent_is_native=float(direction=='native'),
                     log_parent_points=float(np.log1p(len(parent))),log_extra_points=float(np.log1p(len(extra))),
                     added_parent_ratio=len(extra)/max(1,len(parent)))
            own.update({'semantic__'+k:v for k,v in semantic_contrast(evidence[pk],evidence[ok],evidence[uk]).items()})
            added={}
            if not existing_only:
                added={'boundary__'+k:v for k,v in boundary_features(raw9,parent,extra).items()}
                added.update({'view__'+k:v for k,v in region_view_features(parent,extra,frames,excluded).items()})
            assert all(np.isfinite(v) for v in [*own.values(),*added.values()])
            records.append(dict(scene=s,union_id=uid,union_key=uk,parent_key=pk,other_key=ok,direction=direction,
                extra_points=len(extra),extra_sha256=hashlib_sha(extra),excluded_pose_groups=sorted(excluded),
                existing=own,region=added))
    assert {r['union_key'] for r in records}=={g['geometry_key'] for g in pool if any(a['candidate_source']=='pair_union' for a in g['legacy_members'])}
    d.mkdir(parents=True,exist_ok=True);write(d/'records.json',records);write(d/'existing_gate.json',gate)
    write(d/'views.json',dict(raw_frames=sample,pose_groups={str(k):v for k,v in groups.items()},
                            method='existing193 only; no region/view extraction' if existing_only else 'continuous evidence, source pose groups excluded; both joint and separate retained'))
    r=dict(complete=True,scene=s,unions=parent_checks,directional_rows=len(records),sample_frames=len(sample),
           independent_view_rows=sum(not r['region'].get('view__unknown',1) for r in records),GT_read=False,
           inputs=inputs,outputs={str(p):sha(p) for p in d.iterdir() if p.is_file()})
    freeze_json(d/'receipt.json',r);return r


def hashlib_sha(p):
    import hashlib
    return hashlib.sha256(np.asarray(p,dtype='<i8').tobytes()).hexdigest()


def features(out):
    c=read(out/'contract.json');sys.addaudithook(AdmissionGuard(c,'features').hook)
    base.verify_files(c['code']);assert read(out/'tests_verified.json')['passed']
    outputs={};inputs={};totals=Counter();schemas=None
    for n,s in enumerate(c['scenes'],1):
        stage(out,'features',scene=s,completed=n-1,total=200)
        r=feature_scene(out,c,s);outputs.update(r['outputs']);outputs[str(out/'features'/s/'receipt.json')]=sha(out/'features'/s/'receipt.json');inputs.update(r['inputs'])
        totals.update({k:r[k] for k in ('unions','directional_rows','sample_frames','independent_view_rows')})
        for row in read(out/'features'/s/'records.json'):
            schema={k:sorted(row[k]) for k in ('existing','region')}
            if schemas is None:schemas=schema
            assert schemas==schema,('feature schema',s)
        print(f'Features {n}/200 {s}: {r["unions"]} U; {r["independent_view_rows"]}/{r["directional_rows"]} independent visible rows',flush=True)
    base.verify_files(c['code'])
    freeze_json(out/'features_frozen.json',dict(complete=True,outputs=outputs,inputs=inputs,counts=dict(totals),
        schemas=schemas,GT_read=False,reserve100_read=False,contract_sha256=sha(out/'contract.json')))


def labels(out):
    c=read(out/'contract.json');guard=AdmissionGuard(c,'labels');sys.addaudithook(guard.hook)
    seal=read(out/'features_frozen.json');base.verify_files(seal['outputs']);base.verify_files(c['code'])
    ev=base.evaluator();outputs={};stats=Counter()
    for n,s in enumerate(c['scenes'],1):
        stage(out,'labels',scene=s,completed=n-1,total=200);dest=out/'labels'/(s+'.json')
        if dest.exists():data=read(dest)
        else:
            spec=c['bindings'][s];assert sha(spec['gt_path'])==spec['gt_sha256']
            gt=np.loadtxt(spec['gt_path'],dtype=np.int64);assert len(gt)==spec['point_count']
            geometry,points,_=base.load_geometry(spec);pool=source_pool(geometry,('native','track','pair_union'))
            keys=[g['geometry_key'] for g in pool];ids,counts=np.unique(gt,return_counts=True)
            valid=(ids>=1000)&(counts>=100)&np.isin(ids//1000,ev.VALID_CLASS_IDS);ids,counts=ids[valid],counts[valid]
            iou,inter,sizes=base.geometry_overlaps([points[k] for k in keys],gt,ids,counts)
            iou[sizes<100]=0;lookup={k:i for i,k in enumerate(keys)};values=[]
            for r in read(out/'features'/s/'records.json'):
                u,p,o=[lookup[r[k]] for k in ('union_key','parent_key','other_key')]
                target=anchor_targets(iou[p],iou[o],iou[u],ids)
                # Independent direct intersections certify both directional targets.
                if target['anchor'] is not None:
                    g=target['anchor']; gsize=int((gt==g).sum())
                    vals=[]
                    for k in (r['parent_key'],r['other_key'],r['union_key']):
                        hit=int((gt[points[k]]==g).sum());vals.append(hit/(len(points[k])+gsize-hit) if len(points[k])>=100 else 0.)
                    assert abs(vals[2]-target['absolute_iou'])<1e-14
                    assert abs(vals[2]-max(vals[:2])-target['delta_iou'])<1e-14
                values.append(dict(union_key=r['union_key'],direction=r['direction'],**target))
            data=dict(scene=s,GT=len(ids),GT_sha256=spec['gt_sha256'],values=values,
                      feature_sha256=sha(out/'features'/s/'records.json'))
            write(dest,data)
        outputs[str(dest)]=sha(dest);stats['GT']+=data['GT'];stats['directions']+=len(data['values'])
        stats['reliable']+=sum(v['reliable'] for v in data['values'])
        stats['positive_reliable']+=sum(v['reliable'] and v['gain']>0 for v in data['values'])
    freeze_json(out/'labels_frozen.json',dict(complete=True,outputs=outputs,counts=dict(stats),
        policy=c['labels'],features_sha256=sha(out/'features_frozen.json'),reserve100_read=False))
    print('Labels frozen:',dict(stats),flush=True)


def matrix(records, schemas, kind):
    return np.asarray([[r['existing'][k] for k in schemas['existing']]+
                       ([r['region'][k] for k in schemas['region']] if kind=='region_oof' else [])
                       for r in records],dtype=float).reshape(len(records),len(schemas['existing'])+(len(schemas['region']) if kind=='region_oof' else 0))


def fit(out):
    c=read(out/'contract.json');guard=AdmissionGuard(c,'fit');sys.addaudithook(guard.hook)
    base.verify_files(c['code']);fs=read(out/'features_frozen.json');ls=read(out/'labels_frozen.json')
    base.verify_files(fs['outputs']);allrows={s:read(out/'features'/s/'records.json') for s in c['scenes']};outputs={}
    for fold in range(5):
        train=[s for s in c['scenes'] if c['folds'][s]!=fold];test=[s for s in c['scenes'] if c['folds'][s]==fold]
        assert len(train)==160 and len(test)==40 and not {s[:9] for s in train}&{s[:9] for s in test}
        guard.training=set(train);d=out/'models'/f'fold{fold}';rp=d/'receipt.json'
        if rp.exists():
            r=read(rp);base.verify_files(r['outputs']);outputs.update(r['outputs']);outputs[str(rp)]=sha(rp);continue
        stage(out,'fit',fold=fold);records=[];targets=[];label_inputs={}
        for s in train:
            p=out/'labels'/(s+'.json');assert sha(p)==ls['outputs'][str(p)];label_inputs[str(p)]=sha(p)
            data=read(p);assert data['feature_sha256']==sha(out/'features'/s/'records.json')
            assert len(allrows[s])==len(data['values'])
            for r,t in zip(allrows[s],data['values']):
                assert (r['union_key'],r['direction'])==(t['union_key'],t['direction'])
                records.append(r);targets.append(t)
        reliable=np.asarray([v['reliable'] for v in targets]);assert reliable.sum()>80
        assert len({targets[i]['gain'] for i in np.flatnonzero(reliable)})>1
        d.mkdir(parents=True);local={}
        for kind in LEARNED:
            x=matrix(records,fs['schemas'],kind);models={}
            for target in ('gain','absolute_iou'):
                use=reliable if target=='gain' else np.ones(len(records),dtype=bool)
                rr=[r for r,ok in zip(records,use) if ok]
                yy=np.asarray([v[target] for v,ok in zip(targets,use) if ok],dtype=float)
                model=HistGradientBoostingRegressor(**PARAMS).fit(x[use],yy,sample_weight=sample_weights(rr))
                models[target]=model
            path=d/(kind+'.joblib');joblib.dump(dict(models=models,schemas=fs['schemas'],kind=kind,fold=fold,training_scenes=train,params=PARAMS),path);local[str(path)]=sha(path)
            for s in test:
                rr=allrows[s];xx=matrix(rr,fs['schemas'],kind)
                gain=models['gain'].predict(xx) if len(rr) else np.empty(0)
                quality=np.clip(models['absolute_iou'].predict(xx),0,1) if len(rr) else np.empty(0)
                result=dict(scene=s,fold=fold,kind=kind,model_sha256=local[str(path)],
                    decisions=[dict(union_key=r['union_key'],direction=r['direction'],gain=float(g),absolute_iou=float(q)) for r,g,q in zip(rr,gain,quality)],
                    kept=sorted(admission_keys(rr,gain,quality)))
                p=out/'oof'/s/(kind+'.json');write(p,result);local[str(p)]=sha(p)
        freeze_json(rp,dict(complete=True,fold=fold,train=train,test=test,fit_rows=len(records),reliable_rows=int(reliable.sum()),
            label_inputs=label_inputs,heldout_labels_read=False,outputs=local))
        outputs.update(local);outputs[str(rp)]=sha(rp);print(f'Fitted fold{fold}: {len(records)} rows, test scenes={len(test)}; no held-out labels',flush=True)
    freeze_json(out/'oof_frozen.json',dict(complete=True,folds=5,scenes=200,outputs=outputs,
        operating_point=c['operating_point'],hyperparameter_search=False,AP_read=False,reserve100_read=False))


def predict(out, arms=ARMS, learned=LEARNED):
    """Reuse actual-pool scoring for compatible admission decision artifacts."""
    c=read(out/'contract.json');sys.addaudithook(AdmissionGuard(c,'predict').hook)
    base.verify_files(c['code']);base.verify_files(read(out/'oof_frozen.json')['outputs'])
    assert sha(c['B_model'])==c['B_model_sha256'];model=joblib.load(c['B_model']);outputs={};totals=Counter()
    for n,s in enumerate(c['scenes'],1):
        stage(out,'B26_predictions',scene=s,completed=n-1,total=200);d=out/'predictions'/s
        if (d/'receipt.json').exists():r=read(d/'receipt.json');base.verify_files(r['outputs'])
        else:
            gs,points,ev=base.load_geometry(c['bindings'][s]);spec=c['bindings'][s]
            allU={a['union_key'] for a in read(out/'features'/s/'records.json')}
            keep=dict(NT=set(),NTU=allU,existing_gate={k for k,v in read(out/'features'/s/'existing_gate.json').items() if v})
            keep.update({a:set(read(out/'oof'/s/(a+'.json'))['kept']) for a in learned})
            d.mkdir(parents=True);counts={}
            for a in arms:
                pool=admitted_pool(gs,keep[a]);keys=[g['geometry_key'] for g in pool]
                pp={k:points[k] for k in keys};ee={k:ev[k] for k in keys}
                members,x=features_and_members(pool,pp,ee,spec['point_count'])
                selected,scores=learned_selection(members,x,model)
                previous.check_features(pool,points,ev,spec['point_count'],members,x)
                # Recompute both controls from actual pools, not old Full context.
                if a in ('NT','NTU'):
                    old='A' if a=='NT' else 'D';tag='nt' if a=='NT' else 'ntu'
                    assert selected==read(PRIOR/'scenes'/s/(old+'_selected.json'))
                    assert np.array_equal(x,np.load(PRIOR/'scenes'/s/(tag+'_features.npy'),allow_pickle=False))
                    assert np.array_equal(scores,np.load(PRIOR/'scenes'/s/(tag+'_scores.npy'),allow_pickle=False))
                assert selected==sorted([{**m,'score':float(z)} for m,z in zip(members,scores)],key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:800]
                write(d/(a+'_selected.json'),selected)
                counts[a]=dict(kept_U=len(keep[a]),eligible_members=len(members),geometry=len(pool),selected=len(selected),
                    selected_U=sum(m['geometry_key'] in allU for m in selected),
                    scores_sha256=hashlib_sha_bytes(scores),features_sha256=hashlib_sha_bytes(x))
            write(d/'kept.json',{a:sorted(v) for a,v in keep.items()})
            r=dict(complete=True,scene=s,counts=counts,original_U=len(allU),NT_NTU_controls_exact=True,
                actual_pool_features_checked=True,outputs={str(p):sha(p) for p in d.iterdir() if p.is_file()})
            freeze_json(d/'receipt.json',r)
        outputs.update(r['outputs']);outputs[str(d/'receipt.json')]=sha(d/'receipt.json');totals['original_U']+=r['original_U']
        for a in arms:totals[a+'_kept_U']+=r['counts'][a]['kept_U']
        if n%10==0:print(f'B26 predictions {n}/200; gates and contexts recomputed',flush=True)
    freeze_json(out/'predictions_frozen.json',dict(complete=True,scenes=200,predictions=len(arms)*len(c['scenes']),outputs=outputs,counts=dict(totals),
        AP_read=False,new_rule_OOF=True,reserve100_read=False,production_modified=False))
    print('All actual-pool selections sealed.',dict(totals),flush=True)


def hashlib_sha_bytes(a):
    import hashlib
    return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()


def evaluate(out, arms=ARMS, reference_root=PRIOR, verify_controls=True, report_group='development200'):
    """Original matching/AP, with an explicit arm list for reuse by new gates."""
    c=read(out/'contract.json');guard=AdmissionGuard(c,'evaluate');sys.addaudithook(guard.hook)
    seal=read(out/'predictions_frozen.json');assert seal['complete']
    base.verify_files(seal['outputs']);base.verify_files(c['code']);ev=base.evaluator()
    target=out/'evaluation';target.mkdir(exist_ok=True);GT=0;diagnostics={};receipts={}
    for n,s in enumerate(c['scenes'],1):
        stage(out,'original_matching',scene=s,completed=n-1,total=len(c['scenes']));d=target/s
        if (d/'receipt.json').exists():r=read(d/'receipt.json');base.verify_files(r['outputs'])
        else:
            spec=c['bindings'][s];assert sha(spec['gt_path'])==spec['gt_sha256']
            gt=np.loadtxt(spec['gt_path'],dtype=np.int64);gs,points,_=base.load_geometry(spec)
            pool=source_pool(gs,('native','track','pair_union'));keys=[g['geometry_key'] for g in pool];sizes={g['geometry_key']:g['point_count'] for g in pool}
            selected={a:read(out/'predictions'/s/(a+'_selected.json')) for a in arms}
            unique={key(m):m for a in arms for m in selected[a]};universe=[unique[k] for k in sorted(unique)]
            masks=np.zeros((len(gt),len(universe)),dtype=bool)
            for i,m in enumerate(universe):masks[points[m['geometry_key']],i]=True
            gm,pm=ev.assign_instances_for_scan(dict(pred_masks=masks,pred_classes=np.asarray([m['class_index'] for m in universe]),
                pred_scores=np.asarray([m['score'] for m in universe])),spec['gt_path']);del masks
            census={k:dict(zip(*np.unique(gt[points[k]],return_counts=True))) for k in keys}
            # Certify the original match with direct intersections, retaining void and small GT.
            for ci,label in enumerate(ev.CLASS_LABELS):
                mm=[m for m in universe if m['class_index']==ci and sizes[m['geometry_key']]>=100]
                assert len(mm)==len(pm[label])
                for m,p in zip(mm,pm[label]):
                    cc=census[m['geometry_key']]
                    assert p['void_intersection']==sum(v for gid,v in cc.items() if gid//1000 not in set(ev.VALID_CLASS_IDS))
                    assert {g['instance_id']:g['intersection'] for g in p['matched_gt']}=={g['instance_id']:int(cc[g['instance_id']]) for g in gm[label] if cc.get(g['instance_id'],0)}
                    for g in p['matched_gt']:g.pop('matched_pred',None)
            d.mkdir();local={};cover={};oldmembers=rows(reference_root/'scenes'/s/'nt_members.jsonl')
            ntkeys={g['geometry_key'] for g in source_pool(gs)}
            ids,counts=np.unique(gt,return_counts=True);ok=(ids>=1000)&(counts>=100)&np.isin(ids//1000,ev.VALID_CLASS_IDS);ids,counts=ids[ok],counts[ok]
            iou,inter,ps=base.geometry_overlaps([points[k] for k in keys],gt,ids,counts)
            classes=np.asarray([list(ev.VALID_CLASS_IDS).index(int(g)//1000) for g in ids])
            for a in arms:
                match=project_multiclass(dict(gt=gm,pred=pm),universe,selected[a],sizes,ev.CLASS_LABELS,ev.VALID_CLASS_IDS)
                write(d/(a+'_matches.json'),base.serializable_matches(match))
                local[a],cover[a]=member_diagnostics(keys,ps,iou,inter,classes,selected[a],oldmembers,ntkeys)
            effects={}
            for a in arms[2:]:
                effects[a]={b:{str(t):dict(gained=len(cover[a][str(t)]-cover[b][str(t)]),lost=len(cover[b][str(t)]-cover[a][str(t)])) for t in (.5,.75,.9)} for b in ('NT','NTU')}
            write(d/'diagnosis.json',dict(GT=len(ids),arms=local,coverage_effects=effects))
            r=dict(complete=True,GT=len(ids),GT_sha256=spec['gt_sha256'],original_match_direct_intersections=True,
                outputs={str(p):sha(p) for p in d.iterdir() if p.is_file()})
            freeze_json(d/'receipt.json',r)
        GT+=r['GT'];receipts[str(d/'receipt.json')]=sha(d/'receipt.json');diagnostics[s]=read(d/'diagnosis.json')
        if n%10==0:print(f"Original matching {n}/{len(c['scenes'])}",flush=True)
    AP={g:{} for g in c['reporting_groups']};arrays={};prior=read(reference_root/'evaluation/summary.json') if verify_controls else None
    # One arm at a time bounds memory; AP is pooled, never averaged by scene/fold.
    for a in arms:
        matches={s:read(target/s/(a+'_matches.json')) for s in c['scenes']}
        for group,scans in c['reporting_groups'].items():
            stage(out,'pooled_AP',arm=a,group=group)
            x,*_=ev.evaluate_matches({s:matches[s] for s in scans});AP[group][a]=base.metrics(ev,x)
            p=target/(group+'_'+a+'_AP.npy');np.save(p,x,allow_pickle=False);arrays[str(p)]=sha(p)
            if verify_controls and a in ('NT','NTU'):
                old='A' if a=='NT' else 'D'
                assert np.array_equal(x,np.load(reference_root/'evaluation'/(group+'_'+old+'_AP.npy'),allow_pickle=False),equal_nan=True)
                assert AP[group][a]==prior['AP'][group][old]
        print(a,AP[report_group][a],flush=True)
    changes={g:{a:{b:{m:100*(AP[g][a][m]-AP[g][b][m]) for m in ('ap','ap50','ap25')} for b in ('NT','NTU')} for a in arms[2:]} for g in AP}
    aggs={g:dict(GT=sum(diagnostics[s]['GT'] for s in scans),
        arms=previous.sum_dict([diagnostics[s]['arms'] for s in scans]),
        coverage_effects=previous.sum_dict([diagnostics[s]['coverage_effects'] for s in scans])) for g,scans in c['reporting_groups'].items()}
    freeze_json(target/'summary.json',dict(complete=True,AP_units='fraction',AP=AP,changes_pp=changes,AP_arrays=arrays,
        GT=GT,diagnostics=aggs,scene_receipts=receipts,scope=c['scope'],reserve100_read=c.get('reserve100_evaluation',False),
        new_visual_inference=False,new_geometry=False,B26_retrained=False,production_modified=False,
        no_threshold_search=not c.get('threshold_search',False),
        threshold_search_scope=c.get('threshold_search_scope','none'),
        no_significance_claim=True,control_arrays_exact=2*len(c['reporting_groups']) if verify_controls else 0))
    write(out/'execution.json',dict(status='complete',stage='evaluated'))


def verify(out):
    c=read(out/'contract.json');sys.addaudithook(AdmissionGuard(c,'verify').hook)
    base.verify_files(c['code']);base.verify_files(c['inputs'])
    for name in ('features_frozen.json','oof_frozen.json','predictions_frozen.json'):
        base.verify_files(read(out/name)['outputs'])
    fs=read(out/'features_frozen.json');base.verify_files(fs['inputs'])
    summary=read(out/'evaluation/summary.json');base.verify_files(summary['AP_arrays']);base.verify_files(summary['scene_receipts'])
    for fold in range(5):
        r=read(out/'models'/f'fold{fold}'/'receipt.json')
        assert not set(r['train'])&set(r['test']) and len(r['train'])==160 and len(r['test'])==40
        assert {Path(p).stem for p in r['label_inputs']}==set(r['train'])
        assert all(c['folds'][s]==fold for s in r['test'])
    ev=base.evaluator();uuids={a:set() for a in ARMS};selections=0
    for s in c['scenes']:
        receipts=read(out/'evaluation'/s/'receipt.json');base.verify_files(receipts['outputs'])
        rr=read(out/'features'/s/'records.json')
        keep=read(out/'predictions'/s/'kept.json')
        for a in LEARNED:
            decisions=read(out/'oof'/s/(a+'.json'))
            assert decisions['fold']==c['folds'][s]
            expect=admission_keys(rr,[v['gain'] for v in decisions['decisions']],[v['absolute_iou'] for v in decisions['decisions']])
            assert sorted(expect)==decisions['kept']==keep[a]
        for a in ARMS:
            selected=read(out/'predictions'/s/(a+'_selected.json'));match=read(out/'evaluation'/s/(a+'_matches.json'))
            pool=read(PRIOR/'scenes'/s/'ntu_geometry.json');sizes={g['geometry_key']:g['point_count'] for g in pool}
            valid=[m for m in selected if sizes[m['geometry_key']]>=100];byclass=defaultdict(list)
            for m in valid:byclass[m['class_index']].append(m)
            current=set();indices=[]
            for ci,label in enumerate(ev.CLASS_LABELS):
                assert len(byclass[ci])==len(match['pred'][label])
                for m,p in zip(byclass[ci],match['pred'][label]):
                    assert m['score']==p['confidence'] and sizes[m['geometry_key']]==p['vert_count']
                    current.add(p['uuid']);indices.append(p['pred_id'])
                    for g in p['matched_gt']:
                        gg=next(t for t in match['gt'][label] if t['instance_id']==g['instance_id'])
                        pp=next(t for t in gg['matched_pred'] if t['uuid']==p['uuid'])
                        assert pp['confidence']==p['confidence'] and pp['intersection']==g['intersection']
            assert not current&uuids[a] and len(current)==len(valid);uuids[a]|=current
            assert sorted(indices)==list(range(len(valid)));selections+=1
    scalar_checks=0
    for g in c['reporting_groups']:
        for a in ARMS:
            x=np.load(out/'evaluation'/(g+'_'+a+'_AP.npy'),allow_pickle=False)
            assert x.shape==(1,198,10)
            calc=dict(ap=float(np.nanmean(x[0,:,:9])),ap50=float(np.nanmean(x[0,:,0])),ap25=float(np.nanmean(x[0,:,9])))
            for m in calc:assert abs(calc[m]-summary['AP'][g][a][m])<1e-14;scalar_checks+=1
    done=dict(complete=True,scenes=200,selections=selections,GT=summary['GT'],AP_scalar_checks=scalar_checks,
        pooled_NT_NTU_controls_exact=18,physical_scene_folds_verified=True,fold_label_exclusion_verified=True,
        original_matching_both_directions_verified=True,all_source_and_artifact_hashes_verified=True,
        reserve100_read=False,production_modified=False,summary_sha256=sha(out/'evaluation/summary.json'))
    freeze_json(out/'final_verification.json',done)
    scores=summary['AP']['development200'];positive=[a for a in ARMS[2:] if scores[a]['ap']>scores['NT']['ap']]
    freeze_json(out/'acceptance.json',dict(**done,development_positive_arms=positive,AP=scores,
        adopted_policy=None,reserve100_started=False,
        next_step='Review prespecified OOF results and feature evidence before freezing one policy; reserve100 not consumed by this runner.',
        tests=read(out/'tests_verified.json')))
    write(out/'execution.json',dict(status='complete',stage='verified'))
    print('First-round admission complete and verified.',json.dumps(scores),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','features','labels','fit','predict','evaluate','verify'))
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();out=a.output.resolve()
    assert out.is_relative_to(ROOT/'.local')
    with threadpool_limits(limits=1):
        try:globals()[a.command](out)
        except Exception as exc:
            if out.exists():write(out/'execution.json',dict(status='failed',stage=a.command,error=repr(exc)))
            raise
