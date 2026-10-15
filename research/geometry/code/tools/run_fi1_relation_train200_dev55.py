#!/usr/bin/env python3
"""Existing 200/55 relation baseline; one new fit, one preselected old model."""
import argparse
from collections import Counter
import copy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from PIL import Image
import run_fi1_relation_training_scale as scale

a=scale.a
HERE=Path(__file__).resolve().parent
OUT=a.EXP/'fi1_relation_train200_dev55_20261015_v1'
SPLIT=a.ROOT/'sl_bundle/protected_sources/fi1_sl_single200_dev55_20261010_v1/contract.json'
DEV=scale.DEV
SMALL=a.EXP/'fi1_relation_training_scale_exposed52_20261015_v2'
TRAIN_FRONT=Path('/media/jia/软件1/scannet_additional_train_stream/fi1_G10_full597_20260919/frontend')
EVAL_FRONT=Path('/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921/dev/frontend')
ARMS=('reference82','train200')


def guard(allow=(),gt=False):
    allowed={str(Path(p).resolve()) for p in allow}
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT))
        if writing and not p.is_relative_to(OUT):raise PermissionError('Write outside new200/55 experiment: '+str(p))
        if not gt and not writing and any(k in str(p) for k in ('ground_truth','/labels/','/evaluation/','opportunity_diagnostics','_evidence.csv','/results.json')) and str(p) not in allowed:
            raise PermissionError('GT/outcome read not allowed at this stage: '+str(p))
    sys.addaudithook(hook)


def verify(refs):
    for p,h in refs.items():assert a.sha(p)==h,p


def contract():
    c=a.read(OUT/'contract.json');verify(c['code']);verify(c['source_files']);return c


def physical(scenes):return {s.rsplit('_',1)[0] for s in scenes}


def prepare():
    guard();a.reserve();assert not OUT.exists()
    old,dev,sm=map(a.read,(SPLIT,DEV,SMALL/'contract.json'))
    train,test=old['train_scenes'],old['scenes'];small=sm['splits']['0']['double']
    assert len(train)==200 and len(test)==55 and len(physical(test))==29 and len(small)==82
    assert set(train)==set(dev['scenes']) and set(train).isdisjoint(test) and physical(train).isdisjoint(physical(test))
    assert set(small)<=set(train)
    assert a.read(TRAIN_FRONT/'contract.json')['assets']==a.read(EVAL_FRONT/'contract.json')['assets']
    jobs={};inventory=[];refs={str(p):a.sha(p) for p in (SPLIT,DEV,SMALL/'contract.json',TRAIN_FRONT/'contract.json',EVAL_FRONT/'contract.json')}
    rec=a.read(SMALL/'models/fold0/double_receipt.json');assert rec['scenes']==small
    refs.update(rec['outputs']);refs[str(SMALL/'models/fold0/double_receipt.json')]=a.sha(SMALL/'models/fold0/double_receipt.json')
    for group,ss,front,binding in (('train200',train,TRAIN_FRONT,dev['bindings']),('dev55',test,EVAL_FRONT,old['bindings'])):
        for s in ss:
            b=binding[s];mp=front/'manifests'/f'{s}.json';mf=a.read(mp)
            assert mf['scene_name']==s
            paths=[mp,Path(b['geometry']),Path(b['points']),Path(b['evidence']),Path(b['gt_path'])]
            paths += [front/'arms/D/stage_receipts'/f'{s}_{stage}.json' for stage in ('merged','tracks','hierarchy')]
            paths += [Path(v['path']) for v in mf['files'].values()]
            chosen=set(map(str,mf['arms']['D']))
            for f in mf['frames']:
                if str(f['frame_id']) not in chosen:continue
                cache=front/'frame_cache/grounded_sam'/s/('frame%06d'%int(f['frame_id']))
                paths += [cache/name for name in ('receipt.json','observations.json','visibility.npy')]
                paths += [Path(f[k]['path']) for k in ('color','depth','pose')]
            missing=[str(p) for p in paths if not p.is_file()]
            inventory.append(dict(scene=s,group=group,missing=missing,frames=len(chosen)))
            for p in (mp,Path(b['geometry']),Path(b['points'])):refs[str(p)]=a.sha(p)
            jobs[s]=dict(frontend=str(front),GT_sha256=b['gt_sha256'],job=dict(GT=b['gt_path'],
                binding={k:b[k] for k in ('geometry','points','point_count')},
                paths=dict(manifest=str(mp),processed=mf['files']['processed']['path'])))
    code=[HERE/n for n in ('run_fi1_relation_train200_dev55.py','evaluate_fi1_relation_train200_dev55.py',
        'verify_fi1_relation_train200_dev55.py','test_fi1_relation_train200_dev55.py',
        'run_fi1_relation_training_scale.py','run_fi1_cross_track_dual.py','fi1_cross_track_dual_core.py',
        'run_fi1_pair_relation.py','fi1_pair_relation_core.py','fi1_relation_controls_core.py',
        'audit_fi1_t_source_chain.py','run_fi1_nt_region_identity.py','evaluate_fi1_nt_region_identity.py')]
    a.seal(OUT/'inventory.json',dict(complete=True,scenes=inventory,missing_scenes=[q['scene'] for q in inventory if q['missing']],GT_read=False))
    a.seal(OUT/'contract.json',dict(version=OUT.name,time_unix=time.time(),train_scenes=train,scenes=test,
        train_physical=sorted(physical(train)),dev_physical=sorted(physical(test)),small_scenes=small,small_fold=0,
        jobs=jobs,source_files=refs,code={str(p):a.sha(p) for p in code},model=sm['model'],
        candidate_parameters=scale.candidate.core.PARAMETERS,feature_dim=154,
        baseline='Preselected scale-v2 double fold0, exactly82 scenes; no best-fold selection, no refit.',
        training='All predefined200 scans; clear same/different only; same C scene-equal natural-class weights normalized to fitting row count. Mixed/unknown never binary negatives.',
        cache_policy='Preserve raw candidate generation, masks, endpoints and full unions. For RGB-D measurement only, use a new isolated visibility view equal to cached visibility AND existing positive-camera-depth/in-image projection validity. No new visible points, no depth tolerance change. Recalculate measurement visibility counts/intersection counts before original154 extraction, keep original thresholds and assertions. Audit every cached D frame, save removed point IDs and whether raw masks contained them. Do not delete scenes, masks, candidates or frames. Require old82 feature arrays/rows to remain exactly equal before fitting; otherwise stop and report nonparity.',
        actions='Both arms raw P(same)>=.9 accept proposal, <=.1 reject(not cannot-link), same eligibility/greedy disjoint partner competition and complete unions. Keep full original NT; deduplicate geometry.',
        seal_order='GT-free geometry/features for all255, then training200 labels and one fit. No dev55 labels/GT until both models scores/actions/full outputs frozen. Postseal all55 GT including mixed/unknown and opportunities.',
        evaluation='Same dev55 for both arms; scan and physical-site macro relationship metrics, mixed/cross/unknown accepts, completeNT value and good endpoint copy harm. Posthoc all proposed unions and safe same-target opportunities, never for model/candidate/threshold selection.',
        gates=dict(safe_gain02_scenes_min=2,confirmed_cross_identity_max=0,unresolved_accept_max=0,
            good_endpoint_loss_gt02_max=0,annotation_unknown_fraction_max=.1,union_target_purity_min=.9),
        exposure='Dev55 existing SL development set, not untouched final holdout. Physical isolated from relation training200; multiple scans per29 sites not independent55 sites. Old52 not comparison denominator.',
        limits=dict(new_fits=1,three_class=False,new_frames=False,new_visual_model=False,feature_redesign=False,
            threshold_search=False,AP=False,production_modified=False,old_seals_modified=False)))
    for p in code:shutil.copyfile(p,OUT/('code_'+p.name))
    (OUT/'tmp').mkdir()
    print('PREPARED train200/dev55 physical',len(physical(train)),len(physical(test)),'missing',sum(bool(q['missing']) for q in inventory),flush=True)
    assert not any(q['missing'] for q in inventory),'Missing scenes are preserved in inventory; no smaller training substitute allowed.'


def raw_geometry(s,c):
    dest=OUT/'geometry/predictions'/s
    if (dest/'receipt.json').exists():verify(a.read(dest/'receipt.json')['inputs']);return dest
    prior=scale.GEOMETRY/'predictions'/s
    if not (prior/'receipt.json').exists():prior=SMALL/'training_geometry/predictions'/s
    if (prior/'receipt.json').exists():
        # Reuse only certified complete geometries from the same source scene.
        seal=a.read((scale.GEOMETRY if prior.parent.parent==scale.GEOMETRY else SMALL/'training_geometry')/('predictions_frozen.json' if prior.parent.parent==scale.GEOMETRY else '../training_geometry_frozen.json'))
        verify({str(p):seal['outputs'][str(p)] for p in prior.iterdir()})
        dest.mkdir(parents=True)
        for p in prior.iterdir():shutil.copyfile(p,dest/p.name)
        verify(a.read(dest/'receipt.json')['inputs'])
    else:
        cfg=dict(frontend=c['jobs'][s]['frontend'])
        scale.geometry_scene(s,c['jobs'][s],cfg,dest)
    return dest


def valid_visibility(visible,pixels,z,valid,width,height):
    in_image=valid & (pixels[:,0]>=0)&(pixels[:,1]>=0)&(pixels[:,0]<width)&(pixels[:,1]<height)
    assert np.all(z[valid]>0)
    return visible & in_image


def measurement_geometry(s,c,rawdir):
    dest=OUT/'measurements/predictions'/s
    if (dest/'receipt.json').exists():verify(a.read(dest/'receipt.json')['inputs']);return dest
    spec=c['jobs'][s];mf=a.read(spec['job']['paths']['manifest']);used={}
    def checked(p,h=None):
        p=Path(p);actual=a.sha(p)
        if h is not None:assert actual==h,p
        used[str(p)]=actual;return p
    data=np.load(checked(mf['files']['processed']['path'],mf['files']['processed']['sha256']),mmap_mode='r')
    assert len(data)==spec['job']['binding']['point_count']
    intrinsic=np.loadtxt(checked(mf['files']['intrinsics']['path'],mf['files']['intrinsics']['sha256']))
    nodes=a.read(rawdir/'nodes.json');pairs=a.read(rawdir/'pairs.json');vp=a.read(rawdir/'visibility.json')
    original=a.read(rawdir/'receipt.json');verify(original['inputs']);used.update(original['inputs'])
    frames={int(f['frame_id']):f for f in mf['frames']};corrections=[];newvp=[]
    dest.mkdir(parents=True)
    with np.load(rawdir/'data.npz') as zz:
        arrays={key:zz[key] for key in zz.files}
    fv=arrays['frame_node_visible'].copy();pointsets={n['node']:arrays['n%04d'%n['node']] for n in nodes}
    intersections={q['pair_id']:np.intersect1d(pointsets[q['left']],pointsets[q['right']]) for q in pairs}
    for j,v in enumerate(vp):
        f=frames[int(v['frame_id'])];assert int(arrays['frame_ids'][j])==int(v['frame_id']) and int(arrays['frame_indices'][j])==int(v['frame_index'])
        pose=np.loadtxt(checked(f['pose']['path'],f['pose']['sha256']))
        rgbsize=Image.open(checked(f['color']['path'],f['color']['sha256'])).size
        depthsize=Image.open(checked(f['depth']['path'],f['depth']['sha256'])).size
        k=scale.visual.core.adapt_intrinsic(intrinsic,rgbsize,depthsize)
        px,z,valid=scale.visual.core.project(data[:,:3],pose,k)
        visible=np.load(checked(v['path'],original['inputs'][v['path']]),allow_pickle=False)
        fixed=valid_visibility(visible,px,z,valid,*depthsize)
        bad=visible & ~fixed;removed=np.flatnonzero(bad)
        path=v['path']
        if len(removed):
            path=str(dest/('visibility_%06d.npy'%int(v['frame_id'])));np.save(path,fixed);used[path]=a.sha(path)
            affected=[]
            cache=Path(v['path']).parent
            for obs in a.read(cache/'observations.json'):
                pp=a.points(obs['point_indices_path']);hit=pp[bad[pp]]
                if len(hit):affected.append(dict(observation_id=obs['observation_id'],point_ids=hit.tolist()))
            corrections.append(dict(frame_id=v['frame_id'],frame_index=v['frame_index'],original=v['path'],replacement=path,
                removed_point_ids=removed.tolist(),camera_depth_m=z[removed].tolist(),
                nonpositive_depth=int((z[removed]<=0).sum()),invalid_or_outside=int((z[removed]>0).sum()),affected_observations=affected))
            for node,pp in pointsets.items():fv[j,node]-=np.count_nonzero(bad[pp])
            for pid,pp in intersections.items():arrays[pid+'_intersection_visible'][j]-=np.count_nonzero(bad[pp])
        newvp.append(dict(v,path=path))
    arrays['frame_node_visible']=fv
    np.savez_compressed(dest/'data.npz',**arrays)
    for name in ('nodes.json','pairs.json'):shutil.copyfile(rawdir/name,dest/name)
    a.seal(dest/'visibility.json',newvp)
    a.seal(dest/'cache_corrections.json',dict(scene=s,GT_used=False,frames_checked=len(vp),corrections=corrections,
        raw_candidates_and_pointsets_unchanged=True,measurement_visibility_only=True))
    a.seal(dest/'receipt.json',dict(inputs=used,GT_used=False,raw_geometry=str(rawdir),frames_checked=len(vp)))
    return dest


def features_scene(s,c,rawdir):
    root=OUT/'scene_data'/s;dest=root/'features'/s
    if (dest/'receipt.json').exists():verify(a.read(dest/'receipt.json')['inputs']);return dest
    # Failed attempts remain intact; restart is explicit and uses a fresh attempt directory.
    assert not dest.exists(),'Partial feature output exists; preserve it and resolve explicitly.'
    measured=measurement_geometry(s,c,rawdir)
    geometry=OUT/'measurements'
    frozen=OUT/'scene_geometry_seals'/s
    a.seal(frozen/'predictions_frozen.json',dict(outputs={str(frozen/'predictions'/s/p.name):a.sha(p) for p in measured.iterdir()}))
    # Original extractor expects SOURCE/predictions and a seal at SOURCE root.
    (frozen/'predictions').mkdir();(frozen/'predictions'/s).symlink_to(measured,target_is_directory=True)
    a.seal(root/'contract.json',dict(jobs={s:c['jobs'][s]}))
    visual=scale.visual
    visual.OUT,visual.SOURCE,visual.PREDECESSOR,a.OUT=root,frozen,root/'no_reuse',root
    visual.guard=lambda *args,**kwargs:None
    visual.contract=lambda:dict(scenes=[s],candidate_count=len(a.read(measured/'pairs.json')),reused_GT_free_feature_files={})
    visual.features()
    return dest


def data_stage():
    guard();c=contract();assert not a.read(OUT/'inventory.json')['missing_scenes']
    results=[]
    for i,s in enumerate(c['train_scenes']+c['scenes']):
        a.reserve();start=time.time();status=OUT/'data_status'/f'{s}.json'
        if status.exists():
            q=a.read(status)
            if q['complete']:verify(q['outputs']);results.append(q);continue
            raise RuntimeError('Prior failed scene preserved: '+s)
        try:
            raw=raw_geometry(s,c);dest=features_scene(s,c,raw)
            corrections=a.read(OUT/'measurements/predictions'/s/'cache_corrections.json')
            outputs={str(p):a.sha(p) for p in raw.iterdir() if p.is_file()}
            outputs.update({str(p):a.sha(p) for p in (OUT/'measurements/predictions'/s).iterdir() if p.is_file()})
            outputs.update({str(p):a.sha(p) for p in dest.iterdir()})
            q=dict(complete=True,scene=s,seconds=time.time()-start,pairs=len(a.read(dest/'rows.json')),
                corrected_frames=len(corrections['corrections']),removed_visibility_points=sum(len(x['removed_point_ids']) for x in corrections['corrections']),outputs=outputs)
            a.seal(status,q);results.append(q)
            print('DATA',i+1,255,s,q['pairs'],'corrections',q['corrected_frames'],q['removed_visibility_points'],'seconds',round(q['seconds'],2),flush=True)
        except Exception:
            a.seal(status,dict(complete=False,scene=s,seconds=time.time()-start,traceback=traceback.format_exc(),excluded=False,model_fits=0))
            raise
    outputs={p:h for q in results for p,h in q['outputs'].items()}
    # The reference model's original inputs must remain unchanged by cache validation.
    parity=[]
    for s in c['small_scenes']:
        old=scale.BASE/'features'/s
        if not old.exists():old=SMALL/'training_data/features'/s
        current=OUT/'scene_data'/s/'features'/s
        with np.load(old/'features.npz') as before,np.load(current/'features.npz') as after:
            for key in before.files:np.testing.assert_array_equal(before[key],after[key])
        assert a.read(old/'rows.json')==a.read(current/'rows.json'),s
        parity.append(s)
    for group,ss in (('train200',c['train_scenes']),('dev55',c['scenes'])):
        rows=[];g=[];v=[]
        for s in ss:
            d=OUT/'scene_data'/s/'features'/s;rows+=a.read(d/'rows.json')
            with np.load(d/'features.npz') as z:g.extend(z['geometry']);v.extend(z['rgbd'])
        dest=OUT/'data'/group;dest.mkdir(parents=True)
        np.savez_compressed(dest/'features.npz',geometry=np.asarray(g).reshape(-1,23),rgbd=np.asarray(v).reshape(-1,131))
        a.seal(dest/'feature_rows.json',rows)
        for name in ('features.npz','feature_rows.json'):outputs[str(dest/name)]=a.sha(dest/name)
    a.seal(OUT/'data_frozen.json',dict(complete=True,GT_used=False,scenes=results,reference82_feature_parity=parity,
        outputs=outputs,total_scenes=255,none_excluded=True,time_unix=time.time()))
    print('ALL255 DATA FROZEN; exact82 feature parity passed',flush=True)


def make_labels(ss,c):
    counts=Counter();outputs={}
    for s in ss:
        spec=c['jobs'][s];gp=spec['job']['GT'];assert a.sha(gp)==spec['GT_sha256']
        gt=np.loadtxt(gp,dtype=np.int64);assert len(gt)==spec['job']['binding']['point_count']
        rr=a.read(OUT/'scene_data'/s/'features'/s/'rows.json')
        with np.load(OUT/'geometry/predictions'/s/'data.npz') as z:
            identities={k:scale.relation.endpoint_label(*np.unique(gt[z['n%04d'%k]],return_counts=True)) for k in {n for q in rr for n in (q['left'],q['right'])}}
        pairs=[dict(pair_id=q['pair_id'],relation=scale.relation.labels(identities[q['left']],identities[q['right']])) for q in rr]
        path=OUT/'labels'/f'{s}.json';a.seal(path,dict(scene=s,GT_path=gp,GT_sha256=spec['GT_sha256'],nodes=identities,pairs=pairs))
        counts.update(q['relation'] for q in pairs);outputs[str(path)]=a.sha(path)
    return dict(complete=True,counts=dict(counts),outputs=outputs)


def labels():
    c=a.read(OUT/'contract.json');guard([c['jobs'][s]['job']['GT'] for s in c['train_scenes']]);c=contract()
    verify(a.read(OUT/'data_frozen.json')['outputs'])
    a.seal(OUT/'training_labels_frozen.json',make_labels(c['train_scenes'],c))


def matrix(group):
    with np.load(OUT/'data'/group/'features.npz') as z:x=np.c_[z['geometry'],z['rgbd']]
    return a.read(OUT/'data'/group/'feature_rows.json'),x


def fit():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    c=a.read(OUT/'contract.json');guard([OUT/'labels'/f'{s}.json' for s in c['train_scenes']]);c=contract()
    verify(a.read(OUT/'data_frozen.json')['outputs']);seal=a.read(OUT/'training_labels_frozen.json');verify(seal['outputs'])
    rr,x=matrix('train200');lookup={}
    for s in c['train_scenes']:lookup.update({(s,q['pair_id']):q['relation'] for q in a.read(OUT/'labels'/f'{s}.json')['pairs']})
    y=np.array([1 if lookup[q['scene'],q['pair_id']]=='same' else 0 if lookup[q['scene'],q['pair_id']]=='different' else -1 for q in rr]);idx=np.flatnonzero(y>=0)
    w=scale.metrics.weights([rr[i]['scene'] for i in idx],x[idx,4]>0,y[idx]);dest=OUT/'models';dest.mkdir()
    start=time.time();model=HistGradientBoostingClassifier(**c['model'])
    with threadpool_limits(4):model.fit(x[idx],y[idx],sample_weight=w)
    joblib.dump(model,dest/'train200.joblib');shutil.copyfile(SMALL/'models/fold0/double.joblib',dest/'reference82.joblib')
    np.savez_compressed(dest/'training.npz',row_indices=idx,y=y[idx],weights=w)
    outputs={str(p):a.sha(p) for p in dest.iterdir()}
    a.seal(dest/'receipt.json',dict(fits=1,baseline_refits=0,seconds=time.time()-start,listed_scenes=c['train_scenes'],
        effective_scenes=sorted({rr[i]['scene'] for i in idx}),rows=len(idx),counts=dict(Counter(y[idx].tolist())),outputs=outputs))
    print('FIT200',len(idx),Counter(y[idx]),'effective scenes',len({rr[i]['scene'] for i in idx}),flush=True)


def predict():
    import joblib
    from threadpoolctl import threadpool_limits
    guard();c=contract();verify(a.read(OUT/'data_frozen.json')['outputs']);rec=a.read(OUT/'models/receipt.json');verify(rec['outputs'])
    rr,x=matrix('dev55');trainrows,trainx=matrix('train200');outputs=dict(rec['outputs']);outputs[str(OUT/'models/receipt.json')]=a.sha(OUT/'models/receipt.json')
    for arm in ARMS:
        model=joblib.load(OUT/'models'/(arm+'.joblib'))
        with threadpool_limits(4):score=model.predict_proba(x)[:,1];train_score=model.predict_proba(trainx)[:,1]
        np.save(OUT/(arm+'_scores.npy'),score);np.save(OUT/(arm+'_train_scores.npy'),train_score)
        pred=[]
        for s in c['scenes']:
            ix=[i for i,q in enumerate(rr) if q['scene']==s];dd=scale.relation.decide([rr[i] for i in ix],score[ix],[rr[i]['eligible'] for i in ix])
            dest=OUT/'predictions'/arm/s;dest.mkdir(parents=True);emitted=set();arrays={}
            with np.load(OUT/'geometry/predictions'/s/'data.npz') as z:
                for q in dd:
                    if q['state']!='accept':continue
                    pp=z[q['pair_id']+'_union'];assert a.digest(pp)==q['union_sha256']
                    if q['duplicate_original_NT'] or q['union_sha256'] in emitted:q['reason']+='_duplicate_geometry_not_emitted';continue
                    emitted.add(q['union_sha256']);arrays[q['pair_id']+'_union']=pp;q['emit']=True
            np.savez_compressed(dest/'unions.npz',**arrays);a.seal(dest/'pairs.json',dd);pred+=dd
            outputs.update({str(p):a.sha(p) for p in dest.iterdir()})
        a.seal(OUT/(arm+'_decisions.json'),pred)
        for name in (arm+'_scores.npy',arm+'_train_scores.npy',arm+'_decisions.json'):outputs[str(OUT/name)]=a.sha(OUT/name)
        print('PREDICT',arm,Counter(q['state'] for q in pred),'emitted',sum(q['emit'] for q in pred),flush=True)
    a.seal(OUT/'predictions_frozen.json',dict(complete=True,GT_used=False,time_unix=time.time(),outputs=outputs,new_fits=1,baseline_refits=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','data','labels','fit','predict'));args=p.parse_args()
    if args.mode=='data':data_stage()
    else:globals()[args.mode]()
