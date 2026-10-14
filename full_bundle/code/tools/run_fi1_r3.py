#!/usr/bin/env python3
"""Frozen R3 experiments, including the supplementary R3c target-only ablation."""
import argparse
from collections.abc import Mapping
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from tools.prepare_fi1_r0_r1_inputs import read,write,sha,digest,load_partition,main_scene
from tools.run_fi1_r2 import load_dataset,rows,save_rows,DEV_GT
from tools.fi1_r2_core import MODEL_PARAMS,FEATURES,select
from tools.fi1_r3_core import EXTRA_FEATURES,pair_overlap,context_features,gt_iou,assign_competition,independent_context,independent_assignment
from tools.audit_fi1_r2_training import checked,require
from evaluate.scannet200 import eval_semantic_instance as evaluator

CLASS_IDS=[int(evaluator.PRED_ID_TO_ID[i]) for i in range(198)]


ARM_SPECS = {
    'R3a': {'features': FEATURES+EXTRA_FEATURES, 'target': 'y'},
    'R3b': {'features': FEATURES+EXTRA_FEATURES, 'target': 'y_competition'},
    'R3c': {'features': FEATURES, 'target': 'y_competition'},
}


def arm_features(data, arm):
    """Supply only declared features; R3c never reads the context array."""
    require(arm in ARM_SPECS, 'unknown experiment arm')
    own=data['X']
    require(own.ndim==2 and own.shape[1]==20 and np.isfinite(own).all(), 'invalid own features')
    result=own if arm=='R3c' else np.column_stack((own,data['context']))
    require(result.shape==(len(own),len(ARM_SPECS[arm]['features'])) and np.isfinite(result).all(),'arm feature shape differs')
    return result


def competition_reference(base):
    """Bind historical data through its completed audit, without re-signing code."""
    old=base/'r3_v1';audit=base/'r3_code_audit_v1'
    si=read(old/'identity.json');ds=read(old/'dataset/summary.json')
    ai=read(audit/'identity.json');au=read(audit/'summary.json')
    require(au['complete'] and au['identity_sha256']==sha(audit/'identity.json') and au['selected_members_and_order_equal'],'original R3 audit differs')
    require(ai['R3_identity_sha256']==sha(old/'identity.json') and ai['dataset_summary_sha256']==sha(old/'dataset/summary.json'),'original R3 audit binding differs')
    require(ds['complete'] and ds['identity_sha256']==digest(si),'original R3 dataset identity differs')
    return {str(p):sha(p) for p in [old/'identity.json',old/'dataset/summary.json',audit/'identity.json',audit/'summary.json']}


def verify_competition_reference(base, scene, y, assignment, members_sha):
    old=base/'r3_v1';rr=read(old/'dataset'/scene/'receipt.json')
    ds=read(old/'dataset/summary.json');record=next(r for r in ds['rows'] if r['scene_name']==scene)
    require(rr['summary']==record and rr['identity_sha256']==digest(read(old/'identity.json')) and record['R1_eligible_sha256']==members_sha,'reference scene binding differs')
    cache=old/'dataset'/scene/'context.npz';checked({str(cache):rr['outputs']['context.npz']})
    with np.load(cache,allow_pickle=False) as c:
        require(np.array_equal(y,c['y_competition']) and np.array_equal(assignment,c['assigned_gt']),'R3c target differs from frozen R3b')
    return {str(old/'dataset'/scene/'receipt.json'):sha(old/'dataset'/scene/'receipt.json'),str(cache):sha(cache)}


def code_files():
    paths=[Path(__file__).resolve(),ROOT/'tools/fi1_r3_core.py',ROOT/'tools/run_fi1_r2.py',
        ROOT/'tools/fi1_r2_core.py',ROOT/'tools/audit_fi1_r2_training.py',ROOT/'tools/prepare_fi1_r0_r1_inputs.py',
        ROOT/'docs/FI1_R3_PREREGISTRATION_20260913.md',ROOT/'docs/FI1_R3C_PREREGISTRATION_20260914.md',
        ROOT/'tools/audit_fi1_r3_fit.py',ROOT/'tools/analyze_fi1_r3.py',Path(evaluator.__file__)]
    return {str(p):sha(p) for p in paths}


def run(base,out,arms=('R3a','R3b'),fit_only=False):
    arms=tuple(arms)
    require(arms in (('R3a','R3b'),('R3c',)), 'only original paired arms or supplementary R3c are approved')
    require(not out.exists(), 'refusing to overwrite an existing run')
    if arms==('R3c',):require(fit_only, 'R3c requires fit, independent audit, then evaluate stages')
    source=base/'r2_v1/dataset';di,ds,data=load_dataset(source)
    partition=load_partition(base/'approved_split_v1');require(di['partition']==partition,'approved partition differs')
    require(list(evaluator.opt['min_region_sizes'])==[100] and di['class_index_to_dataset_id']==CLASS_IDS,'GT evaluation settings differ')
    audit=read(base/'r2_code_audit_v1/summary.json');require(audit['complete'],'R2 independent audit required')
    ai=read(base/'r2_code_audit_v1/identity.json');require(audit['identity_sha256']==sha(base/'r2_code_audit_v1/identity.json') and ai['dataset_summary_sha256']==sha(source/'summary.json'),'R2 audited dataset differs')
    reports={r['scene_name']:r for r in ds['rows']};require(len(reports)==260,'dataset coverage differs')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['dataset','models','predictions','evaluation']:(out/name).mkdir()
    identity={'version':'fi1_r3c_target_only_v1' if arms==('R3c',) else 'fi1_r3_context_competition_v1','partition':partition,'partition_id':digest(partition),
        'source_dataset':str(source),'source_summary_sha256':sha(source/'summary.json'),
        'source_audit_sha256':sha(base/'r2_code_audit_v1/summary.json'),'code':code_files(),
        'features':FEATURES if arms==('R3c',) else FEATURES+EXTRA_FEATURES,'model_parameters':MODEL_PARAMS,
        'arms':{arm:ARM_SPECS[arm] for arm in arms},
        'competition_reference':competition_reference(base) if arms==('R3c',) else {},
        'uses_learned_R2_score_as_feature':False,'new_visual_inference':False,'validation_is_blind':False,'whole_system_oof':False}
    write(out/'identity.json',identity)
    execution={'status':'preparing','new_training':False,'new_ap':False}
    def status(value):
        execution['status']=value;write(out/'execution.json',execution)
    status('preparing')
    preparation=[]
    for scene in sorted(reports):
        r=reports[scene];receipt=read(source/scene/'receipt.json')
        role=next(k for k,v in partition['roles'].items() if scene in v);require(r['role']==role,'scene role differs')
        checked(receipt['inputs'])
        members=rows(r['R1_eligible']);gs=read(r['geometry_file']);keys=[g['geometry_key'] for g in gs]
        with np.load(r['points_file'],allow_pickle=False) as p:points=[p[g['array_key']] for g in gs]
        # Frozen prepared XYZ establishes the complete point domain, including
        # vertices not present in any proposal; no validation GT access here.
        prepared=[Path(p) for p in receipt['inputs'] if p.endswith(f'/{scene[5:]}.npy')]
        require(len(prepared)==1,'prepared point domain ambiguous')
        domain=len(np.load(prepared[0],mmap_mode='r',allow_pickle=False))
        iou,coverage=pair_overlap(points,domain)
        cx=context_features(members,keys,iou,coverage)
        replay=independent_context(members,keys,iou,coverage)
        error=float(np.max(np.abs(cx-replay)));require(error<=1e-14,'independent same-class context differs')
        # Independently audit geometric pair intersections via sorted sets.
        sampled=np.unique(np.linspace(0,len(points)-1,min(8,len(points))).astype(int))
        max_pair_error=0.
        for i in sampled:
            for j in range(len(points)):
                inter=len(np.intersect1d(points[i],points[j],assume_unique=True))
                union=len(points[i])+len(points[j])-inter
                expected=inter/union if union else 0
                max_pair_error=max(max_pair_error,abs(iou[i,j]-expected))
        require(max_pair_error==0,'point overlap audit differs')
        arrays={'context':cx};assignment_count=0;removed=0;changed=0
        if role=='train':
            gt=np.loadtxt(source/'ground_truth_train'/f'{scene}.txt',dtype=np.int64)
            require(len(gt)==domain,'training GT point domain differs')
            overlap,ids=gt_iou(points,gt,CLASS_IDS)
            y,assignment=assign_competition(members,keys,overlap,ids,CLASS_IDS)
            yy,aa=independent_assignment(members,keys,overlap,ids,CLASS_IDS)
            require(np.array_equal(y,yy) and np.array_equal(assignment,aa),'independent one-to-one assignment differs')
            lookup={k:i for i,k in enumerate(keys)}
            max_y=np.zeros(len(members))
            for c in range(198):
                jj=np.flatnonzero(ids//1000==CLASS_IDS[c]);rr=[i for i,m in enumerate(members) if m['class_index']==c]
                if len(jj) and rr:
                    max_y[rr]=overlap[np.ix_([lookup[members[i]['geometry_key']] for i in rr],jj)].max(axis=1)
            require(np.array_equal(max_y,data[scene]['y']),'R2 maximum IoU label replay differs')
            require((y<=max_y).all(),'assignment exceeds maximum IoU')
            assigned=assignment[assignment>=0];require(len(assigned)==len(set(assigned.tolist())),'GT assigned more than once')
            arrays.update(y_competition=y,assigned_gt=assignment)
            if arms==('R3c',):
                identity_reference=verify_competition_reference(base,scene,y,assignment,r['R1_eligible_sha256'])
            else:identity_reference={}
            assignment_count=int((y>0).sum());removed=int(((max_y>0)&(y==0)).sum());changed=int((max_y!=y).sum())
            data[scene]['y_competition']=y
        data[scene]['context']=cx
        target=out/'dataset'/scene;target.mkdir();np.savez(target/'context.npz',**arrays)
        record={'scene_name':scene,'role':role,'member_count':len(members),'geometry_count':len(gs),
            'R1_eligible':r['R1_eligible'],'R1_eligible_sha256':r['R1_eligible_sha256'],
            'context_max_replay_error':error,'sampled_pair_error':max_pair_error,'sampled_pair_rows':len(sampled),
            'positive_competition_targets':assignment_count if role=='train' else None,
            'positive_maxIoU_targets_removed':removed if role=='train' else None,'changed_targets':changed if role=='train' else None,
            'independent_assignment_exact':True if role=='train' else None,'GT_read':role=='train',
            'frozen_R3b_target_exact':True if role=='train' and arms==('R3c',) else None}
        write(target/'receipt.json',{'identity_sha256':digest(identity),'source_receipt_sha256':sha(source/scene/'receipt.json'),
            'inputs':{**receipt['inputs'],**(identity_reference if role=='train' else {})},'outputs':{'context.npz':sha(target/'context.npz')},'summary':record})
        preparation.append(record);print('R3 prepared/audited',scene,'positive assignments',assignment_count,'removed',removed,flush=True)
    write(out/'dataset/summary.json',{'complete':True,'identity_sha256':digest(identity),'scene_count':260,'rows':preparation,
        'nontraining_GT_read':False,'context_all_rows_independently_replayed':True,'assignment_all_training_rows_independently_replayed':True})
    # Save a separate immutable training identity: model fitting starts only
    # after the two arms and full dataset have been fixed and audited.
    write(out/'training_identity.json',{'dataset_summary_sha256':sha(out/'dataset/summary.json'),
        'source_summary_sha256':sha(source/'summary.json'),'identity_sha256':sha(out/'identity.json'),'code':code_files()})
    prepared_reports={r['scene_name']:r for r in preparation};train=partition['roles']['train'];folds=partition['training_heldout_folds']
    require(sorted(s for f in folds for s in f)==train,'fold coverage differs')
    model_reports=[]
    with threadpool_limits(limits=8):
        for arm in arms:
            target_field=ARM_SPECS[arm]['target']
            for number in range(6):
                heldout=folds[number] if number<5 else partition['roles']['development_validation']+partition['roles']['diagnostic']
                fitting=[s for s in train if s not in heldout]
                require(not set(fitting)&set(heldout),'fit/prediction scan overlap')
                if number<5:require(not {main_scene(s) for s in fitting}&{main_scene(s) for s in heldout},'fold main-scene overlap')
                name=f'{arm}_fold{number}' if number<5 else f'{arm}_full200'
                execution['new_training']=True;status('fitting_'+name);started=time.monotonic()
                X=np.concatenate([arm_features(data[s],arm) for s in fitting]);y=np.concatenate([data[s][target_field] for s in fitting])
                w=np.concatenate([data[s]['weights'] for s in fitting]);w*=len(w)/w.sum()
                model=HistGradientBoostingRegressor(**MODEL_PARAMS).fit(X,y,sample_weight=w)
                model_path=out/'models'/f'{name}.joblib';joblib.dump(model,model_path);saved=joblib.load(model_path)
                require(model.n_iter_==100 and model.n_features_in_==len(ARM_SPECS[arm]['features']),'incomplete/wrong-dimensional HGB fit')
                metrics=[]
                for s in heldout:
                    members=rows(reports[s]['R1_eligible']);scores=np.clip(model.predict(arm_features(data[s],arm)),0,1)
                    require(np.array_equal(scores,np.clip(saved.predict(arm_features(data[s],arm)),0,1)),'persisted model score replay differs')
                    chosen=select(members,scores)
                    expected=sorted([{**r,'score':float(v)} for r,v in zip(members,scores)],key=lambda r:(-r['score'],not r['has_original_native_alias'],r['geometry_key'],r['class_index']))[:600]
                    require(chosen==expected,'top600 independent selection differs')
                    dest=out/'predictions'/s;dest.mkdir(exist_ok=True)
                    np.save(dest/f'{arm}_eligible_scores.npy',scores);save_rows(dest/f'{arm}_selected.jsonl',chosen)
                    native_before={(r['geometry_key'],k) for r in members for k in r['native_alias_keys']}
                    native_after={(r['geometry_key'],k) for r in chosen for k in r['native_alias_keys']}
                    write(dest/f'{arm}_receipt.json',{'model':str(model_path),'model_sha256':sha(model_path),
                        'source_member_file':reports[s]['R1_eligible'],'source_member_sha256':reports[s]['R1_eligible_sha256'],
                        'eligible_member_count':len(members),'selected_member_count':len(chosen),
                        'selected_native_alias_count':len(native_after),'evicted_native_alias_count':len(native_before-native_after),
                        'selected_expanded_classes':sum(r['expanded_class'] for r in chosen),
                        'persisted_model_replay_exact':True,'independent_top600_exact':True,
                        'outputs':{f'{arm}_eligible_scores.npy':sha(dest/f'{arm}_eligible_scores.npy'),f'{arm}_selected.jsonl':sha(dest/f'{arm}_selected.jsonl')}})
                    if number<5:
                        metrics.append({'scene_name':s,**{f'mse_{t}':float(np.dot(data[s]['weights'],(scores-data[s][t])**2)) for t in ['y','y_competition']}})
                record={'name':name,'arm':arm,'feature_count':model.n_features_in_,'target':target_field,'fit_scenes':fitting,'heldout_scenes':heldout,
                    'fit_rows':len(y),'iterations':model.n_iter_,'seconds':time.monotonic()-started,'model_sha256':sha(model_path),
                    'fit_positive_target_main_scenes':len({main_scene(s) for s in fitting if (data[s][target_field]>0).any()}),
                    'heldout_positive_target_main_scenes':len({main_scene(s) for s in heldout if (data[s][target_field]>0).any()}) if number<5 else None,
                    'heldout_metrics':metrics}
                model_reports.append(record);write(out/'fit_progress.json',model_reports);print('R3 fit complete',name,round(record['seconds'],1),'seconds',flush=True)
    execution['new_training']=True
    write(out/'fit_summary.json',{'complete':True,'model_count':6*len(arms),'prediction_scene_count_per_arm':260,'rows':model_reports,
        'training_identity_sha256':sha(out/'training_identity.json'),'dataset_summary_sha256':sha(out/'dataset/summary.json')})
    checked(identity['code']);checked(identity['competition_reference'])
    if fit_only:
        status('fit_complete_pending_independent_audit');return
    status('evaluating')
    old_eval=read(base/'r2_v1/evaluation/summary.json');gt_binding=read(base/'r2_v1/evaluation/gt_binding.json')
    require(sha(base/'r2_v1/evaluation/gt_binding.json')==old_eval['gt_binding_sha256'],'GT evaluation identity differs')
    require(sha(DEV_GT/'summary.json')==gt_binding['gt_summary_sha256'],'GT summary differs')
    gt_records={r['scene_name']:r for r in gt_binding['rows']}
    roles={k:partition['roles'][k] for k in ['development_validation','diagnostic']}
    require(gt_binding['roles']==roles and set(gt_records)=={s for v in roles.values() for s in v},'GT role coverage differs')
    for s,g in gt_records.items():
        checked(g['files']);require(str(DEV_GT/f'{s}.txt') in g['files'],'verified/consumed GT path differs')
    geometry={s:{g['geometry_key']:g for g in read(reports[s]['geometry_file'])} for s in gt_records}
    selected={};budget={};audit_rows=[]
    with threadpool_limits(limits=4):
        for arm in arms:
            for s in sorted(reports):
                dest=out/'predictions'/s;rr=read(dest/f'{arm}_receipt.json');checked({str(dest/k):h for k,h in rr['outputs'].items()})
                checked({rr['source_member_file']:rr['source_member_sha256'],rr['model']:rr['model_sha256']})
                full_or_fold='full200' if s in gt_records else f'fold{next(i for i,f in enumerate(folds) if s in f)}'
                require(Path(rr['model'])==out/'models'/f'{arm}_{full_or_fold}.joblib','wrong fitted fold model')
                scores=np.load(dest/f'{arm}_eligible_scores.npy',allow_pickle=False)
                model=joblib.load(rr['model']);replay=np.clip(model.predict(arm_features(data[s],arm)),0,1)
                require(np.array_equal(scores,replay),'evaluation model replay differs')
                actual=rows(dest/f'{arm}_selected.jsonl');members=rows(reports[s]['R1_eligible'])
                expected=sorted([{**r,'score':float(v)} for r,v in zip(members,scores)],key=lambda r:(-r['score'],not r['has_original_native_alias'],r['geometry_key'],r['class_index']))[:600]
                require(actual==expected,'evaluation member replay differs')
                audit_rows.append({'scene_name':s,'arm':arm,'member_count':len(members),'receipt_sha256':sha(dest/f'{arm}_receipt.json')})
                if s in gt_records:selected[s,arm]=actual
        # Same evaluator run includes frozen R2 control with fully checked files.
        for s in gt_records:
            old=base/'r2_v1/fit/predictions'/s;rr=read(old/'receipt.json')
            checked({str(old/k):h for k,h in rr['outputs'].items()})
            selected[s,'R2']=rows(old/'R2_selected.jsonl')
    write(out/'evaluation/prediction_audit.json',{'complete':True,'rows':audit_rows,'scene_arm_count':260*len(arms)})
    class Predictions(Mapping):
        def __init__(self,scenes,arm):self.scenes,self.arm=scenes,arm
        def __iter__(self):return iter(self.scenes)
        def __len__(self):return len(self.scenes)
        def __getitem__(self,s):
            if s not in self.scenes:raise KeyError(s)
            rr=selected[s,self.arm];mask=np.zeros((gt_records[s]['point_count'],len(rr)),dtype=bool)
            with np.load(reports[s]['points_file'],allow_pickle=False) as pts:
                for i,r in enumerate(rr):mask[pts[geometry[s][r['geometry_key']]['array_key']],i]=True
            return {'pred_masks':mask,'pred_classes':np.array([r['class_index'] for r in rr],dtype=np.int64),'pred_scores':np.array([r['score'] for r in rr],dtype=np.float64)}
    AP={}
    for role,scenes in roles.items():
        AP[role]={}
        for arm in [*arms,'R2']:
            with (out/'evaluation'/f'{role}_{arm}.log').open('x') as log,contextlib.redirect_stdout(log):
                av,*_=evaluator.evaluate(Predictions(scenes,arm),str(DEV_GT),str(out/'evaluation'/f'{role}_{arm}.csv'),dataset='scannet200')
            AP[role][arm]={k:float(av[v]) for k,v in {'ap':'all_ap','ap50':'all_ap_50%','ap25':'all_ap_25%'}.items()}
            if arm=='R2':require(AP[role][arm]==old_eval['AP'][role],'R2 control AP changed')
            write(out/'evaluation/progress.json',AP);print('R3 AP',role,arm,AP[role][arm],flush=True)
    checked(identity['code'])
    write(out/'summary.json',{'complete':True,'AP':AP,'model_count':6*len(arms),'new_training':True,'new_ap':True,
        'identity_sha256':sha(out/'identity.json'),'fit_summary_sha256':sha(out/'fit_summary.json'),
        'prediction_audit_sha256':sha(out/'evaluation/prediction_audit.json'),'R2_AP_control_exact':True,
        'whole_system_oof':False,'validation_is_blind':False,'R4_completed':False})
    execution['new_ap']=True;status('complete')


def evaluate_r3c(base,out):
    """Evaluate a frozen, independently refitted R3c; training groups precede dev."""
    from tools.analyze_fi1_r3 import configure_evaluator,compact_metrics,public_json,validate_ap_array
    identity=read(out/'identity.json');fit=read(out/'fit_summary.json')
    audit_root=out/'independent_audit';audit=read(audit_root/'summary.json');ai=read(audit_root/'identity.json')
    require(list(identity['arms'])==['R3c'] and identity['arms']['R3c']==ARM_SPECS['R3c'],'not approved R3c')
    checked(identity['code']);checked(identity['competition_reference'])
    require(fit['complete'] and fit['model_count']==6 and fit['training_identity_sha256']==sha(out/'training_identity.json'),'incomplete R3c fit')
    ti=read(out/'training_identity.json')
    require(ti['identity_sha256']==sha(out/'identity.json') and ti['dataset_summary_sha256']==sha(out/'dataset/summary.json') and ti['source_summary_sha256']==sha(base/'r2_v1/dataset/summary.json'),'training identity differs')
    require(audit['complete'] and audit['model_count']==6 and audit['scene_arm_count']==260 and audit['selected_members_and_order_equal'],'independent six-model audit required')
    require(audit['identity_sha256']==sha(audit_root/'identity.json') and ai['R3_identity_sha256']==sha(out/'identity.json') and ai['fit_summary_sha256']==sha(out/'fit_summary.json') and ai['dataset_summary_sha256']==sha(out/'dataset/summary.json'),'audit chain differs')
    checked(ai['code'])
    final=out/'summary.json';ed=out/'evaluation'
    if final.exists():
        old=read(final);require(old['complete'],'incomplete final result')
        require(old['identity_sha256']==sha(out/'identity.json') and old['fit_summary_sha256']==sha(out/'fit_summary.json') and old['audit_summary_sha256']==sha(audit_root/'summary.json'),'completed result dependency differs')
        checked(old['evaluation_inputs']);checked({str(ed/k):h for k,h in old['evaluation_outputs'].items()})
        print('R3c already evaluated; verified read-only',flush=True);return old
    require(not any(ed.iterdir()),'partial evaluation retained; use a reviewed new attempt, never overwrite')
    source=base/'r2_v1/dataset';di,ds,data=load_dataset(source)
    # Loading features verifies the original dataset chain; no fitting occurs here.
    del data
    part=load_partition(base/'approved_split_v1');require(part==identity['partition']==di['partition'],'partition differs')
    reports={r['scene_name']:r for r in ds['rows']}
    groups={f'fold{i}':scans for i,scans in enumerate(part['training_heldout_folds'])}
    groups.update({f'source_{name}':part['source_scans'][name] for name in ['official100','ncs_train100']})
    groups.update({role:part['roles'][role] for role in ['development_validation','diagnostic']})
    old_analysis=base/'r3_analysis_v2_20260914';frozen=read(base/'postrun_code_audit_v1_20260914/frozen_artifact_hashes.json')
    inputs={str(p):sha(p) for p in [out/'identity.json',out/'fit_summary.json',out/'dataset/summary.json',out/'training_identity.json',audit_root/'identity.json',audit_root/'summary.json',source/'summary.json']}
    def bind(path,expected):
        path=Path(path);checked({str(path):expected});inputs[str(path)]=expected
    bind(old_analysis/'summary.json',frozen[str(old_analysis/'summary.json')])
    controls=read(old_analysis/'summary.json')['evaluations'];AP={};control_classes={}
    configure_evaluator()
    for group,scans in groups.items():
        AP[group]={}
        for arm in ['R2','R3a','R3b']:
            dest=old_analysis/'evaluations'/f'{group}_{arm}.json';npy=dest.with_suffix('.npy')
            for p in (dest,npy):bind(p,frozen[str(p)])
            old=read(dest);array=np.load(npy,allow_pickle=False);validate_ap_array(array)
            require(old['scene_names']==scans and old['scans']==len(scans) and old['removed_source'] is None,'control group differs')
            require(old['AP']==controls[group][arm]==compact_metrics(array),'control AP differs')
            AP[group][arm]=old['AP'];control_classes[group,arm]=old['classes']
    bindings=read(base/'r2_v1/evaluation/gt_binding.json');old_eval=read(base/'r2_v1/evaluation/summary.json')
    bind(base/'r2_v1/evaluation/gt_binding.json',old_eval['gt_binding_sha256'])
    gt_records={r['scene_name']:r for r in bindings['rows']}
    require(bindings['roles']=={r:part['roles'][r] for r in ['development_validation','diagnostic']},'development GT roles differ')
    matches={};association=[];execution=read(out/'execution.json')
    with threadpool_limits(limits=4):
        for group,scans in groups.items():
            execution.update(status='evaluating_'+group,new_training=True,new_ap=True);write(out/'execution.json',execution)
            for scene in scans:
                if scene in matches:continue
                report=reports[scene];pred=out/'predictions'/scene;receipt=read(pred/'R3c_receipt.json')
                role=report['role'];is_train=role=='train'
                wanted='fold'+str(next(i for i,fold in enumerate(part['training_heldout_folds']) if scene in fold)) if is_train else 'full200'
                require(Path(receipt['model'])==out/'models'/f'R3c_{wanted}.joblib','evaluation consumed wrong fold/full model')
                bind(pred/'R3c_receipt.json',sha(pred/'R3c_receipt.json'));bind(receipt['model'],receipt['model_sha256'])
                for filename,h in receipt['outputs'].items():bind(pred/filename,h)
                bind(report['geometry_file'],report['geometry_sha256']);bind(report['points_file'],report['points_sha256']);bind(report['R1_eligible'],report['R1_eligible_sha256'])
                gtfile=source/'ground_truth_train'/f'{scene}.txt' if is_train else DEV_GT/f'{scene}.txt'
                if is_train:
                    sr=read(source/scene/'receipt.json');require(str(gtfile) in sr['inputs'],'training GT not bound')
                    bind(gtfile,sr['inputs'][str(gtfile)])
                else:bind(gtfile,gt_records[scene]['files'][str(gtfile)])
                count=len(np.loadtxt(gtfile,dtype=np.int64))
                if not is_train:require(count==gt_records[scene]['point_count'],'GT point count differs')
                selected=rows(pred/'R3c_selected.jsonl');members=rows(report['R1_eligible'])
                scores=np.load(pred/'R3c_eligible_scores.npy',allow_pickle=False)
                expected=sorted([{**r,'score':float(v)} for r,v in zip(members,scores)],key=lambda r:(-r['score'],not r['has_original_native_alias'],r['geometry_key'],r['class_index']))[:600]
                require(scores.shape==(len(members),) and np.isfinite(scores).all() and expected==selected,'evaluation member/score alignment differs')
                geometry={g['geometry_key']:g for g in read(report['geometry_file'])}
                mask=np.zeros((count,len(selected)),dtype=bool)
                with np.load(report['points_file'],allow_pickle=False) as points:
                    for j,r in enumerate(selected):mask[points[geometry[r['geometry_key']]['array_key']],j]=True
                gg,pp=evaluator.assign_instances_for_scan({'pred_masks':mask,'pred_classes':np.array([r['class_index'] for r in selected]),'pred_scores':np.array([r['score'] for r in selected])},str(gtfile))
                matches[scene]={'gt':gg,'pred':pp};del mask
                association.append({'scene_name':scene,'role':role,'selected_members':len(selected),'model':wanted,'GT_points':count,'fresh_point_association':True})
            arrays,*_=evaluator.evaluate_matches({s:matches[s] for s in scans});validate_ap_array(arrays)
            AP[group]['R3c']=compact_metrics(arrays)
            np.save(ed/f'{group}_R3c_AP.npy',arrays)
            write(ed/f'{group}_R3c_classes.json',public_json(evaluator.compute_averages(arrays)['classes']))
            write(ed/'progress.json',{'complete':False,'AP':AP,'evaluated_groups':[g for g,a in AP.items() if 'R3c' in a]})
            print('R3c AP',group,AP[group]['R3c'],'R3b_minus_R3c_pp',100*(AP[group]['R3b']['ap']-AP[group]['R3c']['ap']),flush=True)
    require(len(matches)==260 and len(association)==260,'evaluation scene coverage differs')
    write(ed/'association_audit.json',{'complete':True,'rows':association,'all_260_fresh_point_associations':True})
    checked(identity['code']);checked(inputs)
    result={'complete':True,'AP':AP,'model_count':6,'new_training':True,'new_ap':True,'new_visual_inference':False,
        'identity_sha256':sha(out/'identity.json'),'fit_summary_sha256':sha(out/'fit_summary.json'),'audit_summary_sha256':sha(audit_root/'summary.json'),
        'evaluation_inputs':inputs,'evaluation_outputs':{p.name:sha(p) for p in ed.iterdir() if p.is_file()},
        'controls':'hash-verified frozen R2/R3a/R3b analysis, not refitted this run',
        'whole_system_oof':False,'validation_is_blind':False}
    write(final,result);execution.update(status='complete',new_ap=True);write(out/'execution.json',execution)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=list(ARM_SPECS),default=['R3a','R3b'])
    p.add_argument('--stage',choices=['all','fit','evaluate'],default='all')
    args=p.parse_args();existed=args.output.exists()
    try:
        if args.stage=='evaluate':evaluate_r3c(args.base.resolve(),args.output.resolve())
        else:run(args.base.resolve(),args.output.resolve(),args.arms,fit_only=args.stage=='fit')
    except Exception as error:
        # A rejected invocation must never add failure files to a historical run.
        if not existed and args.output.exists():write(args.output/'failure.json',{'error':repr(error)})
        raise


if __name__=='__main__':main()
