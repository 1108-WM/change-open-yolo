#!/usr/bin/env python3
"""Frozen data/feature-controlled ablations over existing full200 R3b evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from tools.prepare_fi1_r0_r1_inputs import read,write,sha,digest,load_partition
from tools.fi1_r2_core import FEATURES,MODEL_PARAMS,geometry_features,select
from tools.fi1_r3_core import EXTRA_FEATURES,pair_overlap,context_features
from tools.prepare_fi1_r3b_val312 import no_gt_io

BASE=ROOT/'.local/fi1_r0_r4_preparation_20260913'
VAL=ROOT/'.local/fi1_r3b_val312_execution_20260914/continued_after_alpha_key_fix'
PREP=ROOT/'.local/fi1_r3b_val312_preparation_20260914'
Q=['own_geometry_quality_mean','own_geometry_quality_range','own_geometry_quality_missing']
P=['log1p_parent_count','parent_iou_mean','child_added_fraction_mean','parent_removed_fraction_mean']
ARMS={'B':[],'Q':list(range(3)),'P':list(range(3,7)),'QP':list(range(7))}
REPLAY_ATOL=1e-12

def require(ok,msg):
    if not ok:raise ValueError(msg)

def rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f]

def verify(files):
    for path,h in files.items():require(sha(path)==h,'bound file changed: '+str(path))

def replay_scores(first,second):
    require(first.shape==second.shape and np.isfinite(first).all() and np.isfinite(second).all(),'invalid replay scores')
    error=float(np.max(np.abs(first-second),initial=0))
    require(error<=REPLAY_ATOL,'refit score differs beyond floating tolerance')
    return {'max_abs_error':error,'bitwise_equal':bool(np.array_equal(first,second))}

def member_identity(selected):
    return [{k:v for k,v in row.items() if k!='score'} for row in selected]

def code():
    names=['fi1_geometry_feature_ablation.py','fi1_r2_core.py','fi1_r3_core.py','prepare_fi1_r0_r1_inputs.py','prepare_fi1_r3b_val312.py','fi1_r0_r1_protocol.py']
    return {str(ROOT/'tools'/n):sha(ROOT/'tools'/n) for n in names}

def source_roots():
    r={n:BASE/'dv3_generation_v2'/n for n in ['official100','ncs_train100','ncs_validation60']}
    r['val312']=Path('/media/jia/软件1/scannet_validation_stream/fi1_d_v3_dm_sms1_val312_local_20260902/class_agnostic_repro_20260902/17_fi1_d_v3_frozen_inference')
    return r

def indexes():
    result={};files={}
    for name,root in source_roots().items():
        summary=read(root/'summary.json')
        require(summary['model_package_sha256']=='757ebcaa5744f5a765be533c732a0f6eaa821a9bdff22ff9669d322c44fa92f0','upstream model package differs')
        aa={};cc={};relations=defaultdict(list)
        for field in ['summary.json','stage_a/deployment_quality_predictions.jsonl','stage_c/stage_c_v2_refined_union_plan.jsonl','stage_b/candidate_relations.jsonl']:
            files[str(root/field)]=sha(root/field)
        for row in rows(root/'stage_a/deployment_quality_predictions.jsonl'):
            require(row['ground_truth_usage']=='none' and row['fold_index']==-1,'unexpected upstream prediction identity')
            require(row['geometry_key'] not in aa,'duplicate stage A geometry')
            aa[row['geometry_key']]=row
        for row in rows(root/'stage_c/stage_c_v2_refined_union_plan.jsonl'):
            key=(row['scene_name'],row['original_union_geometry_key'],row['temporary_refined_sha256'])
            require(key not in cc,'duplicate stage C candidate');cc[key]=row
        for row in rows(root/'stage_b/candidate_relations.jsonl'):
            if row['pair_union_parent_relation']:
                relations[row['scene_name']].append(row)
        result[name]=(aa,cc,relations)
    return result,files

def extra_features(scene,geometry,points,index):
    """Use only own quality and exact pointset parent changes, never GT/parent quality fallback."""
    aa,cc,relations=index;keys=[g['geometry_key'] for g in geometry]
    by_key=dict(zip(keys,points));aliases={a:g['geometry_key'] for g in geometry for a in g['legacy_geometry_keys']}
    parents=defaultdict(set)
    for rel in relations.get(scene,[]):
        ka,kb=rel['geometry_key_a'],rel['geometry_key_b']
        require(ka in aliases and kb in aliases,'unresolved union relation')
        if rel['source_a']=='pair_union' and rel['source_b'] in ['native','track']:
            parents[aliases[ka]].add(aliases[kb])
        elif rel['source_b']=='pair_union' and rel['source_a'] in ['native','track']:
            parents[aliases[kb]].add(aliases[ka])
        else:raise ValueError('ambiguous union parent direction')
    quality={};missing=0
    for g,p in zip(geometry,points):
        raw=np.asarray(p,dtype='<i8').tobytes();h1=hashlib.sha1(raw).hexdigest();h2=hashlib.sha256(raw).hexdigest()
        require(h2==g['points_sha256'] and len(p)==g['point_count'],'pointset mismatch')
        values=[]
        for a in g['legacy_geometry_keys']:
            require(a in aa and aa[a]['scene_name']==scene and aa[a]['geometry_hash']==h1,'own quality point identity differs')
            values.append(float(aa[a]['deployment_unified_quality']))
        for plan in g['dv3_plans']:
            if not plan['append_only']:continue
            parent=plan['original_union_geometry_key'];require(parent in aliases,'unresolved refined parent')
            parents[g['geometry_key']].add(aliases[parent])
            cr=cc[(scene,parent,h2)]
            require(cr['append_eligible_refined_union'] and cr['temporary_refined_point_count']==len(p),'refined quality identity differs')
            require(cr['original_union_sha256']==hashlib.sha256(np.asarray(by_key[aliases[parent]],dtype='<i8').tobytes()).hexdigest(),'refined parent point identity differs')
            values.append(float(cr['corrected_oof_temporary_refined_quality']))
        require(np.isfinite(values).all(),'invalid own quality')
        quality[g['geometry_key']]=[float(np.mean(values)),float(np.ptp(values)),0.] if values else [0.,0.,1.]
        missing+=not bool(values)
    out=[]
    for g,p in zip(geometry,points):
        # Equal pointsets cannot be their own parent relation after deduplication.
        ps=parents[g['geometry_key']]-{g['geometry_key']};stats=[]
        for parent in sorted(ps):
            pp=by_key[parent];inter=len(np.intersect1d(p,pp,assume_unique=True));union=len(p)+len(pp)-inter
            stats.append([inter/union if union else 0,(len(p)-inter)/len(p) if len(p) else 0,
                          (len(pp)-inter)/len(pp) if len(pp) else 0])
        pv=[float(np.log1p(len(ps))),*(np.mean(stats,axis=0).tolist() if stats else [0.,0.,0.])]
        out.append(quality[g['geometry_key']]+pv)
    result=np.asarray(out,dtype=np.float64)
    require(result.shape==(len(geometry),7) and np.isfinite(result).all(),'invalid feature shape/values')
    return result,{'own_quality_missing':missing,'parent_geometries':int((result[:,3]>0).sum()),'geometries':len(geometry)}

def matrix(data,arm):
    base=np.column_stack((data['X'],data['context']))
    return np.column_stack((base,data['extra'][:,ARMS[arm]])) if ARMS[arm] else base

def prepare(out):
    no_gt_io();require(not out.exists(),'refusing existing preparation output')
    out.mkdir(parents=True);(out/'dataset').mkdir()
    part=load_partition(BASE/'approved_split_v1');report=read(BASE/'r2_v1/dataset/summary.json')['rows']
    jobs=[{**r,'group':'train' if r['scene_name'] in part['roles']['train'] else 'dev55'} for r in report if r['scene_name'] in part['roles']['train']+part['roles']['development_validation']]
    for scene in (PREP/'inventory/val312.txt').read_text().split():
        jobs.append({'scene_name':scene,'source':'val312','group':'val312','geometry_file':str(PREP/'geometry_binding'/scene/'geometry.json'),
            'points_file':str(PREP/'geometry_binding'/scene/'points.npz'),'R1_eligible':str(VAL/'full/predictions'/scene/'R1_eligible.jsonl')})
    ix,files=indexes();records=[]
    with threadpool_limits(limits=4):
        for r in sorted(jobs,key=lambda x:x['scene_name']):
            scene=r['scene_name'];gf=Path(r['geometry_file']);pf=Path(r['points_file']);mf=Path(r['R1_eligible'])
            inputs={};g=read(gf);members=rows(mf)
            for path,field in [(gf,'geometry_sha256'),(pf,'points_sha256'),(mf,'R1_eligible_sha256')]:
                h=sha(path);require(field not in r or r[field]==h,'original dataset input changed');inputs[str(path)]=h
            with np.load(pf,allow_pickle=False) as arrays:points=[arrays[row['array_key']] for row in g]
            extra,stats=extra_features(scene,g,points,ix[r['source']]);lookup={row['geometry_key']:i for i,row in enumerate(g)}
            extra=extra[[lookup[m['geometry_key']] for m in members]]
            if r['source']!='val312':
                fp=BASE/'r2_v1/dataset'/scene/'features.npz';cp=BASE/'r3_v1/dataset'/scene/'context.npz'
                for path,parent in [(fp,fp.parent),(cp,cp.parent)]:
                    rr=read(parent/'receipt.json');inputs[str(path)]=rr['outputs'][path.name];verify({str(path):inputs[str(path)]})
                with np.load(fp,allow_pickle=False) as a:values={k:a[k] for k in a.files if k in ['X','weights']}
                with np.load(cp,allow_pickle=False) as a:
                    values['context']=a['context']
                    if r['group']=='train':values['y_competition']=a['y_competition']
            else:
                ep=VAL/'full/alpha_bound'/scene/'evidence.jsonl';er=read(ep.parent/'receipt.json');inputs[str(ep)]=er['outputs'][ep.name];verify({str(ep):inputs[str(ep)]})
                evidence={r['geometry_key']:r for r in rows(ep)}
                gx={row['geometry_key']:geometry_features(row,evidence[row['geometry_key']]) for row in g}
                domain=read(PREP/'geometry_binding'/scene/'receipt.json')['summary']['point_count']
                iou,cov=pair_overlap(points,domain)
                values={'X':np.asarray([gx[m['geometry_key']][m['class_index']] for m in members]),'context':context_features(members,[row['geometry_key'] for row in g],iou,cov)}
            values['extra']=extra
            require(values['X'].shape==(len(members),20) and values['context'].shape==(len(members),6),'base features differ')
            for v in values.values():require(np.isfinite(v).all(),'nonfinite dataset')
            folder=out/'dataset'/scene;folder.mkdir();np.savez(folder/'features.npz',**values)
            rec={'scene_name':scene,'group':r['group'],'source':r['source'],'geometry_file':str(gf),'points_file':str(pf),'members':str(mf),'rows':len(members),'feature_availability':stats}
            write(folder/'receipt.json',{'inputs':inputs,'outputs':{'features.npz':sha(folder/'features.npz')},'summary':rec})
            records.append(rec);print('feature ablation prepared',scene,r['group'],stats,flush=True)
    verify(files)
    protocol=ROOT/'docs/FI1_CONTROLLED_ABLATIONS_PROTOCOL_20260914.md'
    identity={'version':'fi1_full200_geometry_ablation','partition':part,'features':FEATURES+EXTRA_FEATURES,'quality_features':Q,'parent_features':P,
        'arms':ARMS,'parameters':MODEL_PARAMS,'upstream_inputs':files,'code':code(),'protocol_sha256':sha(protocol),
        'quality_is_fixed_upstream_deployment_not_OOF':True,'whole_system_oof':False,'validation_is_blind':False,'new_Qwen':False}
    write(out/'identity.json',identity)
    write(out/'dataset/summary.json',{'complete':True,'rows':records,'identity_sha256':digest(identity),
        'receipts':{str(out/'dataset'/r['scene_name']/'receipt.json'):sha(out/'dataset'/r['scene_name']/'receipt.json') for r in records}})

def load_data(out):
    identity=read(out/'identity.json');verify(identity['upstream_inputs'])
    summary=read(out/'dataset/summary.json');require(summary['complete'] and summary['identity_sha256']==digest(identity),'incomplete dataset')
    verify(summary['receipts'])
    execution=out/'execution_identity.json'
    if execution.exists():
        amended=read(execution);require(amended['prepared_identity_sha256']==sha(out/'identity.json'),'prepared identity changed')
        verify(amended['code']);verify(amended['amendment_inputs'])
        for path,h in identity['code'].items():
            old=out/'failed_exact_replay_attempt'/Path(path).name if Path(path).resolve()==Path(__file__).resolve() else Path(path)
            verify({str(old):h})
        identity={**identity,'execution_identity':amended}
    else:verify(identity['code'])
    return identity,summary

def fit(out):
    no_gt_io();identity,summary=load_data(out)
    train=identity['partition']['roles']['train'];reports={r['scene_name']:r for r in summary['rows']};data={}
    for scene in train:
        folder=out/'dataset'/scene;rr=read(folder/'receipt.json');verify(rr['inputs']);verify({str(folder/n):h for n,h in rr['outputs'].items()})
        with np.load(folder/'features.npz',allow_pickle=False) as a:data[scene]={k:a[k] for k in a.files}
    y=np.concatenate([data[s]['y_competition'] for s in train]);w=np.concatenate([data[s]['weights'] for s in train]);w*=len(w)/w.sum()
    (out/'models').mkdir(exist_ok=True);fit_rows=[]
    with threadpool_limits(limits=8):
        for arm in ARMS:
            path=out/'models'/(arm+'.joblib');receipt=path.with_suffix('.json')
            if receipt.exists():
                r=read(receipt);require(r['identity_sha256']==digest(identity),'fit identity differs');verify({str(path):r['model_sha256'],str(path.with_name(arm+'_independent.joblib')):r['audit_model_sha256']});fit_rows.append(r);continue
            X=np.concatenate([matrix(data[s],arm) for s in train]);started=time.monotonic()
            print('fit',arm,X.shape,flush=True)
            model=HistGradientBoostingRegressor(**MODEL_PARAMS).fit(X,y,sample_weight=w)
            joblib.dump(model,path)
            # Fresh independent estimator fit with the same frozen data/weights.
            audit=HistGradientBoostingRegressor(**MODEL_PARAMS).fit(X,y.copy(),sample_weight=w.copy())
            auditpath=out/'models'/(arm+'_independent.joblib');joblib.dump(audit,auditpath)
            check_idx=np.unique(np.linspace(0,len(y)-1,min(10000,len(y))).astype(int))
            refit_check=replay_scores(model.predict(X[check_idx]),audit.predict(X[check_idx]))
            baseline_check=None
            if arm=='B':
                old=joblib.load(BASE/'r3_v1/models/R3b_full200.joblib')
                baseline_check=replay_scores(model.predict(X[check_idx]),old.predict(X[check_idx]))
            r={'arm':arm,'identity_sha256':digest(identity),'fit_scenes':train,'feature_count':X.shape[1],
                'model_sha256':sha(path),'audit_model_sha256':sha(auditpath),'sampled_training_replay_rows':len(check_idx),
                'seconds':time.monotonic()-started,'new_training':True,'refit_check':refit_check,'baseline_check':baseline_check}
            write(receipt,r);fit_rows.append(r);del X,model,audit
    write(out/'fit_summary.json',{'complete':True,'rows':fit_rows,'identity_sha256':digest(identity),'independent_models':4,'models':4,'new_training':True})

def predict(out):
    no_gt_io();identity,summary=load_data(out);require(read(out/'fit_summary.json')['complete'],'fit incomplete')
    models={};audits={}
    for arm in ARMS:
        p=out/'models'/(arm+'.joblib');r=read(p.with_suffix('.json'));require(r['identity_sha256']==digest(identity),'model identity differs');verify({str(p):r['model_sha256'],str(p.with_name(arm+'_independent.joblib')):r['audit_model_sha256']})
        models[arm]=joblib.load(p);audits[arm]=joblib.load(p.with_name(arm+'_independent.joblib'))
    records=[]
    with threadpool_limits(limits=4):
        for r in summary['rows']:
            if r['group']=='train':continue
            scene=r['scene_name'];folder=out/'dataset'/scene;rr=read(folder/'receipt.json');verify(rr['inputs']);verify({str(folder/n):h for n,h in rr['outputs'].items()})
            with np.load(folder/'features.npz',allow_pickle=False) as a:data={k:a[k] for k in a.files}
            ms=rows(r['members']);target=out/'predictions'/scene;target.mkdir(parents=True,exist_ok=True)
            if (target/'receipt.json').exists():
                old=read(target/'receipt.json');require(old['dataset_receipt_sha256']==sha(folder/'receipt.json') and old['identity_sha256']==digest(identity),'prediction input differs');verify({str(target/n):h for n,h in old['outputs'].items()});records.append(r);continue
            replay_checks={}
            for arm in ARMS:
                x=matrix(data,arm);scores=np.clip(models[arm].predict(x),0,1);second=np.clip(audits[arm].predict(x),0,1)
                replay_checks[arm]=replay_scores(scores,second)
                chosen=select(ms,scores)
                require(member_identity(chosen)==member_identity(select(ms,second)),'independent refit selected members/order differ')
                expected=sorted([{**m,'score':float(s)} for m,s in zip(ms,scores)],key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:600]
                require(chosen==expected,'independent selection differs')
                if arm=='B':
                    oldfolder=VAL/'full/predictions'/scene if r['group']=='val312' else BASE/'r3_v1/predictions'/scene
                    old=np.load(oldfolder/('R3b_scores.npy' if r['group']=='val312' else 'R3b_eligible_scores.npy'),allow_pickle=False)
                    replay_checks['frozen_baseline']=replay_scores(scores,old)
                    require(member_identity(chosen)==member_identity(rows(oldfolder/'R3b_selected.jsonl')),'all-row frozen baseline member/order replay differs')
                np.save(target/(arm+'_scores.npy'),scores)
                with (target/(arm+'_selected.jsonl')).open('w') as f:
                    for row in chosen:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            write(target/'receipt.json',{'dataset_receipt_sha256':sha(folder/'receipt.json'),'identity_sha256':digest(identity),
                'outputs':{p.name:sha(p) for p in target.iterdir() if p.is_file()},'replay_checks':replay_checks,'selected_members_and_order_exact':True})
            records.append(r);print('feature ablation predictions',scene,r['group'],flush=True)
    write(out/'prediction_summary.json',{'complete':True,'rows':records,'scene_count':len(records),'identity_sha256':digest(identity),
        'baseline_selected_members_and_order_exact':True,'refit_scores_within_atol':REPLAY_ATOL})

def evaluate(out):
    from tools.analyze_fi1_r3 import configure_evaluator,compact_metrics,public_json,validate_ap_array
    from evaluate.scannet200 import eval_semantic_instance as ev
    identity,ds=load_data(out);ps=read(out/'prediction_summary.json');require(ps['complete'] and ps['scene_count']==367 and ps['identity_sha256']==digest(identity),'complete dev55+val312 predictions required')
    configure_evaluator();ed=out/'evaluation';ed.mkdir(exist_ok=True);result={}
    dev_gt={r['scene_name']:r for r in read(BASE/'r2_v1/evaluation/gt_binding.json')['rows']}
    for group in ['dev55','val312']:
        matches={a:{} for a in ARMS};group_rows=[r for r in ps['rows'] if r['group']==group]
        for r in group_rows:
            scene=r['scene_name'];pred=out/'predictions'/scene;rr=read(pred/'receipt.json');require(rr['identity_sha256']==digest(identity),'prediction identity differs');verify({str(pred/n):h for n,h in rr['outputs'].items()})
            dr=read(out/'dataset'/scene/'receipt.json');require(rr['dataset_receipt_sha256']==sha(out/'dataset'/scene/'receipt.json'),'dataset receipt changed');verify(dr['inputs'])
            if group=='dev55':
                gt=next(Path(k) for k in dev_gt[scene]['files'] if k.endswith('/'+scene+'.txt'));verify({str(gt):dev_gt[scene]['files'][str(gt)]})
            else:
                gt=Path('/home/jia/Wm/Dataset/scannet200/ground_truth')/(scene+'.txt');er=read(VAL/'evaluation'/scene/'receipt.json');verify({str(gt):er['inputs'][str(gt)]})
            count=len(np.loadtxt(gt,dtype=np.int64));gs={g['geometry_key']:g for g in read(r['geometry_file'])}
            with np.load(r['points_file'],allow_pickle=False) as points:
                for arm in ARMS:
                    chosen=rows(pred/(arm+'_selected.jsonl'));mask=np.zeros((count,len(chosen)),dtype=bool)
                    for j,m in enumerate(chosen):mask[points[gs[m['geometry_key']]['array_key']],j]=True
                    gg,pp=ev.assign_instances_for_scan({'pred_masks':mask,'pred_classes':np.array([m['class_index'] for m in chosen]),'pred_scores':np.array([m['score'] for m in chosen])},str(gt))
                    matches[arm][scene]={'gt':gg,'pred':pp};del mask
            print('feature ablation GT association',group,scene,flush=True)
        result[group]={}
        for arm in ARMS:
            ap,*_=ev.evaluate_matches(matches[arm]);validate_ap_array(ap)
            if arm=='B':
                fp=BASE/'r3_analysis_v2_20260914/evaluations/development_validation_R3b.npy' if group=='dev55' else VAL/'evaluation/R3b_AP.npy'
                require(np.array_equal(ap,np.load(fp),equal_nan=True),'baseline AP not exactly reproduced')
            result[group][arm]=compact_metrics(ap);np.save(ed/(group+'_'+arm+'.npy'),ap)
            write(ed/(group+'_'+arm+'_classes.json'),public_json(ev.compute_averages(ap)['classes']))
            print('feature ablation AP',group,arm,result[group][arm],flush=True)
        write(ed/'progress.json',result);del matches
    write(out/'summary.json',{'complete':True,'AP':result,'new_training':True,'train_count':200,'new_visual_inference':False,'Qwen':False,
        'features':{'Q':Q,'P':P},'fixed_geometry_members_targets_parameters':True,'identity_sha256':digest(identity),
        'baseline_AP_exact':True,'independent_refit_scores_within_atol':REPLAY_ATOL,'refit_selected_members_and_order_exact':True,'post_hoc_development_not_blind':True,
        'code':code(),'outputs':{str(p):sha(p) for p in ed.iterdir() if p.is_file()}})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','fit','predict','evaluate']);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.output.resolve())
