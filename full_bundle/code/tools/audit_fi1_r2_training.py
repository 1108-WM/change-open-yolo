#!/usr/bin/env python3
"""Audit every R2 training label, independently refit all six heads and replay AP.

Uses sparse mask matrix intersections rather than the production target code.
Does not call production R2 preparation, training, or selection functions.
Historical code, models, predictions and AP remain unchanged.
"""
import argparse
from collections.abc import Mapping
import contextlib
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from tools.prepare_fi1_r0_r1_inputs import read,write,sha,digest,load_partition
from evaluate.scannet200 import eval_semantic_instance as evaluator

PARAMS=dict(loss='squared_error',learning_rate=.05,max_iter=100,max_leaf_nodes=7,
    max_depth=3,min_samples_leaf=100,l2_regularization=1.,early_stopping=False,random_state=20260913)
CLASS_IDS=[int(evaluator.PRED_ID_TO_ID[c]) for c in range(198)]


def require(condition,message):
    if not condition:raise ValueError(message)


def checked(files):
    for p,h in files.items():require(sha(p)==h,f'changed input: {p}')


def rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f]


def check_fit_roles(partition,fit_rows):
    train=partition['roles']['train'];folds=partition['training_heldout_folds']
    expected={f'fold{i}':([s for s in train if s not in fold],fold) for i,fold in enumerate(folds)}
    expected['full200']=(train,partition['roles']['development_validation']+partition['roles']['diagnostic'])
    require(len(fit_rows)==len(expected) and {r['model'] for r in fit_rows}==set(expected),'model coverage differs')
    predicted=[]
    for r in fit_rows:
        fitting,heldout=expected[r['model']]
        require(r['fit_scenes']==fitting and r['prediction_scenes']==heldout,'fit/heldout assignment differs from approved partition')
        require(not set(fitting)&set(heldout),'scan leakage')
        if r['model']!='full200':
            require(not {s.split('_')[0] for s in fitting}&{s.split('_')[0] for s in heldout},'main-scene leakage')
        predicted+=heldout
    require(len(predicted)==len(set(predicted))==sum(len(x) for x in partition['roles'].values()),'prediction role coverage differs')
    return expected


def sparse_targets(point_sets,gt,class_ids,minimum=100):
    """Full sparse mask/GT intersection table, independently of histogram targets."""
    valid_ids=[int(i) for i in np.unique(gt) if i>=1000 and int(i)//1000 in class_ids]
    gt_masks=[gt==i for i in valid_ids]
    keep=[j for j,m in enumerate(gt_masks) if np.count_nonzero(m)>=minimum]
    valid_ids=[valid_ids[j] for j in keep];gt_masks=[gt_masks[j] for j in keep]
    size=np.array([len(p) for p in point_sets],dtype=np.int64)
    target=np.zeros((len(point_sets),len(class_ids)),dtype=np.float64)
    if not valid_ids:return target,0
    indices=np.concatenate(point_sets) if point_sets else np.empty(0,dtype=np.int64)
    indptr=np.r_[0,np.cumsum(size)]
    pred=sparse.csr_matrix((np.ones(len(indices),dtype=np.int64),indices,indptr),shape=(len(point_sets),len(gt)))
    truth=sparse.csr_matrix(np.column_stack(gt_masks),dtype=np.int64)
    intersection=(pred@truth).toarray()
    gt_sizes=np.asarray(truth.sum(axis=0)).ravel()
    iou=intersection/(size[:,None]+gt_sizes[None,:]-intersection)
    classes=np.array(valid_ids)//1000
    for c in np.unique(classes):target[:,class_ids.index(int(c))]=iou[:,classes==c].max(axis=1)
    target[size<minimum]=0
    return target,len(valid_ids)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve();out=args.output.resolve()
    dataset=base/'r2_v1/dataset';fit=base/'r2_v1/fit';evaluation=base/'r2_v1/evaluation'
    partition=load_partition(base/'approved_split_v1');di=read(dataset/'identity.json');ds=read(dataset/'summary.json')
    fi=read(fit/'identity.json');fs=read(fit/'summary.json');es=read(evaluation/'summary.json')
    require(di['partition']==partition and di['partition_id']==digest(partition),'dataset partition differs')
    require(ds['complete'] and ds['identity_sha256']==digest(di),'dataset identity differs')
    require(fs['complete'] and fs['dataset_summary_sha256']==sha(dataset/'summary.json') and fs['identity_sha256']==sha(fit/'identity.json'),'fit identity differs')
    require(fi['dataset']==str(dataset) and fi['partition_id']==digest(partition),'fit data path/partition differs')
    require(es['fit_summary_sha256']==sha(fit/'summary.json'),'evaluation fit identity differs')
    require(di['protocol']['model']==PARAMS and fi['protocol']==di['protocol'],'model protocol differs')
    require(di['class_index_to_dataset_id']==CLASS_IDS and list(evaluator.opt['min_region_sizes'])==[100],'target evaluator configuration differs')
    for files in [di['code'],di['input_files'],fi['code'],*di['source_files'].values()]:checked(files)
    previous_audit=base/'r2_v1/dataset_audit/summary.json';pa=read(previous_audit)
    require(pa['complete'] and pa['dataset_summary_sha256']==sha(dataset/'summary.json'),'independent feature audit stale')
    expected=check_fit_roles(partition,fs['rows'])
    reports={r['scene_name']:r for r in ds['rows']}
    require(len(reports)==len(ds['rows'])==260 and set(reports)=={s for v in partition['roles'].values() for s in v},'dataset scene coverage differs')
    require({p.stem for p in (dataset/'ground_truth_train').glob('*.txt')}==set(partition['roles']['train']),'training GT role coverage differs')
    out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();(out/'scores').mkdir()
    write(out/'identity.json',{'partition_id':digest(partition),'dataset_summary_sha256':sha(dataset/'summary.json'),
        'fit_summary_sha256':sha(fit/'summary.json'),'original_AP_sha256':sha(evaluation/'summary.json'),
        'prior_all_feature_audit_sha256':sha(previous_audit),'audit_code_sha256':sha(__file__),'params':PARAMS})
    data={};weight_by_scene={};target_audits=[]
    for scene in sorted(reports):
        r=reports[scene];role=next(k for k,v in partition['roles'].items() if scene in v)
        require(r['role']==role,'dataset row role differs')
        receipt=read(dataset/scene/'receipt.json')
        require(receipt['summary']==r and receipt['identity_sha256']==digest(di),'scene receipt differs')
        checked(receipt['inputs']);checked({str(dataset/scene/k):h for k,h in receipt['outputs'].items()})
        require(receipt['outputs']['features.npz']==sha(dataset/scene/'features.npz'),'consumed feature identity differs')
        members=rows(r['R1_eligible']);keys=np.array([x['geometry_key'] for x in members])
        require(len(members)==r['member_count'] and len({(x['geometry_key'],x['class_index']) for x in members})==len(members),'duplicate/missing members')
        unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
        w=1./(len(unique)*counts[inverse]);weight_by_scene[scene]=w
        with np.load(dataset/scene/'features.npz',allow_pickle=False) as a:
            require(set(a.files)==({'X','weights','y'} if role=='train' else {'X','weights'}),'label role isolation failed')
            require(np.array_equal(a['weights'],w),'row weighting differs')
            require(a['X'].shape==(len(members),20) and np.isfinite(a['X']).all(),'feature schema differs')
            data[scene]={k:a[k] for k in a.files}
        if role!='train':continue
        gt=np.loadtxt(dataset/'ground_truth_train'/f'{scene}.txt',dtype=np.int64)
        gs=read(r['geometry_file']);lookup={g['geometry_key']:i for i,g in enumerate(gs)}
        with np.load(r['points_file'],allow_pickle=False) as p:
            points=[p[g['array_key']] for g in gs]
        for g,points_i in zip(gs,points):
            require(len(points_i)==g['point_count'] and np.array_equal(points_i,np.unique(points_i)),'point count/order differs')
            require(not len(points_i) or (points_i[0]>=0 and points_i[-1]<len(gt)),'point domain differs')
            require(hashlib.sha256(np.asarray(points_i,dtype='<i8').tobytes()).hexdigest()==g['points_sha256'],'point hash differs')
        targets,valid_count=sparse_targets(points,gt,CLASS_IDS)
        y=np.array([targets[lookup[m['geometry_key']],m['class_index']] for m in members])
        error=float(np.max(np.abs(y-data[scene]['y'])))
        require(error==0.,'full independent supervision differs')
        require(valid_count==r['valid_gt_count'],'valid GT count differs')
        target_audits.append({'scene_name':scene,'geometry_count':len(gs),'member_count':len(y),'max_target_error':error})
        print('all target rows and provenance verified',scene,flush=True)
    write(out/'targets.json',{'complete':True,'scene_count':len(target_audits),'rows':target_audits,
        'geometry_count':sum(x['geometry_count'] for x in target_audits),'target_row_count':sum(x['member_count'] for x in target_audits),'max_target_error':max(x['max_target_error'] for x in target_audits)})
    model_reports=[];selections={}
    with threadpool_limits(limits=8):
        for name,(fitting,heldout) in expected.items():
            actual=next(r for r in fs['rows'] if r['model']==name)
            old_path=fit/'models'/f'{name}.joblib';require(sha(old_path)==actual['model_sha256'],'model artifact differs')
            old=joblib.load(old_path)
            require(all(old.get_params()[k]==v for k,v in PARAMS.items()),'persisted model hyperparameters differ')
            X=np.concatenate([data[s]['X'] for s in fitting]);y=np.concatenate([data[s]['y'] for s in fitting])
            w=np.concatenate([weight_by_scene[s] for s in fitting]);w*=len(w)/w.sum()
            require(len(y)==actual['fit_row_count'] and np.isclose(w.sum(),actual['sample_weight_sum']),'actual fitting row/weight count differs')
            fresh=HistGradientBoostingRegressor(**PARAMS).fit(X,y,sample_weight=w)
            require(fresh.n_iter_==old.n_iter_==100,'tree iteration count differs')
            joblib.dump(fresh,out/'models'/f'{name}.joblib')
            errors=[];same_selections=True
            for s in heldout:
                r=reports[s];original_receipt=read(fit/'predictions'/s/'receipt.json')
                require(original_receipt['model']==str(old_path) and original_receipt['model_sha256']==sha(old_path),'scene uses wrong fold/full model')
                checked({str(fit/'predictions'/s/k):h for k,h in original_receipt['outputs'].items()})
                old_scores=np.load(fit/'predictions'/s/'R2_eligible_scores.npy',allow_pickle=False)
                scores=np.clip(fresh.predict(data[s]['X']),0,1)
                require(scores.shape==old_scores.shape,'prediction coverage differs')
                error=float(np.max(np.abs(scores-old_scores)));errors.append(error)
                require(error<=1e-12,'independent refit prediction differs')
                np.save(out/'scores'/f'{s}.npy',scores)
                members=rows(r['R1_eligible'])
                chosen=sorted(range(len(members)),key=lambda i:(-float(scores[i]),not members[i]['has_original_native_alias'],members[i]['geometry_key'],members[i]['class_index']))[:600]
                selection=[{**members[i],'score':float(scores[i])} for i in chosen]
                original=rows(fit/'predictions'/s/'R2_selected.jsonl')
                same=selection==original;same_selections &= same
                require([{k:v for k,v in x.items() if k!='score'} for x in selection]==[{k:v for k,v in x.items() if k!='score'} for x in original],'refit changes selected members/ties')
                selections[s]=selection
            model_reports.append({'model':name,'fit_scans':len(fitting),'heldout_scans':len(heldout),
                'max_refit_score_error':max(errors),'selected_rows_exact_equal':same_selections,'model_sha256':sha(out/'models'/f'{name}.joblib')})
            print('independent refit',name,model_reports[-1],flush=True)
    write(out/'refit.json',{'complete':True,'models':model_reports,'prediction_scene_count':len(selections)})
    # Re-evaluate using scores from these independently refitted models.
    binding=read(evaluation/'gt_binding.json')
    require(sha(evaluation/'gt_binding.json')==es['gt_binding_sha256'],'GT binding identity differs')
    require(binding['roles']=={k:partition['roles'][k] for k in ['development_validation','diagnostic']},'evaluation roles differ')
    for r in binding['rows']:checked(r['files'])
    gt_by_scene={r['scene_name']:r for r in binding['rows']}
    gt_root=Path('/media/jia/软件1/scannet_validation_stream/dm_sms1_ncs_validation60_20260820/ground_truth_once_20260822')
    geometry={s:{g['geometry_key']:g for g in read(reports[s]['geometry_file'])} for s in gt_by_scene}
    class Predictions(Mapping):
        def __init__(self,scenes):self.scenes=scenes
        def __iter__(self):return iter(self.scenes)
        def __len__(self):return len(self.scenes)
        def __getitem__(self,scene):
            if scene not in self.scenes:raise KeyError(scene)
            selected=selections[scene];mask=np.zeros((gt_by_scene[scene]['point_count'],len(selected)),dtype=bool)
            with np.load(reports[scene]['points_file'],allow_pickle=False) as pts:
                for i,r in enumerate(selected):mask[pts[geometry[scene][r['geometry_key']]['array_key']],i]=True
            return {'pred_masks':mask,'pred_classes':np.array([r['class_index'] for r in selected],dtype=np.int64),
                'pred_scores':np.array([r['score'] for r in selected],dtype=np.float64)}
    AP={}
    for role in ['development_validation','diagnostic']:
        with (out/f'{role}_AP.log').open('x') as log,contextlib.redirect_stdout(log):
            averages,*_=evaluator.evaluate(Predictions(partition['roles'][role]),str(gt_root),str(out/f'{role}_AP.csv'),dataset='scannet200')
        AP[role]={k:float(averages[v]) for k,v in {'ap':'all_ap','ap50':'all_ap_50%','ap25':'all_ap_25%'}.items()}
        require(AP[role]==es['AP'][role],'independently refitted AP differs')
        print('refit AP exact',role,AP[role],flush=True)
    write(out/'summary.json',{'complete':True,'full_target_audit_sha256':sha(out/'targets.json'),
        'refit_sha256':sha(out/'refit.json'),'identity_sha256':sha(out/'identity.json'),'AP':AP,
        'all_six_AP_metrics_exact_equal':True,'whole_system_oof':False,
        'limitations':['Same sklearn implementation and frozen visual evidence','Same ScanNet AP evaluator, not an independent AP implementation','Historical RGB and upstream supervision limitations remain']})


if __name__=='__main__':main()
