#!/usr/bin/env python3
"""Independently assemble the approved folds and refit all R3 heads.

Verify saved context/target roles and source hashes, independently rebuild
weights, replay every eligible score and selected member. Does not call the
R3 experiment's training or selection implementation.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from tools.prepare_fi1_r0_r1_inputs import read,write,sha,digest,load_partition
from tools.audit_fi1_r2_training import checked,require,PARAMS


def rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f]


def audit_feature_matrix(own,context,arm):
    require(arm in ('R3a','R3b','R3c'),'unknown audited arm')
    require(own.ndim==2 and own.shape[1]==20 and np.isfinite(own).all(),'own feature shape differs')
    if arm=='R3c':return own.copy()
    require(context.shape==(len(own),6) and np.isfinite(context).all(),'context feature shape differs')
    return np.concatenate([own,context],axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',type=Path)
    args=p.parse_args();base=args.base.resolve();out=args.output.resolve();run=(args.run or base/'r3_v1').resolve();source=base/'r2_v1/dataset'
    part=load_partition(base/'approved_split_v1');ident=read(run/'identity.json');summary=read(run/'fit_summary.json');ds=read(run/'dataset/summary.json')
    ti=read(run/'training_identity.json')
    require(summary['complete'] and summary['training_identity_sha256']==sha(run/'training_identity.json'),'R3 fitted identity differs')
    require(ti['identity_sha256']==sha(run/'identity.json') and ti['dataset_summary_sha256']==sha(run/'dataset/summary.json') and ti['source_summary_sha256']==sha(source/'summary.json'),'R3 data chain differs')
    require(ident['partition']==part and ident['partition_id']==digest(part) and ds['identity_sha256']==digest(ident),'R3 partition differs')
    require(ident['model_parameters']==PARAMS and ident['uses_learned_R2_score_as_feature'] is False,'R3 model/protocol differs')
    checked(ident['code']);checked(ti['code'])
    arms=list(ident['arms'])
    require(arms in (['R3a','R3b'],['R3c']),'unapproved arm set')
    # Audit declares its own feature and target contracts; it does not import
    # the experiment's selector or ARM_SPECS.
    from tools.fi1_r2_core import FEATURES
    from tools.fi1_r3_core import EXTRA_FEATURES
    for arm in arms:
        if isinstance(ident['arms'][arm],dict):
            require(ident['arms'][arm]=={'features':FEATURES if arm=='R3c' else FEATURES+EXTRA_FEATURES,
                'target':'y' if arm=='R3a' else 'y_competition'},'arm feature/target contract differs')
    checked(ident.get('competition_reference',{}))
    original=read(source/'summary.json');oi=read(source/'identity.json');require(original['identity_sha256']==digest(oi),'R2 source identity differs')
    checked(oi['code']);records={r['scene_name']:r for r in original['rows']}
    require(len(ds['rows'])==len(records)==260 and {r['scene_name'] for r in ds['rows']}==set(records),'scene coverage differs')
    expected={}
    train=part['roles']['train'];folds=part['training_heldout_folds']
    for arm in arms:
        for i in range(6):
            held=folds[i] if i<5 else part['roles']['development_validation']+part['roles']['diagnostic']
            fitting=[s for s in train if s not in held]
            if i<5:require(not {s.split('_')[0] for s in fitting}&{s.split('_')[0] for s in held},'main-scene overlap')
            expected[f'{arm}_fold{i}' if i<5 else f'{arm}_full200']=(arm,fitting,held)
    require(len(summary['rows'])==6*len(arms) and {r['name'] for r in summary['rows']}==set(expected),'model coverage differs')
    out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir()
    write(out/'identity.json',{'R3_identity_sha256':sha(run/'identity.json'),'fit_summary_sha256':sha(run/'fit_summary.json'),
        'dataset_summary_sha256':sha(run/'dataset/summary.json'),'audit_code_sha256':sha(__file__),'code':{str(Path(__file__).resolve()):sha(__file__)},'same_sklearn_implementation':True})
    data={};members={};assignment_stats=[]
    for r in ds['rows']:
        s=r['scene_name'];role=next(k for k,v in part['roles'].items() if s in v)
        require(r['role']==role and r['GT_read']==(role=='train'),'R3 label role mismatch')
        rr=read(run/'dataset'/s/'receipt.json');orig=read(source/s/'receipt.json')
        require(rr['summary']==r and rr['identity_sha256']==digest(ident) and rr['source_receipt_sha256']==sha(source/s/'receipt.json'),'scene receipt differs')
        checked(rr['inputs']);checked({str(run/'dataset'/s/k):h for k,h in rr['outputs'].items()})
        checked({str(source/s/k):h for k,h in orig['outputs'].items()})
        require(sha(records[s]['R1_eligible'])==r['R1_eligible_sha256']==records[s]['R1_eligible_sha256'],'member identity differs')
        m=rows(records[s]['R1_eligible']);members[s]=m
        keys=np.array([x['geometry_key'] for x in m]);u,inv,count=np.unique(keys,return_inverse=True,return_counts=True)
        w=1/(len(u)*count[inv])
        with np.load(source/s/'features.npz',allow_pickle=False) as a,np.load(run/'dataset'/s/'context.npz',allow_pickle=False) as c:
            require(set(c.files)==({'context','y_competition','assigned_gt'} if role=='train' else {'context'}),'R3 saved label role leakage')
            require(set(a.files)==({'X','y','weights'} if role=='train' else {'X','weights'}),'R2 saved label role leakage')
            require(np.array_equal(a['weights'],w),'independent weight mismatch')
            require(c['context'].shape==(len(m),6) and np.isfinite(c['context']).all(),'context shape/values differ')
            require(a['X'].shape==(len(m),20) and np.isfinite(a['X']).all(),'own feature shape differs')
            data[s]={'X':audit_feature_matrix(a['X'],c['context'],arms[0]),'w':w}
            if role=='train':
                y=c['y_competition'];assignment=c['assigned_gt']
                require(y.shape==assignment.shape==(len(m),) and np.isfinite(y).all() and ((y>=0)&(y<=a['y'])).all(),'invalid competition target')
                positive=assignment[assignment>=0];require(len(positive)==len(np.unique(positive)),'GT capacity violation')
                require(np.array_equal(y>0,assignment>=0),'assignment positivity mismatch')
                require(int((y>0).sum())==r['positive_competition_targets'],'positive target count mismatch')
                data[s].update(R3a=a['y'],R3b=y,R3c=y)
                if arms==['R3c']:
                    reference=base/'r3_v1/dataset'/s
                    old=read(reference/'receipt.json');path=reference/'context.npz'
                    require(str(path) in rr['inputs'] and rr['inputs'][str(path)]==old['outputs']['context.npz'],'missing audited target reference')
                    checked({str(path):old['outputs']['context.npz']})
                    with np.load(path,allow_pickle=False) as previous:
                        require(np.array_equal(y,previous['y_competition']) and np.array_equal(assignment,previous['assigned_gt']),'frozen R3b target mismatch')
                assignment_stats.append({'scene_name':s,'positive_R2_targets':int((a['y']>0).sum()),'positive_R3_targets':int((y>0).sum())})
    refits=[]
    with threadpool_limits(limits=8):
        for name,(arm,fitting,held) in expected.items():
            old=next(r for r in summary['rows'] if r['name']==name)
            require(old['fit_scenes']==fitting and old['heldout_scenes']==held and old['arm']==arm,'recorded fit roles differ')
            require(old['target']==('y' if arm=='R3a' else 'y_competition'),'recorded target differs')
            if 'feature_count' in old:require(old['feature_count']==(20 if arm=='R3c' else 26),'recorded feature dimension differs')
            old_path=run/'models'/f'{name}.joblib';require(sha(old_path)==old['model_sha256'],'original model changed')
            model_before=joblib.load(old_path);require(model_before.n_features_in_==(20 if arm=='R3c' else 26),'model consumes wrong feature count');require(all(model_before.get_params()[k]==v for k,v in PARAMS.items()),'model parameters differ')
            X=np.concatenate([data[s]['X'] for s in fitting]);y=np.concatenate([data[s][arm] for s in fitting]);w=np.concatenate([data[s]['w'] for s in fitting]);w*=len(w)/w.sum()
            require(len(y)==old['fit_rows'],'training row coverage differs')
            fresh=HistGradientBoostingRegressor(**PARAMS).fit(X,y,sample_weight=w)
            require(fresh.n_iter_==100,'refit incomplete');joblib.dump(fresh,out/'models'/f'{name}.joblib')
            max_error=0.;identical_scores=True
            for s in held:
                dest=run/'predictions'/s;receipt=read(dest/f'{arm}_receipt.json')
                require(receipt['model']==str(old_path) and receipt['model_sha256']==sha(old_path),'wrong fold/deployment model consumed')
                checked({str(dest/k):h for k,h in receipt['outputs'].items()})
                before=np.load(dest/f'{arm}_eligible_scores.npy',allow_pickle=False);after=np.clip(fresh.predict(data[s]['X']),0,1)
                require(before.shape==after.shape==(len(members[s]),),'prediction row coverage differs')
                error=float(np.max(np.abs(before-after)));max_error=max(max_error,error);identical_scores &= np.array_equal(before,after)
                require(error<=1e-12,'refit score differs beyond floating tolerance')
                order=sorted(range(len(after)),key=lambda i:(-float(after[i]),not members[s][i]['has_original_native_alias'],members[s][i]['geometry_key'],members[s][i]['class_index']))[:600]
                new=[{k:v for k,v in members[s][i].items() if k!='score'} for i in order]
                actual=rows(dest/f'{arm}_selected.jsonl')
                require(new==[{k:v for k,v in r.items() if k!='score'} for r in actual],'refit selected member/alias/order mismatch')
            record={'name':name,'max_refit_score_error':max_error,'scores_exact_equal':bool(identical_scores),'selected_members_and_order_equal':True,'prediction_scans':len(held),'model_sha256':sha(out/'models'/f'{name}.joblib')}
            refits.append(record);write(out/'progress.json',refits);print('R3 independent refit passed',record,flush=True)
    checked(ident['code']);checked(ti['code'])
    write(out/'summary.json',{'complete':True,'model_count':6*len(arms),'scene_arm_count':260*len(arms),'max_refit_score_error':max(r['max_refit_score_error'] for r in refits),
        'models':refits,'training_assignment_counts':assignment_stats,'identity_sha256':sha(out/'identity.json'),
        'selected_members_and_order_equal':True,'new_AP':False,'whole_system_oof':False})


if __name__=='__main__':main()
