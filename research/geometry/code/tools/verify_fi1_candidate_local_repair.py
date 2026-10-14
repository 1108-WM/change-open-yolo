#!/usr/bin/env python3
"""Independent point action/IoU/fold/selection replay after prediction sealing."""
from pathlib import Path
from collections import Counter
import sys,csv,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_candidate_local_repair as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def independent_iou(points,gt,ids,sizes):
    if len(points)<100:return np.zeros(len(ids))
    values,counts=np.unique(gt[points],return_counts=True);map_counts=dict(zip(values,counts));hit=np.array([map_counts.get(i,0) for i in ids])
    return hit/(len(points)+sizes-hit)


def verify():
    out=run.OUT;c=run.read(out/'contract.json');count=Counter()
    for mapping in (c['code'],c['inputs'],c['frozen_SL_models']):run.audit.prior.old.verify_files(mapping)
    for name in ('plan','features','labels','models','predictions','evaluation'):run.audit.prior.old.verify_files(run.read(out/(name+'_frozen.json'))['outputs'])
    physical={}
    for s in c['scenes']:
        if s[:9] in physical:assert physical[s[:9]]==c['folds'][s]
        physical[s[:9]]=c['folds'][s]
    for fold in range(5):
        h=run.read(out/'models'/f'fold{fold}'/'history.json');expected=[s for s in c['scenes'] if c['folds'][s]!=fold]
        assert h['train_scenes']==expected and len(expected)==160 and len(h['test_scenes'])==40
        assert set(s[:9] for s in h['train_scenes']).isdisjoint(s[:9] for s in h['test_scenes'])
        assert len(h['epochs'])==12 and [e['epoch'] for e in h['epochs']]==list(range(1,13));assert h['shared_between_arms'];count['folds']+=1
        held=next(s for s in c['scenes'] if c['folds'][s]==fold)
        for mode,path in [('train',out/'labels'/held/'labels.npz'),('predict',Path(c['jobs'][held]['GT'])),('features',out/'labels'/held/'labels.npz'),('plan',run.audit.OUT/'diagnosis'/'forbidden.json')]:
            try:run.Guard(c,mode,fold).hook('open',(str(path),'r',0))
            except PermissionError:count['guard_rejections']+=1
            else:raise AssertionError((mode,path))
    with (out/'evaluation/candidates.csv').open() as f:reported={(r['scene'],r['candidate_id'],r['arm']):r for r in csv.DictReader(f)}
    with (out/'evaluation/targets.csv').open() as f:target_report={(r['scene'],r['arm'],int(r['gt_id'])):r for r in csv.DictReader(f)}
    for s in c['scenes']:
        rows=run.read(out/'plan'/s/'candidates.json')
        if s in c['fixed10']:
            old=run.read(run.audit.OUT/'plan'/s/'selected.json')
            for r,o in zip(rows,old):assert r['views']==o['views'];count['frozen10_views_replayed']+=2
        job=c['jobs'][s];gt=np.loadtxt(job['GT'],dtype=np.int64);ids,sizes=run.audit.prior.valid_targets(gt,c['class_ids']);gg,pp=run.audit.prior.old.load_nt(job)
        baseline=np.zeros(len(ids))
        for p in pp.values():baseline=np.maximum(baseline,independent_iou(p,gt,ids,sizes))
        best={a:baseline.copy() for a in run.core.ARMS}
        with np.load(out/'features'/s/'data.npz') as z,np.load(out/'predictions'/s/'predictions.npz') as prediction:
            for r in rows:
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];assert np.array_equal(parent,pp[r['parent']]);before=independent_iou(parent,gt,ids,sizes)
                for a in run.core.ARMS:
                    probability=prediction[cid+'_'+a+'_probability'].astype(np.float32);seen=z[cid+'_'+a+'_groups']>=2
                    parent_flag=np.isin(halo,parent);selected=parent_flag.copy()
                    selected[parent_flag&seen&(probability[:,1]>=.8)]=False
                    selected[~parent_flag&seen&(probability[:,0]>=.8)]=True
                    p=halo[selected] if selected.sum()>=100 else parent
                    assert np.array_equal(p,prediction[cid+'_'+a+'_points']);assert np.array_equal(np.isin(halo[~seen],p),parent_flag[~seen]);count['independent_action_replays']+=1
                    after=independent_iou(p,gt,ids,sizes);best[a]=np.maximum(best[a],after);report=reported[(s,cid,a)]
                    deleted=np.setdiff1d(parent,p);added=np.setdiff1d(p,parent);assert len(added)==int(report['added']) and len(deleted)==int(report['deleted'])
                    if report['parent_target']:
                        gid=int(report['parent_target']);k=int(np.flatnonzero(ids==gid)[0]);assert abs(float(report['parent_iou'])-before[k])<1e-12 and abs(float(report['repaired_iou'])-after[k])<1e-12
                        assert int(report['deleted_target'])==int((gt[deleted]==gid).sum()) and int(report['added_target'])==int((gt[added]==gid).sum());count['known_parent_geometry_replays']+=1
        for a in run.core.ARMS:
            for i,gid in enumerate(ids):
                r=target_report[(s,a,int(gid))];assert abs(float(r['original_NT_best'])-baseline[i])<1e-12 and abs(float(r['expanded_NT_best'])-best[a][i])<1e-12;count['pool_target_replays']+=1
        print('verified',s,flush=True)
    assert count['folds']==5 and count['guard_rejections']==20 and count['frozen10_views_replayed']==160
    receipt=dict(complete=True,independent_point_and_geometry_replay=True,counts=dict(count),contract_sha256=sha(out/'contract.json'),summary_sha256=sha(out/'summary.json'),predictions_seal_sha256=sha(out/'predictions_frozen.json'))
    freeze_json(out/'independent_verified.json',receipt);freeze_json(out/'completion_verified.json',dict(complete=True,training_completed=True,OOF_predictions_completed=True,geometry_completed=True,independent_verified_sha256=sha(out/'independent_verified.json'),SL_preserved=True,AP_run=False))
    run.stage('complete_verified',**dict(count))

if __name__=='__main__':verify()
