#!/usr/bin/env python3
"""Independent fold, proposal/action, pool, supervision and CPU model replay."""
from pathlib import Path
from collections import Counter
import csv,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_deletion_blocks as run
from tools.fi1_frontend_protocol import sha256_file as sha


def iou(p,gt,ids,sizes):
    if len(p)<100:return np.zeros(len(ids))
    # Independent histogram implementation (evaluation uses unique).
    histogram=Counter(gt[p].tolist());hit=np.array([histogram[int(g)] for g in ids]);return hit/(len(p)+sizes-hit)


def verify():
    c=run.read(run.OUT/'contract.json');counts=Counter();model_replays=[];torch.set_num_threads(2)
    for mapping in (c['code'],c['source_bindings'],c['frozen_SL_models']):run.previous.original.audit.prior.old.verify_files(mapping)
    for stage in ('labels','generators','training_blocks','verifiers','predictions','evaluation'):run.previous.original.audit.prior.old.verify_files(run.read(run.OUT/(stage+'_frozen.json'))['outputs'])
    for fold in range(5):
        holdout=[s for s in c['scenes'] if c['folds'][s]==fold];train=[s for s in c['scenes'] if c['folds'][s]!=fold]
        for inner in [i for i in range(5) if i!=fold]+[None]:
            d=run.OUT/'generators'/f'outer{fold}'/('full' if inner is None else f'inner{inner}');h=run.read(d/'history.json')
            expected=[s for s in c['scenes'] if c['folds'][s] not in ({fold} if inner is None else {fold,inner})]
            test=[s for s in c['scenes'] if c['folds'][s]==(fold if inner is None else inner)]
            assert h['train_scenes']==expected and h['predict_scenes']==test and len(h['epochs'])==12 and not h['points_sampled_with_replacement']
            assert set(s[:9] for s in expected).isdisjoint(s[:9] for s in test+holdout);counts['proposal_models']+=1
            guard=run.Guard(c,'train',expected)
            for s in set(test+holdout):
                try:guard.hook('open',(str(run.OUT/'labels'/s/'data.npz'),'r',0))
                except PermissionError:counts['fold_label_rejections']+=1
                else:raise AssertionError('nested fold leak')
        blockmeta=run.read(run.OUT/'training_blocks'/f'outer{fold}'/'meta.json');assert blockmeta['scenes']==train and all(s in train for s,cid,k in blockmeta['keys']);counts['training_blocks']+=len(blockmeta['keys'])
        for kind in ('summary','relation'):
            h=run.read(run.OUT/'verifiers'/f'outer{fold}'/kind/'history.json');assert h['train_scenes']==train and len(h['epochs'])==40 and h['each_block_once_per_epoch'] and h['positive_cost']<=5;counts['verifier_models']+=1
        guard=run.Guard(c,'verifier_train');guard.outer=fold
        for other in range(5):
            if other==fold:continue
            try:guard.hook('open',(str(run.OUT/'training_blocks'/f'outer{other}'/'data.npz'),'r',0))
            except PermissionError:counts['block_fold_rejections']+=1
            else:raise AssertionError('verifier outer label leak')
        for mode,path in [('predict',run.OUT/'labels'/train[0]/'meta.json'),('predict',Path(c['jobs'][holdout[0]]['GT'])),('train',Path(c['jobs'][train[0]]['GT'])),('predict',run.ROOT/'.local/fi1_local_edit_ablation_independent_analysis_20261012/component_probe.json')]:
            try:run.Guard(c,mode,train).hook('open',(str(path),'r',0))
            except PermissionError:counts['inference_GT_rejections']+=1
            else:raise AssertionError('inference/diagnostic access leak')
    with (run.OUT/'evaluation/candidates.csv').open() as f:reported={(r['scene'],r['candidate_id'],r['arm']):r for r in csv.DictReader(f)}
    with (run.OUT/'evaluation/targets.csv').open() as f:targets={(r['scene'],r['gt_id'],r['arm']):r for r in csv.DictReader(f)}
    totals={a:Counter() for a in run.ARMS};summary=run.read(run.OUT/'summary.json')
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=run.previous.original.audit.prior.valid_targets(gt,c['class_ids']);geometry,nt=run.previous.original.audit.prior.old.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,iou(p,gt,ids,sizes))
        best={a:baseline.copy() for a in run.ARMS};fold=c['folds'][s];rawpath=run.OUT/'generators'/f'outer{fold}'/'full'/'proposals'/s/'data.npz'
        data=run.examples([s]);metadata={r['candidate_id']:r for r in run.read(run.OUT/'labels'/s/'meta.json')}
        with np.load(rawpath) as proposals,np.load(run.OUT/'predictions'/s/'predictions.npz') as pr,np.load(run.PREVIOUS/'predictions'/s/'predictions.npz') as old,np.load(run.OUT/'labels'/s/'data.npz') as labels:
            for key,e in sorted(data.items()):
                cid=key[1];parent=e['parent'];halo=e['halo'];seen=e['visible']>=2
                base={r['candidate_id']:r for r in run.read(run.SOURCE/'labels'/s/'meta.json')}[cid]
                y,meta=run.core.stable_supervision(gt,parent,halo,base,c['class_ids']);assert np.array_equal(y,labels[cid+'_y']) and meta['stable']==metadata[cid]['stable'];counts['identity_supervisions']+=1
                for source in ('B','G'):
                    probability=proposals[cid+'_'+source+'_probability'];blocks=run.core.deletion_blocks(parent,halo,probability,e['visible'],e['neighbors']);sealed=run.unpack(proposals,cid,'_'+source)
                    assert len(blocks)==len(sealed) and all(np.array_equal(a,b) for a,b in zip(blocks,sealed));counts['GT_free_block_partitions']+=1
                    if source=='B':assert np.array_equal(probability,old[cid+'_B_probability']);counts['frozen_B_probability_replays']+=1
                    expected=run.core.delete(parent,halo,blocks);assert np.array_equal(expected,pr[cid+'_'+source+'_raw_points'])
                    for kind in ('summary','relation'):
                        a=source+'_'+kind;prob=pr[cid+'_'+a+'_block_prediction'];chosen=prob[:,0].astype(np.float32)>=.9;expected=run.core.delete(parent,halo,blocks,chosen);assert np.array_equal(expected,pr[cid+'_'+a+'_points']);counts['block_selection_replays']+=len(blocks)
                cdeleted=np.setdiff1d(parent,old[cid+'_C_points']);cpoints=np.setdiff1d(parent,cdeleted);cpoints=parent if len(cpoints)<100 else cpoints;assert np.array_equal(cpoints,pr[cid+'_C_gate_points'])
                for a in run.ARMS:
                    p=pr[cid+'_'+a+'_points'];assert np.isin(p,parent).all() and np.isin(halo[~seen],p).tolist()==np.isin(halo[~seen],parent).tolist();remove=np.setdiff1d(parent,p);report=reported[s,cid,a];counts['point_actions']+=1
                    assert len(remove)==int(report['deleted']);counts_arm=totals[a];counts_arm.update(parents=1,actions=bool(len(remove)),deleted=len(remove));best[a]=np.maximum(best[a],iou(p,gt,ids,sizes))
                    if report['target']:
                        gid=int(report['target']);k=int(np.flatnonzero(ids==gid)[0]);before=iou(parent,gt,ids,sizes)[k];after=iou(p,gt,ids,sizes)[k]
                        assert abs(before-float(report['parent_iou']))<1e-12 and abs(after-float(report['repaired_iou']))<1e-12
                        lost=int((gt[remove]==gid).sum());pollution=int(((gt[remove]>=1000)&(gt[remove]!=gid)).sum());assert lost==int(report['deleted_target']) and pollution==int(report['deleted_other'])
                        counts_arm.update(deleted_target=lost,deleted_other=pollution,unknown_edits=len(remove)-lost-pollution)
                    else:counts_arm['unknown_edits']+=len(remove)
        for a in run.ARMS:
            for j,g in enumerate(ids):
                r=targets[s,str(g),a];assert abs(float(r['original_NT_best'])-baseline[j])<1e-12 and abs(float(r['expanded_NT_best'])-best[a][j])<1e-12;counts['pool_targets']+=1
        print('independent deletion geometry',s,flush=True)
    for a,cnt in totals.items():
        for k,v in cnt.items():assert v==summary['arms'][a].get(k,0),(a,k,v,summary['arms'][a].get(k,0))
    # CPU proposal and verifier replays for all folds. Regions recomputed independently
    # from point inputs for selected first two candidates (not error/GT selected).
    for fold in range(5):
        s=next(s for s in c['scenes'] if c['folds'][s]==fold);data=run.examples([s]);model=run.previous.original.core.CandidateLocalSegmenter().eval();model.load_state_dict(torch.load(run.OUT/'generators'/f'outer{fold}'/'full'/'model.pt',map_location='cpu',weights_only=True));verifiers={}
        for kind in ('summary','relation'):
            v=run.core.BlockVerifier(kind).eval();v.load_state_dict(torch.load(run.OUT/'verifiers'/f'outer{fold}'/kind/'model.pt',map_location='cpu',weights_only=True));verifiers[kind]=v
        with np.load(run.OUT/'generators'/f'outer{fold}'/'full'/'proposals'/s/'data.npz') as raw,np.load(run.OUT/'predictions'/s/'predictions.npz') as pred:
            for key in sorted(data)[:2]:
                cid=key[1];e=data[key]
                with torch.inference_mode():logits,_,_=run.forward(model,[e],'cpu');prob=logits.softmax(1).numpy().astype(np.float16)
                diff=float(np.abs(prob.astype(np.float32)-raw[cid+'_G_probability'].astype(np.float32)).max());assert diff<=.001;model_replays.append(dict(outer=fold,scene=s,candidate=cid,kind='generator',maximum_difference=diff))
                # GPU quantized point probabilities are the sealed action input.
                for source in ('B','G'):
                    blocks=run.unpack(raw,cid,'_'+source);regions=run.core.region_features(e,raw[cid+'_'+source+'_probability'],blocks).astype(np.float16);assert np.array_equal(regions,raw[cid+'_'+source+'_regions'])
                    for kind,v in verifiers.items():
                        with torch.inference_mode():out=run.core.decoded(v(torch.tensor(regions.astype(np.float32)))).numpy().astype(np.float16) if len(regions) else np.empty((0,5),np.float16)
                        saved=pred[cid+'_'+source+'_'+kind+'_block_prediction'];diff=float(np.abs(out.astype(np.float32)-saved.astype(np.float32)).max(initial=0));assert diff<=.001
                        # Quantization close to the threshold is explicitly checked.
                        points=run.core.delete(e['parent'],e['halo'],blocks,out[:,0].astype(np.float32)>=.9);assert np.array_equal(points,pred[cid+'_'+source+'_'+kind+'_points']);model_replays.append(dict(outer=fold,scene=s,candidate=cid,kind=source+'_'+kind,maximum_difference=diff))
    assert counts['proposal_models']==25 and counts['verifier_models']==10 and counts['point_actions']==11193 and counts['pool_targets']==4689*7
    run.seal(run.OUT/'independent_verified.json',dict(complete=True,counts=dict(counts),model_replays=model_replays,contract_sha256=sha(run.OUT/'contract.json'),summary_sha256=sha(run.OUT/'summary.json')))
    run.seal(run.OUT/'completion_verified.json',dict(complete=True,training_models=35,outer_pointsets=11193,nested_crossfit=True,GT_free_blocks=True,SL_preserved=True,AP_run=False,independent_verified_sha256=sha(run.OUT/'independent_verified.json')));run.stage('complete_verified',**dict(counts))


if __name__=='__main__':verify()
