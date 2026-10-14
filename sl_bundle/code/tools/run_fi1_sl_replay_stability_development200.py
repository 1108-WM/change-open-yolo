#!/usr/bin/env python3
"""Real SL replay and two preregistered repetitions of the exact L method."""
from pathlib import Path
import argparse, inspect, json, shutil, sys, time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import torch
from tools import fi1_sl_inference as inference
from tools import run_fi1_s_semantic_rank_residual_development200 as original
from tools import run_fi1_s_fixed_L_rescore_development200 as fixed
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
OUT=ROOT/'.local/fi1_sl_replay_stability_development200_20261010_v1'
L=original.OUT;S=original.S;SL=fixed.OUT
SEEDS=(20261010,20261011,20261012)
read=original.read;ctl=original.ctl


def stage(name,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=name,updated=time.time(),**kw));print(name,kw,flush=True)


def prepare():
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>=50*1024**3
    c=read(L/'contract.json');sc=read(S/'contract.json')
    c.update(version=OUT.name,seeds=list(SEEDS),B_model=sc['B_model'],B_model_sha256=sc['B_model_sha256'],
        inference='Rebuild symmetric NT pool, complete X26, frozen B26 selects800; real L weights + training normalization score these fixed members. Original8192 row shapes with zero padding; no cached inference input.',
        repetitions='Exactly original fit function, folds,32hidden,loss,20epochs,S800. Only initialization/pair/anchor/shuffle seed changes. Report all three seeds, no best-seed selection.',
        final_validation='This is exposed development200, downstream OOF only. Final single model and genuinely independent identities require separate frozen protocol; no independent GT accessed in this experiment.')
    c['inputs'][sc['B_model']]=sc['B_model_sha256']
    for s in c['scenes']:
        q=c['bindings'][s]['evidence'];c['inputs'][q]=sc['inputs'][q]
    c['dataset_source']=str(L/'dataset');c['original_SL']=str(SL)
    c['dataset_seal_sha256']=sha(L/'dataset_frozen.json')
    c['original_fit_source_sha256']=__import__('hashlib').sha256(inspect.getsource(original.fit).encode()).hexdigest()
    c['original_models']={str(q):sha(q) for q in (L/'models').glob('fold*/*') if q.is_file()}
    c['SL_reference_seal_sha256']=sha(SL/'predictions_frozen.json')
    for n in ('tools/fi1_sl_inference.py','tools/run_fi1_sl_replay_stability_development200.py',
              'tools/run_fi1_sl_replay_stability_development200.sh','tests/test_fi1_sl_inference.py'):
        q=ROOT/n;c['code'][str(q)]=sha(q)
    ctl.base.verify_files(c['code']);ctl.base.verify_files(c['original_models'])
    OUT.mkdir();freeze_json(OUT/'contract.json',c)
    for q in c['code']:
        dest=OUT/'code_snapshot'/Path(q).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dest)
    stage('prepared',seeds=SEEDS)


def arm(seed):return OUT/'replay' if seed==SEEDS[0] else OUT/f'seed_{seed}'


def replay(c):
    sys.addaudithook(original.Guard(c,'predict').hook);torch.set_num_threads(4)
    ctl.base.verify_files(c['code']);ctl.base.verify_files(c['original_models'])
    b=joblib.load(c['B_model']);assert sha(c['B_model'])==c['B_model_sha256']
    models={f:inference.load_L(L/'models'/f'fold{f}'/'model.pt',L/'models'/f'fold{f}'/'normalization.npz') for f in range(5)}
    root=arm(SEEDS[0]);outputs={}
    for j,s in enumerate(c['scenes']):
        stage('real_frontend_replay',scene=s,completed=j,total=200);d=root/'predictions'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():ctl.base.verify_files(read(d/'receipt.json')['outputs']);outputs.update(read(d/'receipt.json')['outputs']);continue
        for n in ('geometry','points','evidence'):
            q=c['bindings'][s][n];assert sha(q)==c['inputs'][q]
        members,x26,scores=inference.build_pool(c['bindings'][s],b)
        rows,ca,indices,delta=inference.rescore_pool(members,x26,scores,*models[c['folds'][s]])
        freeze_json(d/'members.json',members)
        np.savez_compressed(d/'pool.npz',x26=x26,scores=scores,indices=indices,delta=delta)
        freeze_json(d/'semantic_SL.json',rows);freeze_json(d/'class_agnostic_SL.json',ca)
        r=dict(complete=True,outputs={str(d/n):sha(d/n) for n in ('members.json','pool.npz','semantic_SL.json','class_agnostic_SL.json')},
               fold=c['folds'][s],pool_members=len(members),context='full S pool',GT_used=False,cached_scores_used=False)
        freeze_json(d/'receipt.json',r);outputs.update(r['outputs'])
    freeze_json(root/'predictions_frozen.json',dict(complete=True,outputs=outputs,raw_GT_read=False))


def verify(c,seed):
    # No labels/matching needed to verify identities, scores, reload or positivity.
    sys.addaudithook(original.Guard(c,'verify').hook);torch.set_num_threads(4)
    root=arm(seed);ctl.base.verify_files(read(root/'predictions_frozen.json')['outputs']);diffs=0;members_count=0
    models_root=L/'models' if seed==SEEDS[0] else root/'models'
    models={f:inference.load_L(models_root/f'fold{f}'/'model.pt',models_root/f'fold{f}'/'normalization.npz') for f in range(5)}
    for j,s in enumerate(c['scenes']):
        stage('verify_inference',seed=seed,scene=s,completed=j)
        if seed==SEEDS[0]:
            d=root/'predictions'/s;pool=read(d/'members.json')
            with np.load(d/'pool.npz',allow_pickle=False) as z:x26,base=z['x26'],z['scores']
            assert pool==ctl.base.rows(S/'predictions'/s/'members.jsonl')
            with np.load(S/'predictions'/s/'pool.npz',allow_pickle=False) as z:
                assert np.array_equal(x26,z['x26']) and np.array_equal(base,z['scores'])
        else:
            pool=ctl.base.rows(S/'predictions'/s/'members.jsonl')
            with np.load(S/'predictions'/s/'pool.npz',allow_pickle=False) as z:x26,base=z['x26'],z['scores']
        rows,ca,indices,delta=inference.rescore_pool(pool,x26,base,*models[c['folds'][s]])
        assert rows==read(root/'predictions'/s/'semantic_SL.json') and ca==read(root/'predictions'/s/'class_agnostic_SL.json')
        if seed==SEEDS[0]:
            assert rows==read(SL/'predictions'/s/'semantic_SL.json') and ca==read(SL/'predictions'/s/'class_agnostic_SL.json')
            with np.load(L/'predictions'/s/'scores.npz',allow_pickle=False) as z:
                actual=inference.core.adjusted(base[indices],delta);diffs+=int((actual!=z['full'][indices]).sum())
                assert np.array_equal(actual,z['full'][indices])
        srows=read(S/'predictions'/s/'semantic_S.json')
        assert {ctl.shared.key(m) for m in rows}=={ctl.shared.key(m) for m in srows}
        assert all(m['score']>0 for m in rows);members_count+=len(rows)
    freeze_json(root/'inference_verified.json',dict(complete=True,scenes=200,members=members_count,bitwise_score_differences=diffs,
        full_pool_context_exact=seed==SEEDS[0],model_reload_exact=True,S_member_identity_exact=True,all_selected_positive=True,
        predictions_seal_sha256=sha(root/'predictions_frozen.json'),raw_GT_read=False))


def load_training_scenes(c,scenes,labels=True):
    arrays=[];ys=[];ids=[]
    for i,s in enumerate(scenes):
        source=L/'dataset'/s;ctl.base.verify_files(read(source/'receipt.json')['outputs'])
        with np.load(source/'features.npz',allow_pickle=False) as z:arrays.append(dict(z));ids.append(np.full(len(z['S']),i,dtype=np.int16))
        if labels:
            with np.load(source/'labels.npz',allow_pickle=False) as z:ys.append(dict(z))
    result={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]};result['scene_ids']=np.concatenate(ids)
    if labels:result.update({k:np.concatenate([a[k] for a in ys]) for k in ys[0]})
    return result


def fit(c,seed,fold):
    assert seed in SEEDS[1:] and fold in range(5)
    root=arm(seed);root.mkdir(exist_ok=True)
    if not (root/'dataset_frozen.json').exists():freeze_json(root/'dataset_frozen.json',dict(complete=True,source=str(L/'dataset_frozen.json'),sha256=c['dataset_seal_sha256']))
    assert sha(L/'dataset_frozen.json')==c['dataset_seal_sha256']
    # Execute original frozen function without editing it or its historical files.
    assert __import__('hashlib').sha256(inspect.getsource(original.fit).encode()).hexdigest()==c['original_fit_source_sha256']
    original.OUT=root;original.core.SEED=seed;original.load_scenes=load_training_scenes
    original.fit(c,fold)


def predict(c,seed):
    assert seed in SEEDS[1:];sys.addaudithook(original.Guard(c,'predict').hook);torch.set_num_threads(4)
    root=arm(seed);ctl.base.verify_files(c['code']);outputs={}
    models={f:inference.load_L(root/'models'/f'fold{f}'/'model.pt',root/'models'/f'fold{f}'/'normalization.npz') for f in range(5)}
    for j,s in enumerate(c['scenes']):
        stage('repetition_inference',seed=seed,scene=s,completed=j);d=root/'predictions'/s;d.mkdir(parents=True,exist_ok=True)
        pool=ctl.base.rows(S/'predictions'/s/'members.jsonl')
        with np.load(S/'predictions'/s/'pool.npz',allow_pickle=False) as z:x26,base=z['x26'],z['scores']
        rows,ca,_,_=inference.rescore_pool(pool,x26,base,*models[c['folds'][s]])
        freeze_json(d/'semantic_SL.json',rows);freeze_json(d/'class_agnostic_SL.json',ca)
        outputs.update({str(d/n):sha(d/n) for n in ('semantic_SL.json','class_agnostic_SL.json')})
    freeze_json(root/'predictions_frozen.json',dict(complete=True,outputs=outputs,raw_GT_read=False,seed=seed))


def evaluate(c,seed):
    root=arm(seed);fixed.OUT=root;fixed.evaluate(c)
    if seed==SEEDS[0]:
        refs=read(SL/'evaluation_summary.json');here=read(root/'evaluation_summary.json')
        for g in c['reporting_groups']:
            for v in original.VIEWS:
                assert here['AP'][g][v]['SL']==refs['AP'][g][v]['SL']
                assert np.array_equal(np.load(root/'evaluation'/f'{g}_{v}_SL_AP.npy'),
                    np.load(SL/'evaluation'/f'{g}_{v}_SL_AP.npy'),equal_nan=True)
        freeze_json(root/'AP_replay_verified.json',dict(complete=True,arrays_exact=16,all_metrics_exact=True))


def learnability(c,seed):
    """Same frozen heldout diagnostic pairs for all seeds; full-pool scoring.

    Labels/pairs are read only after all five models and SL predictions are sealed.
    These diagnostics never select checkpoint/seed or modify training.
    """
    root=arm(seed);assert read(root/'predictions_frozen.json')['complete'];torch.set_num_threads(4);stats={}
    for fold in range(5):
        scenes=[s for s in c['scenes'] if c['folds'][s]==fold]
        a=load_training_scenes(c,scenes,labels=False)
        model,mean,std=inference.load_L(root/'models'/f'fold{fold}'/'model.pt',root/'models'/f'fold{fold}'/'normalization.npz')
        scores=original.core.adjusted(a['S'],original.infer(model,a['x'],mean,std))
        with np.load(L/f'heldout_fold{fold}_pairs.npz',allow_pickle=False) as z:pairs=dict(z)
        stats[fold]=dict(train=read(root/'models'/f'fold{fold}'/'training_metrics.json'),
            heldout_before=original.core.ranking_metrics(a['S'],pairs),heldout_after=original.core.ranking_metrics(scores,pairs))
    freeze_json(root/'learnability.json',dict(complete=True,folds=stats,heldout_pairs='original frozen identical across seeds',model_selection=False))


def coverage(c,seed):
    root=arm(seed);ev=ctl.base.evaluator();ts=list(map(float,ev.opt['overlaps']));counts={};recovered=0
    for s in c['scenes']:
        old=read(S/'predictions'/s/'semantic_S.json');new=read(root/'predictions'/s/'semantic_SL.json')
        assert {ctl.shared.key(m) for m in old}=={ctl.shared.key(m) for m in new}
        census=read(c['jobs'][s]['target_census']);ious=census['ious']
        def hits(rows,t,semantic,positive):
            return {int(g) for m in rows if not positive or m['score']>0 for g,v in ious[m['geometry_key']].items()
                    if v>t and (not semantic or int(g)//1000==c['class_ids'][m['class_index']])}
        dd={}
        for t in ts:
            dd[f'{t:.2f}']={}
            for semantic in (True,False):
                for positive in (True,False):
                    a=hits(old,t,semantic,positive);b=hits(new,t,semantic,positive);assert a==b
                    dd[f'{t:.2f}'][f'{semantic}_{positive}']=len(b)
        rec=read(ROOT/'.local/fi1_s_c_selection_scope_development200_20261010_v1/scenes'/s/'members.json')['recovered_targets']
        for r in rec:assert int(r['gt_id']) in hits(new,min(ts,key=lambda t:abs(t-.75)),True,True)
        recovered+=len(rec);counts[s]=dd
    assert recovered==424
    freeze_json(root/'coverage_verified.json',dict(complete=True,all10_thresholds_semantic_CA_presence_positive_exact=True,recovered424=recovered,scenes=counts))


def report(c):
    summaries={seed:read(arm(seed)/'evaluation_summary.json')['AP'] for seed in SEEDS}
    stats={};lines=['# SL真实推理重放与三种子稳定性','',
        '完整S对称资格池计算X26，冻结B26选择800，L模型与训练折归一化提供最终置信。原种子重放成员／分数逐位一致，16个AP数组精确复现。两个追加种子固定20261011、20261012，仅改变原训练流程的随机源；三套均报告，不挑最好。','',
        '|组|种子|类别AP (%)|实际去类AP (%)|','|---|---|---:|---:|']
    for g in c['reporting_groups']:
        stats[g]={}
        for seed in SEEDS:lines.append(f"|{g}|{seed}|{100*summaries[seed][g]['semantic']['SL']['ap']:.6f}|{100*summaries[seed][g]['class_agnostic']['SL']['ap']:.6f}|")
        for v in original.VIEWS:
            stats[g][v]={}
            for metric in ('ap','ap50','ap25'):
                values=[summaries[seed][g][v]['SL'][metric] for seed in SEEDS]
                stats[g][v][metric]=dict(mean=float(np.mean(values)),min=float(min(values)),max=float(max(values)),values=values)
        vals=[stats[g][v]['ap'] for v in original.VIEWS]
        lines.append(f"|{g}|均值 [范围]|{100*vals[0]['mean']:.6f} [{100*vals[0]['min']:.6f}, {100*vals[0]['max']:.6f}]|{100*vals[1]['mean']:.6f} [{100*vals[1]['min']:.6f}, {100*vals[1]['max']:.6f}]|")
    gates={seed:all(summaries[seed]['development200'][v]['SL']['ap']>summaries[seed]['development200'][v][ref]['ap'] for v in original.VIEWS for ref in ('S','C','R')) for seed in SEEDS}
    for seed in SEEDS:assert read(arm(seed)/'coverage_verified.json')['recovered424']==424
    lines+=['',f'三套分别超过S/C/R双AP：{gates}。全部固定S800及原10档覆盖／424正分保护；不代表每类／每折AP无损。',
        '', '均值由每套完整分组AP计算，不用折AP平均替代完整200评测；仅三个预定种子，范围不是置信区间。原五折仅下游头OOF，开发200已经暴露。最终单模型和独立评测安排见final_validation_plan.md。生产Full未替换，U/U′未追加。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    freeze_json(OUT/'summary.json',dict(complete=True,seeds=list(SEEDS),groups=stats,each_seed_both_AP_pass=gates,
        all_S_coverage_preserved=True,all_recovered424_preserved=True,production_modified=False))
    stage('results_complete')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','replay','verify','fit','predict','learnability','evaluate','coverage','report']);p.add_argument('--seed',type=int,default=SEEDS[0]);p.add_argument('--fold',type=int)
    a=p.parse_args()
    if a.stage=='prepare':prepare();return
    c=read(OUT/'contract.json')
    if a.stage=='fit':fit(c,a.seed,a.fold)
    elif a.stage in ('verify','predict','learnability','evaluate','coverage'):globals()[a.stage](c,a.seed)
    else:globals()[a.stage](c)


if __name__=='__main__':main()
