#!/usr/bin/env python3
"""Fixed S800 member identities, frozen out-of-fold L scores, original AP."""
from pathlib import Path
from collections import Counter
import argparse,shutil,subprocess,sys,time,warnings
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_s_semantic_rank_residual_development200 as parent
from tools import run_fi1_sr_cross_control_development200 as cross
from tools import run_fi1_sr_score_factorization_development200 as factor
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
from tools.fi1_sr_cross_control import fixed_rescore,independent_rescore_match,key
OUT=ROOT/'.local/fi1_s_fixed_L_rescore_development200_20261010_v1'
S,L=parent.S,parent.OUT;ctl=parent.ctl;read=parent.read;VIEWS=parent.VIEWS;ARM='SL'


def stage(name,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=name,updated=time.time(),**kw));print(name,kw,flush=True)


def prepare():
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>=50*1024**3
    c=read(L/'contract.json');assert read(L/'independent_verified.json')['complete'] and read(L/'handoff_verified.json')['complete']
    for receipt in ('completion_verified.json','independent_verified.json','supplemental_verified.json'):
        ctl.base.verify_files(read(L/receipt)['outputs'])
    # Live-document hashes in the older handoff are historical snapshots, not immutable data.
    inputs={str(root/n):sha(root/n) for root,n in ((S,'contract.json'),(S,'predictions_frozen.json'),
        (L,'contract.json'),(L,'predictions_frozen.json'),(L,'inference_verified.json'),(L,'independent_verified.json'))}
    labels={str(L/'evaluation_summary.json'):sha(L/'evaluation_summary.json'),str(L/'learnability.json'):sha(L/'learnability.json')}
    ss=read(S/'predictions_frozen.json');ls=read(L/'predictions_frozen.json')
    for s in c['scenes']:
        for root,seal,names in ((S,ss,('members.jsonl','pool.npz','semantic_S.json','class_agnostic_S.json','semantic_C.json','class_agnostic_C.json')),
                                (L,ls,('scores.npz','semantic_L.json','class_agnostic_L.json'))):
            for n in names:
                q=root/'predictions'/s/n;inputs[str(q)]=seal['outputs'][str(q)]
        q=c['jobs'][s]['nt_geometry.json'];inputs[q]=c['inputs'][q]
        for view in VIEWS:
            q=S/'evaluation'/s/f'{view}_S_matches.json';labels[str(q)]=c['label_inputs'][str(q)]
        for q in (Path(c['jobs'][s]['target_census']),ROOT/'.local/fi1_s_c_selection_scope_development200_20261010_v1/scenes'/s/'members.json'):
            labels[str(q)]=c['label_inputs'][str(q)]
    for group in c['reporting_groups']:
        for view in VIEWS:
            q=L/'evaluation'/f'{group}_{view}_L_AP.npy';labels[str(q)]=read(L/'evaluation_summary.json')['AP_arrays'][str(q)]
    code=dict(c['code'])
    for n in ('tools/run_fi1_s_fixed_L_rescore_development200.py','tools/run_fi1_s_fixed_L_rescore_development200.sh',
              'tools/verify_fi1_s_fixed_L_rescore_development200.py'):
        q=ROOT/n;code[str(q)]=sha(q)
    contract=dict(version=OUT.name,scenes=c['scenes'],folds=c['folds'],reporting_groups=c['reporting_groups'],jobs=c['jobs'],
        class_ids=c['class_ids'],inputs=inputs,label_inputs=labels,code=code,arms=[ARM],budget=800,
        policy='S chooses exactly800 geometry-class identities. Frozen fivefold heldout L full scores rescore/reorder these800 only. No reselect/drop zero/small masks/refill/GT exception.',
        CA='Exact pointset max of actual fixed800 members, recompute winner and order, no refill or independent full-pool CA score.',
        GT_policy='No raw GT, training labels, model loading or model inference. Predict/verify forbid matching/census until predictions sealed. Afterwards rescore cached S intersections/ignore metadata bidirectionally and original evaluator recomputes AP.',
        acceptance='Both primaryAP strictly>S,C,R; zero lost recovered424 members/positive and zero lost S coverage at every original evaluator threshold, semantic andCA. Old90/new110/folds and delta fromL reported, not silently turned into additional hard gates.',
        failure='If lower gains but incomplete gates, retain conservative control only. If gain gone, evidence of reliance on reselection; do not auto-train. No weight/bound/threshold/K scan or Q/F,U/Uprime.',
        interpretation='S duplicate pairing selects highestS representative by construction. L inversion measures change in S representative order; cannot by itself establish worsening actual duplicates or justify stronger mimic loss.',
        scope='Exposed development200 old90/new110; original5 physical folds for downstream L only; upstream/S not whole-system blind. Passing is development candidate, no production deployment or unregistered heldout access.',
        new_training=False,new_model_inference=False,new_geometry=False,U_Uprime_added=False,production_modified=False)
    ctl.base.verify_files(inputs);ctl.base.verify_files(labels);ctl.base.verify_files(code)
    OUT.mkdir();freeze_json(OUT/'contract.json',contract)
    for q in code:
        dest=OUT/'code_snapshot'/Path(q).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dest)
    test=subprocess.run([sys.executable,'-m','pytest','tests/test_fi1_sr_cross_control.py','tests/test_fi1_class_agnostic_ablation.py','-q'],capture_output=True,text=True)
    (OUT/'tests.log').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stdout+test.stderr
    freeze_json(OUT/'tests_verified.json',dict(passed=True,log_sha256=sha(OUT/'tests.log')));stage('prepared')


def load_pool(s):
    pool=ctl.base.rows(S/'predictions'/s/'members.jsonl')
    with np.load(L/'predictions'/s/'scores.npz',allow_pickle=False) as z:scores=z['full']
    assert scores.shape==(len(pool),);return pool,scores


def predict(c):
    sys.addaudithook(cross.Guard('predict').hook);ctl.base.verify_files(c['inputs']);ctl.base.verify_files(c['code']);outputs={};receipts={}
    for j,s in enumerate(c['scenes']):
        stage('predict',scene=s,completed=j,total=200);d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True)
        pool,scores=load_pool(s);selected=read(S/'predictions'/s/'semantic_S.json');assert len(selected)==800
        rows=fixed_rescore(selected,pool,scores);geom=ctl.collapse_members(rows)
        assert {key(m) for m in rows}=={key(m) for m in selected}
        assert {g['geometry_key'] for g in geom}=={g['geometry_key'] for g in read(S/'predictions'/s/'class_agnostic_S.json')}
        np.savez_compressed(d/'scores.npz',full=scores)
        freeze_json(d/'semantic_SL.json',rows);freeze_json(d/'class_agnostic_SL.json',geom)
        receipt=dict(complete=True,fold=c['folds'][s],raw_GT_read=False,outputs={str(d/n):sha(d/n) for n in ('scores.npz','semantic_SL.json','class_agnostic_SL.json')})
        freeze_json(d/'receipt.json',receipt);outputs.update(receipt['outputs']);receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    freeze_json(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,receipts=receipts,raw_GT_read=False,new_model_inference=False))


def verify(c):
    sys.addaudithook(cross.Guard('verify_predictions').hook);ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs']);counts=Counter()
    for s in c['scenes']:
        pool,scores=load_pool(s);ix={key(m):i for i,m in enumerate(pool)};selected=read(S/'predictions'/s/'semantic_S.json')
        expected=sorted([{**m,'score':float(scores[ix[key(m)]])} for m in selected],key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))
        actual=read(OUT/'predictions'/s/'semantic_SL.json');assert expected==actual and len(actual)==800
        assert all((m['score']>0)==(scores[ix[key(m)]]>0) for m in actual)
        assert all(m['score']<=0 or scores[ix[key(m)]]>0 for m in selected)
        best={}
        for m in actual:
            k=m['geometry_key']
            if k not in best or (m['score'],-m['class_index'])>(best[k]['score'],-best[k]['class_index']):best[k]=m
        ca=sorted([dict(geometry_key=k,score=m['score'],winning_class_index=m['class_index']) for k,m in best.items()],key=lambda m:(-m['score'],m['geometry_key']))
        assert ca==read(OUT/'predictions'/s/'class_agnostic_SL.json');counts.update(members=800,geometries=len(ca),zero_scores=sum(m['score']==0 for m in actual))
    freeze_json(OUT/'inference_verified.json',dict(complete=True,counts=dict(counts),independent_rank_metadata_collapse=True,
        predictions_seal_sha256=sha(OUT/'predictions_frozen.json'),raw_GT_read=False))


def evaluate(c):
    sys.addaudithook(cross.Guard('evaluate').hook);ctl.base.verify_files(c['code']);ctl.base.verify_files(c['label_inputs']);ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs'])
    assert read(OUT/'inference_verified.json')['predictions_seal_sha256']==sha(OUT/'predictions_frozen.json')
    ev=ctl.base.evaluator();ca,_,_=ctl.class_agnostic_evaluator();receipts={}
    for j,s in enumerate(c['scenes']):
        stage('cached_matching',scene=s,completed=j,total=200);d=OUT/'evaluation'/s;d.mkdir(parents=True,exist_ok=True)
        sizes={g['geometry_key']:g['point_count'] for g in read(c['jobs'][s]['nt_geometry.json'])};outputs={}
        for view in VIEWS:
            old=read(S/'predictions'/s/f'{view}_S.json');new=read(OUT/'predictions'/s/f'{view}_SL.json');template=read(S/'evaluation'/s/f'{view}_S_matches.json')
            if view=='semantic':projected=ctl.project_multiclass(template,old,new,sizes,ev.CLASS_LABELS,ev.VALID_CLASS_IDS)
            else:projected=ctl.project_matches(template,[m['geometry_key'] for m in old if sizes[m['geometry_key']]>=100],new,ca.CLASS_LABELS[0])
            independent=independent_rescore_match(template,old,new,sizes,view=='class_agnostic');assert projected==independent
            q=d/f'{view}_SL_matches.json';freeze_json(q,projected);outputs[str(q)]=sha(q)
        receipt=dict(complete=True,outputs=outputs,independent_bidirectional_projection_exact=True,raw_GT_read=False);freeze_json(d/'receipt.json',receipt);receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    freeze_json(OUT/'matches_frozen.json',dict(complete=True,receipts=receipts,raw_GT_read=False))
    refs=read(L/'evaluation_summary.json')['AP'];ap={g:{v:{a:refs[g][v][a] for a in ('S','C','R','L')} for v in VIEWS} for g in c['reporting_groups']};arrays={}
    for view,evaluator in (('semantic',ev),('class_agnostic',ca)):
        matches={s:read(OUT/'evaluation'/s/f'{view}_SL_matches.json') for s in c['scenes']}
        for group,ss in c['reporting_groups'].items():
            stage('AP',view=view,group=group);subset={s:matches[s] for s in ss};x,*_=evaluator.evaluate_matches(subset)
            ap[group][view]['SL']=ctl.base.metrics(evaluator,x);q=OUT/'evaluation'/f'{group}_{view}_SL_AP.npy';np.save(q,x,allow_pickle=False);arrays[str(q)]=sha(q)
            if group=='development200':
                repeat,*_=evaluator.evaluate_matches(dict(reversed(list(subset.items()))));assert np.allclose(x,repeat,atol=1e-12,rtol=0,equal_nan=True)
            print('result',group,view,ap[group][view]['SL'],flush=True)
    freeze_json(OUT/'evaluation_summary.json',dict(complete=True,AP=ap,AP_arrays=arrays,full200_reverse_replayed=2,raw_GT_read=False))


def diagnostics(c):
    sys.addaudithook(cross.Guard('diagnostics').hook)
    # Reuse registered read-only coverage/rank helper in the isolated directory.
    factor.OUT=OUT;factor.ARMS=('SL',);factor.diagnostics(c)


def report(c):
    ap=read(OUT/'evaluation_summary.json')['AP'];d=read(OUT/'diagnostics.json')['groups']['development200']['SL'];main=ap['development200']
    both=all(main[v]['SL']['ap']>main[v][a]['ap'] for v in VIEWS for a in ('S','C','R'))
    coverage=not d['counts']['recovered_member_lost'] and not d['counts']['recovered_positive_lost'] and all(z['S']['semantic_lost']==0 and z['S']['CA_lost']==0 for z in d['coverage'].values())
    accepted=both and coverage
    result='通过双AP与S覆盖保护，可登记为开发候选，进入后续稳定性／独立验证；生产不自动替换。' if accepted else ('有排序收益但未通过全部门槛，只登记保守对照，不事后放宽验收。' if any(main[v]['SL']['ap']>main[v]['S']['ap'] for v in VIEWS) else '固定S集合未保留收益，L收益依赖重选；后继预算监督需另立合同，不自动训练。')
    lines=['# 固定S800，封存L分数：SL对照','',result,'',
        'S决定名单，L提供置信分数；相同成员身份／类别／点集，不重选、不删零分或小掩码、不补位。去类仅从这些成员同点集取最高分。原L模型和训练方法保持冻结，没有训练或新模型推理。',
        '', '|分组|方案|类别AP／AP50／AP25 (%)|实际去类AP／AP50／AP25 (%)|','|---|---|---:|---:|']
    for g in ap:
        for a in ('S','C','R','L','SL'):
            val=['／'.join(f'{100*ap[g][v][a][k]:.6f}' for k in ('ap','ap50','ap25')) for v in VIEWS];lines.append(f'|{g}|{a}|{val[0]}|{val[1]}|')
    lines+=['', '|固定条件效应|类别ΔAP(pp)|去类ΔAP(pp)|','|---|---:|---:|']
    for before,after,label in (('S','SL','固定S集合换L分数'),('SL','L','固定L分数换L集合'),('S','L','L总变化')):
        vals=[100*(main[v][after]['ap']-main[v][before]['ap']) for v in VIEWS];lines.append(f'|{label}|{vals[0]:+.6f}|{vals[1]:+.6f}|')
    lines+=['',f"双AP超过S/C/R={both}；S全部档位覆盖及424保护={coverage}；正式开发门={accepted}。424成员缺失{d['counts']['recovered_member_lost']}，无正分（含缺失）{d['counts']['recovered_positive_lost']}。",
        '', '相对L的新增覆盖取舍、所有目标逐档与424实际分数／全局同类排名见supplemental_report.md／diagnostics.json。两段差值按代数相加，不解释为独立可加机制收益。',
        '', '修正重复指标解释：最高S成员被标为代表，S的代表对重复排序天然占优。L重复逆序率上升表示改变S代表顺序，不能独立证明实际重复变多、去重能力变差；当前尚未证明学会重复抑制，不凭该数增大模仿S损失。旧报告保持封存，本项及live文档补充修正。',
        '', '仅暴露开发200＝旧90＋新110，原五折是L下游头OOF而非整系统盲测。预测封存前禁止缓存GT；随后复用S原双向匹配的交集／忽略信息，由原evaluator重新计算AP，未复用TP/FP决定。本轮无原始GT、模型加载、训练、U/U′或生产修改。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');freeze_json(OUT/'decision.json',dict(complete=True,both_AP_passed=both,coverage_passed=coverage,accepted=accepted,interpretation=result,
        next_training=False,production_modified=False,U_Uprime_added=False,duplicate_interpretation_corrected=True))
    files=[OUT/n for n in ('contract.json','tests_verified.json','predictions_frozen.json','inference_verified.json','matches_frozen.json','evaluation_summary.json','diagnostics.json','decision.json','report.md')]
    freeze_json(OUT/'completion_verified.json',dict(complete=True,outputs={str(q):sha(q) for q in files},raw_GT_read=False,new_training=False,new_model_inference=False,production_modified=False))
    write_json(OUT/'execution.json',dict(status='complete',stage='complete',resume_required=False,accepted=accepted))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','verify','evaluate','diagnostics','report']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:globals()[a.stage](read(OUT/'contract.json'))


if __name__=='__main__':main()
