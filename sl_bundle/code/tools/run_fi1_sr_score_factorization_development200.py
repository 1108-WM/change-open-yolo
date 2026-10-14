#!/usr/bin/env python3
"""P0 selected maxima, P1 full maxima fixed set, P2 full-pool K800."""
from pathlib import Path
from collections import Counter
import argparse
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_sr_cross_control_development200 as cross
from tools.fi1_sr_score_factorization import factorize
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha

OUT=ROOT/'.local/fi1_sr_score_factorization_development200_20261010_v1'
AUDIT=ROOT/'.local/fi1_sr_cross_independent_audit_20261010_v1'
S,R=cross.S,cross.R;ARMS=('P0','P1','P2');VIEWS=cross.VIEWS
read=cross.read;ctl=cross.ctl


def stage(name,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=name,updated=time.time(),**kw));print(name,kw,flush=True)


def prepare():
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>=50*1024**3
    c=read(cross.OUT/'contract.json');original=read(R/'contract.json')
    assert read(cross.OUT/'final_verified.json')['complete'] and read(AUDIT/'verified.json')['complete']
    ctl.base.verify_files(read(AUDIT/'verified.json')['outputs'])
    inputs=dict(c['inputs']);labels=dict(c['label_inputs']);code=dict(c['code'])
    cross_seal=read(cross.OUT/'predictions_frozen.json')
    for n in ('report.md','verified.json'):inputs[str(AUDIT/n)]=sha(AUDIT/n)
    for n in ('contract.json','final_verified.json','predictions_frozen.json','evaluation_summary.json'):
        inputs[str(cross.OUT/n)]=sha(cross.OUT/n)
    for s in c['scenes']:
        for n in ('geometry','points'):
            q=original['bindings'][s][n];inputs[q]=original['inputs'][q]
        for n in ('semantic_SR.json','class_agnostic_SR.json'):
            q=cross.OUT/'predictions'/s/n;inputs[str(q)]=cross_seal['outputs'][str(q)]
        for view in VIEWS:
            q=S/'evaluation'/s/f'{view}_C_matches.json';labels[str(q)]=read(S/'evaluation'/s/'receipt.json')['outputs'][str(q)]
        for view in VIEWS:
            q=cross.OUT/'evaluation'/s/f'{view}_SR_matches.json';labels[str(q)]=sha(q)
        q=ROOT/'.local/fi1_s_c_selection_scope_development200_20261010_v1/scenes'/s/'members.json'
        labels[str(q)]=sha(q)
    for n in ('tools/run_fi1_sr_score_factorization_development200.py','tools/fi1_sr_score_factorization.py',
              'tools/run_fi1_sr_score_factorization_development200.sh','tests/test_fi1_sr_score_factorization.py'):
        q=ROOT/n;code[str(q)]=sha(q)
    contract=dict(version=OUT.name,scenes=c['scenes'],folds=c['folds'],reporting_groups=c['reporting_groups'],
        jobs=c['jobs'],bindings=original['bindings'],class_ids=c['class_ids'],inputs=inputs,label_inputs=labels,code=code,
        arms=ARMS,budget=800,views=VIEWS,
        P0='Within S800 geometry maxima: R_max * S_member/S_selected_max; fixed800, zero R maxima preserved.',
        P1='Full S eligibility maxima: R_full_max * S_member/S_full_max; fixedS800, no refill.',
        P2='Same full-pool scores as P1, original deterministic member ties and reselectK800.',
        relative='Maximum-normalized relative scores, not normalized class probabilities. No multiplier clipping/epsilon/minimum/fusion weight.',
        zero_policy='P0 requires positive S denominator; zero selected R max yields zero geometry. Full pool currently all maxima positive; unforeseen zero S or R max falls back to original S for that geometry.',
        GT_policy='No targets during prediction/verification. All predictions sealed first; approved development200 GT permitted only for original evaluator matching of newly selected P2 members, exact bindings/hash whitelist. No heldout data.',
        invariants='P0 geometry keys/scores exactly match SR, class winner may differ; P0 CA AP arrays exactly SR. P0/P1 fixedS800; P1/P2 share exact full scores. No geometry/eligibility modification.',
        diagnostics='All original evaluator IoU thresholds; target presence and positive score separate; all target gains/losses vsS/C/R, recovered424, scene/global correct-member ranks, geometric maxima interface942/zero339, class ratios and budget entry/exit.',
        gates='P0 structure support if semantic AP>=S and >SR with fixed identical SR CA; P1 support if semantic>=S and CA>C. P2 acceptance strictly semantic>S,C,R and CA>S,C,R, zero lost recovered424 and no S>.75 semantic/CA target loss; report old90/new110/folds heterogeneity.',
        followup='Stop at these3 fixed tests; no new head, parameter/weight/clipping/threshold/K search or U/Uprime. P2 success yields development candidate only, no production deployment.',
        new_training=False,new_model_inference=False,new_geometry=False,U_Uprime_output=False,production_modified=False,
        scope='Exposed development200; R new head OOF physical folds, upstream/S historical training overlap, not whole-system blind acceptance.')
    ctl.base.verify_files(inputs);ctl.base.verify_files(code)
    OUT.mkdir(parents=True);freeze_json(OUT/'contract.json',contract)
    for q in code:
        dest=OUT/'code_snapshot'/Path(q).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dest)
    result=subprocess.run([sys.executable,'-m','pytest','tests/test_fi1_sr_score_factorization.py','tests/test_fi1_sr_cross_control.py','tests/test_fi1_class_agnostic_ablation.py','-q'],capture_output=True,text=True)
    (OUT/'tests.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stdout+result.stderr
    freeze_json(OUT/'tests_verified.json',dict(passed=True,log_sha256=sha(OUT/'tests.log')));stage('prepared')


def predict(c):
    sys.addaudithook(cross.Guard('predict').hook);ctl.base.verify_files(c['inputs']);ctl.base.verify_files(c['code'])
    outputs={};receipts={};total=Counter()
    for j,s in enumerate(c['scenes']):
        stage('predict',scene=s,completed=j,total=200);d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():r=read(d/'receipt.json');ctl.base.verify_files(r['outputs'])
        else:
            members,old=cross.pool_scores(s);selected=read(S/'predictions'/s/'semantic_S.json');ix={cross.key(m):i for i,m in enumerate(members)}
            indexes=[ix[cross.key(m)] for m in selected]
            p0,meta0=factorize(members,old['S'],old['R'],indexes);full,metaf=factorize(members,old['S'],old['R'])
            assert all(m['S_max']>0 and m['R_max']>0 and not m['fallback'] for m in metaf.values())
            results={'P0':cross.fixed_rescore(selected,members,p0),'P1':cross.fixed_rescore(selected,members,full),
                     'P2':cross.source.previous.rank(members,full)[:800]}
            np.savez_compressed(d/'scores.npz',P0=p0,full=full)
            local={str(d/'scores.npz'):sha(d/'scores.npz')}
            for a,rows in results.items():
                geom=ctl.collapse_members(rows)
                if a=='P0':
                    sr=read(cross.OUT/'predictions'/s/'class_agnostic_SR.json')
                    assert [(m['geometry_key'],m['score']) for m in geom]==[(m['geometry_key'],m['score']) for m in sr]
                for view,value in zip(VIEWS,(rows,geom)):
                    q=d/f'{view}_{a}.json';freeze_json(q,value);local[str(q)]=sha(q)
            freeze_json(d/'maxima.json',dict(selected=meta0,full=metaf));local[str(d/'maxima.json')]=sha(d/'maxima.json')
            counts=dict(full_geometries=len(metaf),selected_geometries=len(meta0),selected_R_max_zero=sum(v['R_max']==0 for v in meta0.values()),
                        selected_full_R_max_higher=sum(metaf[g]['R_max']>v['R_max'] for g,v in meta0.items()),
                        selected_S_max_differs=sum(metaf[g]['S_max']!=v['S_max'] for g,v in meta0.items()))
            r=dict(complete=True,outputs=local,counts=counts,raw_GT_read=False);freeze_json(d/'receipt.json',r)
        outputs.update(r['outputs']);total.update(r['counts']);receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    assert total['full_geometries']==14150 and total['selected_R_max_zero']==339 and total['selected_full_R_max_higher']==942 and total['selected_S_max_differs']==0
    freeze_json(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,receipts=receipts,counts=dict(total),raw_GT_read=False))


def verify(c):
    sys.addaudithook(cross.Guard('verify_predictions').hook)
    ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs']);ctl.base.verify_files(c['code'])
    counts=Counter()
    for s in c['scenes']:
        members,old=cross.pool_scores(s);ix={cross.key(m):i for i,m in enumerate(members)};selected=read(S/'predictions'/s/'semantic_S.json')
        scope={m['geometry_key'] for m in selected};groups={g:[] for g in scope};allgroups={}
        for i,m in enumerate(members):allgroups.setdefault(m['geometry_key'],[]).append(i)
        for m in selected:groups[m['geometry_key']].append(ix[cross.key(m)])
        with np.load(OUT/'predictions'/s/'scores.npz',allow_pickle=False) as z:p0,full=z['P0'],z['full']
        for name,gg,values in (('selected',groups,p0),('full',allgroups,full)):
            for g,indexes in gg.items():
                ms=max(old['S'][indexes]);mr=max(old['R'][indexes]);assert ms>0
                expected=np.asarray([mr*(float(old['S'][i])/float(ms)) for i in indexes])
                assert np.array_equal(expected,values[indexes]);assert max(values[indexes])==mr
                # Ratio invariance where nonzero scale; zero geometries truthfully zero.
                if mr>0:assert np.allclose(values[indexes]/mr,old['S'][indexes]/ms,atol=2e-16,rtol=2e-16)
                else:assert name=='selected' and not values[indexes].any()
                counts['geometry_formula_checks']+=1;counts['member_formula_checks']+=len(indexes)
        for a,values in (('P0',p0),('P1',full),('P2',full)):
            base=selected if a!='P2' else members
            expected=sorted([{**m,'score':float(values[ix[cross.key(m)]])} for m in base],key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:800]
            assert expected==read(OUT/'predictions'/s/f'semantic_{a}.json')
            best={}
            for m in sorted(expected,key=lambda m:(-m['score'],m['class_index'])):best.setdefault(m['geometry_key'],dict(geometry_key=m['geometry_key'],score=m['score'],winning_class_index=m['class_index']))
            ca=sorted(best.values(),key=lambda m:(-m['score'],m['geometry_key']));assert ca==read(OUT/'predictions'/s/f'class_agnostic_{a}.json')
    freeze_json(OUT/'inference_verified.json',dict(complete=True,counts=dict(counts),independent_formula_rank_collapse=True,
        predictions_seal_sha256=sha(OUT/'predictions_frozen.json'),raw_GT_read=False))


def evaluate(c):
    assert read(OUT/'inference_verified.json')['predictions_seal_sha256']==sha(OUT/'predictions_frozen.json')
    guard=cross.source.ReadGuard(c,'evaluate');sys.addaudithook(guard.hook)
    ctl.base.verify_files(c['code']);ctl.base.verify_files(c['label_inputs']);ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs'])
    ev=ctl.base.evaluator();ca,valid,uid=ctl.class_agnostic_evaluator();receipts={}
    for j,s in enumerate(c['scenes']):
        stage('matching',scene=s,completed=j,total=200);d=OUT/'evaluation'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():r=read(d/'receipt.json');ctl.base.verify_files(r['outputs'])
        else:
            b=c['bindings'][s];assert sha(b['gt_path'])==b['gt_sha256'];gt=np.loadtxt(b['gt_path'],dtype=np.int64)
            _,points=ctl.shared.load_points(b);sizes={k:len(p) for k,p in points.items()}
            sr={a:read(OUT/'predictions'/s/f'semantic_{a}.json') for a in ARMS}
            refs={a:read((R if a=='R' else S)/'predictions'/s/f'semantic_{a}.json') for a in ('S','C','R')}
            rows={**sr,**refs};universe=list({cross.key(m):m for values in rows.values() for m in values}.values())
            template=ctl.shared.assign(ev,universe,points,b['gt_path'],len(gt))
            semantic={a:ctl.project_multiclass(template,universe,rr,sizes,ev.CLASS_LABELS,ev.VALID_CLASS_IDS) for a,rr in rows.items()}
            edges=sum(ctl.shared.check_match(semantic[a],rows[a],points,gt,ev) for a in rows)
            geom={a:ctl.collapse_members(rr) for a,rr in rows.items()};mapped=ctl.remap_gt(gt,valid,uid)
            def loader(path):
                assert Path(path).resolve()==Path(b['gt_path']).resolve();return mapped.copy()
            ca.util_3d.load_ids=loader;cu=list({m['geometry_key']:m for vv in geom.values() for m in vv}.values())
            tm=ctl.shared.assign(ca,cu,points,b['gt_path'],len(gt),True);kk=[m['geometry_key'] for m in cu if sizes[m['geometry_key']]>=100]
            agnostic={a:ctl.project_matches(tm,kk,vv,ca.CLASS_LABELS[0]) for a,vv in geom.items()}
            edges+=sum(ctl.shared.check_match(agnostic[a],geom[a],points,mapped,ca,True) for a in rows)
            for a in ('S','C','R'):
                for view,matches in (('semantic',semantic),('class_agnostic',agnostic)):
                    original=read((R if a=='R' else S)/'evaluation'/s/f'{view}_{a}_matches.json')
                    assert ctl.shared.canonical_match(matches[a])==ctl.shared.canonical_match(original)
            assert ctl.shared.canonical_match(agnostic['P0'])==ctl.shared.canonical_match(read(cross.OUT/'evaluation'/s/'class_agnostic_SR_matches.json'))
            local={}
            for view,matches in (('semantic',semantic),('class_agnostic',agnostic)):
                for a in ARMS:
                    q=d/f'{view}_{a}_matches.json';freeze_json(q,ctl.base.serializable_matches(matches[a]));local[str(q)]=sha(q)
            r=dict(complete=True,outputs=local,GT_sha256=b['gt_sha256'],independent_intersection_edges=edges,
                   controls_matches_exact=True,P0_CA_SR_exact=True);freeze_json(d/'receipt.json',r)
        receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    freeze_json(OUT/'matches_frozen.json',dict(complete=True,receipts=receipts,GT_scans=len(guard.opened_gt),predictions_before_GT=True))
    ap={g:{v:{} for v in VIEWS} for g in c['reporting_groups']};arrays={}
    for view in VIEWS:
        evaluator=ev if view=='semantic' else ca
        for a in ARMS:
            matches={s:read(OUT/'evaluation'/s/f'{view}_{a}_matches.json') for s in c['scenes']}
            for group,ss in c['reporting_groups'].items():
                stage('AP',view=view,arm=a,group=group);mm={s:matches[s] for s in ss};x,*_=evaluator.evaluate_matches(mm)
                ap[group][view][a]=ctl.base.metrics(evaluator,x)
                q=OUT/'evaluation'/f'{group}_{view}_{a}_AP.npy';np.save(q,x,allow_pickle=False);arrays[str(q)]=sha(q)
                if a=='P0' and view=='class_agnostic':assert np.array_equal(x,np.load(cross.OUT/'evaluation'/f'{group}_class_agnostic_SR_AP.npy'),equal_nan=True)
                if group=='development200':
                    repeat,*_=evaluator.evaluate_matches(dict(reversed(list(mm.items()))));assert np.allclose(x,repeat,rtol=0,atol=1e-12,equal_nan=True)
                print('result',group,view,a,ap[group][view][a],flush=True)
    # All S/R group metrics already frozen by cross; C main groups already frozen by S.
    old=read(cross.OUT/'evaluation_summary.json')['AP'];base=read(S/'evaluation_summary.json')['AP']
    for group in ap:
        for view in VIEWS:
            ap[group][view].update(S=old[group][view]['SS'],R=old[group][view]['RR'],SR=old[group][view]['SR'])
            if group in base:ap[group][view]['C']=base[group][view]['C']
    freeze_json(OUT/'evaluation_summary.json',dict(complete=True,AP=ap,AP_arrays=arrays,P0_all8_CA_arrays_SR_exact=True,full200_reverse_AP_replayed=6))


def diagnostics(c):
    assert read(OUT/'evaluation_summary.json')['complete'];ev=ctl.base.evaluator();ts=list(map(float,ev.opt['overlaps']));t75=min(ts,key=lambda t:abs(t-.75))
    details={};totals={};rec_total=0
    selected_class_scores={a:{} for a in ARMS};full_class_scores={}
    for j,s in enumerate(c['scenes']):
        stage('diagnostics',scene=s,completed=j,total=200)
        census=read(c['jobs'][s]['target_census']);ov={k:{int(g):v for g,v in v.items()} for k,v in census['ious'].items()};ids=list(map(int,census['targets']))
        rows={a:read(OUT/'predictions'/s/f'semantic_{a}.json') for a in ARMS}
        rows.update({a:read((R if a=='R' else S)/'predictions'/s/f'semantic_{a}.json') for a in ('S','C','R')})
        sizes={g['geometry_key']:g['point_count'] for g in read(c['jobs'][s]['nt_geometry.json'])}
        recovered=read(ROOT/'.local/fi1_s_c_selection_scope_development200_20261010_v1/scenes'/s/'members.json')['recovered_targets'];recids={int(r['gt_id']) for r in recovered};rec_total+=len(recids)
        pool,old=cross.pool_scores(s)
        with np.load(OUT/'predictions'/s/'scores.npz',allow_pickle=False) as z:full=z['full']
        for i,m in enumerate(pool):full_class_scores.setdefault(m['class_index'],[]).append(float(full[i]))
        for a in ARMS:
            for m in rows[a]:selected_class_scores[a].setdefault(m['class_index'],[]).append(m['score'])
        ranking=cross.source.core.order(pool,full);rank={i:j+1 for j,i in enumerate(ranking)}
        data={}
        def hits(a,t,semantic,positive=False):
            return {g for m in rows[a] for g,v in ov[m['geometry_key']].items() if v>t and (not semantic or g//1000==c['class_ids'][m['class_index']]) and (not positive or m['score']>0)}
        for a in ARMS:
            cover={}
            for t in ts:
                actual=hits(a,t,True);positive=hits(a,t,True,True);ca=hits(a,t,False);pc=hits(a,t,False,True)
                refs={}
                for b in ('S','C','R'):
                    ref=hits(b,t,True);cr=hits(b,t,False)
                    refs[b]=dict(semantic_gained=sorted(actual-ref),semantic_lost=sorted(ref-actual),CA_gained=sorted(ca-cr),CA_lost=sorted(cr-ca))
                cover[f'{t:.2f}']=dict(semantic=len(actual),positive=len(positive),CA=len(ca),positive_CA=len(pc),vs=refs)
            match=read(OUT/'evaluation'/s/f'semantic_{a}_matches.json');status=cross.source.previous.member_status(rows[a],match,sizes,c['class_ids'])
            rec=[]
            for g in recids:
                correct=[m for m in rows[a] if c['class_ids'][m['class_index']]==g//1000 and ov[m['geometry_key']][g]>t75]
                eligible=[i for i,m in enumerate(pool) if c['class_ids'][m['class_index']]==g//1000 and ov[m['geometry_key']][g]>t75]
                good=min(eligible,key=lambda i:rank[i]);m=pool[good]
                # Within-class global rank computed later across all scenes.
                rec.append(dict(gt_id=g,member_present=bool(correct),positive_present=any(m['score']>0 for m in correct),
                    best_scene_full_rank=rank[good],best_full_geometry=m['geometry_key'],class_index=m['class_index'],best_full_score=float(full[good]),
                    best_selected_score=max((m['score'] for m in correct),default=None)))
            before={cross.key(m) for m in rows['S']};after={cross.key(m) for m in rows[a]}
            data[a]=dict(coverage=cover,status=dict(Counter(status.values())),recovered424=rec,
                entered=len(after-before),exited=len(before-after),zero_score_members=sum(m['score']==0 for m in rows[a]))
        details[s]=data
    assert rec_total==424
    full_class_scores={k:np.sort(v) for k,v in full_class_scores.items()}
    selected_class_scores={a:{k:np.sort(v) for k,v in vv.items()} for a,vv in selected_class_scores.items()}
    def rank_interval(values,score):
        return [int(len(values)-np.searchsorted(values,score,side='right')+1),int(len(values)-np.searchsorted(values,score,side='left'))]
    for dd in details.values():
        for a,d in dd.items():
            for r in d['recovered424']:
                ci=r['class_index'];r['global_class_full_rank_interval']=rank_interval(full_class_scores[ci],r['best_full_score'])
                r['global_class_selected_rank_interval']=None if r['best_selected_score'] is None else rank_interval(selected_class_scores[a][ci],r['best_selected_score'])
    for group,ss in c['reporting_groups'].items():
        totals[group]={}
        for a in ARMS:
            stat=Counter();cover={f'{t:.2f}':{b:Counter() for b in ('S','C','R')} for t in ts};absolute={f'{t:.2f}':Counter() for t in ts}
            for s in ss:
                d=details[s][a];stat.update(d['status']);stat.update(entered=d['entered'],exited=d['exited'],zero_score_members=d['zero_score_members'],
                    recovered_member_lost=sum(not r['member_present'] for r in d['recovered424']),recovered_positive_lost=sum(not r['positive_present'] for r in d['recovered424']))
                for t,v in d['coverage'].items():
                    absolute[t].update({k:v[k] for k in ('semantic','positive','CA','positive_CA')})
                    for b,r in v['vs'].items():cover[t][b].update({k:len(v) for k,v in r.items()})
            totals[group][a]=dict(counts=dict(stat),absolute_coverage={t:dict(v) for t,v in absolute.items()},coverage={t:{b:dict(v) for b,v in refs.items()} for t,refs in cover.items()})
    freeze_json(OUT/'diagnostics.json',dict(complete=True,scenes=details,groups=totals,recovered424=rec_total))


def report(c):
    results=read(OUT/'evaluation_summary.json');d=read(OUT/'diagnostics.json');ap=results['AP'];main=ap['development200'];stats=d['groups']['development200']
    p0=main['semantic']['P0']['ap']>=main['semantic']['S']['ap'] and main['semantic']['P0']['ap']>main['semantic']['SR']['ap']
    p1=main['semantic']['P1']['ap']>=main['semantic']['S']['ap'] and main['class_agnostic']['P1']['ap']>main['class_agnostic']['C']['ap']
    p2=all(main[v]['P2']['ap']>main[v][a]['ap'] for v in VIEWS for a in ('S','C','R'))
    p2=p2 and stats['P2']['counts']['recovered_member_lost']==0 and stats['P2']['counts']['recovered_positive_lost']==0
    p2=p2 and stats['P2']['coverage']['0.75']['S']['semantic_lost']==0 and stats['P2']['coverage']['0.75']['S']['CA_lost']==0
    if not p0:meaning='P0未恢复S类别AP，R几何最高分与S类别比例仍不能同时保留两项收益；不直接训练模仿R最高分的倍率头。'
    elif not p1:meaning='P0支持而P1不支持；全池最大值接口引入损失，先处理未选类别抬升几何最高分。'
    elif not p2:meaning='P0/P1支持但P2未验收；真实预算竞争或严格指标／覆盖门未通过，不部署或默认追加几何。'
    else:meaning='P2通过预注册双口径及覆盖门，可登记开发候选；无需新增倍率头，生产仍保持冻结。'
    lines=['# R几何最高分 × S类别比例：P0/P1/P2','',meaning,'',
        '固定NT点集、S资格、原五折和K800，不训练、不模型推理、不截断倍率。pS为相对最高分，不是归一化概率。P0选中成员范围取最高分、固定S800；P1全池最高分、固定S800；P2同P1全池评分、重选800。',
        '', '|臂|类别AP (%)|实际去类AP (%)|424成员损失|424正分损失（含缺失）|','|---|---:|---:|---:|---:|']
    for a in ('S','C','R','SR')+ARMS:
        counts=stats[a]['counts'] if a in ARMS else {}
        lines.append(f"|{a}|{100*main['semantic'][a]['ap']:.6f}|{100*main['class_agnostic'][a]['ap']:.6f}|{counts.get('recovered_member_lost','参考')}|{counts.get('recovered_positive_lost','参考')}|")
    lines+=['', '|分组|臂|类别AP (%)|实际去类AP (%)|','|---|---|---:|---:|']
    for group in c['reporting_groups']:
        if group=='development200':continue
        for a in ARMS:lines.append(f"|{group}|{a}|{100*ap[group]['semantic'][a]['ap']:.6f}|{100*ap[group]['class_agnostic'][a]['ap']:.6f}|")
    lines+=['',f'结构／接口／真实预算门：P0={p0}，P1={p1}，P2={p2}。三臂全部按冻结合同执行，不依P0结果改后两臂。',
        '', 'P0去类几何分数与SR逐项精确一致，8个去类AP数组精确复现；339零最高分如实保留。P1全池接口影响942个S已选几何；14150几何全池S/R最大值均正，没有使用fallback。全部48个新AP数组聚合，6个全200新数组逆场景重算核验；原S/C/R匹配记录精确复现。',
        '', 'GT使用：所有新预测及独立公式、排序、去类核验先封存；P2新增成员随后读取白名单开发200已登记GT进行原匹配，逐点交集／忽略核对。没有训练或其他留出集；开发OOF不是整系统盲测。',
        '', '逐档全部目标新增／损失、FP/ignore、小掩码、424见证及P2最佳全池场景排名在diagnostics.json；零分仍参与原AP，正分仅辅助诊断。比例保留不保证同类别跨几何／跨场景排序无损。',
        '', '本轮结论直接用于是否保留该结构，不再扫融合权重、倍率上下限或预算。不加入U/U′，不启动新头训练或替换生产Full。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    freeze_json(OUT/'decision.json',dict(complete=True,P0_supported=p0,P1_supported=p1,P2_accepted=p2,interpretation=meaning,
        next_head_training=False,U_Uprime_added=False,production_modified=False))
    files=[OUT/n for n in ('contract.json','tests_verified.json','predictions_frozen.json','inference_verified.json','matches_frozen.json','evaluation_summary.json','diagnostics.json','decision.json','report.md')]
    freeze_json(OUT/'completion_verified.json',dict(complete=True,outputs={str(p):sha(p) for p in files},new_training=False,new_model_inference=False,production_modified=False))
    write_json(OUT/'execution.json',dict(status='complete',stage='complete',resume_required=False,P2_accepted=p2))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','verify','evaluate','diagnostics','report']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:globals()[a.stage](read(OUT/'contract.json'))


if __name__=='__main__':main()
