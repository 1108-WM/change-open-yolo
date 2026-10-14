#!/usr/bin/env python3
"""Frozen member-set x score factorial; no training/model inference/raw GT."""
from pathlib import Path
from collections import Counter
import argparse
import json
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_s_member_quality_competition_development200 as source
from tools.fi1_sr_cross_control import fixed_rescore,independent_rescore_match,key
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha,write_json

OUT=ROOT/'.local/fi1_sr_cross_control_development200_20261010_v1'
S=source.S;R=source.OUT;AUDIT=ROOT/'.local/fi1_rqf_independent_code_audit_20261010_v1'
ARMS=('SS','SR','RS','RR');VIEWS=source.VIEWS
read=source.read;ctl=source.ctl


def stage(name,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=name,updated=time.time(),**kw))
    print(name,kw,flush=True)


class Guard:
    def __init__(self,mode):self.mode=mode
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=Path(args[0]).resolve();parts=p.parts;st=str(p)
        if any(x.startswith('ground_truth') for x in parts):raise PermissionError('raw GT prohibited throughout')
        if any(x in st for x in ('/Dataset/scannet200/','/scannetv2_test/','/scannet200_test/','/paper_real_figure_assets_','/paper_furniture_focused_')):
            raise PermissionError('out-of-scope data prohibited')
        if '/.local/' in st and 'val312' in st and '/code_snapshot/' not in st:raise PermissionError('validation outputs prohibited')
        if self.mode in ('predict','verify_predictions') and ('/evaluation/' in st or '/labels/' in st or p.name in ('labels.npz','target_census.json')):
            raise PermissionError('cached GT/matches prohibited until prediction seal')
        if p.suffix=='.joblib':raise PermissionError('no model loading in fixed-score replay')


def prepare():
    assert not OUT.exists()
    assert shutil.disk_usage(ROOT).free>=50*1024**3
    c=read(R/'contract.json');p=read(S/'contract.json')
    for root,name in ((R,'final_verified.json'),(S,'completion_verified.json'),(AUDIT,'verified.json')):
        assert read(root/name)['complete']
        receipt=read(root/name)
        hashes=receipt.get('outputs',receipt.get('artifacts',{}))
        assert hashes
        ctl.base.verify_files({str(root/q) if not Path(q).is_absolute() else q:h for q,h in hashes.items()})
    groups=dict(c['reporting_groups'])
    for f in range(5):groups[f'fold{f}']=[s for s in c['scenes'] if c['folds'][s]==f]
    assert all(len(groups[f'fold{f}'])==40 for f in range(5))
    inputs={str(root/n):sha(root/n) for root,n in ((S,'contract.json'),(S,'predictions_frozen.json'),(S,'completion_verified.json'),
        (R,'contract.json'),(R,'predictions_frozen.json'),(R,'final_verified.json'),(AUDIT,'report.md'),(AUDIT,'verified.json'))}
    labels={};ss=read(S/'predictions_frozen.json');rs=read(R/'predictions_frozen.json')
    risk_seal=read(source.previous.RISK/'labels_frozen.json');jobs={}
    for s in c['scenes']:
        jobs[s]=dict(c['jobs'][s])
        q=c['jobs'][s]['nt_geometry.json'];inputs[q]=c['inputs'][q]
        for root,seal,names in ((S,ss,('pool.npz','members.jsonl','semantic_S.json','class_agnostic_S.json')),
                                (R,rs,('scores.npz','semantic_R.json','class_agnostic_R.json'))):
            for n in names:
                q=root/'predictions'/s/n;inputs[str(q)]=seal['outputs'][str(q)]
            receipt=read(root/'evaluation'/s/'receipt.json')
            alias='S' if root==S else 'R'
            for view in VIEWS:
                q=root/'evaluation'/s/f'{view}_{alias}_matches.json';labels[str(q)]=receipt['outputs'][str(q)]
        q=jobs[s]['target_census'];labels[q]=risk_seal['outputs'][q]
        for name in ('diagnostics/summary.json',):
            q=S/name;labels[str(q)]=sha(q)
    for root,alias in ((S,'S'),(R,'R')):
        for group in c['reporting_groups']:
            for view in VIEWS:
                q=root/'evaluation'/f'{group}_{view}_{alias}_AP.npy';labels[str(q)]=sha(q)
    code=dict(c['code'])
    for n in ('tools/run_fi1_sr_cross_control_development200.py','tools/fi1_sr_cross_control.py',
              'tests/test_fi1_sr_cross_control.py','tools/run_fi1_sr_cross_control_development200.sh'):
        q=ROOT/n;code[str(q)]=sha(q)
    contract=dict(version=OUT.name,scenes=c['scenes'],folds=c['folds'],reporting_groups=groups,
                  jobs=jobs,class_ids=c['class_ids'],inputs=inputs,label_inputs=labels,code=code,
                  arms=ARMS,budget=800,views=VIEWS,
                  definitions={a:dict(member_set=a[0],score=a[1]) for a in ARMS},
                  policy='Exactly the source arm800 member identities; swap full-pool frozen S/R score, reorder within fixed set only, retain zero scores and <100 masks; no refill or clipping change.',
                  projection='Exact pointset max score of selected members, recompute winning class, no geometric refill; source geometry identity fixed.',
                  evaluation='Reuse sealed source matching including void/small GT; score-independent intersections unchanged; independently rebuild both match directions; original evaluator AP.',
                  GT_policy='No raw GT ever; predictions and independent set/score/collapse checks sealed before reading cached matches/census.',
                  diagnostics='All original strict IoU thresholds, positive-score coverage separate from mere member presence, recovered424, score-zero target witnesses, class AP changes, same-geometry class winners, member entries/exits and global FP pressure.',
                  attribution='Report within-set score effects and within-score set effects plus interaction; two paths sum arithmetically but are order-dependent, not independent additive causal AP contributions.',
                  decision='Diagnose whether R score benefit persists on S set and whether SR preserves semantic AP; assess need for preserving intra-geometry class order. No automatic geometry-calibrator training.',
                  failure='No parameter/weight/threshold/K search. Inconsistent identity/control AP aborts; retain old outputs.',
                  new_training=False,new_model_inference=False,new_geometry=False,U_Uprime_output=False,production_modified=False,
                  scope='Exposed development200, fixed prior five physical folds; mechanism diagnosis, not deployable or end-to-end blind acceptance.')
    ctl.base.verify_files(inputs);ctl.base.verify_files(code)
    OUT.mkdir(parents=True);freeze_json(OUT/'contract.json',contract)
    for q in code:
        dest=OUT/'code_snapshot'/Path(q).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dest)
    run=subprocess.run([sys.executable,'-m','pytest','tests/test_fi1_sr_cross_control.py','tests/test_fi1_class_agnostic_ablation.py','-q'],capture_output=True,text=True)
    (OUT/'tests.log').write_text(run.stdout+run.stderr);assert run.returncode==0,run.stdout+run.stderr
    freeze_json(OUT/'tests_verified.json',dict(passed=True,log_sha256=sha(OUT/'tests.log')))
    stage('prepared')


def pool_scores(s):
    members=ctl.base.rows(S/'predictions'/s/'members.jsonl')
    with np.load(S/'predictions'/s/'pool.npz',allow_pickle=False) as z:ss=z['scores']
    with np.load(R/'predictions'/s/'scores.npz',allow_pickle=False) as z:rs=z['R']
    assert ss.shape==rs.shape==(len(members),)
    return members,dict(S=ss,R=rs)


def predict(c):
    sys.addaudithook(Guard('predict').hook);ctl.base.verify_files(c['inputs']);ctl.base.verify_files(c['code'])
    outputs={};receipts={}
    for i,s in enumerate(c['scenes']):
        stage('predictions',scene=s,completed=i,total=200)
        d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():
            r=read(d/'receipt.json');ctl.base.verify_files(r['outputs'])
        else:
            pool,scores=pool_scores(s)
            sets=dict(S=read(S/'predictions'/s/'semantic_S.json'),R=read(R/'predictions'/s/'semantic_R.json'))
            assert all(len(rows)==800 for rows in sets.values())
            local={}
            for arm in ARMS:
                rows=fixed_rescore(sets[arm[0]],pool,scores[arm[1]])
                assert {key(m) for m in rows}=={key(m) for m in sets[arm[0]]}
                ca=ctl.collapse_members(rows)
                if arm in ('SS','RR'):
                    root=S if arm=='SS' else R;alias=arm[0]
                    assert rows==sets[alias]
                    assert ca==read(root/'predictions'/s/f'class_agnostic_{alias}.json')
                for view,values in zip(VIEWS,(rows,ca)):
                    q=d/f'{view}_{arm}.json';freeze_json(q,values);local[str(q)]=sha(q)
            r=dict(complete=True,outputs=local,raw_GT_read=False,fixed_members=800,SS_RR_exact=True)
            freeze_json(d/'receipt.json',r)
        outputs.update(r['outputs']);receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    freeze_json(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,receipts=receipts,raw_GT_read=False,new_model_inference=False))


def verify_predictions(c):
    sys.addaudithook(Guard('verify_predictions').hook)
    seal=read(OUT/'predictions_frozen.json');ctl.base.verify_files(seal['outputs']);ctl.base.verify_files(c['inputs'])
    counters=Counter()
    for s in c['scenes']:
        pool,scores=pool_scores(s);ix={key(m):i for i,m in enumerate(pool)}
        originals=dict(S=read(S/'predictions'/s/'semantic_S.json'),R=read(R/'predictions'/s/'semantic_R.json'))
        for arm in ARMS:
            rows=[{**m,'score':float(scores[arm[1]][ix[key(m)]])} for m in originals[arm[0]]]
            rows.sort(key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))
            assert rows==read(OUT/'predictions'/s/f'semantic_{arm}.json')
            best={}
            for m in sorted(rows,key=lambda m:(-m['score'],m['class_index'])):
                best.setdefault(m['geometry_key'],dict(geometry_key=m['geometry_key'],score=m['score'],winning_class_index=m['class_index']))
            collapsed=sorted(best.values(),key=lambda m:(-m['score'],m['geometry_key']))
            assert collapsed==read(OUT/'predictions'/s/f'class_agnostic_{arm}.json')
            assert len(rows)==800
            counters.update(members=len(rows),zero_score_members=sum(m['score']==0 for m in rows),geometries=len(collapsed))
    freeze_json(OUT/'inference_verified.json',dict(complete=True,independent_set_score_rank_collapse=True,counts=dict(counters),
                predictions_seal_sha256=sha(OUT/'predictions_frozen.json'),raw_GT_read=False))


def evaluate(c):
    assert read(OUT/'inference_verified.json')['predictions_seal_sha256']==sha(OUT/'predictions_frozen.json')
    sys.addaudithook(Guard('evaluate').hook);ctl.base.verify_files(c['label_inputs']);ctl.base.verify_files(c['code'])
    ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs'])
    ev=ctl.base.evaluator();ca,_,_=ctl.class_agnostic_evaluator();receipts={}
    for i,s in enumerate(c['scenes']):
        stage('cached_matching',scene=s,completed=i,total=200)
        d=OUT/'evaluation'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():
            r=read(d/'receipt.json');ctl.base.verify_files(r['outputs'])
        else:
            geometry=read(c['jobs'][s]['nt_geometry.json']);sizes={g['geometry_key']:g['point_count'] for g in geometry}
            local={}
            for arm in ARMS:
                root=S if arm[0]=='S' else R;alias=arm[0]
                for view in VIEWS:
                    before=read(root/'predictions'/s/f'{view}_{alias}.json')
                    after=read(OUT/'predictions'/s/f'{view}_{arm}.json')
                    template=read(root/'evaluation'/s/f'{view}_{alias}_matches.json')
                    if view=='semantic':
                        projected=ctl.project_multiclass(template,before,after,sizes,ev.CLASS_LABELS,ev.VALID_CLASS_IDS)
                    else:
                        keys=[m['geometry_key'] for m in before if sizes[m['geometry_key']]>=100]
                        projected=ctl.project_matches(template,keys,after,ca.CLASS_LABELS[0])
                    independent=independent_rescore_match(template,before,after,sizes,view=='class_agnostic')
                    assert projected==independent,(s,arm,view)
                    if arm in ('SS','RR'):assert projected==template
                    q=d/f'{view}_{arm}_matches.json';freeze_json(q,projected);local[str(q)]=sha(q)
            r=dict(complete=True,outputs=local,independent_bidirectional_projection_exact=True,raw_GT_read=False)
            freeze_json(d/'receipt.json',r)
        receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    freeze_json(OUT/'matches_frozen.json',dict(complete=True,receipts=receipts,raw_GT_read=False,new_intersection_matching=False))
    ap={g:{v:{} for v in VIEWS} for g in c['reporting_groups']};arrays={}
    for view in VIEWS:
        evaluator=ev if view=='semantic' else ca
        for arm in ARMS:
            matches={s:read(OUT/'evaluation'/s/f'{view}_{arm}_matches.json') for s in c['scenes']}
            for group,scenes in c['reporting_groups'].items():
                stage('AP',view=view,arm=arm,group=group)
                subset={s:matches[s] for s in scenes}
                arr,*_=evaluator.evaluate_matches(subset)
                ap[group][view][arm]=ctl.base.metrics(evaluator,arr)
                q=OUT/'evaluation'/f'{group}_{view}_{arm}_AP.npy';np.save(q,arr,allow_pickle=False);arrays[str(q)]=sha(q)
                if group=='development200' and arm in ('SR','RS'):
                    repeat,*_=evaluator.evaluate_matches(dict(reversed(list(subset.items()))))
                    assert np.allclose(arr,repeat,rtol=0,atol=1e-12,equal_nan=True)
                if group in ('development200','historical90','new110') and arm in ('SS','RR'):
                    root=S if arm=='SS' else R;alias=arm[0]
                    assert np.array_equal(arr,np.load(root/'evaluation'/f'{group}_{view}_{alias}_AP.npy'),equal_nan=True)
                print('result',group,view,arm,ap[group][view][arm],flush=True)
            del matches
    freeze_json(OUT/'evaluation_summary.json',dict(complete=True,AP=ap,AP_arrays=arrays,baseline_AP_exact=True,
                mixed_full200_reverse_replayed=True,raw_GT_read=False))


def diagnostics(c):
    sys.addaudithook(Guard('diagnostics').hook)
    assert read(OUT/'evaluation_summary.json')['complete']
    ev=ctl.base.evaluator();ts=list(map(float,ev.opt['overlaps']));t75=min(ts,key=lambda t:abs(t-.75))
    details={};group_counts={};pressure_data={a:{t:dict(front={},fp={}) for t in ts} for a in ARMS}
    recovered_total=0
    for i,s in enumerate(c['scenes']):
        stage('diagnostics',scene=s,completed=i,total=200)
        pool,scores=pool_scores(s);ix={key(m):j for j,m in enumerate(pool)}
        census=read(c['jobs'][s]['target_census']);ov={k:{int(g):v for g,v in v.items()} for k,v in census['ious'].items()}
        sizes={g['geometry_key']:g['point_count'] for g in read(c['jobs'][s]['nt_geometry.json'])}
        rows={a:read(OUT/'predictions'/s/f'semantic_{a}.json') for a in ARMS}
        ca={a:read(OUT/'predictions'/s/f'class_agnostic_{a}.json') for a in ARMS}
        sets={a:{key(m) for m in v} for a,v in rows.items()}
        recovered=read(ROOT/'.local/fi1_s_c_selection_scope_development200_20261010_v1/scenes'/s/'members.json')['recovered_targets']
        recids={int(r['gt_id']) for r in recovered};recovered_total+=len(recids)
        data={}
        def supporters(a,g,t,semantic=True,positive=False):
            return [m for m in rows[a] if ov[m['geometry_key']][g]>t and
                    (not semantic or c['class_ids'][m['class_index']]==g//1000) and (not positive or m['score']>0)]
        for a in ARMS:
            match=read(OUT/'evaluation'/s/f'semantic_{a}_matches.json')
            status=source.previous.member_status(rows[a],match,sizes,c['class_ids'])
            cmatch=read(OUT/'evaluation'/s/f'class_agnostic_{a}_matches.json')
            cover={}
            for t in ts:
                present={g for g in map(int,census['targets']) if supporters(a,g,t)}
                positive={g for g in present if supporters(a,g,t,positive=True)}
                geom={g for g in map(int,census['targets']) if supporters(a,g,t,False)}
                cover[f'{t:.2f}']=dict(semantic_present=len(present),semantic_positive=len(positive),ca_present=len(geom))
                cp=[]
                for p in next(iter(cmatch['pred'].values())):
                    hits=[g for g in p['matched_gt'] if g['instance_id']>=1000 and g['vert_count']>=100 and g['intersection']/(g['vert_count']+p['vert_count']-g['intersection'])>t]
                    ignored=p['void_intersection']+sum(g['intersection'] for g in p['matched_gt'] if g['instance_id']<1000 or g['vert_count']<100)
                    if not hits and ignored/p['vert_count']<=t:cp.append(p['confidence'])
                pressure_data[a][t]['fp'][s]=cp
                pressure_data[a][t]['front'].update({(s,g):max(m['score'] for m in ca[a] if ov[m['geometry_key']][g]>t) for g in geom})
            rec={}
            for g in recids:
                selected=supporters(a,g,t75)
                rec[str(g)]=dict(member_present=bool(selected),positive_score_present=any(m['score']>0 for m in selected),
                                 best_selected_score=max((m['score'] for m in selected),default=None),
                                 selected_members=[dict(geometry_key=m['geometry_key'],class_index=m['class_index'],score=m['score']) for m in selected])
            data[a]=dict(status=dict(Counter(status.values())),coverage=cover,
                         zero_score_members=sum(m['score']==0 for m in rows[a]),
                         zero_score_valid_matches=sum(m['score']==0 and status[key(m)]=='valid_sameclass_match' for m in rows[a]),
                         recovered424=rec)
        comparisons={}
        for left,right in (('SS','SR'),('SS','RS'),('SR','RR'),('RS','RR'),('SS','RR')):
            before={m['geometry_key']:m for m in ca[left]};after={m['geometry_key']:m for m in ca[right]};common=before.keys()&after.keys()
            comparisons[left+'→'+right]=dict(member_entered=len(sets[right]-sets[left]),member_exited=len(sets[left]-sets[right]),
                geometry_entered=len(after.keys()-before.keys()),geometry_exited=len(before.keys()-after.keys()),
                shared_geometry_winning_class_changed=sum(before[g]['winning_class_index']!=after[g]['winning_class_index'] for g in common),
                shared_geometry_score_up=sum(after[g]['score']>before[g]['score'] for g in common),
                shared_geometry_score_down=sum(after[g]['score']<before[g]['score'] for g in common))
        lost_targets=[]
        for g in map(int,census['targets']):
            if supporters('SS',g,t75) and (not supporters('RR',g,t75) or not supporters('RR',g,t75,positive=True)):
                lost_targets.append(dict(gt_id=g,arms={a:dict(member_present=bool(supporters(a,g,t75)),
                    positive_score_present=bool(supporters(a,g,t75,positive=True)),
                    best_score=max((m['score'] for m in supporters(a,g,t75)),default=None)) for a in ARMS}))
        details[s]=dict(arms=data,comparisons=comparisons,S_positive_loss_witnesses=lost_targets)
    assert recovered_total==424
    from bisect import bisect_left
    global_pressure={}
    for group,scenes in c['reporting_groups'].items():
        scene_set=set(scenes);stats={a:Counter() for a in ARMS};comp=default_counter={pair:Counter() for pair in next(iter(details.values()))['comparisons']}
        cover={a:{f'{t:.2f}':Counter() for t in ts} for a in ARMS}
        for s in scenes:
            for a in ARMS:
                d=details[s]['arms'][a];stats[a].update(d['status']);stats[a].update(zero_score_members=d['zero_score_members'],zero_score_valid_matches=d['zero_score_valid_matches'])
                stats[a].update(recovered424_member_lost=sum(not r['member_present'] for r in d['recovered424'].values()),
                                recovered424_positive_lost=sum(not r['positive_score_present'] for r in d['recovered424'].values()))
                for t,v in d['coverage'].items():cover[a][t].update(v)
            for pair,values in details[s]['comparisons'].items():comp[pair].update(values)
        group_counts[group]=dict(arms={a:dict(v) for a,v in stats.items()},coverage={a:{t:dict(v) for t,v in values.items()} for a,values in cover.items()},comparisons={p:dict(v) for p,v in comp.items()})
        global_pressure[group]={}
        for t in ts:
            table={};prefix={}
            for a in ARMS:
                fps=sorted(v for s in scenes for v in pressure_data[a][t]['fp'][s]);front={k:v for k,v in pressure_data[a][t]['front'].items() if k[0] in scene_set}
                prefix[a]={k:len(fps)-bisect_left(fps,v) for k,v in front.items()}
                table[a]=dict(FP=len(fps),FP_median_score=float(np.median(fps)) if fps else None,covered_targets=len(front))
            for a in ARMS:
                common=prefix[a].keys()&prefix['SS'].keys()
                table[a]['vs_SS']=dict(common_targets=len(common),more_FP_ahead=sum(prefix[a][k]>prefix['SS'][k] for k in common),
                                      fewer_FP_ahead=sum(prefix[a][k]<prefix['SS'][k] for k in common))
            global_pressure[group][f'{t:.2f}']=table
    freeze_json(OUT/'diagnostics.json',dict(complete=True,scenes=details,groups=group_counts,global_CA_rank_pressure=global_pressure,
                recovered424_total=recovered_total,raw_GT_read=False,positive_score_distinguished=True))
    class_changes={}
    threshold_index=np.flatnonzero(np.asarray(ev.opt['overlaps'])!=.25)
    for group in c['reporting_groups']:
        arrays={a:np.load(OUT/'evaluation'/f'{group}_semantic_{a}_AP.npy') for a in ARMS}
        # Original macro-mean across valid class/IoU cells; missing GT remains NaN.
        own={a:np.nanmean(v[:,:,threshold_index],axis=(0,2)) for a,v in arrays.items()}
        valid=np.isfinite(own['SS']);count=int(valid.sum());records=[]
        for j in np.flatnonzero(valid):
            values={a:float(own[a][j]) for a in ARMS}
            records.append(dict(class_index=int(j),class_id=c['class_ids'][j],label=ev.CLASS_LABELS[j],AP=values,
                macro_contribution_pp={a:100*(values[a]-values['SS'])/count for a in ARMS}))
        class_changes[group]=dict(valid_classes=count,classes=records)
    freeze_json(OUT/'class_AP_changes.json',dict(complete=True,groups=class_changes))


def report(c):
    result=read(OUT/'evaluation_summary.json');diagnostic=read(OUT/'diagnostics.json');ap=result['AP'];main=ap['development200'];counts=diagnostic['groups']['development200']
    effects={}
    for group,values in ap.items():
        effects[group]={}
        for view,arms in values.items():
            x={a:arms[a]['ap'] for a in ARMS}
            effects[group][view]=dict(score_on_S_pp=100*(x['SR']-x['SS']),score_on_R_pp=100*(x['RR']-x['RS']),
                set_at_S_pp=100*(x['RS']-x['SS']),set_at_R_pp=100*(x['RR']-x['SR']),
                interaction_pp=100*(x['RR']-x['RS']-x['SR']+x['SS']),total_R_minus_S_pp=100*(x['RR']-x['SS']))
    signal=dict(R_CA_gain_survives_S_set=main['class_agnostic']['SR']['ap']>main['class_agnostic']['SS']['ap'],
                SR_semantic_not_lower_than_SS=main['semantic']['SR']['ap']>=main['semantic']['SS']['ap'],
                SR_recovered424_positive_losses=counts['arms']['SR']['recovered424_positive_lost'])
    lines=['# S/R 固定成员集合 × 分数交叉对照','',
           '开发200，固定N/T点集、S资格与K800。每组保留指定源方案的800成员身份，只交换封存全池分数、在集合内排序，不补位、不删零分、不重训；实际去类取已选成员同点集最高分。',
           '', '|成员集合|分数|类别相关 AP (%)|实际输出去类 AP (%)|424成员缺失|424正分缺失|','|---|---|---:|---:|---:|---:|']
    for a in ARMS:
        v=counts['arms'][a];lines.append(f"|{a[0]}|{a[1]}|{100*main['semantic'][a]['ap']:.6f}|{100*main['class_agnostic'][a]['ap']:.6f}|{v['recovered424_member_lost']}|{v['recovered424_positive_lost']}|")
    lines+=['','|固定条件下的改变|类别相关 ΔAP (pp)|去类 ΔAP (pp)|','|---|---:|---:|']
    for key,label in [('score_on_S_pp','S集合换R分'),('score_on_R_pp','R集合换R分'),('set_at_S_pp','S分换R集合'),('set_at_R_pp','R分换R集合'),('interaction_pp','交互项')]:
        lines.append(f"|{label}|{effects['development200']['semantic'][key]:+.6f}|{effects['development200']['class_agnostic'][key]:+.6f}|")
    lines+=['', '两条路径的差值可按代数相加得到RR−SS，但有交互，不能解释为互相独立、可相加的集合和评分收益。固定集合保留的零分预测仍参与原AP；“成员存在”“正分存在”分别报告，不能将零分简单当作已恢复有效排序。',
            '', '|分组|臂|类别相关 AP (%)|实际输出去类 AP (%)|','|---|---|---:|---:|']
    for group in c['reporting_groups']:
        if group=='development200':continue
        for a in ARMS:lines.append(f"|{group}|{a}|{100*ap[group]['semantic'][a]['ap']:.6f}|{100*ap[group]['class_agnostic'][a]['ap']:.6f}|")
    lines+=['',f'信号：R去类收益能在S集合上保留={signal["R_CA_gain_survives_S_set"]}；SR类别AP不低于SS={signal["SR_semantic_not_lower_than_SS"]}；SR的424正分缺失={signal["SR_recovered424_positive_losses"]}。',
            '', '这四组没有限制同几何类别次序，所以即使SR仍有去类收益，也不能直接证明正倍率几何校准 a(g)·sS(g,c) 会获益。下一训练需另行冻结监督、倍率与排序约束；不自动训练或选择融合权重。',
            '', '诊断：diagnostics.json包含各原IoU档、逐目标成员存在／零分、同几何赢家变化及全局去类FP排序；class_AP_changes.json包含逐类宏平均贡献；effects.json包含两个固定条件效应与交互。',
            '', '64个AP数组来自原evaluator；旧SS/RR三分组12个数组精确复现，混合全200四数组逆场景顺序复算一致。200场景双向缓存匹配独立重建一致，没有读取原始GT、训练、模型推理或新点集；五折是原头的OOF身份，不称整系统盲验收。生产Full和所有历史回执保持冻结。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    freeze_json(OUT/'effects.json',dict(complete=True,effects=effects,signals=signal,no_new_training=True))
    files=[OUT/n for n in ('contract.json','tests_verified.json','predictions_frozen.json','inference_verified.json','matches_frozen.json','evaluation_summary.json','diagnostics.json','class_AP_changes.json','effects.json','report.md')]
    freeze_json(OUT/'completion_verified.json',dict(complete=True,outputs={str(q):sha(q) for q in files},raw_GT_read=False,new_training=False,new_model_inference=False,production_modified=False))
    write_json(OUT/'execution.json',dict(status='complete',stage='complete',resume_required=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','verify_predictions','evaluate','diagnostics','report']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:globals()[a.stage](read(OUT/'contract.json'))


if __name__=='__main__':main()
