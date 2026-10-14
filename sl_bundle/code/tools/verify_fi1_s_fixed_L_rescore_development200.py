#!/usr/bin/env python3
"""Postseal fixed-set checks, all-target coverage and L reselection tradeoff."""
from pathlib import Path
from collections import Counter
import sys,warnings,bisect
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_s_fixed_L_rescore_development200 as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def main():
    out=run.OUT;c=run.read(out/'contract.json');completion=run.read(out/'completion_verified.json');assert completion['complete']
    sys.addaudithook(run.cross.Guard('postseal').hook)
    for hashes in (completion['outputs'],c['inputs'],c['label_inputs'],c['code'],run.read(out/'predictions_frozen.json')['outputs']):run.ctl.base.verify_files(hashes)
    ap=run.read(out/'evaluation_summary.json')['AP'];diag=run.read(out/'diagnostics.json');ev=run.ctl.base.evaluator();ts=np.asarray(ev.opt['overlaps']);t75=int(np.argmin(abs(ts-.75)))
    checks=Counter();effects={};coverage={};rec_stats={};allrows={};class_scores={};rankchecks=[]
    for s in c['scenes']:
        print('independent_fixed_SL_verify',s,flush=True)
        run.ctl.base.verify_files(run.read(out/'evaluation'/s/'receipt.json')['outputs'])
        pool,scores=run.load_pool(s);ix={run.key(m):i for i,m in enumerate(pool)}
        rows={a:run.read((run.S if a=='S' else run.L if a=='L' else out)/'predictions'/s/f'semantic_{a}.json') for a in ('S','L','SL')}
        assert len(rows['SL'])==800 and {run.key(m) for m in rows['S']}=={run.key(m) for m in rows['SL']}
        assert all(m['score']==scores[ix[run.key(m)]] for m in rows['SL'])
        assert all({k:v for k,v in m.items() if k!='score'}=={k:v for k,v in pool[ix[run.key(m)]].items() if k!='score'} for m in rows['SL'])
        checks['member_metadata_scores']+=800
        for m in rows['SL']:class_scores.setdefault(m['class_index'],[]).append(m['score'])
        for v in run.VIEWS:
            before=run.read(run.S/'predictions'/s/f'{v}_S.json');after=run.read(out/'predictions'/s/f'{v}_SL.json');sizes={g['geometry_key']:g['point_count'] for g in run.read(c['jobs'][s]['nt_geometry.json'])}
            template=run.read(run.S/'evaluation'/s/f'{v}_S_matches.json')
            expected=run.independent_rescore_match(template,before,after,sizes,v=='class_agnostic');assert expected==run.read(out/'evaluation'/s/f'{v}_SL_matches.json');checks['bidirectional_match_replays']+=1
        census=run.read(c['jobs'][s]['target_census']);ids=np.asarray(list(map(int,census['targets'])));saved=diag['scenes'][s]['SL'];hit={}
        for a,rr in rows.items():
            table=np.asarray([[census['ious'][m['geometry_key']][str(g)] for g in ids] for m in rr]);good=table[:,:,None]>ts[None,None,:]
            same=np.asarray([c['class_ids'][m['class_index']] for m in rr])[:,None]==ids[None,:]//1000
            pos=np.asarray([m['score']>0 for m in rr])[:,None,None]
            hit[a]=dict(semantic=(good & same[:,:,None]).any(axis=0),positive=(good & same[:,:,None] & pos).any(axis=0),CA=good.any(axis=0),positive_CA=(good & pos).any(axis=0))
        assert all(np.array_equal(hit['S'][k],hit['SL'][k]) for k in hit['S'])
        coverage[s]={}
        for ti,t in enumerate(ts):
            actual=saved['coverage'][f'{t:.2f}'];assert all(actual[k]==int(hit['SL'][k][:,ti].sum()) for k in hit['SL'])
            coverage[s][f'{t:.2f}']={}
            for view in ('semantic','CA'):
                before,after=hit['L'][view][:,ti],hit['SL'][view][:,ti]
                coverage[s][f'{t:.2f}'][view]=dict(SL_gained_vsL=list(map(int,ids[after & ~before])),SL_lost_vsL=list(map(int,ids[before & ~after])))
            checks['all_target_threshold_checks']+=1
        recovered=run.read(ROOT/'.local/fi1_s_c_selection_scope_development200_20261010_v1/scenes'/s/'members.json')['recovered_targets'];recids={int(r['gt_id']) for r in recovered}
        assert {r['gt_id'] for r in saved['recovered424']}==recids
        rec_stats[s]={}
        for r in saved['recovered424']:
            gi=int(np.flatnonzero(ids==r['gt_id'])[0]);assert r['member_present']==bool(hit['SL']['semantic'][gi,t75]);assert r['positive_present']==bool(hit['SL']['positive'][gi,t75]);assert r['member_present'] and r['positive_present']
            correct=[m for m in rows['SL'] if c['class_ids'][m['class_index']]==r['gt_id']//1000 and census['ious'][m['geometry_key']][str(r['gt_id'])]>ts[t75]]
            best=max(m['score'] for m in correct);assert r['best_selected_score']==best;rankchecks.append(r)
            rec_stats[s][str(r['gt_id'])]=dict(SL_member=True,SL_positive=True,L_member=bool(hit['L']['semantic'][gi,t75]))
            checks['recovered424']+=1
    class_scores={k:sorted(v) for k,v in class_scores.items()}
    for r in rankchecks:
        values=class_scores[r['class_index']];score=r['best_selected_score'];expected=[len(values)-bisect.bisect_right(values,score)+1,len(values)-bisect.bisect_left(values,score)]
        assert r['global_class_selected_rank_interval']==expected;checks['recovered_class_rank_checks']+=1
    for group,ss in c['reporting_groups'].items():
        effects[group]={v:{'fixed_S_score_effect_pp':100*(ap[group][v]['SL']['ap']-ap[group][v]['S']['ap']),
                          'L_reselection_effect_pp':100*(ap[group][v]['L']['ap']-ap[group][v]['SL']['ap']),
                          'total_L_effect_pp':100*(ap[group][v]['L']['ap']-ap[group][v]['S']['ap'])} for v in run.VIEWS}
        for v in run.VIEWS:
            x=effects[group][v];assert abs(x['fixed_S_score_effect_pp']+x['L_reselection_effect_pp']-x['total_L_effect_pp'])<1e-12
            arr=np.load(out/'evaluation'/f'{group}_{v}_SL_AP.npy');summary=run.read(out/'evaluation_summary.json');assert sha(out/'evaluation'/f'{group}_{v}_SL_AP.npy')==summary['AP_arrays'][str(out/'evaluation'/f'{group}_{v}_SL_AP.npy')]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                independent=dict(ap=float(np.nanmean(arr[0,:,~np.isclose(ts,.25)])),ap50=float(np.nanmean(arr[0,:,np.isclose(ts,.5)])),ap25=float(np.nanmean(arr[0,:,np.isclose(ts,.25)])))
            assert all(abs(independent[k]-ap[group][v]['SL'][k])<1e-14 for k in independent);checks['AP_array_aggregates']+=1
    tradeoffs={}
    for group,ss in c['reporting_groups'].items():
        tradeoffs[group]={}
        for ti,t in enumerate(ts):
            tradeoffs[group][f'{t:.2f}']={v:{k:sum(len(coverage[s][f'{t:.2f}'][v][k]) for s in ss) for k in ('SL_gained_vsL','SL_lost_vsL')} for v in ('semantic','CA')}
    mainap=ap['development200'];decision=run.read(out/'decision.json');both=all(mainap[v]['SL']['ap']>mainap[v][a]['ap'] for v in run.VIEWS for a in ('S','C','R'))
    assert decision['both_AP_passed']==both and decision['coverage_passed'] and decision['accepted']==both
    analysis=out/'supplemental_analysis.json';freeze_json(analysis,dict(complete=True,checks=dict(checks),AP_effects=effects,
        all_target_SL_vsL=coverage,groups_SL_vsL=tradeoffs,recovered424=rec_stats,decision_exact=True,
        duplicate_interpretation='The fixed S representative definition favors S by construction. Rank change is not proof of true duplicate deterioration.'))
    lines=['# 固定S800使用L分数：覆盖与排序效应核验','',decision['interpretation'],'',
        '独立核查固定成员160000条、400双向匹配投影、16新AP数组聚合、2000逐档覆盖；所有S覆盖及正分覆盖精确一致，424正确类全保留。',
        '', '|口径|固定S集合换L分数(pp)|L分数下重选集合(pp)|L总收益(pp)|','|---|---:|---:|---:|']
    for view,z in effects['development200'].items():lines.append(f"|{view}|{z['fixed_S_score_effect_pp']:+.6f}|{z['L_reselection_effect_pp']:+.6f}|{z['total_L_effect_pp']:+.6f}|")
    lines+=['','|IoU|SL相比L语义新增／损失|SL相比L去类新增／损失|','|---|---:|---:|']
    for t,z in tradeoffs['development200'].items():lines.append(f"|>{t}|{z['semantic']['SL_gained_vsL']}／{z['semantic']['SL_lost_vsL']}|{z['CA']['SL_gained_vsL']}／{z['CA']['SL_lost_vsL']}|")
    restored=sum(not r['L_member'] for ss in rec_stats.values() for r in ss.values())
    lines+=['',f'相对L恢复{restored}个424子群成员；SL恢复S成员选择，放弃L重选的新机会。新增／损失逐个目标见supplemental_analysis.json，424分数和实际全局同类排名见diagnostics.json。',
        '', '固定S集合是可以GT-free执行的政策：S先选800，L对名单提供分数；模型推理仍需真实自身特征／训练归一化，不能用当前覆盖名单作为推理输入。本轮复用封存分数，没有新模型调用。',
        '', '重复配对解释修正：当前S代表由最高S分定义，因此S逆序率天然占优。L逆序增加不能独立证明实际重复恶化；可以确定的只是尚未证明去重能力。旧冻结报告保留，当前解释在本报告与live文档更正。',
        '', '八分组（全200、旧90、新110、五折）AP/AP50/AP25和S/C/R/L对照见report.md。通过只是暴露开发候选，后续稳定性／独立验收需符合数据边界；不自动替换生产或启用已消费保留集、不训练预算模块、不加U/U′。']
    report=out/'supplemental_report.md';report.write_text('\n'.join(lines)+'\n')
    freeze_json(out/'independent_verified.json',dict(complete=True,code_sha256=sha(__file__),counts=dict(checks),all_S_coverage_exact=True,all424_member_positive_present=True,
        outputs={str(analysis):sha(analysis),str(report):sha(report),str(out/'completion_verified.json'):sha(out/'completion_verified.json')},
        raw_GT_read=False,new_training=False,new_model_inference=False,production_modified=False))
    print('Independent SL verification complete',dict(checks),decision,flush=True)


if __name__=='__main__':main()
