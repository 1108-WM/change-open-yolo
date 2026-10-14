#!/usr/bin/env python3
"""Independent scores/AP aggregation, coverage/FP effects and final reports."""
from pathlib import Path
from collections import Counter
import json,sys,hashlib,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_witnessed_completion_sl as run
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
read=run.read;OUT=run.OUT;ARMS=run.ARMS;VIEWS=run.VIEWS

def main():
    c=read(OUT/'contract.json');assert read(OUT/'evaluation_summary.json')['complete']
    run.ctl.base.verify_files(c['code']);run.ctl.base.verify_files(read(OUT/'evidence_frozen.json')['outputs']);run.ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs']);run.ctl.base.verify_files(read(OUT/'matches_frozen.json')['outputs'])
    import torch
    torch.set_num_threads(4);models={f:run.sl.load_L(run.L/'models'/f'fold{f}'/'model.pt',run.L/'models'/f'fold{f}'/'normalization.npz') for f in range(5)}
    cv=read(OUT/'coverage.json');ev=run.ctl.base.evaluator();thresholds=list(map(float,ev.opt['overlaps']));t75=min(thresholds,key=lambda t:abs(t-.75));counts=Counter();details={};totals={g:{a:{'coverage':{f'{t:.2f}':Counter() for t in thresholds},'members':Counter()} for a in ARMS} for g in c['groups']};rec_total=0
    for i,s in enumerate(c['scenes']):
        print('verify selected scores',i,s,flush=True);d=OUT/'predictions'/s;em=read(d/'members.json');new={g['geometry_key'] for g in read(run.geom.OUT/'geometry'/s/'geometry.json')};gs,pts,evidence=run.expanded(c,s)
        own=read(OUT/'evidence'/s/'evidence.json');assert {r['geometry_key'] for r in own}==new
        for r in own:
            assert r['ground_truth_read'] is False and r['ap_computed'] is False
            assert r['alpha']['acquisition_complete'] and r['yolo']['acquisition_complete'];assert hashlib.sha256(pts[r['geometry_key']].astype('<i8').tobytes()).hexdigest()==r['points_sha256']
        with np.load(d/'extended_pool.npz') as z:x26=z['x26'];baseline=z['scores']
        # Independent selected-member relation summaries from point intersections.
        member_index={run.ctl.shared.key(m):j for j,m in enumerate(em)}
        selected_probe=read(d/'semantic_append.json')[:20]
        for probe in selected_probe:
            j=member_index[run.ctl.shared.key(probe)];m=em[j];p=pts[m['geometry_key']]
            neighbors=[r for r in em if r['class_index']==m['class_index'] and r['geometry_key']!=m['geometry_key']]
            inter=np.array([len(np.intersect1d(p,pts[r['geometry_key']],assume_unique=True)) for r in neighbors]);sizes=np.array([len(pts[r['geometry_key']]) for r in neighbors]);ov=inter/(len(p)+sizes-inter);ss=np.array([r['score'] for r in neighbors]);strong=ov>.5
            expected=np.array([ov.max(initial=0),np.log1p(strong.sum()),(ov*ss).max(initial=0),m['score']-ss[strong].max() if strong.any() else 0,np.mean(ss[strong]>m['score']) if strong.any() else 0,(inter/len(p)).max(initial=0)])
            assert np.array_equal(x26[j,20:],expected);counts['independent_context_rows']+=1
        for g in gs:
            if g['geometry_key'] in new:
                ownx=run.src.geometry_features(g,evidence[g['geometry_key']]);assert np.array_equal(ownx[:,14:18],np.tile([0,0,0,1],(198,1)))
                assert all(a['class_index']==-1 for a in run.src.original_aliases(g))
        # Full-pool original batch inference independently checks padded selected-only path.
        model,mean,std=models[c['folds'][s]];x=np.column_stack((x26,baseline,run.sl.core.base_logits(baseline))).astype(np.float32);deltas=[]
        with torch.no_grad():
            for start in range(0,len(x),8192):deltas.append(model(torch.from_numpy(((x[start:start+8192]-mean)/std).astype(np.float32))).numpy().astype(float))
        final=run.sl.core.adjusted(baseline,np.concatenate(deltas));index={run.ctl.shared.key(m):j for j,m in enumerate(em)}
        rows={a:read(d/f'semantic_{a}.json') for a in ARMS};sets={a:{run.ctl.shared.key(m) for m in rows[a]} for a in ARMS};lookup={a:{run.ctl.shared.key(m):m for m in rows[a]} for a in ARMS}
        for a in ('context_NT','append'):
            expected=sorted([{**m,'score':float(baseline[j])} for j,m in enumerate(em) if a=='append' or m['geometry_key'] not in new],key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:800]
            expected=[{**m,'score':float(final[index[run.ctl.shared.key(m)]])} for m in expected];expected.sort(key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))
            assert expected==rows[a];counts['model_score_rows_exact']+=len(expected)
        assert rows['SL']==read(run.SL/'predictions'/s/'semantic_SL.json')
        for a in ARMS:
            best={}
            for m in rows[a]:
                k=m['geometry_key']
                if k not in best or (m['score'],-m['class_index'])>(best[k]['score'],-best[k]['class_index']):best[k]=m
            ca=sorted([dict(geometry_key=k,score=m['score'],winning_class_index=m['class_index']) for k,m in best.items()],key=lambda m:(-m['score'],m['geometry_key']))
            assert ca==read(d/f'class_agnostic_{a}.json');counts['CA_outputs_checked']+=len(ca)
        rec=read(ROOT/'.local/fi1_s_c_selection_scope_development200_20261010_v1/scenes'/s/'members.json')['recovered_targets'];ids=np.array(cv[s]['gt_ids']);recids={int(r['gt_id']) for r in rec};rec_total+=len(recids);recmask=np.isin(ids,list(recids));scene={}
        semantic_matches={a:read(OUT/'evaluation'/s/f'semantic_{a}_matches.json') for a in ARMS}
        positive={a:{f'{t:.2f}':{g['instance_id'] for gg in semantic_matches[a]['gt'].values() for g in gg if g['instance_id'] in ids for p in g['matched_pred'] if p['confidence']>0 and p['intersection']/(p['vert_count']+g['vert_count']-p['intersection'])>t} for t in thresholds} for a in ARMS}
        for a in ARMS:
            value=cv[s]['arms'][a];ref=cv[s]['arms']['SL'];cov={}
            for t in thresholds:
                cc=Counter()
                for view in VIEWS:
                    now=np.array(value[view])>t;before=np.array(ref[view])>t
                    cc[view+'_covered']=int(now.sum());cc[view+'_gained']=int((now&~before).sum());cc[view+'_lost']=int((before&~now).sum())
                    cc[view+'_recovered424_lost']=int((before&~now&recmask).sum())
                tag=f'{t:.2f}';cc['semantic_positive_lost']=len(positive['SL'][tag]-positive[a][tag]);cc['semantic_positive_recovered424_lost']=len((positive['SL'][tag]-positive[a][tag])&recids)
                cov[tag]=dict(cc)
            ms=Counter(entered=len(sets[a]-sets['SL']),exited=len(sets['SL']-sets[a]),new_members=sum(r['geometry_key'] in new for r in rows[a]),new_selected_geometries=len({r['geometry_key'] for r in rows[a] if r['geometry_key'] in new}),zero_members=sum(r['score']==0 for r in rows[a]))
            sizes={k:len(p) for k,p in pts.items()};match=read(OUT/'evaluation'/s/f'semantic_{a}_matches.json');status=run.src.member_status(rows[a],match,sizes,c['class_ids'],t75);newstatus=Counter(status[run.ctl.shared.key(m)] for m in rows[a] if m['geometry_key'] in new)
            oldmatch=read(OUT/'evaluation'/s/'semantic_SL_matches.json');oldstatus=run.src.member_status(rows['SL'],oldmatch,sizes,c['class_ids'],t75)
            entered=Counter(status[k] for k in sets[a]-sets['SL']);exited=Counter(oldstatus[k] for k in sets['SL']-sets[a])
            ms.update({'new_'+k:v for k,v in newstatus.items()});ms.update({'entered_'+k:v for k,v in entered.items()});ms.update({'exited_'+k:v for k,v in exited.items()})
            scene[a]=dict(coverage=cov,members=dict(ms),new_status=dict(newstatus),entered_status=dict(entered),exited_status=dict(exited))
            for g,ss in c['groups'].items():
                if s not in ss:continue
                totals[g][a]['members'].update(ms)
                for t,cc in cov.items():totals[g][a]['coverage'][t].update(cc)
        details[s]=scene
    assert rec_total==424
    totals={g:{a:dict(members=dict(v['members']),coverage={t:dict(cc) for t,cc in v['coverage'].items()}) for a,v in arms.items()} for g,arms in totals.items()}
    summary=read(OUT/'evaluation_summary.json');ap=summary['AP'];aggregate_checks=0
    for g in c['groups']:
        for view in VIEWS:
            for a in ARMS:
                q=OUT/'evaluation'/f'{g}_{view}_{a}_AP.npy';z=np.load(q);overlaps=np.asarray(ev.opt['overlaps']);expected=dict(ap=float(np.nanmean(z[:,:,overlaps!=.25])),ap50=float(np.nanmean(z[:,:,np.isclose(overlaps,.5)])),ap25=float(np.nanmean(z[:,:,overlaps==.25])))
                assert all(abs(expected[k]-ap[g][view][a][k])<1e-14 for k in expected);assert sha(q)==summary['AP_arrays'][str(q)];aggregate_checks+=1
    freeze_json(OUT/'diagnostics.json',dict(complete=True,groups=totals,scenes=details,recovered424=rec_total,member_states='Evaluator-compatible overlap/ignore status; valid_sameclass_match includes duplicates, not unique TP or AP contribution.'))
    freeze_json(OUT/'independent_verified.json',dict(complete=True,source_sha256=sha(__file__),counts=dict(counts),AP_aggregation_checks=aggregate_checks,original_SL_reference_exact=True,full_pool_model_inference_matches_selected_only=True,geometry_verified_sha256=sha(run.geom.OUT/'independent_geometry_verified.json'),outputs={str(OUT/n):sha(OUT/n) for n in ('diagnostics.json','evaluation_summary.json','predictions_frozen.json','matches_frozen.json')}))
    geo=read(run.geom.OUT/'geometry_quality.json')['groups']['development200'];op=read(run.geom.OUT/'opportunity_action_gap.json')['counts'];diag=read(run.geom.diag.OUT/'summary.json')['groups']['development200'];main=ap['development200'];stat=totals['development200']['append'];semgain=100*(main['semantic']['append']['ap']-main['semantic']['SL']['ap']);cagain=100*(main['class_agnostic']['append']['ap']-main['class_agnostic']['SL']['ap']);protected=all(not v['semantic_lost'] and not v['class_agnostic_lost'] for v in stat['coverage'].values());accepted=semgain>0 and cagain>0 and protected
    lines=['# 固定SL：NT几何缺口诊断与首版局部补全','',f'本轮完成一种几何动作和真实双AP对照。追加方案相对SL类别{semgain:+.6f}pp、实际去类{cagain:+.6f}pp；全部档位覆盖保护={protected}，双AP及保护共同通过={accepted}。不改生产，不根据本轮结果扫描门槛。','',
        '诊断在原开发200五折留出上进行，SL单模型／开发55保持封存。GT仅用于解释范围误差与封存后评测；生成动作和自身语义推理不读GT。200已用于方法选择，不称盲测。','',
        '|877个NT>.75缺口：最佳候选类型|数量|','|---|---:|','|较纯但欠覆盖|260|','|覆盖较全但污染|336|','|欠覆盖且污染|94|','|纯度／覆盖率均>.75但IoU不足|110|','|评测有效NT无交集|77|','',
        '同时检查每类来源前5候选、最高IoU前12候选两两并集、30帧物理可见性与原始掩码。731个缺口在30帧总并集中至少半数目标可见，只有166个在原始掩码总并集中达到半数覆盖；18个无可见点、176个无原始掩码支持。30帧没有支持不等于全部视角前端漏检。404个最佳候选的污染包含其他有效实例。','',
        'GT挑选的两两并集可修复60个>.75缺口，保守观测裁剪仅2个。它们是特定搜索与观测范围的机会，不是全部可能修复上限，也不能直接作为预测。选择实现一种一跳局部补全：N/T双向供给缺失区域、至少两个姿态组共同掩码支持、逐点支持、物理接触、增长预算；分离证据否决，有同分歧义则弃权。所有原NT保留，未知点不自动补入，禁止传递增长和父类别／分数继承。','',
        '|阈值|新增覆盖（相对整个NT池）|NT已覆盖目标|合格新点集|不合格新点集|','|---|---:|---:|---:|---:|']
    for t,z in geo.items():lines.append(f"|>{t}|{z['new_over_complete_NT']}|{z['already_NT_covered']}|{z['qualified_new_geometries']}|{z['nonqualified_new_geometries']}|")
    extent=read(run.geom.OUT/'extent_quality_supplement.json')['rows'];polluted=sum(r['added_pollution_points']>r['added_target_points'] for r in extent)
    lines+=['','201个新点集多数覆盖已有目标，不能把合格新点集101个解释成新增101个目标。全部NT保留意味着池级覆盖没有损失，最终预算输出仍可能损失。',f'按新点集最佳匹配GT解释追加区域，{polluted}/201个动作新增污染点多于新增该目标点。共同掩码与空间接触没有充分证明同实例身份；即使整体IoU较高，追加区域也可能方向错误。该诊断不进入推理规则。小于100点候选的IoU按原评测置零，无交集类型不等于所有物理碎片完全无目标点。','',f"60个理论两两并集机会中，{op['has_direction_pair_overlap_excluded']}个被固定pair重叠范围排除；{op['has_direction_insufficient_independent_joint']}个至少一方向缺独立共同支持；{op['has_direction_anchor_identity_budget']}个至少一方向超过锚定增长预算；{op['has_direction_separation_counterevidence']}个有分离反证。各项可重叠。对这60个最佳oracle配对，当前动作没有形成>.75修复，实际新增1个来自另一配对。说明范围搜索与观测支持接口没有把理论互补机会有效转成输出；不据此直接放宽合并门槛。",'',
        '新增点集全部重新投影并获取自身YOLO40视角票数及Alpha20视角×三尺度证据；不套用父类，不用Alpha top1或SMS过滤。冻结198文本向量从旧模型资产复制，未读取val312场景／结果。新来源显式借用refined_union特征槽、类=-1；这是既有26维模型的兼容扩展，来源输入分布发生变化，原NT泛化验证不自动延伸。','',
        '|分组|方案|类别AP (%)|实际去类AP (%)|','|---|---|---:|---:|']
    for g in c['groups']:
        for a in ARMS:lines.append(f"|{g}|{a}|{100*ap[g]['semantic'][a]['ap']:.6f}|{100*ap[g]['class_agnostic'][a]['ap']:.6f}|")
    witness=read(OUT/'new_geometry_cover_witnesses.json')['rows']
    lines+=['',f"三个新覆盖见证均有自身Alpha／融合top1正确类，B26场景排名63／30／39并实际进入追加输出；分别为scene0017_02 GT52021（>.90）、scene0051_00 GT77032（>.75）、scene0198_00 GT59008（>.90）。它们是真实局部有效例子，但不能替代整体AP和退出代价。详见new_geometry_cover_witnesses.json。",'',
        'SL为完整NT原参考；context_NT以扩池上下文计算特征／分数，只在NT中选800；append使用相同扩池分数，允许新几何参加800竞争。各组均由B26选择名单、对应五折L提供置信，实际去类从已选成员同点集最高分得到，不补位、不独立重排几何。context_NT−SL含上下文及其名单变化；append−context_NT在相同评分池下分离追加预算效应。差值不是独立可加机制贡献。','', '|IoU档位|追加语义新增／损失|追加去类新增／损失|424语义损失|','|---|---|---|---:|']
    for t,v in stat['coverage'].items():lines.append(f"|>{t}|{v['semantic_gained']}／{v['semantic_lost']}|{v['class_agnostic_gained']}／{v['class_agnostic_lost']}|{v['semantic_recovered424_lost']}|")
    lines+=['',f"追加输出成员统计：`{json.dumps(stat['members'],ensure_ascii=False)}`。有效同类匹配包含重复，进入／退出FP计数不能直接换算AP损失。详见diagnostics.json各组与每场景。",'',
        '独立验证重建4689个目标／877缺口与201动作；5项范围／分离／姿态去重测试通过。所有三臂预测封存后原evaluator重匹配，双向交集／void独立点计数核对，原SL成员／分数／16AP逐位复现；扩池320000成员的完整批次模型推理与仅选中评分精确一致，48AP数组独立聚合。旧模型、预测、AP、生产代码和开发55结果均未修改。','',
        '后续几何研究应先处理互补碎片的关联证据与被门槛排除的修复区域，并区分“可见却没有有效掩码”“其他NT已有点但身份证据不足”；混合污染需另一个独立分离动作合同。本轮不自动扩大合并、不同时实现裁剪、不继续调SL语义头。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    freeze_json(OUT/'decision.json',dict(complete=True,accepted=accepted,semantic_delta_pp=semgain,class_agnostic_delta_pp=cagain,all_coverage_protected=protected,new_geometry_gain=geo,SL_weights_unchanged=True,next_action_not_implemented=True,production_modified=False,dev55_read=False,no_parameter_scan=True))
    freeze_json(OUT/'completion_verified.json',dict(complete=True,outputs={str(OUT/n):sha(OUT/n) for n in ('contract.json','report.md','decision.json','diagnostics.json','independent_verified.json','evaluation_summary.json')},new_training=False,production_modified=False))
    write_json(OUT/'execution.json',dict(status='complete',stage='complete',resume_required=False,accepted=accepted))
    write_json(run.geom.OUT/'execution.json',dict(status='complete',stage='geometry_and_downstream_completed',resume_required=False,downstream=str(OUT)))
    write_json(run.geom.diag.OUT/'execution.json',dict(status='complete',stage='independent_verified',resume_required=False,verified=str(run.geom.OUT/'independent_geometry_verified.json')))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):main()
