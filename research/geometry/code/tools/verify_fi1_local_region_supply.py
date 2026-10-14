#!/usr/bin/env python3
"""Independent point partitions, annotation and complete NT capacity replay."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_local_region_supply as run
from tools import diagnose_fi1_local_region_supply as diagnosis


def histogram_iou(p,gt,ids,sizes):
    if len(p)<100:return np.zeros(len(ids))
    hh=Counter(gt[p].tolist());hit=np.array([hh[int(g)] for g in ids]);return hit/(len(p)+sizes-hit)


def score_removal(parent,halo,indices,gt,ids,sizes):
    mask=np.setdiff1d(parent,halo[indices]);mask=parent if len(mask)<100 else mask
    return histogram_iou(mask,gt,ids,sizes)


def independent_stats(blocks,gt,parent,halo,stable,gid,ids,sizes):
    ann=[];safe=[];deltas=[];before=histogram_iou(parent,gt,ids,sizes);lookup={int(g):k for k,g in enumerate(ids)};hits=Counter(gt[parent].tolist())
    for b in blocks:
        if not stable:ann.append(4);safe.append(False);deltas.append(np.nan);continue
        values=gt[halo[b]];t=int(np.count_nonzero(values==gid));o=int(np.count_nonzero((values>=1000)&(values!=gid)));n=len(b)
        cat=0 if t==n else 1 if o==n else 2 if t and o else 3;h=hits[gid]-t;delta=h/(len(parent)-n+sizes[lookup[gid]]-h)-before[lookup[gid]] if len(parent)-n>=100 else 0.
        ann.append(cat);safe.append(cat==1 and delta>1e-12);deltas.append(delta)
    ann=np.asarray(ann);safe=np.asarray(safe,bool);length=np.asarray([len(b) for b in blocks]);sizebin=np.searchsorted([16,64,256],length,side='right')
    union=np.unique(np.concatenate(blocks)) if blocks else np.empty(0,int)
    stats=Counter(regions=len(blocks),safe_actions=int(safe.sum()),safe_parents=int(safe.any()),mixed_parents=int((ann==2).any()),unknown_parents=int((ann==4).any()),target_parents=int((ann==0).any()),parents_with_regions=int(bool(blocks)),safe64=int((safe&(length>=64)).sum()),safe256=int((safe&(length>=256)).sum()),covered_unique_points=len(union),single_block_target_gain02=int(np.any(np.asarray(deltas)>=.02-1e-12)))
    for k,name in enumerate(diagnosis.CATEGORIES):stats[name]=int((ann==k).sum())
    for k,name in enumerate(run.core.SIZE_NAMES):stats['size_'+name]=int((sizebin==k).sum());stats['safe_size_'+name]=int((safe&(sizebin==k)).sum())
    if stable:
        values=gt[halo[union]];stats.update(covered_pollution=int(((values>=1000)&(values!=gid)).sum()),covered_target=int((values==gid).sum()),covered_unknown=int((values<1000).sum()))
    return stats


def verify_scene(s):
    c=run.read(run.OUT/'contract.json');fold0=c['folds'][s];counts=Counter();job=c['jobs'][s];raw=np.load(job['paths']['processed'],mmap_mode='r');gt=np.loadtxt(job['GT'],dtype=np.int64)
    ids,sizes=run.old.previous.original.audit.prior.valid_targets(gt,c['class_ids']);_,nt=run.old.previous.original.audit.prior.old.load_nt(job);base=np.zeros(len(ids))
    for p in nt.values():base=np.maximum(base,histogram_iou(p,gt,ids,sizes))
    meta={r['candidate_id']:r for r in run.read(run.PREVIOUS/'labels'/s/'meta.json')};geometry_meta={r['candidate_id']:r for r in run.read(run.OUT/'regions'/s/'meta.json')}
    best={f:{a:{mode:base.copy() for mode in ('point','union','single','either')} for a in run.ARMS} for f in range(5) if f!=fold0};visible=base.copy();stats={f:{a:Counter() for a in run.ARMS} for f in best}
    with np.load(run.SOURCE/'features'/s/'data.npz') as features,np.load(run.SOURCE/'plan'/s/'domains.npz') as domains,np.load(run.OUT/'regions'/s/'regions.npz') as bank,np.load(run.OUT/'diagnosis'/s/'data.npz') as reported:
        assert np.array_equal(ids,reported['gt_ids']) and np.max(np.abs(base-reported['NT_best']),initial=0)<1e-12
        for row in run.read(run.SOURCE/'plan'/s/'candidates.json'):
            cid=row['candidate_id'];parent=features[cid+'_parent'];halo=features[cid+'_halo'];seen=features[cid+'_'+run.OBS+'_groups'];nn=features[cid+'_neighbors'];points=np.asarray(raw[halo,:9],np.float32)
            actual=run.old.unpack(bank,cid);rebuilt,metadata,budgets,trace=run.core.propose(points[:,:3],points[:,3:6],points[:,6:9],parent,halo,seen,nn,domains[cid+'_roles'][3])
            assert len(actual)==len(rebuilt) and all(np.array_equal(a,b) for a,b in zip(actual,rebuilt));assert metadata==geometry_meta[cid]['regions'] and trace==geometry_meta[cid]['trace'];counts['partition_replays']+=1
            eligible=np.flatnonzero(np.isin(halo,parent)&(seen>=2));assert np.array_equal(run.core.union(actual),eligible)
            for b,indices in budgets.items():assert np.array_equal(indices,bank[cid+'_budget'+b]) and len(indices)<=int(b) and len(indices)==len(np.unique(indices));counts['budget_replays']+=1
            stable=meta[cid]['stable'];gid=meta[cid]['target'];before=histogram_iou(parent,gt,ids,sizes)
            parent_hist=Counter(gt[parent].tolist());parent_hits=np.array([parent_hist[int(g)] for g in ids]);valid_lookup={int(g):k for k,g in enumerate(ids)}
            if stable:
                pollution=eligible[(gt[halo[eligible]]>=1000)&(gt[halo[eligible]]!=gid)];visible=np.maximum(visible,score_removal(parent,halo,pollution,gt,ids,sizes))
            geometry={}
            for a in run.ARMS[2:]:
                selected=actual if a=='geometry_raw' else [actual[k] for k in budgets[a.removeprefix('geometry')]];geometry[a]=selected
            # Independently derive labels directly, not diagnosis.labels().
            for k,b in enumerate(actual):
                if stable:
                    values=gt[halo[b]];t=int((values==gid).sum());o=int(((values>=1000)&(values!=gid)).sum());u=len(b)-t-o
                    category=0 if t==len(b) else 1 if o==len(b) else 2 if t and o else 3
                    gi=valid_lookup[gid];remaining_hit=parent_hist[gid]-t
                    delta=remaining_hit/(len(parent)-len(b)+sizes[gi]-remaining_hit)-before[gi] if len(parent)-len(b)>=100 else 0.;safe=category==1 and delta>1e-12
                    assert abs(delta-reported[cid+'_delta'][k])<1e-12
                else:t=o=0;u=len(b);category=4;safe=False;assert np.isnan(reported[cid+'_delta'][k])
                assert (t,o,u)==tuple(reported[cid+'_counts'][k]);assert category==reported[cid+'_category'][k] and safe==reported[cid+'_safe'][k];counts['region_annotation_replays']+=1
            for fold in best:
                path=run.PREVIOUS/'generators'/f'outer{fold}'/f'inner{fold0}'/'proposals'/s/'data.npz'
                with np.load(path) as oldg:
                    groups=dict(geometry);groups['G_cached']=run.old.unpack(oldg,cid,'_G');groups['G_float32']=run.core.probability_blocks(parent,halo,oldg[cid+'_G_probability'],seen,nn)
                    for a,blocks in groups.items():
                        stats[fold][a].update(independent_stats(blocks,gt,parent,halo,stable,gid,ids,sizes))
                        if not stable:continue
                        coverage=run.core.union(blocks);correct=coverage[(gt[halo[coverage]]>=1000)&(gt[halo[coverage]]!=gid)]
                        safe_blocks=[]
                        for b in blocks:
                            values=gt[halo[b]]
                            if len(parent)-len(b)>=100 and (values>=1000).all() and (values!=gid).all():safe_blocks.append(b)
                        ps=score_removal(parent,halo,correct,gt,ids,sizes);us=score_removal(parent,halo,run.core.union(safe_blocks),gt,ids,sizes);ss=before.copy()
                        for b in safe_blocks:
                            removed_hist=Counter(gt[halo[b]].tolist());remaining=parent_hits-np.array([removed_hist[int(g)] for g in ids]);ss=np.maximum(ss,remaining/(len(parent)-len(b)+sizes-remaining));counts['whole_single_actions_replayed']+=1
                        for mode,score in (('point',ps),('union',us),('single',ss),('either',np.maximum(us,ss))):best[fold][a][mode]=np.maximum(best[fold][a][mode],score)
        assert np.max(np.abs(visible-reported['visible_known_other_best']),initial=0)<1e-12
        for fold,arms in best.items():
            for a,modes in arms.items():
                for mode,score in modes.items():assert np.max(np.abs(score-reported['fold'+str(fold)+'_'+a+'_'+mode]),initial=0)<1e-12;counts['pool_target_replays']+=len(ids)
                actual_stats=run.read(run.OUT/'diagnosis'/s/'stats.json')['folds'][str(fold)][a]
                for key,value in stats[fold][a].items():assert value==actual_stats.get(key,0),(s,fold,a,key,value,actual_stats.get(key,0));counts['sample_census_replays']+=1
    return counts


def report(summary,verified):
    lines=['# 局部区域提议覆盖与样本供给报告','','**本报告只有GT-free区域提议与原外层训练范围后置诊断，没有新训练、学习删除或AP。**','',
           '固定开发200/1599父N-T、GD10六帧、原可靠可见性。几何区域直接用原始XYZ/RGB/法向、冻结8邻接；三尺度.05/.12/.30米×两偏移，格内还能按边界拆分。N/T分歧仅优先级；不读取G概率、superpoint或GT。所有200场景区域封存后才诊断。主预算96，48/192和预算前只作固定对照，不事后挑主规则。',
           '', '五套训练范围均160场景，彼此重叠，不能把相加当独立样本或正式OOF。G_cached使用该外层训练内部的原内交叉提议；G_float32只在旧概率上明确float32门，不改原封存模型/结果。',
           '', '|外层训练范围|提议|区域|安全动作|安全父/场景|安全区≥64 / ≥256点|整块筛查NT增量≥.02|检查通过|','|---|---|---:|---:|---:|---:|---:|---|']
    for fold,data in summary['folds'].items():
        for arm in run.ARMS:
            m=data['arms'][arm];passed=data['main_supply_passed'] if arm=='geometry96' else '—'
            lines.append(f"|{fold}|{arm}|{m['regions']}|{m['safe_actions']}|{m['safe_parents']} / {m['safe_scene']}|{m['safe64']} / {m['safe256']}|{m['either_pool_gain02']}|{passed}|")
    lines+=['','上表安全动作指稳定父、整区全为已知其他实例、删除后>=100点且父IoU改善；是训练GT标签，不是推理安全概率。整块筛查取每个父的全部安全区联合删除或其中一个区，之后按完整NT池取最佳；两种模式分别保存，不求所有组合最优。未知标注保留，实际删点去重。',
            '', '|外层训练范围|主预算96纯目标区|纯污染区|混合区|标注未知区|身份未知区|安全区尺寸中位数/最大|','|---|---:|---:|---:|---:|---:|---:|']
    for fold,data in summary['folds'].items():
        m=data['arms']['geometry96'];q=summary['sizes'][fold]['geometry96']['safe_size_quantiles'];size='—' if not q else f'{q[1]:g} / {q[3]:g}'
        lines.append(f"|{fold}|{m['target_only']}|{m['known_pollution_only']}|{m['target_neighbor_mixed']}|{m['annotation_unknown']}|{m['identity_unknown']}|{size}|")
    lines+=['','200开发场景的训练诊断并集（不是折外效果）：可靠可见已知污染逐点筛查可提供'+str(summary['union200_visible_capacity'])+'个≥.02目标；各几何供给的覆盖内逐点/纯块联合/单块/二者最佳分别为：',
            '', '|几何供给|覆盖内正确点|纯块联合|单块|二者最佳|','|---|---:|---:|---:|---:|']
    for a,m in summary['union200_geometry_capacity'].items():lines.append(f"|{a}|{m['point']}|{m['union']}|{m['single']}|{m['either']}|")
    lines+=['','固定96的供给门全部通过：'+str(summary['all_training_supply_passed'])+'。每折具体未通过条件：']
    for fold,data in summary['folds'].items():lines.append('- fold'+fold+'：'+str([k for k,v in data['main_supply_gate'].items() if not v]))
    lines+=['','该门仅检查是否存在跨场景、跨尺度样本与潜在几何价值，不证明区域归属可以学会。即使通过，本轮也不启动验证器；后续应使用相同区域来源训练/测试，先验收归属和实际后果预测相对常数基线。若未通过，先改提议/预算，不能从本GT筛查挑区域当预测。',
            '', '区域样本可空间重叠，统计不是独立点数；跨尺度精确同点集去重，近似重复保留作为多尺度提议。安全/混合/未知、N/T、每父逐区点数与delta在diagnosis/*/regions.csv；逐父各折对照在evaluation/candidates.csv。六个IoU档与真实目标变化见evaluation/pool_targets.csv。','',
            '核验：'+str(verified)+'。全部原始点域分区、预算、GT后置标注与完整NT诊断重放；访问检查覆盖原外层留出、原诊断和学习提议隔离。没有原GT分段超点、新SAM/DINO、轨迹或SL改动。开发上游已暴露，不称全系统盲测。']
    (run.OUT/'report.md').write_text('\n'.join(lines)+'\n')
    decision='主预算96供给检查通过，可作为下一轮同源区域归属学习的候选输入；本轮不训练，GT筛查不是预测收益。' if summary['all_training_supply_passed'] else '主预算96供给检查未全部通过，停在区域提议阶段；先修改区域产生/预算，不自动训练。'
    (run.OUT/'decision.md').write_text(decision+'\n\n详见report.md、summary.json及各折样本/池目标诊断；48/192不替换预注册主预算。\n')


def verify():
    c=run.read(run.OUT/'contract.json')
    for mapping in (c['code'],c['source_bindings'],c['frozen_SL_models']):run.old.previous.original.audit.prior.old.verify_files(mapping)
    for name in ('regions','diagnosis'):run.old.previous.original.audit.prior.old.verify_files(run.read(run.OUT/(name+'_frozen.json'))['outputs'])
    counts=Counter();summary=run.read(run.OUT/'summary.json')
    for fold in range(5):
        s=next(s for s in c['scenes'] if c['folds'][s]==fold)
        for mode,path in [('generate',run.SOURCE/'labels'/s/'meta.json'),('diagnose',Path(c['jobs'][s]['GT'])),('diagnose',run.OUT/'diagnosis'/s/'stats.json'),('generate',run.PREVIOUS/'generators/outer0/full/model.pt'),('generate',ROOT/'.local/fi1_deletion_blocks_independent_analysis_20261013/audit.json')]:
            try:run.Guard(c,mode,fold).hook('open',(str(path),'r',0))
            except PermissionError:counts['access_rejections']+=1
            else:raise AssertionError('stage or fold leak')
        data=summary['folds'][str(fold)];m=data['arms']['G_cached'];expected=len(run.read(run.PREVIOUS/'training_blocks'/f'outer{fold}'/'meta.json')['keys']);assert m['regions']==expected
        assert data['main_supply_gate']==diagnosis.gate(data['arms']['geometry96']);counts['training_scopes_verified']+=1
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(verify_scene,s) for s in c['scenes']])):
            counts.update(f.result())
            if (i+1)%10==0:print('independent region supply replay',i+1,'/200',flush=True)
    assert counts['partition_replays']==1599 and counts['budget_replays']==4797
    report(summary,dict(counts));run.seal(run.OUT/'independent_verified.json',dict(complete=True,counts=dict(counts),contract_sha256=run.sha(run.OUT/'contract.json'),summary_sha256=run.sha(run.OUT/'summary.json')))
    run.seal(run.OUT/'completion_verified.json',dict(complete=True,regions_generated_before_GT=True,training_scope_only=True,new_training=False,new_actions=False,SL_preserved=True,AP_run=False,independent_sha256=run.sha(run.OUT/'independent_verified.json'),report_sha256=run.sha(run.OUT/'report.md'),decision_sha256=run.sha(run.OUT/'decision.md')));run.stage('complete_verified',**dict(counts))


if __name__=='__main__':verify()
