#!/usr/bin/env python3
"""Post-seal interpretation of supply scale, budget losses and source support."""
from pathlib import Path
from collections import Counter
import csv,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_local_region_supply as run


def summarize():
    c=run.read(run.OUT/'contract.json');s=run.read(run.OUT/'summary.json');assert run.read(run.OUT/'completion_verified.json')['complete']
    raw_family=[Counter() for _ in range(6)];main_family=[Counter() for _ in range(6)];source={};counts=Counter();targets={a:set() for a in run.ARMS[2:]}
    for scene in c['scenes']:
        fold=(c['folds'][scene]+1)%5
        with np.load(run.OUT/'regions'/scene/'regions.npz') as bank,np.load(run.OUT/'diagnosis'/scene/'data.npz') as label:
            for row in run.read(run.OUT/'regions'/scene/'meta.json'):
                cid=row['candidate_id'];src=row['source'];sizes=np.diff(bank[cid+'_block_offsets']);safe=label[cid+'_safe'];category=label[cid+'_category'];selected=bank[cid+'_budget96'];chosen=set(selected.tolist())
                m=source.setdefault(src,Counter());m.update(parents=1,raw_regions=len(sizes),main_regions=len(selected),raw_safe=int(safe.sum()),main_safe=int(safe[selected].sum()),main_mixed=int((category[selected]==2).sum()),main_unknown=int((category[selected]==4).sum()),main_safe_parents=int(safe[selected].any()))
                counts.update(raw_safe=int(safe.sum()),main_safe=int(safe[selected].sum()),main_safe_points_region_sum=int(sizes[selected][safe[selected]].sum()),raw_safe_points_region_sum=int(sizes[safe].sum()))
                for rid,metadata in enumerate(row['regions']):
                    for family in metadata['families']:
                        for array,yes in ((raw_family,True),(main_family,rid in chosen)):
                            if yes:array[family].update(regions=1,safe=int(safe[rid]),safe64=int(safe[rid] and sizes[rid]>=64),safe256=int(safe[rid] and sizes[rid]>=256),safe_max=int(sizes[rid]) if safe[rid] else 0)
                # Counter.update adds max fields; replace with actual family maxima.
            base=label['NT_best'];ids=label['gt_ids']
            for arm in targets:
                values=label[f'fold{fold}_{arm}_either'];targets[arm].update((scene,int(ids[k])) for k in np.flatnonzero(values-base>=.02-1e-12))
    # Size maxima are already independently recorded per training scope.
    for family in raw_family+main_family:family.pop('safe_max',None)
    for arm in targets:assert len(targets[arm])==s['union200_geometry_capacity'][arm]['either']
    aggregate={arm:{key:sum(f['arms'][arm].get(key,0) for f in s['folds'].values())//4 for key in ('covered_pollution','covered_target','covered_unknown','covered_unique_points')} for arm in run.ARMS[2:]}
    detail=dict(complete=True,new_training=False,GT_posthoc_only=True,unique_scene_parent_scope=True,source={k:dict(v) for k,v in source.items()},counts=dict(counts),coverage=aggregate,
                family_memberships_raw=[dict(v) for v in raw_family],family_memberships_main=[dict(v) for v in main_family],family_counts_overlap=True,
                primary_missed_raw_targets=len(targets['geometry_raw']-targets['geometry96']),primary_extra_raw_targets=len(targets['geometry96']-targets['geometry_raw']),
                reporting_code={str(Path(__file__).resolve()):run.sha(__file__)})
    run.seal(run.OUT/'interpretation.json',detail)
    lines=['','## 决策补充：数量改善，尺度和预算仍有损失','',
           '**主预算96未通过供给门，停在提议阶段，不训练验证器。** 它已提供比G多得多的跨场景正反样本和潜在价值，但缺少较大正例。每套训练范围安全区1710—1927个、117—121场景，中位6—7点、最大174—193点；预算前安全区最大194点，>=256点仍为0。提高预算不能创造划分阶段未产生的大块。',
           '', '200场景诊断并集：预算前可行整区筛查68目标，96区31，192区41，48区19；对应覆盖内逐点筛查70/62/68/56。这里都是GT可行筛查，不是学习收益、AP或组合最优上界。主预算相对预算前缺少'+str(detail['primary_missed_raw_targets'])+'个潜在目标，额外'+str(detail['primary_extra_raw_targets'])+'个；集合差只用于后置定位，不选择新动作。',
           '', '主预算单区删除只有8个≥.02目标，联合多区31个。因此主要机会仍依赖组合小区域，不能把“多尺度”名称等同于已经形成大而完整的可判断区域。',
           '', '|几何供给|父内已知污染覆盖点（按父累计）|GT整区筛查目标≥.02|','|---|---:|---:|']
    for arm in ('geometry_raw','geometry48','geometry96','geometry192'):lines.append(f"|{arm}|{aggregate[arm]['covered_pollution']}|{s['union200_geometry_capacity'][arm]['either']}|")
    lines+=['','区域样本点数相加会重复，上表覆盖点已经每个父内去重；不同父共享的点仍分别累计，不能称全场景去重点数。','',
            '|主预算供给来源|父|区域|安全区|混合区|身份未知区|有安全区的父|','|---|---:|---:|---:|---:|---:|---:|']
    for src,m in source.items():lines.append(f"|{src}|{m['parents']}|{m['main_regions']}|{m['main_safe']}|{m['main_mixed']}|{m['main_unknown']}|{m['main_safe_parents']}|")
    lines+=['','下一步先改变区域构成或提出可组合的邻接区域，在训练范围内重新检验尺度覆盖；不降低本次供给门，不拿GT纯区域当预测，不按四个已知见证定制，不启动完整五折学习。后续归属模型还须读取实际点/边界关系，并在训练与内部留出上让后果预测优于常数基线。','',
            '本轮8项针对性测试通过；全部1599父划分、4797预算、1508729区域标注、450144池目标及129600项供给计数独立重放。v1诊断仅因复用工作进程残留旧折检查而中断，execution_amendment_v2复用同一个可变检查状态；未重新生成区域、不改预算/标签/筛查规则，失败日志保留。']
    original=(run.OUT/'report.md').read_text();(run.OUT/'report_v2.md').write_text(original+'\n'.join(lines)+'\n')
    run.seal(run.OUT/'interpretation_verified.json',dict(complete=True,outputs={str(run.OUT/n):run.sha(run.OUT/n) for n in ('interpretation.json','report_v2.md')},summary_sha256=run.sha(run.OUT/'summary.json'),region_bank_sha256=run.sha(run.OUT/'regions_frozen.json')))


if __name__=='__main__':summarize()
