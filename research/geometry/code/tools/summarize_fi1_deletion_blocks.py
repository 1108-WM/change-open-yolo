#!/usr/bin/env python3
"""Read-only interpretation of sealed nested deletion-block results."""
from pathlib import Path
import csv,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_deletion_blocks as run


def summarize():
    out=run.OUT;s=run.read(out/'summary.json');c=run.read(out/'contract.json');assert run.read(out/'completion_verified.json')['complete']
    training={};fits={kind:[] for kind in ('summary','relation')}
    for fold in range(5):
        meta=run.read(out/'training_blocks'/f'outer{fold}'/'meta.json');details=meta['details']
        with np.load(out/'training_blocks'/f'outer{fold}'/'data.npz') as z:y=z['y']
        training[str(fold)]=dict(blocks=len(y),safe=int(y[:,0].sum()),stable=sum(r['stable'] for r in details),unknown=len(details)-sum(r['stable'] for r in details),
                               parents=len({(scene,cid) for scene,cid,k in meta['keys']}),scenes=len({scene for scene,cid,k in meta['keys']}),pollution=sum(r['pollution_deleted'] for r in details),target=sum(r['target_deleted'] for r in details))
        for kind in fits:fits[kind].append(run.read(out/'verifiers'/f'outer{fold}'/kind/'history.json')['training_fit'])
    with (out/'evaluation/candidates.csv').open() as f:rows=list(csv.DictReader(f))
    with (out/'evaluation/blocks.csv').open() as f:blocks=list(csv.DictReader(f))
    stable_error={};rawproposal={source:dict(blocks=0,points=0,parents=set()) for source in ('B','G')}
    for r in blocks:
        p=rawproposal[r['source']];p['blocks']+=1;p['points']+=int(r['points']);p['parents'].add((r['scene'],r['candidate_id']))
    for source in ('B','G'):
        p=rawproposal[source];p['parents']=len(p['parents'])
        for kind in ('summary','relation'):
            rr=[r for r in blocks if r['source']==source and r['stable']=='True' and r['delta']]
            stable_error[source+'_'+kind]=dict(blocks=len(rr),true_delta_MAE=float(np.mean([abs(float(r[kind+'_expected_delta'])-float(r['delta'])) for r in rr])) if rr else None)
    cases={arm:sorted([r for r in rows if r['arm']==arm and r.get('parent_delta')],key=lambda r:float(r['parent_delta']))[:3] for arm in run.ARMS}
    gains={arm:[r for r in rows if r['arm']==arm and r.get('pool_delta') and float(r['pool_delta'])>=.02-1e-12 and float(r['parent_delta'])>0] for arm in run.ARMS}
    extra=dict(complete=True,training_nested_proposals=training,formal_training_fit=fits,raw_proposal_capacity=rawproposal,stable_delta_error=stable_error,severe_cases=cases,pool_gain_parent_cases=gains,
               frozen_B_distribution_transfer=True,unknown_delta_zero_is_abstention_target_not_true_IoU=True,code={str(Path(__file__).resolve()):run.sha(__file__)})
    run.seal(out/'interpretation.json',extra)
    primary=s['arms']['B_relation'];status='本版区域关系选择取得真实删除池增量，但整体几何验收未通过。' if primary['pool_gain02'] and not primary['geometry_passed'] else '本版区域关系选择通过预注册几何门。' if primary['geometry_passed'] else '本版区域关系选择没有取得至少0.02的删除池级增量，假设尚未得到支持。'
    lines=['# 局部删除块：最终核验与判断','','**'+status+'**','',
           '复用原开发200物理五折、1599个N/T父、0.15米点域、GD10最多6帧、冻结逐点特征和竞争候选。纯删除、原NT全保留。20个内层生成器训练120/预测40场景，5个外层生成器训练160/预测40；10个块验证器只训练该外层四折的内交叉提议。原全局OOF B/C不作验证器训练数据。',
           '', '监督修正：全点域每点一次，父内污染CE代价上限3，Dice及好父保持在完整域计算；不稳定父逐点全未知、不强行追逐最佳GT的少量交集。1202确定身份、397未知身份、355好父保持。块每轮一次，父/场景等权、正例代价由训练数据决定且最多5。','',
           '|实际纯删除输出|动作|父提升≥.02|完整NT提升≥.02|池价值目标|污染删除|目标误删|已知删除精度|好父损失>.02 / >.10|漂移|通过|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for arm in run.ARMS:
        m=s['arms'][arm];precision='—' if m['known_delete_precision'] is None else f"{m['known_delete_precision']:.2%}"
        lines.append(f"|{arm}|{m['actions']}|{m.get('parent_improved_02',0)}|{m['pool_gain02']}|{m['joint_value_targets']}|{m.get('deleted_other',0)}|{m.get('deleted_target',0)}|{precision}|{m.get('good_degraded',0)} / {m.get('good_loss_gt_0.1',0)}|{m.get('drift',0)}|{m['geometry_passed']}|")
    lines+=['','B_raw使用封存B原始概率、两组可靠可见门生成纯删除（在加点之前），C_gate拆出封存C实际删除；最低100点都在最终纯删除后执行。S是普通七区域摘要MLP；R是块—保留邻域/父/竞争区域的显式关系网络。没有补点。',
            '', '**提议分布限制必须保留：** 为修正原过度采样/身份监督并避免堆叠泄漏，验证器用新G的严格内交叉提议训练，再分别测试G及封存B。B结果同时涉及这种提议分布迁移，不能单独把失败或成功归给区域网络能力。没有实现“相同B配方内交叉提议＋只改变块选择”的完全隔离对照。',
            '', '|外层折|内交叉块|确定身份块|安全正块|有块的父/场景|','|---|---:|---:|---:|---:|']
    for fold,m in training.items():lines.append(f"|{fold}|{m['blocks']}|{m['stable']}|{m['safe']}|{m['parents']} / {m['scenes']}|")
    lines+=['','原始折外提议规模：'+str(rawproposal)+'。纠正监督后是否仍有充足删除提议，必须和验证器效果一起看；不能把更少动作直接称为纠错进展。',
            '', '**正式训练内拟合**（40轮正式验证器，不是另训的小pilot）：']
    for kind,ff in fits.items():
        for fold,m in enumerate(ff):lines.append(f"- {kind}/fold{fold}：选{m['selected_blocks']}块，其中安全{m['selected_safe']}；删污染{m['selected_pollution']}、误删目标{m['selected_target_loss']}，已知精度{m['known_precision']}。")
    lines+=['','训练集表现不用于选轮/门；门固定p(safe)>=.9。全部后果监督针对模型实际提议，而不是父候选理想可修状态。未知父的0增量回归是弃权目标，不是已知实际IoU，后置稳定身份的真实delta误差另见interpretation/stable_delta_error。块级选中统计在最低点数回退之前，实际执行以点集CSV为准。',
            '', '**GT纯块机会（后置诊断）**：'+str(s['oracle'])+'。known-other筛查复用历史最佳父GT身份；stable-only仅确定身份，与新监督一致。它们不进入学习输出，也不是最优oracle。',
            '', '|严格IoU覆盖>|原完整NT|'+'|'.join(run.ARMS)+'|','|---|'+'---:|'*(len(run.ARMS)+1)]
    for t in (.5,.75,.78,.9,.95,.98):lines.append('|'+str(t)+'|'+str(s['baseline_coverage'][str(t)])+'|'+'|'.join(str(s['arms'][a]['coverage_'+str(t)]) for a in run.ARMS)+'|')
    lines+=['','父退化/好父尾部、未知/范围外/混合、小实例、N/T与每折分别见summary/breakdown和evaluation/candidates.csv。池增量、父改善必须同时看；原NT保留不保证未来SL K800保护。没有自身语义、SL上下文重算或AP，当前结果不能作为最终实例AP。',
            '', '区域关系参考核对[SSTNet第4.4节](https://arxiv.org/pdf/2108.07478)：论文CliqueNet用图交互修剪提议叶区域。本R只学块与各参照区的成对关系并聚合，没有其语义树和完整三层CliqueNet，不声称整套复现。',
            '', '核验：19相关测试，25提议模型+10验证器的原物理嵌套折、标签/诊断访问拒绝、1599封存B概率、全部11193实际点集与32823完整池目标重放，以及每折CPU生成器/验证器重放。原SL、前序源码/模型/预测哈希保持。',
            '', '首次入口只因源标签哈希被训练访问门误拦在训练前停止；execution_amendment_v2把完整性哈希放到折门前、许可当前外层的块监督写入。原合同、代码快照、失败日志保留，损失/拆块/门不变。',
            '', '如果缺少真实池级收益，不继续叠加接受开关。本版结果用于定位训练提议覆盖、动作归属特征与区域划分问题；开发200上游已暴露，仅新模块物理OOF，不能概括为几何没有空间。']
    (out/'audited_report.md').write_text('\n'.join(lines)+'\n')
    run.seal(out/'interpretation_verified.json',dict(complete=True,outputs={str(out/n):run.sha(out/n) for n in ('interpretation.json','audited_report.md')},summary_sha256=run.sha(out/'summary.json'),completion_sha256=run.sha(out/'completion_verified.json')))


if __name__=='__main__':summarize()
