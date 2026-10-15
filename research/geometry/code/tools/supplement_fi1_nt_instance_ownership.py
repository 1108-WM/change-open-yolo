#!/usr/bin/env python3
"""Keep oracle populations distinct; append transparent posthoc reporting supplement."""
from pathlib import Path
from collections import Counter
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_nt_instance_ownership as run
D=run.D;OUT=run.OUT


def main():
    c=D.read(OUT/'contract.json')
    D.verify(D.read(OUT/'completion_verified.json')['outputs'])
    D.verify(D.read(OUT/'failure_diagnosis_frozen.json')['outputs'])
    rows=[];previous={};allblocks=[];allunions=[]
    for scene in c['scenes']:
        previous.update({(scene,b['block_id']):b for b in D.read(run.BASE/'evaluation'/scene/'blocks.json')})
        allblocks+=D.read(OUT/'evaluation'/scene/'blocks.json')
        allunions+=D.read(OUT/'evaluation'/scene/'unions.json')
    for b in allblocks:
        old=previous[b['scene'],b['block_id']]
        if old.get('valuable_point_contributor'):
            r=dict(b,valuable_pure_union_contributor=old.get('valuable_pure_contributor',False),
                pure_block_in_point_oracle_opportunity=old['block_GT']['ownership']=='target_only')
            rows.append(r)
    pure=[b for b in rows if b['valuable_pure_union_contributor']]
    pure_in_point=[b for b in rows if b['pure_block_in_point_oracle_opportunity']]
    def counts(rs):
        return dict(blocks=len(rs),scenes=len({r['scene'] for r in rs}),
            states=dict(Counter(r['state'] for r in rs)),reasons=dict(Counter(r['reason'] for r in rs)),
            accepted_target=sum(r['clipped_GT']['target'] for r in rs if r['state']=='accept'),
            nonaccepted_target=sum(r['clipped_GT']['target'] for r in rs if r['state']!='accept'))
    native_failure=D.read(OUT/'failure_diagnosis/valuable_blocks.json')
    stable=[r for r in allunions if r['arm']=='relation_C' and r['stable'] and r['changed']]
    r=D.read(OUT/'summary.json')['modes']['relation_C']
    errata=dict(
        reason='Initial posthoc failure summary used pure_oracle_contribution>0 inside the 32 point-opportunity blocks, yielding27 pure blocks, while its prose called these the21 contributors from valuable pure-whole-block unions. Distinguish the populations. Prediction, evaluation, rule and decisions are unchanged.',
        point_opportunity_contributors=counts(rows),
        pure_blocks_inside_point_opportunity_parents=counts(pure_in_point),
        valuable_pure_union_contributors=counts(pure),
        wrong_output_pointsets=False,wrong_evaluation_counts=False,new_rule=False,
        original_summary_path=str(OUT/'failure_diagnosis/summary.json'))
    dest=OUT/'failure_diagnosis_v2';dest.mkdir()
    D.seal(dest/'summary.json',errata)
    D.seal(dest/'valuable_pure_union_blocks.json',pure)
    lines=[]
    for name,rr in [('点级有价值union贡献块',rows),('其中纯目标块（父的点级oracle有价值）',pure_in_point),('整块有价值union贡献的纯块',pure)]:
        cc=counts(rr);lines.append(f"| {name} | {cc['blocks']} | {cc['scenes']} | {cc['states']} |")
    text=f'''# 当前失败定位补充（有效版）

本文件区分两种oracle机会块集合；前版failure_diagnosis/report.md将27块称为21块的文字错误已明确披露，前版封存不修改。完整预测、GT评估、三态规则和停止判定均不变。主报告report.md的全量数值不受影响。

| GT后诊断集合 | 块数 | 物理场景数 | 实际固定原型输出 |
|---|---:|---:|---|
{chr(10).join(lines)}

这些集合仅用于事后解释，不能拿来选择输出。完整207父及所有负例早已评估。

32个点级机会块中11块接受、21块无法确定，无拒绝；其中16块父缺合格参照、5块待补块缺合格参照。全部1026个unknown块中1009缺参照、17存在多组竞争。当前瓶颈主要是跨源参照覆盖不足，而不是竞争机制能清晰判定所有邻物。

32块中仅11块存在原N同时多数覆盖父与块，恰好都被接收。没有原始多数连接的关键碎片不能靠这项证书找回；这也不证明其他弱连接、逐帧来源或连续关系没有身份信息，当前原型没有使用它们作为接收依据。

scene0078/c004的9个接收块都是纯目标，C保留341目标点且无邻物，父IoU有所改善，但关键r11的177点缺合格参照被遗漏，实际联合输出相对完整NT仍为−0.020216。不能GT后补上这一块。scene0301两个原baseline有收益的父均无跨源参照，收益被全部过滤。scene0002/r1、scene0085、scene0369/c006、scene0688的困难机会仍多为unknown。

稳定父55个动作中，52个修补副本不如或等于已有最佳N，也不如或等于实际连接N；即使归属正确，也主要在复现已有池覆盖。剩余3个副本相对完整NT的最大增量仍不足.02；全池连续增量总和只有{r['same_parent_pool']['pool_delta_sum']:.9f}，没有任何预定gain02目标或阈值跨越。

原型的稳定新增点为10019目标、353邻物、73未知，精度95.92%（未知计入分母），并非完全安全。8个接收块仍带邻物。例：scene0199/c006接受209个全邻物点，连接N有5115目标与496邻物（8.84%污染），父副本IoU下降.029332。这说明即便N整体大部分正确，小污染也可给局部错误块一个完全错误的连接；仅检查“巨大的混合N”或整体N纯度不够。另有11个身份未定父产生动作，按父scope保留已知/未知分布而不强加GT目标。

本轮没有通过“保留新增池价值”这一必要条件。保护好副本与整体精度改善不能补偿它；不把几乎不操作判成成功。停止本轮证书与分组，不事后放松阈值或重组后覆盖结果。当前N/T一致性证书失败不等于所有实例关系无法学习，也不等于后续AP必无收益。

下一轮若研究观测，候选应从本轮全部unknown/冲突或疑似错误连接中按无GT方式先固定，包含负例和身份不明父；先追查原T每帧来源、深度可见性和自动mask关联，再判断是否确需补充能同时看清父与块的帧。该协议及任何新规则另行封存；本轮未新增帧、训练或运行AP。
'''
    (dest/'report.md').write_text(text)
    D.seal(OUT/'failure_diagnosis_v2_frozen.json',dict(complete=True,outputs={str(p):D.sha(p) for p in dest.iterdir()},
        code_sha256=D.sha(Path(__file__)),reporting_correction_only=True,predictions_changed=False))
    (OUT/'README.md').write_text('''# N/T归属原型：结果入口

本轮已实现三态归属原型并完成52场景测试。减少污染和好副本损害，但完整NT新增价值归零，未通过开发验收；停止本固定原型，未开展独立场景或AP。

- [主报告](report.md)：规则、预定验收、全量结果及关键机会
- [当前有效失败定位](failure_diagnosis_v2/report.md)：解释关键机会遗漏、连接N冗余、假连接；区分21/27两类oracle块集合
- [全量指标](summary.json)、[停止决策](decision.json)、[独立核验](verification.json)
- [GT前合同](contract.json)、[预测封存](predictions_frozen.json)、[GT后评估封存](evaluation_frozen.json)

失败定位v1中的27块是“点级机会父中贡献的纯块”，不是“纯整块union有价值的21块”。v2修正这一汇总文字，所有封存与原文件保留，预测、规则、GT评估及判定均不变化。

代码：run_fi1_nt_instance_ownership.py（prepare/generate/evaluate/report），fi1_nt_instance_ownership_core.py，verify_fi1_nt_instance_ownership.py，test_fi1_nt_instance_ownership.py。GT前快照在code_snapshot。后置定位代码diagnose_fi1_nt_instance_ownership_failures.py及supplement_fi1_nt_instance_ownership.py不新增点集或规则。

环境：.venvs/yoloe_gpu/bin/python，OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4。已封存输出禁止覆盖；核验可重新运行verify，从头复现须另建输出目录。未训练、未修改生产或项目阅读入口、未提交GitHub。
''')
    paths=[OUT/'README.md',OUT/'report.md',OUT/'completion_verified.json',OUT/'failure_diagnosis_v2_frozen.json',OUT/'verification.json',Path(__file__)]
    for f in ('predictions_frozen.json','evaluation_frozen.json'):
        D.verify(D.read(OUT/f)['outputs']);paths.append(OUT/f)
    D.seal(OUT/'delivery_verified.json',dict(complete=True,current_reports=[str(OUT/'report.md'),str(dest/'report.md')],
        outputs={str(p):D.sha(p) for p in paths},original_seals_unchanged=True,training=False,AP_run=False,github_push=False))
    print(errata,flush=True)


if __name__=='__main__':main()
