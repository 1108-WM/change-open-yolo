#!/usr/bin/env python3
"""Readable final interpretation of sealed ablation results, without new outputs."""
from pathlib import Path
from collections import Counter
import sys,csv
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from tools import run_fi1_local_edit_ablation as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def main():
    out=run.OUT;c=run.read(out/'summary.json');fit=run.read(out/'pilot_summary.json');segment=run.read(out/'pilot_segmentation_vs_gate.json')
    lines=['# 固定GD10训练消融：最终核验与决策','','**纠错平衡能增加删除，但未产生删除池级收益；本版显式身份与候选头相对B降低风险，却未保住全部补点收益。B/C均失败，不接SL。**','',
           '同1599非GT N/T父、0.15米域、冻结GD10最多6帧/DINO与原开发200物理五折。A复用封存预测，原A训练双观测臂；A0/B/C只训练GD10、同逐轮候选顺序/随机种子/12轮，故B-A0检验监督，C-B检验身份表示+候选门整体。A0/B为50179参数，C89103；15模型及10训练内pilot完成。','',
           '|臂|池增益≥.02目标|池价值目标|只删除池增益≥.02|删除召回|正确删污染|目标误删|邻居误加|好父损失>.02 / >.10|漂移|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a,m in c['arms'].items():lines.append(f"|{a}|{m['full_pool_gain02']}|{m['joint_value_targets']}|{m['delete_only_pool_gain02']}|{m['delete_recall']:.2%}|{m['deleted_other']}|{m['deleted_target']}|{m['added_other']}|{m['good_degraded']} / {m['good_loss_gt_0.1']}|{m['drift']}|")
    lines+=['','分母固定：可见确认父污染153126点、父外目标195153点；点数按候选累计。池价值包括>=.02或跨>.75/.90/.95；好父468个，尾部基于原IoU>=.90。只删/只补为已预测动作的后置拆解，没有按GT挑策略作为实际输出。','',
            'B相对A0：删除召回.07%→16.50%，目标误删8→53124、邻居误加2612→16288，好父>.02退化9→90。N尤其受损：800N中改善≥.02仅10、退化>.02达132；T改善290、退化26。B只补诊断有7池价值、实际增删6，删除还抵消了部分补点。只增强纠错权重并不能解决可靠归属。','',
            'C相对B：目标误删53124→3581、邻居误加16288→6163、>.10好父损失16→7、漂移3→0；但删除召回16.50%→1.10%，补点池价值7→4。C相对A/A0并未整体更安全：误删仍高，N改善6/退化16，>.10退化7比A的4和A0的1更多。不能将C对激进B的改善当作超过原基线。','',
            'A0满足旧版保守几何门，但只删除仍0，并保留一例>.50好父大退化；它是输入训练口径控制，不是本轮删除研究成功，也不因此启动SL。B失败在增删精度、目标误删、好父、漂移与未知；C仍失败在增删精度和好父，且未保住B补点池价值。全部版本未满足本轮删除首次池增量要求。','',
            '训练内拟合：B删除召回96.03%—97.61%，补点83.90%—98.89%；C不执行候选门的逐点删除95.87%—98.96%、补点95.14%—98.53%。C候选分类在每折20个pilot样本上20/20，但正式OOF六类准确率55.66%；执行候选门后pilot删除2.22%—97.06%。这是拟合到泛化及身份弃权的落差，不是局部网络连训练纠错都学不会。训练内数据只按训练GT层与hash选，不读本轮外折错误名单；40轮pilot不调参、不热启动，正式模型从头12轮。','',
            '范围监督分为edit776/keep426/outside_scope166/unannotated158/ambiguous71/small2，按本轮明确阈值对所有父分类，不能与旧审计只分析286未知父的143/140/3混算。点输出仍兼容原3类，C另学6类候选状态，后置GT分类不是推理拒绝名单。未知编辑A/A0/B/C=12834/4556/26622/6617；C仍有14个原父匹配未知的编辑，不能说新增unknown表示已解决。','',
            '严重案例：scene0340_00/c003，原父1.0，A→.23497，A0保持1.0，B删除2目标点→.98450，C补23邻居且删3目标→.82895；C错误p(edit)=.94824。scene0384_00/c004四臂为.21648/1.0/1.0/1.0，C p(edit)=.44507而弃权。floor scene0167_00/c006，A删4056，A0不动，B删10，C不动且预测范围外。C能缓解部分具体问题，但候选概率也会高置信错误。','',
            '|完整NT池覆盖严格>|原NT|A|A0|B|C|','|---|---:|---:|---:|---:|---:|']
    for t in (.5,.75,.78,.9,.95,.98):lines.append(f"|{t}|{c['baseline_coverage'][str(t)]}|"+'|'.join(str(c['arms'][a][f'coverage_{t}']) for a in run.ARMS)+'|')
    lines+=['','原NT保留故池级覆盖不会降低，父/好父损失仍真实存在。新几何已生成，但未做自身语义、SL预算或AP，因此不可当最终分割AP。B/C池级有少量真实补点增量，删污染可靠价值仍未证明。','',
            '下一步判断：保持现有观测；不继续简单提高删除损失权重。优先检查候选条件归属的泛化、混合父目标监督与正确保持（特别N），以及候选头如何预测具体建议动作的安全性。本C监督“身份稳定且理想可修”，不等于“当前模型提议的编辑会改善”，这个区别需在训练折内部生成并监督实际建议动作，再另行设计实验；C本轮同时改身份表示和弃权，不能细分二者的独立贡献。未通过的新版本不替换旧模型，不按外折错误调整门。','',
            '核验：11模块测试，15正式模型/10训练内pilot折身份与轮数，30GT/后置错误访问拒绝；4281非GT竞争成员绑定，6396点集动作、1599封存A、56268全NT/只删/只补目标重算及30CPU新模型重放一致。主评测JSON首次遇NumPy整数，evaluation_amendment_v2仅转Python标量；原源码/失败CSV/日志保留，训练预测/门/数值不改。','',
            '证据：contract/source_bindings、各阶段_frozen、evaluation_amendment_v2、models与predictions、summary及evaluation CSV、pilot_summary/pilot_segmentation_vs_gate、independent_verified/completion_verified。A及原SL/生产全部保持；开发200上游已暴露，只是新头物理OOF。']
    (out/'audited_report.md').write_text('\n'.join(lines)+'\n')
    (out/'decision.md').write_text('# 本轮决策\n\nB/C均未通过；只删除池增量仍为0。不接SL/AP、不增加帧或替换原模型。平衡监督能增强删除但产生大量目标误删；本C风险下降伴随召回和补点价值损失，实例条件归属与建议动作安全性仍未解决。\n')
    freeze_json(out/'interpretation_verified.json',dict(complete=True,summary_sha256=sha(out/'summary.json'),code_sha256=sha(Path(__file__)),outputs={str(out/name):sha(out/name) for name in ('audited_report.md','decision.md','pilot_summary.json','pilot_segmentation_vs_gate.json')},no_new_actions_or_threshold_changes=True))
    print('ablation interpretation completed',flush=True)

if __name__=='__main__':main()
