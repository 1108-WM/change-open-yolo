#!/usr/bin/env python3
"""Read sealed outputs only; preserve original training, rules and evaluation receipts."""
from pathlib import Path
import csv,json,sys,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from tools import run_fi1_mask3d_feature_ab as run

def report():
    out=run.OUT;c=run.read(out/'contract.json');summary=run.read(out/'summary.json');assert run.read(out/'completion_verified.json')['complete']
    verification=run.read(out/'independent_verified.json');rules=run.read(out/'rules_frozen.json')['chosen'];grid=run.read(out/'calibration.json')['grid']
    with (out/'evaluation/candidates.csv').open() as f:rows=list(csv.DictReader(f))
    with (out/'evaluation/targets.csv').open() as f:targets=list(csv.DictReader(f))
    sources={str(out/n):run.sha(out/n) for n in ('contract.json','models_frozen.json','probabilities_frozen.json','rules_frozen.json','predictions_frozen.json','evaluation_frozen.json','summary.json','report.md','completion_verified.json','independent_verified.json')}
    training={arm:run.read(out/'models'/arm/'history.json') for arm in ('A','B')};counts=verification['counts']
    timestamps={n:(out/(n+'.json')).stat().st_mtime_ns for n in ('contract','instance_features_frozen','models_frozen','probabilities_frozen','rules_frozen','predictions_frozen','evaluation_frozen')}
    assert list(timestamps.values())==sorted(timestamps.values())
    table=[];breakdowns={};metadata=[]
    for r in rows:
        if r['mode']=='accepted':metadata.append(dict(scene=r['scene'],candidate_id=r['candidate_id'],arm=r['arm'],mode=r['mode'],generation_reference_threshold=.8,execution_threshold=None if rules[r['arm']]['abstain'] else rules[r['arm']]['threshold'],abstain=rules[r['arm']]['abstain']))
    # Original CSV threshold is a generation-reference field on accepted rows; corrected execution metadata is separate.
    with (out/'accepted_execution_metadata.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metadata[0]));w.writeheader();w.writerows(metadata)
    for split in ('fit','selection','calibration','holdout'):
        for arm in ('A','B'):
            for mode in ('generated','accepted'):
                rr=[r for r in rows if r['split']==split and r['arm']==arm and r['mode']==mode]
                for label,selected in [('native',[r for r in rr if r['source']=='native']),('track',[r for r in rr if r['source']=='track'])]:
                    stable=[r for r in selected if r['stable']=='True'];good=[r for r in stable if float(r['parent_iou'])>=.9]
                    breakdowns[f'{split}/{arm}/{mode}/{label}']=dict(parents=len(selected),actions=sum(r['changed']=='True' for r in selected),parent_improved02=sum(float(r['parent_delta'])>=.02-1e-12 for r in stable),parent_degraded02=sum(float(r['parent_delta'])<-.02-1e-12 for r in stable),good_parents=len(good),good_loss02=sum(float(r['parent_delta'])<-.02-1e-12 for r in good),good_loss10=sum(float(r['parent_delta'])<-.1-1e-12 for r in good),worst_good_loss=max([max(0,-float(r['parent_delta'])) for r in good] or [0]),parent_delta_sum=sum(float(r['parent_delta']) for r in stable))
    h=summary['splits']['holdout'];a,b=h['A_generated'],h['B_generated']
    contrasts=dict(registered_generation_gate_passed=summary['B_generation_improved'],pool_gain02_A=a['pool_gain02'],pool_gain02_B=b['pool_gain02'],pool_values_A=a['pool_value_targets'],pool_values_B=b['pool_value_targets'],parent_mean_delta_A=a['parent_delta_sum']/a['stable_parents'],parent_mean_delta_B=b['parent_delta_sum']/b['stable_parents'],B_reduces_neighbor_additions=b.get('added_other',0)<a.get('added_other',0),B_reduces_good_parent_damage=b.get('good_degraded',0)<a.get('good_degraded',0),B_small_continuous_NT_gain=b['pool_delta_sum'],no_permanent_95_gate=True)
    m8={r['arm']:r for r in grid if r['threshold']==.8}
    lines=['# 冻结Mask3D实例特征A/B：有效核验报告','','本轮已完成。固定原N/T父、父+.15米、GD10六帧与112/24/24范围；112内部96拟合/16模型选择。两组192输入/64宽相同网络，只开关新增128维mask_features_head。原点增删、修正全域监督、初始化/优化/候选顺序相同；只训练A/B两套，不增加质量/资格/关系头。','',
      '**结论：实例特征减少误补和好父损害，但尚未兑现本轮要求的完整NT池几何价值。** B在内部留出没有>=.02池提升或跨预定档位的新覆盖；父改善数量下降，损害也下降，连续父IoU略提高。不能把这个取舍写成“所有生成质量均无改善”，也不能由它认定“只缺实例特征”的解释已成立。B未过校准并最终弃权；不叠CliqueNet、不接SL，本轮止于核验。',
      '', '## 接入与身份核验','',f"160场景、{counts['original_index_maps']}原始点的稀疏坐标/inverse map核对，掩码/logit/分数导出前后逐值一致。输入只用processed前9列，网络XYZ/RGB；没有读取GT/实例/原分段列。特征为mask_features_head128维而非候选摘要。原点仍独立输出，不绑定同体素决定。权重冻结。",'',
      '一致性针对同一GT-free无分段输入的导出前后调用；没有替换或声称复现历史使用原segs的N点集。保持原包装器raw_coordinates为归一化颜色的历史约定，本轮不同时修该接口。资产存在与导出一致性不是几何收益证据。',
      '', '## 模型选择与过拟合','', '|组|参数|选中轮|该轮训练/选择损失|第24轮训练/选择损失|优化更新|','|---|---:|---:|---:|---:|---:|']
    for arm,t in training.items():
        chosen=t['epochs'][t['selected_epoch']-1];last=t['epochs'][-1]
        lines.append(f"|{arm}|{t['parameters']}|{t['selected_epoch']}|{chosen['training_loss']:.6f}/{chosen['selection_loss']:.6f}|{last['training_loss']:.6f}/{last['selection_loss']:.6f}|{sum(r['steps'] for r in t['epochs'])}|")
    lines += ['','B内部选择损失更低，后期训练损失下降而选择损失升高；固定选择流程保留早期模型。分类/损失信号不能替代点集收益。96拟合与16选择按物理场景隔离，24校准仅执行设置、24留出仅冻结后检查；该留出已暴露，不称盲测。',
      '', '## 内部留出的生成与接收','', '|组/输出|动作|父改善/退化>=.02|NT>=.02/池价值|补目标/邻居/未知|删污染/目标/未知|好父>.02/>.10|漂移|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode in ('A_generated','B_generated','A_accepted','B_accepted'):
        m=h[mode];unknown_add=m.get('added',0)-m.get('added_target',0)-m.get('added_other',0);unknown_del=m.get('deleted',0)-m.get('deleted_other',0)-m.get('deleted_target',0)
        lines.append(f"|{mode}|{m['actions']}|{m.get('parent_improved_02',0)}/{m.get('parent_degraded_02',0)}|{m['pool_gain02']}/{m['pool_value_targets']}|{m.get('added_target',0)}/{m.get('added_other',0)}/{unknown_add}|{m.get('deleted_other',0)}/{m.get('deleted_target',0)}/{unknown_del}|{m.get('good_degraded',0)}/{m.get('good_loss_gt_0.1',0)}|{m.get('drift',0)}|")
    lines += ['',f"生成A/B连续父IoU增量和{a['parent_delta_sum']:.6f}/{b['parent_delta_sum']:.6f}，153稳定父的平均增量{contrasts['parent_mean_delta_A']:.6f}/{contrasts['parent_mean_delta_B']:.6f}。B连续NT增量和{b['pool_delta_sum']:.6f}，小于本轮.02或跨档价值标准；没有把它隐去或升级成通过。",'',
       f"A/B增点正确比例{a['added_target']/a['added']:.4%}/{b['added_target']/b['added']:.4%}；未知编辑{a['unknown_edits']}/{b['unknown_edits']}，其中稳定父标注未知{a['unknown_annotation_edits']}/{b['unknown_annotation_edits']}、身份不明{a['unknown_identity_edits']}/{b['unknown_identity_edits']}。好父67个，A退化2个/最坏{a['worst_good_loss']:.6f}，B0个；两者都没有留出删除动作。不能把没有执行删除理解为已学会安全去污染。",'',
       '|严格IoU>|原NT覆盖|A生成新增|B生成新增|A接收新增|B接收新增|','|---|---:|---:|---:|---:|---:|']
    for t in (.5,.75,.78,.9,.95,.98):lines.append(f"|{t}|{a['baseline_'+str(t)]}|{a['new_cross_'+str(t)]}|{b['new_cross_'+str(t)]}|{h['A_accepted']['new_cross_'+str(t)]}|{h['B_accepted']['new_cross_'+str(t)]}|")
    lines += ['', '## 校准为什么拒绝B','', 'A原规则.95与.98均满足保护，固定排序选.98；留出10动作/7场景、补401目标点且无确认损害，但仍无池价值，不算几何通过。A在拟合范围接收还有一例>.10严重退化，校准通过不是全范围安全保证。', '',
      f"B固定.8校准补5824正确目标/6480全部增点={5824/6480:.6%}，略低于预定90%；未知564点不是已确认误补。其.7亦未过增点比例，.9/.95另未过未知，.98只有不足3动作支持；没有按已查看留出放宽门。最终B全部弃权，零损害不能作为成功。",'',
      '## 分范围与后续决定','', '拟合、选择、原train112、校准、留出的四输出完整表保留在原report.md；源/旧90新110、连续收益、全部IoU档和96区诊断见summary/evaluation，本补充source_breakdown.json另独立统计N/T父损害。选择范围B有1个真实池价值、增量和.247302，留出未复现；不能挑该正例宣称跨场景有效。', '',
      '正式预注册B生成改善门未过：留出父净改善A22/B17，池价值均0。B有更干净补点和连续改善信号，故结论是未形成要求的池收益，并非实例特征完全没有信息。按本轮停止分支，不追加CliqueNet/质量头/阈值，也不自动扩大五折；下一问题应先核对实例先验与混合父目标条件能否提供有效身份信息。该检查只是后续建议，本次不启动下一实验。','',
      '## 本次核验边界与报告字段补正','',f"8项测试通过；{counts['actual_pointsets']}实际点集、{counts['full_NT_targets']}完整NT目标、{counts['supervision_point_checks']}训练/选择监督点、10校准规则与16 CPU模型回放通过，最大概率差{verification['CPU_max_probability_difference']:.8f}。权重/输入/源码快照/SL哈希及阶段先后核对。核验复算旧预测，不重训Mask3D或重选模型。",'',
      '原evaluation/candidates.csv的accepted行threshold字段记录统一.8生成参考值，真实执行一直来自rules_frozen（A.98/B弃权）。该展示标签不准确，已新增accepted_execution_metadata.csv明确execution_threshold与abstain；原CSV/规则/点集/封存回执保留，不静默改写。',
      '', '没有新帧/轨迹/SAM/DINO推理、CliqueNet、质量头、自身语义、SL或AP。已有Mask3D属于上游预训练，结果只称固定开发对照。']
    (out/'audited_report.md').write_text('\n'.join(lines)+'\n')
    run.seal(out/'source_breakdown.json',breakdowns);run.seal(out/'interpretation.json',dict(complete=True,contrasts=contrasts,stage_order_ns=timestamps,new_predictions=False,new_rules=False,new_training=False,accepted_metadata_correction_only=True))
    run.integrity(sources)
    files=[out/n for n in ('audited_report.md','source_breakdown.json','interpretation.json','accepted_execution_metadata.csv')];run.seal(out/'supplemental_verified.json',dict(complete=True,primary_completion_sha256=run.sha(out/'completion_verified.json'),source_outputs_unchanged=sources,outputs={str(p):run.sha(p) for p in files},new_training=False,new_rules=False,new_predictions=False,reporting_code_sha256=run.sha(Path(__file__))))

if __name__=='__main__':report()
