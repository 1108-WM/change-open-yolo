#!/usr/bin/env python3
"""Summarize sealed learnability, opportunity recall and union contamination."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from tools import run_fi1_fragment_affiliation_learnability as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha

def main():
    out=run.OUT;c=run.diag.read(out/'contract.json');result=run.diag.read(out/'results.json');g=run.diag.read(out/'geometry_supplement.json');assert run.diag.read(out/'independent_verified.json')['complete']
    lines=['# 折外归属与假设并集的实际几何价值','',
        '本轮配对模型只用于可学习性与几何动作诊断。所有输入来自推理可获得的点几何、颜色、法向、原始掩码与RGB／传感器深度；GT身份仅在训练折作监督，所有留出分数封存后检查几何收益。unknown未作为负例，也不从诊断中删除。','',
        '|模型|阈值|接受配对|可靠同实例|可靠异实例|未知|独有并集|>.50新增|>.75新增|>.90新增|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('summary','summary_plus_visual'):
        for th in ('0.5','0.9'):
            d=g['diagnostics'][name][th];new=d['new_over_complete_NT'];lines.append(f"|{name}|{th}|{d['accepted_pairs']}|{d['known_same']}|{d['known_different']}|{d['unknown']}|{d['unique_unions']}|{new['0.50']}|{new['0.75']}|{new['0.90']}|")
    lines+=['','“独有并集”在每个场景按准确点集去重，仍可能与原NT完全相同；精确NT重复单列在JSON中。这些是后置重建的假设点集，不是第二版补全预测。候选池覆盖新增不是可获得的AP增益。','',
        '|模型|阈值|可靠锚方向|新增目标点|新增污染点|多数新增是目标的方向|父覆盖获得／损失>.75|已覆盖目标重复>.75|','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('summary','summary_plus_visual'):
        for th in ('0.5','0.9'):
            d=g['diagnostics'][name][th];lines.append(f"|{name}|{th}|{d['reliable_anchor_directions']}|{d['added_target_points']}|{d['added_pollution_points']}|{d['target_majority_directions']}|{d['parent_improved_targets_0.75']} / {d['parent_lost_targets_0.75']}|{d['redundant_targets_0.75']}|")
    lines+=['','|配对子集|全部配对|未知|摘要配对AP|加入区域视觉配对AP|','|---|---:|---:|---:|---:|']
    for subset,v in g['subset_metrics'].items():
        a=v['model_metrics']['summary'];b=v['model_metrics']['summary_plus_visual'];lines.append(f"|{subset}|{v['all_pairs']}|{v['unknown']}|{a['pair_average_precision'] if a else None}|{b['pair_average_precision'] if b else None}|")
    opportunities=g['oracle60_scores'];examples=g['previous13_counterexample_scores'];scores={}
    lines+=['','|模型|阈值|已知60机会接受|其中26不交接受|旧13异实例反例仍接受|','|---|---:|---:|---:|---:|']
    for name in ('summary','summary_plus_visual'):
        scores[name]={}
        for th in (.5,.9):
            recall=sum(r['retrieved'] and r['scores'][name]>=th for r in opportunities);disjoint=sum(r['retrieved'] and r['pair_iou']==0 and r['scores'][name]>=th for r in opportunities);bad=sum(r['scores'][name]>=th for r in examples)
            scores[name][str(th)]=dict(known60_accepts=recall,disjoint26_accepts=disjoint,previous13_false_accepts=bad)
            lines.append(f'|{name}|{th}|{recall}|{disjoint}|{bad}|')
    lines+=['','这60个是此前GT挑选的机会，不代表全部可修复目标，且其中有些独有区域身份标签可能不可靠。旧13反例对应9个场景—实例对，仅作后置诊断，不作为类别例外或额外训练采样规则。','',
        '实测深度证据包括区域间投影路径上的传感器跳变、缺测以及候选点与传感器表面的遮挡残差；没有把小于5cm三维接触点的delta-z再包装成独立边界证据。投影点的邻域深度环是稀疏点区域邻域，并非精确对象轮廓。','',
        '冻结DINOv2全帧token以可见独有区域点投影池化。一个token仍含周围上下文，小邻物可落入相同token；局部外观相似不等于同实例。增强组同时加入DINO、区域RGB／法向／局部形状和实测深度，因此两组差值只能归因这组区域证据，不能单独称为DINO贡献。模型固定同容量、同种子，按训练场景等总权重，没有使用留出选checkpoint、阈值或容量。','',
        f"预注册几何价值门={result['geometry_value_gate']}；双折外指标改善={result['visual_improves_both']}；{result['folds_pair_AP_improved']}/5折配对average precision改善。判断是否继续几何关联必须结合新覆盖、未知质量和污染，不能只依靠配对分类准确率。即便门通过，本轮也不自动获取自身语义、接入SL或运行完整实例AP。",'',
        '开发200已经反复暴露，只称场景折外归属可学习性，不称整系统盲测或独立最终验收。固定SL及生产Full、开发55、历史模型和预测均未修改；没有同时裁剪或生成第二版输出。']
    (out/'supplemental_report.md').write_text('\n'.join(lines)+'\n')
    freeze_json(out/'opportunity_summary.json',dict(complete=True,scores=scores,new_output_generated=False,source_sha256=sha(__file__)))
    print(scores,flush=True)

if __name__=='__main__':main()
