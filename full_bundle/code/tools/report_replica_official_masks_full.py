#!/usr/bin/env python3
"""Summarize completed official-mask replay without mutating experiment outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '.local/replica_official_masks_full_20261011_v1'


def read(path):
    return json.loads(path.read_text())


def main():
    assert read(OUT / 'execution.json')['complete']
    assert read(OUT / 'verified.json')['complete']
    assert read(OUT / 'agnostic_supplement_verified.json')['complete']
    r = read(OUT / 'evaluation/results.json')
    old = read(ROOT / '.local/replica_registered_full_20261011_v1/evaluation/results.json')
    keys = ['all_ap', 'all_ap_50%', 'all_ap_25%']
    names = {'baseline_official': 'OpenYOLO3D 官方掩码／uniform',
             'baseline_native': '同基线预测／native score', 'Full': 'GD10＋D-v3＋B26／Full／K800'}
    rounded = [round(r['baseline_official']['semantic'][k] * 100, 1) for k in keys]
    paper_match = rounded == [23.7, 28.6, 34.8]
    delta = {view: {k: 100 * (r['Full'][view][k] - r['baseline_official'][view][k]) for k in keys}
             for view in ['semantic', 'actual_class_agnostic']}
    verification = read(OUT / 'verified.json')
    lines = ['# Replica 官方掩码：OpenYOLO3D与登记Full重新评测', '', '日期：2026-10-11。全部8个场景、冻结48类词表及模型；没有训练或调参。旧本地重生成候选结果保留。', '',
             f"基线按论文的一位小数为 **{rounded[0]:.1f}／{rounded[1]:.1f}／{rounded[2]:.1f}**；" + ('与论文表2的23.7／28.6／34.8一致。' if paper_match else '尚未复现论文表2的23.7／28.6／34.8，不调整参数追数。'), '',
             f"Full相对本轮基线类别AP变化 **{delta['semantic']['all_ap']:+.6f}pp**，实际输出去类AP变化 **{delta['actual_class_agnostic']['all_ap']:+.6f}pp**。该结果仅代表冻结生产配置在Replica上的迁移，不自动替换生产或开发SL。", '',
             '## 主结果', '', '百分数；AP沿原Replica评测器使用IoU=.50:.05:.90，另报告AP50、AP25。', '',
             '|方法|类别AP|AP50|AP25|实际去类AP|去类AP50|去类AP25|', '|---|---:|---:|---:|---:|---:|---:|']
    for method in names:
        values = [r[method][view][k] * 100 for view in delta for k in keys]
        lines.append('|' + names[method] + '|' + '|'.join(f'{v:.6f}' for v in values) + '|')
    values = [delta[view][k] for view in delta for k in keys]
    lines += ['|Full−官方掩码基线（百分点）|' + '|'.join(f'{v:+.6f}' for v in values) + '|', '',
              '主基线采用官方上游run_evaluation.py的uniform置信度；native行只改变同一套预测的评测分数。去类合并仅针对实际已选有效成员的精确同点集，取已选最高分，不补位、不使用全池最高分。', '',
              '## 与上轮本地重生成候选的差别', '',
              '|方法|上轮类别AP|官方掩码类别AP|上轮去类AP|官方掩码去类AP|', '|---|---:|---:|---:|---:|']
    for method in ['baseline_official', 'Full']:
        values = [old[method]['semantic']['all_ap'] * 100, r[method]['semantic']['all_ap'] * 100,
                  old[method]['actual_class_agnostic']['all_ap'] * 100, r[method]['actual_class_agnostic']['all_ap'] * 100]
        label = 'OpenYOLO3D／uniform（分别使用两套候选）' if method == 'baseline_official' else names[method]
        lines.append('|' + label + '|' + '|'.join(f'{v:.6f}' for v in values) + '|')
    lines += ['', '本轮直接加载作者原生候选，不重新运行Mask3D或NMS，解决上轮候选生成输入类型／历史NMS差异。Full的原生关系、过滤、U/U′及完整池评分重新计算，不能只替换基线表格数值。', '',
              '## 逐场景结果', '', '|场景|基线类别AP|Full类别AP|基线去类AP|Full去类AP|', '|---|---:|---:|---:|---:|']
    per = {(method, view): read(OUT / 'evaluation' / method / (view + '_per_scene.json'))
           for method in ['baseline_official', 'Full'] for view in delta}
    for row in verification['scenes']:
        s = row['scene']
        values = [per[(method, view)][s]['all_ap'] * 100 for view in delta for method in ['baseline_official', 'Full']]
        lines.append('|' + s + '|' + '|'.join(f'{v:.6f}' for v in values) + '|')
    lines += ['', '逐场景AP不能简单平均成全局AP；全局按类别合并全部场景排序后宏平均。', '',
              '## 冻结身份与实际执行', '',
              '- 原生掩码来自作者官方Google Drive资产包；8个文件CRC、格式、完整点域通过，包内8个网格与本地逐字节一致。',
              '- [官方输入核验](official_input_identity_verified.json)确认4816个原始RGB-D／pose／网格／内参文件及2个基础模型权重（共4818个文件）均与作者包逐字节一致；不读取GT。',
              '- 基线执行作者上游原始赋类／可见视角／IoU函数，frequency1、可见差.4、top40、每几何top1。YOLO检测按完全匹配配置复用，不复用旧原生点集预测。',
              '- Full保持GD10＋历史official100 D-v3七模型＋匹配训练597 B26，NTUU′、K800；native-only资格与生产D-v3准入接口不改，不调用S、C或SL。',
              '- 复用只限与原生候选无关的超点、GD10前端及YOLO-SAM观测／检测；[复用回执](reused_inputs.json)记录11315文件，最终核对未改。原生关系、U/U′、实际新点集自身YOLO／Alpha、完整26维竞争上下文、B26／K800均重算。',
              '- 两方法全8场景预测封存后，才读取GT并评测；[预测封存](predictions_sealed.json)和[评测回执](evaluation/receipt.json)绑定身份。', '',
              '## 核验与复现', '',
              '[原生点集核验](native_domain_verified.json)：基线掩码及顺序与作者文件精确一致；Full原生成员只引用作者点集，没有混入上一轮重生成掩码。',
              f"[独立核验](verified.json)：{verification['input_hashes_checked']}输入／模型／科学源码哈希，完整点序、实际自身证据与资格、独立26维上下文、冻结B26评分和K800精确重放；54AP数组核对及类别缓存AP重放。",
              '[去类补充核验](agnostic_supplement_verified.json)：实际已选点集精确合并／最高分与封存匹配一致，27去类AP数组精确重放。4项必要适配测试通过。核验不重新读取原始GT。', '',
              '环境失败记录保留：初次score误用了D-v3的sklearn1.3.0环境，B26树节点dtype不兼容，未生成评分输出。`score_failed_sklearn130.log`／`execution_failed_sklearn130.json`保存失败；`tools/resume_replica_official_scoring.py`用B26原训练及上轮推理的sklearn1.3.2环境完成评分／评测／核验，不改模型、不重跑封存几何和语义。对应回执为B26_environment_resume.json。', '',
              '入口`tools/run_replica_official_masks_full.py`：隔离prepare／native／backend／evidence／score／evaluate／verify／chain；状态`execution.json`已完成，不自动重跑。旧生产源码、权重、数据和历史AP保留。完整逐类、逐场景结果见[evaluation/results.json](evaluation/results.json)及各方法CSV／JSON；协议见[protocol_notes.md](protocol_notes.md)。', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    write = lambda name, value: (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    write('comparison_summary.json', {'paper_baseline_rounded_match': paper_match, 'paper_table2': [23.7, 28.6, 34.8],
          'baseline_rounded': rounded, 'Full_minus_baseline_pp': delta,
          'dual_AP_improved': all(delta[view]['all_ap'] > 0 for view in delta), 'no_new_training': True})
    write('task_status.json', {'complete': True, 'running': False, 'stage': 'reported_verified',
          'paper_baseline_rounded_match': paper_match, 'results': str(OUT / 'evaluation/results.json'),
          'report': str(OUT / 'report.md'), 'previous_results_unchanged': True, 'no_new_training': True})
    files = [OUT / name for name in ['report.md', 'comparison_summary.json', 'task_status.json', 'verified.json',
             'agnostic_supplement_verified.json', 'official_input_identity_verified.json']]
    write('completion_receipt.json', {'complete': True, 'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    print(json.dumps(read(OUT / 'comparison_summary.json'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
