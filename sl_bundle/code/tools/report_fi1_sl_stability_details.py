#!/usr/bin/env python3
"""All-seed training/heldout diagnostics and inference usage, postseal only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_sl_replay_stability_development200 as run
from tools.fi1_frontend_protocol import freeze_json


def main():
    out=run.OUT;c=run.read(out/'contract.json');summary=run.read(out/'summary.json');allstats={}
    lines=['# 三种子训练与留出排序诊断','',
        '每套五折均固定20轮；留出统计复用原种子的同一批封存配对，避免改变诊断采样。训练配对随预定种子改变。统计只作诊断，不选checkpoint、模型或种子。','',
        '|种子|折|训练配对损失Δ|训练逆序率Δ(pp)|留出配对损失Δ|留出逆序率Δ(pp)|','|---|---|---:|---:|---:|---:|']
    for seed in c['seeds']:
        root=run.L if seed==c['seeds'][0] else run.arm(seed);dd=run.read(root/'learnability.json')['folds'];folds={}
        for f,d in dd.items():
            before=d['train']['baseline'];after=d['train']['residual'];hb=d['heldout_before'];ha=d['heldout_after']
            z=dict(training_loss=after['pair_loss']-before['pair_loss'],training_inversion=after['inversion']-before['inversion'],
                heldout_loss=ha['pair_loss']-hb['pair_loss'],heldout_inversion=ha['inversion']-hb['inversion'])
            folds[f]=z;lines.append(f"|{seed}|{f}|{z['training_loss']:+.6f}|{100*z['training_inversion']:+.6f}|{z['heldout_loss']:+.6f}|{100*z['heldout_inversion']:+.6f}|")
        means={k:float(np.mean([z[k] for z in folds.values()])) for k in next(iter(folds.values()))}
        passed=all(z['training_loss']<0 for z in folds.values()) and means['training_inversion']<0 and means['heldout_loss']<0 and means['heldout_inversion']<0 and sum(z['heldout_loss']<0 and z['heldout_inversion']<0 for z in folds.values())>=3
        allstats[seed]=dict(folds=folds,means=means,original_learnability_rule_passed=bool(passed))
    lines+=['', '重复候选代表是最高S成员；重复配对逆序变化衡量S代表顺序变化，不能单独证明实际重复抑制变好／变差。仍未证明去重能力。',
            '', '|种子|类别ΔAP vs S (pp)|去类ΔAP vs S (pp)|旧90双升|新110双升|语义比S低的折|去类比S低的折|','|---|---:|---:|---|---|---|---|']
    effects={}
    for seed in c['seeds']:
        ap=run.read(run.arm(seed)/'evaluation_summary.json')['AP'];main=ap['development200']
        z={v:{ref:100*(main[v]['SL']['ap']-main[v][ref]['ap']) for ref in ('S','C','R')} for v in run.original.VIEWS}
        groups={g:all(ap[g][v]['SL']['ap']>ap[g][v]['S']['ap'] for v in run.original.VIEWS) for g in ('historical90','new110')}
        bad={v:[f for f in range(5) if ap[f'fold{f}'][v]['SL']['ap']<ap[f'fold{f}'][v]['S']['ap']] for v in run.original.VIEWS}
        effects[seed]=dict(main_delta_pp=z,subgroups_both_above_S=groups,folds_below_S=bad)
        lines.append(f"|{seed}|{z['semantic']['S']:+.6f}|{z['class_agnostic']['S']:+.6f}|{groups['historical90']}|{groups['new110']}|{bad['semantic']}|{bad['class_agnostic']}|")
    lines+=['', '全200的每种子AP、均值和范围见report.md。不能按折均值代替全200 AP，也不将3种子范围称置信区间。固定S名单的代价仍是放弃L重选新增覆盖，本次重复训练没有取消这个策略。']
    (out/'training_stability_report.md').write_text('\n'.join(lines)+'\n')
    freeze_json(out/'training_stability.json',dict(complete=True,training_heldout=allstats,AP_effects=effects,no_model_selection=True))
    s=c['scenes'][0];b=c['bindings'][s];m=run.L/'models'/f"fold{c['folds'][s]}"
    usage=['# SL真实推理入口','',
        '无需封存分数、标签或GT。直接输入实际几何、自身语义证据和权重。例子仅用于现有留出场景；最终部署需改用final_validation_plan.md规定的一个单模型。','',
        '```bash','.venvs/yoloe_gpu/bin/python tools/fi1_sl_inference.py \\',
        f"  --geometry '{b['geometry']}' \\",f"  --points '{b['points']}' \\",f"  --evidence '{b['evidence']}' \\",
        f"  --point-count {b['point_count']} \\",f"  --b26 '{c['B_model']}' \\",
        f"  --l-model '{m/'model.pt'}' \\",f"  --normalization '{m/'normalization.npz'}' \\",
        '  --output .local/sl_single_scene_new_output','```','',
        '输出目录必须尚不存在，避免覆盖已有预测。生成semantic_SL.json、class_agnostic_SL.json及输入／输出哈希回执；分数排完后仍是S原先选出的800个身份，无补位。',
        '', '完整S资格池先算26维特征和B26分数；L只对选中成员提供实际分数。为精确重放保留原8192行批次形状，其余行零填充，非选中行不会输出。网络只有逐行linear/tanh，无批归一化或跨行算子，填充不形成竞争证据。',
        '', '运行环境固定现有.venvs/yoloe_gpu，torch CPU4线程；其他BLAS／CPU平台需按浮点容差核验，不声称任意硬件位级相同。当前200场景和独立CLI烟测均在既定环境精确复现。']
    (out/'inference_usage.md').write_text('\n'.join(usage)+'\n')


if __name__=='__main__':main()
