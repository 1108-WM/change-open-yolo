#!/usr/bin/env python3
"""Explain sealed outcomes; never select new holdout rules or train models."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from tools import run_fi1_region_affiliation as run


def summarize():
    assert run.read(run.OUT/'completion_verified.json')['complete'];summary=run.read(run.OUT/'summary.json');diagnosis={}
    for arm in run.ARMS:
        grid=run.read(run.OUT/'calibration'/f'{arm}.json')['grid'];passing=Counter()
        for row in grid:
            m=row['metrics']
            for name,condition in (('support',m.get('actions',0)>=3 and m['action_scenes']>=2),('precision',m['precision'] is not None and m['precision']>=.9),('target_loss',m['target_loss']<=.01),('unknown',m['unknown_fraction']<=.1),('good',m['good_rate']<=.02),('drift',m.get('drift',0)==0)):passing[name]+=condition
        nonempty=[r for r in grid if r['metrics'].get('deleted',0)>0]
        best=max(nonempty,key=lambda r:r['metrics']['precision']) if nonempty else None
        strict=next(r for r in grid if r['rule']==dict(safe=.98,target=.02,unknown=.02,union_risk=.005,abstain=False))
        history=run.read(run.OUT/'models'/arm/'history.json')
        diagnosis[arm]=dict(grid_rules=len(grid),nonempty_rules=len(nonempty),passing_counts=dict(passing),highest_precision_rejected_calibration_rule=best,strictest_preregistered_rule=strict,model_parameters=history['parameters'],updates=sum(r['updates'] for r in history['epochs']))
    interpret=dict(complete=True,calibration_failure=diagnosis,no_new_rules=True,no_new_actions=True,no_training=True,
                   conclusion='Training fit strong, cross-scene probabilities unsafe; C improves holdout safe ranking but worsens ownership error and calibration ranking. Both calibrations abstain, no joint geometric benefit proven.',
                   caveat='Unknown deletion is not proven wrong deletion; it cannot be certified as known pollution. Zero final damage comes from zero actions, not successful repair. Fixed single split, not full OOF/blind test.')
    run.seal(run.OUT/'interpretation.json',interpret)
    lines=[(run.OUT/'report.md').read_text().rstrip(),'','## 决策解释：训练可拟合，跨场景安全性没有成立','',
           '**本次主测试未证明安全联合删除，也没有得到C的实际几何收益。** 两组各60052参数、24轮、6600次更新，训练内纯污染PR平均精度A99.64%/C98.97%，归属MAE .0123/.0149，显著优于训练常数.2805；不能把本次失败解释为只有几十次更新、训练样本仍只有小量。','',
           '内部留出A/C纯污染PR平均精度6.33%/11.38%，先验1.33%，有排序信号，但训练到跨场景落差很大。归属MAE A .1809、C .2042，均胜过常数.2606；归属交叉熵却为1.6769/2.0060，均差于常数.6065，安全BCE同样差于常数。这组误差组合与错误预测过度自信一致；不能仅凭MAE或排序改善称概率可靠。C在校准集的安全排序反而比A低10.21%→5.88%，不能把留出11.38%当稳定额外贡献。','',
           '### 校准为什么全部拒绝','', '|组|有动作且支持足够的规则|删除精度达标规则|未知占比达标规则|被拒规则最高精度|对应污染/目标/未知点|','|---|---:|---:|---:|---:|---:|']
    for arm,d in diagnosis.items():
        m=d['highest_precision_rejected_calibration_rule']['metrics'];lines.append(f"|{arm}|{d['passing_counts']['support']}/{d['grid_rules']}|{d['passing_counts']['precision']}|{d['passing_counts']['unknown']}|{m['precision']:.1%}|{m.get('other',0)} / {m.get('target',0)} / {m.get('unknown',0)}|")
    lines+=['','所有162条预注册规则都有至少3动作/2场景；全部失败的是删除精度与未知占比，不是动作支持不足。表中是**校准集被拒规则的诊断**，不是正式预测、最优新方法或留出调参。最高精度A的已知子集36/(36+9)=80%，C为18/(18+2)=90%；未知点不能称全错，也不能按正确污染计入安全性。两组分别至少25%/77.78%未知，均越过预注册10%上限，不能通过删去未知评测项改变结论。','',
            '最终A/C均锁定abstain，训练/校准/留出真正动作全部0；正确污染删除、目标误删、未知删除、好父退化、漂移、池级≥.02与六档新增覆盖全部0。**零损害由没有执行造成，不能视作修复成功或C减少了误删。**','',
            '### 机会与证据限制','',
            '同一固定96区域，GT纯污染区联合的后置池增量是训练19、校准2、留出3目标。留出仍有3个可行机会，只是此表示/训练/执行配方没有兑现；这是GT诊断，不保证模型达到，也不用于选择区域或门限。','',
            '留出有>=2帧可见且无歧义父匹配的区域仅20.81%，训练25.74%；按固定匹配评分定义的父/邻居歧义，留出61.54%/56.83%，训练52.61%/48.80%。匹配歧义不等于GT身份错误。当前C是有限覆盖的局部掩码关系实现，不能从这一个分割否定所有GVC局部证据。','',
            '下一步优先检查跨场景的目标/邻居/身份未知表示及概率可靠性，而不是凭本次训练内拟合直接扩大五折。已有96区作为基线保留；区域预算/去重补位可以在后续独立对照，不用留出GT重选规则、不自动加接受开关、不补新帧或接SL。是否更强空间表示/观测有必要，仍需后续固定实验检验。','',
            '核验：8测试通过；100742同源区域、2558真实点集（1279父×2组）、7672池目标（3836目标×2）、2304次CPU区域概率重放，最大GPU/CPU差3.55e-6；13200实际优化更新，16个GT访问拒绝检查，原输入/银行/SL哈希保留。完整源码、模型、概率、校准全部网格与封存回执均可复核。']
    (run.OUT/'report_v2.md').write_text('\n'.join(lines)+'\n')
    run.seal(run.OUT/'interpretation_verified.json',dict(complete=True,source_completion_sha256=run.sha(run.OUT/'completion_verified.json'),source_summary_sha256=run.sha(run.OUT/'summary.json'),reporting_code_sha256=run.sha(Path(__file__)),outputs={str(run.OUT/n):run.sha(run.OUT/n) for n in ('interpretation.json','report_v2.md')}))
    print('interpretation_complete',flush=True)


if __name__=='__main__':summarize()
