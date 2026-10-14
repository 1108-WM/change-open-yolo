#!/usr/bin/env python3
"""Report-only replay of sealed relation metrics, calibration and input identity."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tools import run_fi1_decoupled_region as run


def metrics(frame, constant):
    truth=frame[['same_truth','different_truth','unknown_truth']].to_numpy(dtype=np.float32)
    prob=frame[['same_probability','different_probability','unknown_probability']].to_numpy(dtype=np.float32)
    result=dict(pairs=len(frame),MAE=float(np.abs(truth-prob).mean()),constant_MAE=float(np.abs(truth-constant).mean()),cross_entropy=float(-(truth*np.log(np.clip(prob,1e-8,1))).sum(1).mean()),constant_cross_entropy=float(-(truth*np.log(np.clip(constant,1e-8,1))).sum(1).mean()),classes={})
    for i,name in enumerate(('same','different','unknown')):
        pure=truth[:,i]>=.95;chosen=prob[:,i]>=.9
        result['classes'][name]=dict(PR_average_precision=float(average_precision_score(pure,prob[:,i])) if pure.any() else None,high_confidence=int(chosen.sum()),precision=float(pure[chosen].mean()) if chosen.any() else None,soft_GT_agreement=float(truth[chosen,i].mean()) if chosen.any() else None,parents=len(frame.loc[chosen,['scene','candidate_id']].drop_duplicates()),scenes=frame.loc[chosen,'scene'].nunique())
    return result


def main():
    c=run.read(run.OUT/'contract.json');assert run.read(run.OUT/'completion_verified.json')['complete']
    run.integrity(c['inputs']);run.integrity(c['source_bindings']);run.integrity(c['code']);run.integrity(c['frozen_SL_models'])
    stages=['contexts_frozen','labels_train_frozen','models_frozen','probabilities_frozen','rules_frozen','predictions_frozen','evaluation_frozen','completion_verified']
    times={name:(run.OUT/(name+'.json')).stat().st_mtime for name in stages}
    assert list(times.values())==sorted(times.values())
    assert {p.name for p in (run.OUT/'labels').iterdir()}==set(c['splits']['train'])
    # Explicit cached-label checks supplement the raw-GT stage checks.
    cache_checks=0
    for mode in ('contexts','predict','train','apply','labels_train','calibrate'):
        guard=run.Guard(c,mode)
        for split in c['splits']:
            scene=c['splits'][split][0];allowed=(mode in ('train','labels_train') and split=='train') or (mode=='calibrate' and split=='calibration')
            try:guard.hook('open',(str(run.OUT/'labels'/scene/'data.npz'),))
            except PermissionError:assert not allowed;cache_checks+=1
            else:assert allowed
    frame=pd.read_csv(run.OUT/'evaluation/relations.csv');train=frame[frame.split=='train']
    constant=np.ascontiguousarray(train[['same_truth','different_truth','unknown_truth']].to_numpy(dtype=np.float32)).mean(0)
    summary=run.read(run.OUT/'summary.json');breakdown={};checks=0
    for split in c['splits']:
        part=frame[frame.split==split];value=metrics(part,constant);expected=summary['splits'][split]['relations']
        for key in ('MAE','constant_MAE','cross_entropy','constant_cross_entropy'):
            assert np.isclose(value[key],expected[key],atol=2e-6);checks+=1
        assert value['pairs']==expected['valid_pairs']
        for name,v in value['classes'].items():
            old=expected['classes'][name]
            assert v['high_confidence']==old['predicted_high_confidence']
            for key in ('precision','PR_average_precision'):
                target=old['pure_precision'] if key=='precision' else old[key]
                assert v[key] is None and target is None or np.isclose(v[key],target,atol=2e-6);checks+=1
        breakdown[split]=dict(all=value,reference={name:metrics(part[part.reference==name],constant) for name in part.reference.unique()},parent_scope={name:metrics(part[part.parent_scope==name],constant) for name in part.parent_scope.unique()})
    calibration=run.read(run.OUT/'calibration.json');rejections=Counter();supported=[]
    for row in calibration['grid']:
        m=row['metrics'];fail=[]
        if m.get('actions',0)<3 or m['action_scenes']<2:fail.append('support')
        if m['precision'] is None or m['precision']<.9:fail.append('precision')
        if m['target_loss']>.01:fail.append('target_loss')
        if m['unknown_fraction']>.1:fail.append('unknown')
        if m['good_rate']>.02:fail.append('good_degradation')
        if m.get('drift',0):fail.append('drift')
        assert row['qualified']==(not fail);rejections.update(fail)
        if 'support' not in fail:supported.append(row)
    interpretation=dict(complete=True,metric_checks=checks,cached_label_access_rejections=cache_checks,raw_input_hashes=len(c['inputs']),stage_times=times,relation_breakdown=breakdown,calibration_rejections=dict(rejections),supported_calibration=supported,parameters={name:run.read(run.OUT/'models'/name/'history.json')['parameters'] for name in ('relations','parent')},old_A_unchanged=True,new_rules_searched=False)
    run.seal(run.OUT/'interpretation.json',interpretation)
    lines=[(run.OUT/'report.md').read_text().rstrip(),'','## 封存后补充核对与解释','',
           '本补充仅复算封存CSV和既定校准，不生成新预测、不搜索规则、不改任何门限。训练缓存只含112训练场景，阶段文件时间顺序及全部原始输入哈希一致；原A最终点集保持。','',
           f"独立关系指标复算{checks}项，缓存标签访问拒绝{cache_checks}项，输入哈希{len(c['inputs'])}项。两套模型参数：{interpretation['parameters']}。",'',
           '校准拒绝原因（同一规则可多项失败）：'+str(dict(rejections))+'。','',
           '|校准规则(edit/relation)|动作/场景|正确污染/目标误删/未知|保守精度|目标损失|','|---|---:|---:|---:|---:|']
    for row in supported:
        rule=row['rule'];m=row['metrics'];lines.append(f"|{rule['edit']}/{rule['relation']}|{m['actions']}/{m['action_scenes']}|{m.get('other',0)}/{m.get('target',0)}/{m.get('unknown',0)}|{m['precision']:.4%}|{m['target_loss']:.4%}|")
    lines+=['','以上为被拒绝的校准动作，不能称正式输出或留出收益；最终锁定规则仍以rules_frozen为准。','',
            '|内部留出参考|same PR / ≥.9精度(数量)|different PR / ≥.9精度(数量)|','|---|---|---|']
    for name,value in breakdown['holdout']['reference'].items():
        a=value['classes']['same'];b=value['classes']['different'];lines.append(f"|{name}|{a['PR_average_precision']} / {a['precision']} ({a['high_confidence']})|{b['PR_average_precision']} / {b['precision']} ({b['high_confidence']})|")
    lines+=['','完整分参考/父范围的关系指标见interpretation.json。区域与保留参考可能重叠，统计的关系对不是独立物体；范围外已标注身份参与关系学习不代表允许编辑范围外父。缺测参考不进关系损失。','',
            '新关系任务与旧A/C纯污染分类标签不同，不能直接比较两种PR当作同一任务提升。结论应分别依据关系高精度门、独立父判断、锁定联合动作的池价值与损害。该内部留出此前已详细诊断，本次不是新的盲测，也不是完整五折。']
    (run.OUT/'audited_report.md').write_text('\n'.join(lines)+'\n')
    run.seal(run.OUT/'supplemental_verified.json',dict(complete=True,code_sha256=run.sha(Path(__file__)),report_sha256=run.sha(run.OUT/'audited_report.md'),interpretation_sha256=run.sha(run.OUT/'interpretation.json'),completion_sha256=run.sha(run.OUT/'completion_verified.json'),new_training=False,new_thresholds=False))
    print('report-only supplement verified',checks,'metrics,',len(c['inputs']),'inputs',flush=True)


if __name__=='__main__':main()
