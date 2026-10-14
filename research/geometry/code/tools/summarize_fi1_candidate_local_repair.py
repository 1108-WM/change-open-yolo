#!/usr/bin/env python3
"""Post-prediction aggregate replay and readable three-arm report; no new actions."""
from pathlib import Path
from collections import Counter
import csv,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_candidate_local_repair as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def summarize():
    out=run.OUT;c=run.read(out/'contract.json');sys.addaudithook(run.Guard(c,'evaluate').hook)
    summary=run.read(out/'summary.json')
    with (out/'evaluation/candidates.csv').open() as f:rows=list(csv.DictReader(f))
    with (out/'evaluation/targets.csv').open() as f:targets=list(csv.DictReader(f))
    replay={};value_rows=[]
    for arm in run.core.ARMS:
        counts=Counter();value=set();value_scenes=set();action_scenes=set();worst_good=0.
        for s in c['scenes']:
            gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64)
            with np.load(out/'features'/s/'data.npz') as z,np.load(out/'predictions'/s/'predictions.npz') as prediction:
                for r in (r for r in rows if r['scene']==s and r['arm']==arm):
                    cid=r['candidate_id'];parent=z[cid+'_parent'];p=prediction[cid+'_'+arm+'_points'];add=np.setdiff1d(p,parent);remove=np.setdiff1d(parent,p);edited=len(add)+len(remove)
                    counts.update(parents=1,actions=bool(edited),added=len(add),deleted=len(remove),edits=edited)
                    if edited:action_scenes.add(s)
                    if not r['parent_target']:
                        counts.update(unknown_identity_add=len(add),unknown_identity_delete=len(remove),unknown_identity_actions=bool(edited),unknown_edits=edited)
                        continue
                    gid=int(r['parent_target']);aa=int((gt[add]==gid).sum());ao=int(((gt[add]>=1000)&(gt[add]!=gid)).sum());ad=len(add)-aa-ao
                    dt=int((gt[remove]==gid).sum());do=int(((gt[remove]>=1000)&(gt[remove]!=gid)).sum());du=len(remove)-dt-do
                    counts.update(added_target=aa,added_other=ao,deleted_target=dt,deleted_other=do,unannotated_add=ad,unannotated_delete=du,unknown_edits=ad+du,changed_parent_target=int((gt[parent]==gid).sum()) if edited else 0)
                    delta=float(r['parent_delta']);old=float(r['parent_iou']);counts.update(parent_improved_02=delta>=.02-1e-12,parent_degraded_02=delta<-.02-1e-12,good_parents=old>=.9,good_degraded=old>=.9 and delta<-.02-1e-12,drift=r['drift']=='True')
                    if old>=.9:worst_good=min(worst_good,delta)
                    if r['joint_value']=='True':value.add((s,gid));value_scenes.add(s);value_rows.append(r)
        m=summary['arms'][arm]
        for key in ('parents','actions','added','deleted','edits','unknown_identity_actions','unknown_edits','added_target','added_other','deleted_target','deleted_other','changed_parent_target','parent_improved_02','parent_degraded_02','good_parents','good_degraded','drift'):assert counts[key]==m[key],(arm,key,counts[key],m[key])
        assert len(value)==m['joint_value_targets'] and len(value_scenes)==m['joint_value_scenes'] and len(action_scenes)==m['action_scenes']
        for threshold in (.50,.75,.78,.90,.95,.98):
            original=sum(float(r['original_NT_best'])>threshold for r in targets if r['arm']==arm);expanded=sum(float(r['expanded_NT_best'])>threshold for r in targets if r['arm']==arm)
            assert original==summary['baseline_coverage'][str(threshold)] and expanded==m[f'coverage_{threshold}']
        assert sum(float(r['delta'])>=.02-1e-12 for r in targets if r['arm']==arm)==m['pool_gain_02']
        extra=dict(counts);extra.update(known_assigned_add_precision=counts['added_target']/(counts['added_target']+counts['added_other']),known_assigned_delete_precision=counts['deleted_other']/(counts['deleted_other']+counts['deleted_target']),worst_good_delta=worst_good)
        replay[arm]=extra
    keys={(r['scene'],r['gt_id']) for r in targets};by={(r['scene'],r['gt_id'],r['arm']):r for r in targets}
    diffs=[float(by[s,g,run.core.ARMS[1]]['expanded_NT_best'])-float(by[s,g,run.core.ARMS[0]]['expanded_NT_best']) for s,g in keys]
    paired=dict(GD_better=sum(d>1e-12 for d in diffs),GD_worse=sum(d<-1e-12 for d in diffs),GD_gain02=sum(d>=.02-1e-12 for d in diffs),GD_loss02=sum(d<-.02-1e-12 for d in diffs))
    freeze_json(out/'aggregate_verified.json',dict(complete=True,arms=replay,paired=paired,verifier_code_sha256=sha(Path(__file__)),summary_sha256=sha(out/'summary.json'),no_threshold_or_prediction_changes=True))
    text=['# 真实局部增删：最终核验报告','','**模块产生了少量真实折外池级提升，但两臂均未通过预注册几何门；本轮不接自身语义/SL/AP。**','',
          '开发200原物理五折、每折160训练/40留出，1599非GT父候选（800N/799T），4689有效实例。原10场景80父/双臂选帧原样复用，190场景扩展同规则。每折一个共享双臂模型，冻结DINO；逐点64维输入、原点8邻接、50179参数，固定12轮，不读留出GT选帧/选模型/动作。9493已有图像帧/16833候选—帧绑定，不重跑SAM或轨迹。','',
          '|组别|实际动作|父提升≥.02|父退化>.02|超过全NT≥.02目标|池价值目标/场景|','|---|---:|---:|---:|---:|---:|','|原始NT|0|0|0|0|0|']
    for a,label in zip(run.core.ARMS,('旧30中最多6帧＋模块','完整GD10中最多6帧＋同模块')):
        m=summary['arms'][a];text.append(f"|{label}|{m['actions']}|{m['parent_improved_02']}|{m['parent_degraded_02']}|{m['pool_gain_02']}|{m['joint_value_targets']}/{m['joint_value_scenes']}|")
    text+=['','池价值＝候选改善父候选，同时超过全NT最佳≥.02或跨>.75/.90/.95。5/6价值目标全部在190扩展场景，全部来源于补点，未产生具有池价值的删除动作。GD10包含旧30的5个价值目标并新增1个；逐目标比较GD10优于旧30有9个、较差2个，≥.02额外改善2个。多数父改善已有更好的原NT，不能当池提升。','',
           '|全NT池覆盖（严格大于）|原始NT|旧30＋修复|GD10＋修复|','|---|---:|---:|---:|']
    for threshold in (.50,.75,.78,.90,.95,.98):text.append(f"|{threshold}|{summary['baseline_coverage'][str(threshold)]}|{summary['arms'][run.core.ARMS[0]][f'coverage_{threshold}']}|{summary['arms'][run.core.ARMS[1]][f'coverage_{threshold}']}|")
    text+=['','这是实际OOF预测点集追加完整NT后的几何覆盖，不是理想修复oracle，也不是预算排序后的最终AP。两臂各新增1个>.75和1个>.95，没有>.90新增。原NT全部保留故池覆盖不丢，父退化须单独审查。','',
           '|点级编辑分类（按候选累计）|旧30|GD10|','|---|---:|---:|']
    for key,label in [('added_target','补入正确目标'),('added_other','补入已知相邻实例（误加）'),('unannotated_add','父身份已知但补入标注未知点'),('unknown_identity_add','父身份无法后置确认的补入'),('deleted_other','删除确认相邻实例污染'),('deleted_target','删除目标（误删）'),('unannotated_delete','父身份已知但删除标注未知点'),('unknown_identity_delete','父身份无法后置确认的删除')]:text.append(f"|{label}|{replay[run.core.ARMS[0]][key]}|{replay[run.core.ARMS[1]][key]}|")
    text+=['','未知编辑不能直接称为正确或错误。预注册保守指标将它们保留在全部编辑分母：正确补点占比87.10%/84.07%，确认污染占全部删除91.90%/39.96%。只看已标注且父身份确认的点，补点精度93.81%/93.16%，删除精度99.61%/95.32%；后者不能替换冻结门，尤其GD10的4082删除点来自无法确认父身份的候选。','',
           '旧30原IoU≥.90候选468个中5个退化>.02，GD10为9个；最坏IoU损失分别.52349/.78352。GD10出现1次身份漂移：scene0340_00/c003，原IoU=1的N补入420个相邻实例点，变为.23497。旧30无漂移。未知编辑比例7.18%/13.28%，两臂均无原池/修复候选精确点集重复。','',
           '旧30失败在保守补入正确占比不足.90；GD10还失败在确认删除占比、未知编辑比例和身份漂移。好候选退化率虽满足预注册比例门，个别大幅退化仍说明尚不可靠。未降低阈值、追加GT挑动作、选择最佳轮或把未知当背景。','',
           '示例：scene0214_00目标29010，完整NT原最佳.725732，旧30修到.765393，GD10修到.766454；scene0045_00目标4007，原池.949915，两臂修到.961800/.959253，显示高IoU细化。GD10新增池价值scene0341_00/56058为.826446→.856198。','',
           '判断：已有六帧证据＋局部学习能产生真实增量，但目前主要补回T候选缺失，可靠污染删除的池级贡献没有证明，候选身份与相邻实例区分仍会失败。下一步应先改善候选条件身份与未知回退；当前结果不支持立即增加全帧观测、改SL或宣布几何无机会。','',
           '核验：6基础测试、5折物理分离、20访问拒绝、160原10选帧、3198增删重放、2626可靠父IoU、9378池目标、全部3198可见性/成员输入、20CPU模型概率及动作重放通过；CPU/GPU最大概率差.0004883且动作一致。9493帧/68203原bank行/43184下游行及RGB-D/pose哈希核对通过。','',
           '原v1的features命令分派错误在训练/GT前停止，execution_amendment_v2只补映射到原extract_features，保留run_v1_failed.log与原合同/代码。report.md的“几何上界”指尚未SL预算选择，不能将其误解为GT理想修复。本报告只解释既有封存结果，不改变预测或门。','',
           '资产：models/fold*/model.pt、features/*/data.npz、predictions/*/predictions.npz及decisions；evaluation/candidates.csv/targets.csv/scenes.json；summary与各阶段_frozen、independent_verified、model_replay_verified、observation_bindings_verified、aggregate_verified。开发200上游已暴露，结果仅为新局部头折外验证，不称全系统盲测。']
    (out/'audited_report.md').write_text('\n'.join(text)+'\n')
    print('aggregate replay and audited three-arm report complete',flush=True)

if __name__=='__main__':summarize()
