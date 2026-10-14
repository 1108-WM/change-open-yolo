#!/usr/bin/env python3
"""Separate relation/generalization, parent states, and actual NT geometry."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import average_precision_score,f1_score,confusion_matrix
from tools import run_fi1_decoupled_region as run
from tools.evaluate_fi1_deletion_blocks import iou
from tools.evaluate_fi1_region_affiliation import save_csv,THRESHOLDS


def relation_metrics(samples,constant):
    truth=np.concatenate([r[2] for r in samples]);pr=np.concatenate([r[3] for r in samples]);valid=np.concatenate([r[4] for r in samples]);tt=truth[valid];pp=pr[valid]
    out=dict(valid_pairs=len(tt),MAE=float(np.abs(pp-tt).mean()),constant_MAE=float(np.abs(constant-tt).mean()),cross_entropy=float(-(tt*np.log(np.clip(pp,1e-8,1))).sum(1).mean()),constant_cross_entropy=float(-(tt*np.log(np.clip(constant,1e-8,1))).sum(1).mean()),classes={})
    for k,name in enumerate(('same','different','unknown')):
        pure=tt[:,k]>=.95;selected=pp[:,k]>=.9;parents=set();scenes=set()
        for s,cid,y,p,v in samples:
            if ((p[:,:,k]>=.9)&v).any():parents.add((s,cid));scenes.add(s)
        n=int(selected.sum());correct=int((selected&pure).sum());precision=correct/n if n else None
        out['classes'][name]=dict(pure_pairs=int(pure.sum()),prevalence=float(pure.mean()),PR_average_precision=float(average_precision_score(pure,pp[:,k])) if pure.any() else None,predicted_high_confidence=n,pure_correct_high_confidence=correct,pure_precision=precision,expected_GT_agreement=float(tt[selected,k].mean()) if n else None,parents=len(parents),scenes=len(scenes),high_precision_passed=bool(precision is not None and precision>=.95 and n>=100 and len(parents)>=10 and len(scenes)>=5))
    out['high_precision_instance_relation']=all(out['classes'][k]['high_precision_passed'] for k in ('same','different'))
    return out


def evaluate():
    c=run.read(run.OUT/'contract.json');sys.addaudithook(run.Guard(c,'evaluate').hook);run.integrity(run.read(run.OUT/'predictions_frozen.json')['outputs'])
    splits={s:k for k,ss in c['splits'].items() for s in ss};totals={k:{a:Counter() for a in run.ARMS} for k in c['splits']};relations={k:[] for k in c['splits']};parentdata={k:[] for k in c['splits']};actionscenes={k:{a:set() for a in run.ARMS} for k in c['splits']};oracle={k:Counter() for k in c['splits']};rows=[];targetrows=[];parentrows=[];pairrows=[]
    for ii,s in enumerate(c['scenes']):
        split=splits[s];labels,metarows,gt,ids,sizes=run.supervision(s,c);meta={r['candidate_id']:r for r in metarows};_,nt=run.legacy.original.audit.prior.old.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,iou(p,gt,ids,sizes))
        best={a:baseline.copy() for a in run.ARMS};ideal=baseline.copy();index={int(g):k for k,g in enumerate(ids)}
        with np.load(run.SOURCE/'features'/s/'data.npz') as z,np.load(run.OUT/'probabilities'/s/'data.npz') as pr,np.load(run.OUT/'predictions'/s/'data.npz') as out,np.load(run.REGIONS/'regions'/s/'regions.npz') as bank:
            actions={(r['candidate_id'],r['arm']):r for r in run.read(run.OUT/'predictions'/s/'actions.json')}
            for plan in run.read(run.OUT/'contexts'/s/'meta.json'):
                cid=plan['candidate_id'];m=meta[cid];parent=z[cid+'_parent'];halo=z[cid+'_halo'];valid=pr[cid+'_valid'];prob=pr[cid+'_relations'];truth=labels[cid+'_pair_truth'];relations[split].append((s,cid,truth,prob,valid));parentdata[split].append((m['parent_label'],pr[cid+'_parent']))
                parentrows.append(dict(scene=s,split=split,candidate_id=cid,source=plan['source'],state=m['parent_state'],prediction=run.core.PARENT_STATES[int(np.argmax(pr[cid+'_parent']))],**{name+'_probability':float(pr[cid+'_parent'][k]) for k,name in enumerate(run.core.PARENT_STATES)}))
                rr=bank[cid+'_budget96'];allblocks=run.legacy.prior.old.unpack(bank,cid);blocks=[allblocks[j] for j in rr]
                for j,rid in enumerate(rr):
                    for k,name in enumerate(run.core.REFERENCE_NAMES):
                        if not valid[j,k]:continue
                        pairrows.append(dict(scene=s,split=split,candidate_id=cid,region=int(rid),reference=name,parent_scope=m['scope'],same_truth=float(truth[j,k,0]),different_truth=float(truth[j,k,1]),unknown_truth=float(truth[j,k,2]),same_probability=float(prob[j,k,0]),different_probability=float(prob[j,k,1]),unknown_probability=float(prob[j,k,2])))
                pure=[b for b,safe in zip(blocks,labels[cid+'_safe']) if safe];ideal=np.maximum(ideal,iou(run.legacy.prior.core.remove(parent,halo,run.legacy.prior.core.union(pure)),gt,ids,sizes));before=iou(parent,gt,ids,sizes)
                for arm in run.ARMS:
                    q=out[cid+'_'+arm+'_points'];removed=np.setdiff1d(parent,q);after=iou(q,gt,ids,sizes);best[arm]=np.maximum(best[arm],after);changed=bool(len(removed));t=totals[split][arm];t.update(parents=1,actions=changed,deleted=len(removed),added=0)
                    if changed:actionscenes[split][arm].add(s)
                    record=dict(scene=s,split=split,candidate_id=cid,source=plan['source'],arm=arm,stable=m['stable'],identity_scope=m['scope'],parent_state=m['parent_state'],target=m['target'],deleted=len(removed),reason=actions[cid,arm]['reason'],selected_regions=len(actions[cid,arm]['selected']))
                    if m['stable']:
                        gid=m['target'];k=index[gid];target=int((gt[removed]==gid).sum());other=int(((gt[removed]>=1000)&(gt[removed]!=gid)).sum());unknown=len(removed)-target-other;delta=float(after[k]-before[k]);drift=bool(changed and after.max(initial=0)>0 and np.argmax(after)!=k and (gt[q]==ids[np.argmax(after)]).sum()>=20)
                        t.update(deleted_target=target,deleted_other=other,annotation_unknown_deleted=unknown,changed_parent_target=int((gt[parent]==gid).sum()) if changed else 0,parent_improved02=delta>=.02-1e-12,parent_degraded02=delta<-.02-1e-12,good_parents=before[k]>=.9,good_degraded02=before[k]>=.9 and delta<-.02-1e-12,good_degraded10=before[k]>=.9 and delta<-.1-1e-12,good_degraded25=before[k]>=.9 and delta<-.25-1e-12,drift=drift)
                        record.update(deleted_target=target,deleted_other=other,annotation_unknown_deleted=unknown,parent_iou=float(before[k]),repaired_iou=float(after[k]),parent_delta=delta,NT_best=float(baseline[k]),pool_delta=float(after[k]-baseline[k]),drift=drift)
                    else:t.update(identity_unknown_deleted=len(removed),identity_unknown_actions=changed);t[m['scope']+'_deleted']+=len(removed);record['identity_unknown_deleted']=len(removed)
                    rows.append(record)
        oracle[split]['GT_pure_union_gain02']+=int((ideal-baseline>=.02-1e-12).sum())
        for arm in run.ARMS:
            t=totals[split][arm];t.update(pool_gain02=int((best[arm]-baseline>=.02-1e-12).sum()),targets=len(ids))
            for threshold in THRESHOLDS:t['new_cross_'+str(threshold)]+=int(((baseline<=threshold)&(best[arm]>threshold)).sum())
            for k,g in enumerate(ids):targetrows.append(dict(scene=s,split=split,arm=arm,target=int(g),NT_best=float(baseline[k]),expanded_NT_best=float(best[arm][k]),delta=float(best[arm][k]-baseline[k]),GT_pure_union_best=float(ideal[k])))
        if (ii+1)%20==0:run.stage('post_seal_decoupled_evaluation',completed=ii+1,total=160)
    traintruth=np.concatenate([r[2][r[4]] for r in relations['train']]);constant=traintruth.mean(0);summary=dict(complete=True,splits={},formal_OOF=False,C_enabled=False,old_A_recalibrated=False,SL_modified=False,AP_run=False)
    for split in c['splits']:
        y=np.array([r[0] for r in parentdata[split]]);p=np.stack([r[1] for r in parentdata[split]]);selected=p[:,0]>=.9;positive=y==0
        parentfit=dict(parents=len(y),truth_counts=dict(Counter(y.tolist())),macro_F1=float(f1_score(y,p.argmax(1),labels=list(range(5)),average='macro',zero_division=0)),accuracy=float((p.argmax(1)==y).mean()),edit_PR_average_precision=float(average_precision_score(positive,p[:,0])) if positive.any() else None,edit_prior=float(positive.mean()),high_confidence_edit_predictions=int(selected.sum()),high_confidence_edit_precision=float(positive[selected].mean()) if selected.any() else None,confusion=confusion_matrix(y,p.argmax(1),labels=list(range(5))).tolist(),states=run.core.PARENT_STATES)
        summary['splits'][split]=dict(scenes=len(c['splits'][split]),relations=relation_metrics(relations[split],constant),parent=parentfit,oracle=dict(oracle[split]),arms={})
        for arm in run.ARMS:
            m=dict(totals[split][arm]);n=m['deleted'];target=m.get('deleted_target',0);other=m.get('deleted_other',0);unknown=m.get('annotation_unknown_deleted',0)+m.get('identity_unknown_deleted',0)
            m.update(action_scenes=len(actionscenes[split][arm]),unknown_deleted=unknown,conservative_delete_precision=other/n if n else None,known_delete_precision=other/(other+target) if other+target else None,target_loss=target/max(1,m.get('changed_parent_target',0)),unknown_fraction=unknown/n if n else None,good_degradation_rate=m.get('good_degraded02',0)/max(1,m.get('good_parents',0)))
            m['safe_joint_value']=bool(m['pool_gain02']>=1 and m['action_scenes']>=2 and m['conservative_delete_precision'] is not None and m['conservative_delete_precision']>=.9 and m['target_loss']<=.01 and m['unknown_fraction']<=.1 and m['good_degradation_rate']<=.02 and m.get('drift',0)==0);summary['splits'][split]['arms'][arm]=m
    hold=summary['splits']['holdout'];summary['stop_from_scratch_recipe']=not hold['relations']['high_precision_instance_relation']
    run.seal(run.OUT/'summary.json',summary);d=run.OUT/'evaluation';d.mkdir(exist_ok=True)
    for name,data in (('candidates.csv',rows),('targets.csv',targetrows),('parents.csv',parentrows),('relations.csv',pairrows)):save_csv(d/name,data)
    lines=['# 解耦实例关系与父可编辑性：固定96区独立版本','','原外层0训练范围160场景，复用112训练/24校准/24内部留出、同96区域/原六帧/冻结特征。留出此前已查看，不称盲测；本次模型训练仍不读校准/留出GT。A只复用封存最终点集，不再次校准；C局部特征没有启用。','',
           '关系模型独立学习完整区域与六个固定参考点集的same/different/unknown比例，包括范围外已标注GT身份；不因父不适合编辑就把实例关系全部改为unknown。父模型独立学edit/keep/out_of_scope/unannotated/ambiguous，不读取关系模型预测来训练。两套权重/损失/优化器分离。身份输入移除父成员/角色捷径，保持原点样本，加入对称关系与实际边界点对应；因此这是任务与对应表示的独立新版本，不能声称只改两个输出开关。','',
           '## 实例关系是否泛化','', '|范围|same PR / ≥.9纯关系精度与支持|different PR / ≥.9纯关系精度与支持|关系交叉熵/训练常数|高精度关系成立|','|---|---|---|---:|---|']
    for split in ('train','calibration','holdout'):
        r=summary['splits'][split]['relations'];ss=r['classes']['same'];dd=r['classes']['different'];lines.append(f"|{split}|{ss['PR_average_precision']} / {ss['pure_precision']} ({ss['predicted_high_confidence']})|{dd['PR_average_precision']} / {dd['pure_precision']} ({dd['predicted_high_confidence']})|{r['cross_entropy']:.5f} / {r['constant_cross_entropy']:.5f}|{r['high_precision_instance_relation']}|")
    lines+=['','PR是关系标签的平均精度，不是实例AP。高精度门在新合同预注册：same和different各自≥.9预测需GT纯关系精度≥.95，同时至少100对/10父/5场景。混合关系保留软比例，不能用粗argmax正确率掩盖高置信误判。区域/参考可重叠，关系对不是独立实例数量。','',
            '## 独立父判断','', '|范围|父状态macroF1|edit PR / 先验|≥.9编辑判断精度与数量|','|---|---:|---:|---|']
    for split in ('train','calibration','holdout'):
        p=summary['splits'][split]['parent'];lines.append(f"|{split}|{p['macro_F1']:.5f}|{p['edit_PR_average_precision']} / {p['edit_prior']:.5f}|{p['high_confidence_edit_precision']} ({p['high_confidence_edit_predictions']})|")
    lines+=['','edit标签只在训练/校准/后置评测由GT稳定身份及当前96可改范围的污染机会构造；推理依据独立父模型概率，不能直接用GT排除floor/混合父。五状态与混淆矩阵见summary/parents.csv，范围外不是“关系必未知”。','',
            '## 真正联合删除：封存A与解耦D','', '|范围|组|动作/场景|污染/目标/未知删除|父提升/退化≥.02|好父退化>.02/>.10|漂移|完整NT≥.02|安全联合价值|','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for split in ('train','calibration','holdout'):
        for arm in run.ARMS:
            m=summary['splits'][split]['arms'][arm];lines.append(f"|{split}|{arm}|{m['actions']} / {m['action_scenes']}|{m.get('deleted_other',0)} / {m.get('deleted_target',0)} / {m['unknown_deleted']}|{m.get('parent_improved02',0)} / {m.get('parent_degraded02',0)}|{m.get('good_degraded02',0)} / {m.get('good_degraded10',0)}|{m.get('drift',0)}|{m['pool_gain02']}|{m['safe_joint_value']}|")
    lines+=['','实际动作由新关系置信度和新父编辑概率两项门限决定，只在24校准场景选择16条预注册规则；不是继续搜旧A/C阈值。没有safe动作头、收益回归或C硬规则。点集并集删除、重叠只计一次，检查由关系门限导出的同父风险与>=100点保留，原完整NT保留。','',
            '|内部留出|新增>.5|>.75|>.78|>.90|>.95|>.98|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in run.ARMS:lines.append('|'+arm+'|'+'|'.join(str(hold['arms'][arm]['new_cross_'+str(t)]) for t in THRESHOLDS)+'|')
    lines+=['','锁定校准规则：'+str(run.read(run.OUT/'rules_frozen.json')['chosen'])+'。没有合格规则时弃权，零损害不能当成功。未知删除分开范围外/未标注/含糊父以及稳定父标注未知；不把未知全算错，也不按GT移出验收。','',
            '完整同96纯GT区域联合潜在池价值：'+str({s:v['oracle'] for s,v in summary['splits'].items()})+'；仅后置机会诊断，不参与选择。','',
            '停止当前从头训练配方：'+str(summary['stop_from_scratch_recipe'])+'。若跨场景高精度关系未成立，下一方法需要实例分割先验或提供新边界信息，并允许增删。若关系成立但几何失败，应分开定位父判断/联合执行/区域机会，不能否定关系学习，也不能直接接SL。两种情况本轮都不追加旧开关、改阈值或自动扩大五折。纯删除不是整个几何研究永久准入条件。前序补点确有少量池级收益但安全未过，不能说所有几何改动均无价值。','',
            '固定原SL/生产/轨迹，没有新帧/SAM/DINO/语义/SL/AP；新模型及原始概率、全部校准、实际点集、源码/合同/核验独立保存。']
    (run.OUT/'report.md').write_text('\n'.join(lines)+'\n');run.seal(run.OUT/'evaluation_frozen.json',dict(complete=True,outputs={str(p):run.sha(p) for p in [*d.iterdir(),run.OUT/'summary.json',run.OUT/'report.md']}));run.stage('decoupled_evaluation_complete',stop=summary['stop_from_scratch_recipe'],holdout_geometry=hold['arms'])


if __name__=='__main__':evaluate()
