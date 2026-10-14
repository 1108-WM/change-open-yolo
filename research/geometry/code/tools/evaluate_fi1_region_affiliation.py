#!/usr/bin/env python3
"""Post-seal union deletion geometry and same-source learnability diagnostics."""
from pathlib import Path
from collections import Counter
import csv,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from tools import run_fi1_region_affiliation as run
from tools.evaluate_fi1_deletion_blocks import iou
THRESHOLDS=(.5,.75,.78,.9,.95,.98)


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)


def evaluate():
    c=run.read(run.OUT/'contract.json');sys.addaudithook(run.Guard(c,'evaluate').hook);run.integrity(run.read(run.OUT/'predictions_frozen.json')['outputs'])
    split={s:k for k,ss in c['splits'].items() for s in ss};totals={k:{a:Counter() for a in run.ARMS} for k in c['splits']};basecoverage={k:Counter() for k in c['splits']};classification={k:{a:[] for a in run.ARMS} for k in c['splits']};breakdown={k:{a:{} for a in run.ARMS} for k in c['splits']}
    rows=[];targets=[];regions=[];oracle={k:Counter() for k in c['splits']};actionscenes={k:{a:set() for a in run.ARMS} for k in c['splits']}
    for i,s in enumerate(c['scenes']):
        scope=split[s];annotations,metarows,gt,ids,sizes=run.supervision(s,c);meta={r['candidate_id']:r for r in metarows}
        _,nt=run.original.audit.prior.old.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,iou(p,gt,ids,sizes))
        best={a:baseline.copy() for a in run.ARMS};ideal=baseline.copy();keys={p.astype('<i8').tobytes() for p in nt.values()};newkeys={a:set() for a in run.ARMS};idlookup={int(g):j for j,g in enumerate(ids)}
        with np.load(run.SOURCE/'features'/s/'data.npz') as z,np.load(run.REGIONS/'regions'/s/'regions.npz') as bank,np.load(run.OUT/'predictions'/s/'data.npz') as prediction:
            probabilities={a:np.load(run.OUT/'probabilities'/a/s/'data.npz') for a in run.ARMS}
            actions={(r['candidate_id'],r['arm']):r for r in run.read(run.OUT/'predictions'/s/'actions.json')}
            for plan in run.read(run.SOURCE/'plan'/s/'candidates.json'):
                cid=plan['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];m=meta[cid];before=iou(parent,gt,ids,sizes);k=idlookup.get(m['target']);allblocks=run.prior.old.unpack(bank,cid);rr=bank[cid+'_budget96'];blocks=[allblocks[j] for j in rr]
                pure=[b for b,yes in zip(blocks,annotations[cid+'_safe']) if yes];union=run.prior.core.union(pure);q=run.prior.core.remove(parent,halo,union);ideal=np.maximum(ideal,iou(q,gt,ids,sizes))
                for arm in run.ARMS:
                    pr=probabilities[arm][cid+'_probability'];truth=annotations[cid+'_fractions'];safe=annotations[cid+'_safe'];counts=annotations[cid+'_counts'];classification[scope][arm].append((truth,pr,safe,counts,annotations[cid+'_category']))
                    for j,rid in enumerate(rr):regions.append(dict(scene=s,split=scope,arm=arm,candidate_id=cid,region=int(rid),category=int(annotations[cid+'_category'][j]),points=len(blocks[j]),target_fraction=float(truth[j,0]),other_fraction=float(truth[j,1]),unknown_fraction=float(truth[j,2]),target_probability=float(pr[j,0]),other_probability=float(pr[j,1]),unknown_probability=float(pr[j,2]),safe_probability=float(pr[j,3]),safe=bool(safe[j])))
                    q=prediction[cid+'_'+arm+'_points'];assert np.isin(q,parent).all() and len(q)>=100;removed=np.setdiff1d(parent,q,assume_unique=True);after=iou(q,gt,ids,sizes);best[arm]=np.maximum(best[arm],after);changed=bool(len(removed));t=totals[scope][arm]
                    t.update(parents=1,actions=changed,deleted=len(removed),added=0,targets_known=int(m['stable']));key=q.astype('<i8').tobytes();t['duplicate_original']+=changed and key in keys;t['duplicate_repair']+=changed and key in newkeys[arm];newkeys[arm].add(key)
                    if changed:actionscenes[scope][arm].add(s)
                    record=dict(scene=s,split=scope,candidate_id=cid,source=plan['source'],arm=arm,stable=m['stable'],identity_scope=m['scope'],target=m['target'],original_target=m['original_target'],deleted=len(removed),changed=changed,reason=actions[cid,arm]['reason'],selected_regions=len(actions[cid,arm]['selected']),predicted_union_risk=actions[cid,arm]['risk'])
                    if m['stable']:
                        target=int((gt[removed]==m['target']).sum());other=int(((gt[removed]>=1000)&(gt[removed]!=m['target'])).sum());unknown=len(removed)-target-other;delta=float(after[k]-before[k]);drift=bool(changed and after.max(initial=0)>0 and int(np.argmax(after))!=k and (gt[q]==ids[int(np.argmax(after))]).sum()>=20)
                        t.update(deleted_target=target,deleted_other=other,annotation_unknown_deleted=unknown,changed_parent_target=int((gt[parent]==m['target']).sum()) if changed else 0,parent_improved02=delta>=.02-1e-12,parent_degraded02=delta<-.02-1e-12,good_parents=before[k]>=.9,good_degraded02=before[k]>=.9 and delta<-.02-1e-12,good_degraded10=before[k]>=.9 and delta<-.1-1e-12,good_degraded25=before[k]>=.9 and delta<-.25-1e-12,drift=drift)
                        record.update(deleted_target=target,deleted_other=other,annotation_unknown_deleted=unknown,parent_iou=float(before[k]),repaired_iou=float(after[k]),parent_delta=delta,NT_best=float(baseline[k]),pool_delta=float(after[k]-baseline[k]),drift=drift)
                    else:
                        t.update(identity_unknown_deleted=len(removed),identity_unknown_actions=changed);t[m['scope']+'_deleted']+=len(removed);record.update(identity_unknown_deleted=len(removed))
                        oldidx=int(np.argmax(before)) if len(ids) else None;newidx=int(np.argmax(after)) if len(ids) else None
                        diagnosticdrift=bool(changed and oldidx is not None and newidx!=oldidx and (gt[q]==ids[newidx]).sum()>=20);t['ambiguous_best_identity_changed']+=diagnosticdrift;record['ambiguous_best_identity_changed']=diagnosticdrift
                    rows.append(record)
                    local=Counter(parents=1,actions=changed,deleted=len(removed),deleted_other=record.get('deleted_other',0),deleted_target=record.get('deleted_target',0),unknown=record.get('annotation_unknown_deleted',0)+record.get('identity_unknown_deleted',0),parent_improved02=record.get('parent_delta',0)>=.02-1e-12,parent_degraded02=record.get('parent_delta',0)<-.02-1e-12,drift=record.get('drift',0))
                    for group in (plan['source'],m['scope']):breakdown[scope][arm].setdefault(group,Counter()).update(local)
            for pp in probabilities.values():pp.close()
        for threshold in THRESHOLDS:basecoverage[scope][str(threshold)]+=int((baseline>threshold).sum())
        oracle[scope]['GT_pure_union_gain02']+=int((ideal-baseline>=.02-1e-12).sum());oracle[scope]['targets']+=len(ids)
        for arm in run.ARMS:
            t=totals[scope][arm];t.update(pool_gain02=int((best[arm]-baseline>=.02-1e-12).sum()),targets=len(ids),unique_new=len(newkeys[arm]-keys))
            for threshold in THRESHOLDS:t['new_cross_'+str(threshold)]+=int(((baseline<=threshold)&(best[arm]>threshold)).sum())
            for j,gid in enumerate(ids):targets.append(dict(scene=s,split=scope,arm=arm,target=int(gid),NT_best=float(baseline[j]),expanded_NT_best=float(best[arm][j]),delta=float(best[arm][j]-baseline[j]),GT_pure_union_best=float(ideal[j])))
        if (i+1)%20==0:run.stage('post_seal_internal_geometry',completed=i+1,total=160)
    training=np.concatenate([x[0] for x in classification['train']['A']]);constant=training.mean(0);safeconstant=float(np.concatenate([x[2] for x in classification['train']['A']]).mean())
    summary=dict(complete=True,splits={},original_outer_fold=0,formal_OOF=False,training_regions_same_source=True,holdout_used_for_calibration=False,SL_modified=False,AP_run=False)
    for scope in c['splits']:
        summary['splits'][scope]=dict(scenes=len(c['splits'][scope]),baseline_coverage=dict(basecoverage[scope]),oracle=dict(oracle[scope]),arms={},classification={})
        for arm in run.ARMS:
            m=dict(totals[scope][arm]);other=m.get('deleted_other',0);target=m.get('deleted_target',0);unknown=m.get('annotation_unknown_deleted',0)+m.get('identity_unknown_deleted',0);n=m['deleted']
            m.update(action_scenes=len(actionscenes[scope][arm]),unknown_deleted=unknown,known_delete_precision=other/(other+target) if other+target else None,conservative_delete_precision=other/n if n else None,target_loss=target/max(1,m.get('changed_parent_target',0)),unknown_fraction=unknown/n if n else None,good_degradation_rate=m.get('good_degraded02',0)/max(1,m.get('good_parents',0)),breakdown={k:dict(v) for k,v in breakdown[scope][arm].items()})
            m['safe_joint_value']=m['pool_gain02']>=1 and m['action_scenes']>=2 and m['conservative_delete_precision'] is not None and m['conservative_delete_precision']>=.9 and m['target_loss']<=.01 and m['good_degradation_rate']<=.02 and m.get('drift',0)==0 and m['unknown_fraction']<=.1
            summary['splits'][scope]['arms'][arm]=m
            truth,pr,safe,counts,categories=[np.concatenate([x[k] for x in classification[scope][arm]]) for k in range(5)]
            fit=dict(regions=len(pr),safe_regions=int(safe.sum()),safe_prevalence=float(safe.mean()),safe_average_precision=float(average_precision_score(safe,pr[:,3])) if safe.any() else None,safe_ROC_AUC=float(roc_auc_score(safe,pr[:,3])) if len(np.unique(safe))==2 else None,fraction_MAE=float(np.abs(pr[:,:3]-truth).mean()),constant_fraction_MAE=float(np.abs(constant-truth).mean()),fraction_cross_entropy=float(-(truth*np.log(np.clip(pr[:,:3],1e-8,1))).sum(1).mean()),constant_fraction_cross_entropy=float(-(truth*np.log(np.clip(constant,1e-8,1))).sum(1).mean()),safe_BCE=float(-(safe*np.log(np.clip(pr[:,3],1e-8,1))+(1-safe)*np.log(np.clip(1-pr[:,3],1e-8,1))).mean()),constant_safe_BCE=float(-(safe*np.log(max(safeconstant,1e-8))+(1-safe)*np.log(max(1-safeconstant,1e-8))).mean()),categories={})
            for category in range(5):
                selected=categories==category
                if selected.any():fit['categories'][str(category)]=dict(regions=int(selected.sum()),mean_probability=pr[selected].mean(0).tolist(),fraction_MAE=float(np.abs(pr[selected,:3]-truth[selected]).mean()))
            summary['splits'][scope]['classification'][arm]=fit
    run.seal(run.OUT/'summary.json',summary);d=run.OUT/'evaluation';d.mkdir(exist_ok=True)
    for name,data in (('candidates.csv',rows),('targets.csv',targets),('regions.csv',regions)):save_csv(d/name,data)
    rules=run.read(run.OUT/'rules_frozen.json')['rules'];preflight=run.read(run.OUT/'evidence_preflight.json')['splits']
    lines=['# 同源96区归属学习：A / C 单范围内部留出测试','','固定原外层0训练160场景，按物理场景GT-free哈希划分112训练/24校准/24内部留出；现有N/T父、96预算、GD10六帧/区域规则/骨干与SL不变。两模型同结构同初始化/训练顺序/24轮；A屏蔽C新增20维局部视图证据。没有收益回归、补位、补点或完整五折。','',
           '本开发群体和此前GT供给诊断已暴露；内部留出只对这次参数拟合未见，不能称独立盲测或正式五折效果。规则只由校准24场景决定，内部留出预测先封存再评测。用户明确改为先验证小区域学习，旧>=256纯块供给门不再约束本轮。','',
           '## 实际联合删除与损害','', '|范围|组|动作/场景|正确删污染|误删目标|未知删除|父改善/退化≥.02|好父退化>.02/>.10|漂移|完整NT增益≥.02|安全联合价值|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for scope in ('train','calibration','holdout'):
        for arm in run.ARMS:
            m=summary['splits'][scope]['arms'][arm];lines.append(f"|{scope}|{arm}|{m['actions']} / {m['action_scenes']}|{m.get('deleted_other',0)}|{m.get('deleted_target',0)}|{m['unknown_deleted']}|{m.get('parent_improved02',0)} / {m.get('parent_degraded02',0)}|{m.get('good_degraded02',0)} / {m.get('good_degraded10',0)}|{m.get('drift',0)}|{m['pool_gain02']}|{m['safe_joint_value']}|")
    lines+=['','原NT全池保留，扩池不损失原覆盖；这不掩盖被修改父的退化，也不证明最终K800/AP受保护。删除点按每父并集去重，不同父累计。稳定父的未知标注与身份不确定父编辑分别列在summary和逐候选CSV；identity scope区分out_of_scope/unannotated/ambiguous，不能用未知编辑冒充正确删除。','',
            '|内部留出|新增>.5|>.75|>.78|>.90|>.95|>.98|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in run.ARMS:
        m=summary['splits']['holdout']['arms'][arm];lines.append('|'+arm+'|'+'|'.join(str(m['new_cross_'+str(t)]) for t in THRESHOLDS)+'|')
    lines+=['','## 学习是否成立','', '|范围|组|安全区PR平均精度/先验|归属MAE/训练常数|归属交叉熵/训练常数|','|---|---|---:|---:|---:|']
    for scope in ('train','calibration','holdout'):
        for arm in run.ARMS:
            f=summary['splits'][scope]['classification'][arm];ap=f['safe_average_precision'];lines.append(f"|{scope}|{arm}|{ap if ap is not None else 'NA'} / {f['safe_prevalence']:.5f}|{f['fraction_MAE']:.5f} / {f['constant_fraction_MAE']:.5f}|{f['fraction_cross_entropy']:.5f} / {f['constant_fraction_cross_entropy']:.5f}|")
    lines+=['','区域PR指标衡量pure-known-other标签的排序，不是实例AP；分类改善不能替代联合动作验收。混合区用真实点比例监督且不是安全正例；身份不明确全未知，无强制目标。所有区域每轮一次，权重仅训练折计算，沒有重复少量删除点。','',
            '## 局部证据覆盖与校准','']
    for scope,p in preflight.items():lines.append(f"- {scope}：有>=2帧可靠可见且父匹配无歧义的区域 {p['usable_parent_regions']}/{p['regions']} ({p['usable_parent_region_fraction']:.2%})；父匹配歧义 {p['parent_matching_ambiguity']:.2%}；邻居匹配歧义 {p['neighbor_matching_ambiguity']:.2%}。")
    lines+=['','每视图保存存在/可见/有掩码/父与邻居匹配/歧义/同mask/支持差/质量/缺失等20维。匹配在局部点域进行，只有投影后可靠可见点参加；这是GVC启发的局部归属证据，不是论文复现或整体GVC分数。缺失单独编码，不能当污染。','']
    for arm in run.ARMS:lines.append(f"- {arm}校准选择：{rules[arm]}")
    lines+=['','若没有校准规则同时满足预注册安全与动作支持条件，该组执行弃权；不会用留出结果放宽。联合动作逐点取所选重叠区的最大目标/未知风险，再对真正删除并集检查风险与>=100点保留。此预测风险仍须以实际误删校验，不能视作保证。','',
            '逐区域归属/风险见evaluation/regions.csv，真实动作与父损害见candidates.csv，完整NT池IoU档与GT纯区联合诊断见targets.csv。GT纯区联合仅后置机会诊断，不用于选区。结果范围仅此固定分割，不能从一两个收益目标推断方法成功，也不能直接接SL。']
    (run.OUT/'report.md').write_text('\n'.join(lines)+'\n');run.seal(run.OUT/'evaluation_frozen.json',dict(complete=True,outputs={str(p):run.sha(p) for p in [*d.iterdir(),run.OUT/'summary.json',run.OUT/'report.md']}));run.stage('single_scope_evaluation_complete',holdout=summary['splits']['holdout']['arms'])


if __name__=='__main__':evaluate()
