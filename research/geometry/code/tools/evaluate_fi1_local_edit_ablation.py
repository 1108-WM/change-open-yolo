"""Post-seal four-arm geometry, edit decomposition and scope-separated evaluation."""
from pathlib import Path
from collections import Counter
import csv,sys
import numpy as np
from tools import run_fi1_local_edit_ablation as run
from tools.evaluate_fi1_candidate_local_repair import metrics
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha
THRESHOLDS=(.5,.75,.78,.9,.95,.98)


def evaluate():
    out=run.OUT;c=run.read(out/'contract.json');sys.addaudithook(run.Guard(c,'evaluate').hook)
    run.original.audit.prior.old.verify_files(c['code']);run.original.audit.prior.old.verify_files(c['frozen_SL_models']);run.original.audit.prior.old.verify_files(run.read(out/'predictions_frozen.json')['outputs'])
    totals={a:Counter() for a in run.ARMS};basecoverage=Counter();rows=[];targets=[];scene_rows=[];breakdown={a:{} for a in run.ARMS};confusion=np.zeros((6,6),np.int64)
    value={a:set() for a in run.ARMS};actions={a:set() for a in run.ARMS};mode_value={a:{m:set() for m in ('full','add_only','delete_only')} for a in run.ARMS}
    for i,s in enumerate(c['scenes']):
        run.stage('post_seal_four_arm_geometry',completed=i,scene=s)
        job=c['jobs'][s];assert sha(job['GT'])==c['GT_inputs'][job['GT']];gt=np.loadtxt(job['GT'],dtype=np.int64);ids,sizes=run.original.audit.prior.valid_targets(gt,c['class_ids']);idx={int(g):k for k,g in enumerate(ids)}
        geom,points=run.original.audit.prior.old.load_nt(job);baseline=np.zeros(len(ids))
        for p in points.values():baseline=np.maximum(baseline,run.original.audit.prior.ious(p,gt,ids,sizes))
        original_keys={p.astype('<i8').tobytes() for p in points.values()};newkeys={a:set() for a in run.ARMS};best={a:{m:baseline.copy() for m in ('full','add_only','delete_only')} for a in run.ARMS};scene_counts={a:Counter(targets=len(ids)) for a in run.ARMS}
        for threshold in THRESHOLDS:basecoverage[str(threshold)]+=int((baseline>threshold).sum())
        metadata={r['candidate_id']:r for r in run.read(out/'labels'/s/'meta.json')}
        with np.load(run.SOURCE/'features'/s/'data.npz') as feature,np.load(out/'predictions'/s/'predictions.npz') as prediction:
            for candidate in run.read(run.SOURCE/'plan'/s/'candidates.json'):
                cid=candidate['candidate_id'];meta=metadata[cid];parent=feature[cid+'_parent'];halo=feature[cid+'_halo'];visible=feature[cid+'_'+run.OBS+'_groups']>=2;member=np.isin(halo,parent)
                before=run.original.audit.prior.ious(parent,gt,ids,sizes);gid=meta['target'];known=gid is not None;k=None if not known else idx[gid];old=float(before.max(initial=0));hit=int((gt[parent]==gid).sum()) if known else 0
                delete_opportunity=int((member&visible&(gt[halo]>=1000)&(gt[halo]!=gid)).sum()) if known else 0
                add_opportunity=int((~member&visible&(gt[halo]==gid)).sum()) if known else 0
                for arm in run.ARMS:
                    p=prediction[cid+'_'+arm+'_points'];after=run.original.audit.prior.ious(p,gt,ids,sizes);deleted=np.setdiff1d(parent,p,assume_unique=True);added=np.setdiff1d(p,parent,assume_unique=True);changed=bool(len(deleted)+len(added));counts=scene_counts[arm]
                    counts.update(parents=1,actions=changed,added=len(added),deleted=len(deleted),edits=len(deleted)+len(added),delete_opportunity=delete_opportunity,add_opportunity=add_opportunity,delete_actions=bool(len(deleted)),add_only_actions=bool(len(added) and not len(deleted)),both_actions=bool(len(added) and len(deleted)))
                    if changed:actions[arm].add(s)
                    key=p.astype('<i8').tobytes();counts['duplicate_original']+=changed and key in original_keys;counts['duplicate_repair']+=changed and key in newkeys[arm];newkeys[arm].add(key)
                    deleted_only=np.setdiff1d(parent,deleted,assume_unique=True);deleted_only=parent if len(deleted_only)<100 else deleted_only
                    added_only=np.union1d(parent,added)
                    for mode,mask in (('full',p),('add_only',added_only),('delete_only',deleted_only)):best[arm][mode]=np.maximum(best[arm][mode],run.original.audit.prior.ious(mask,gt,ids,sizes))
                    record=dict(scene=s,fold=c['folds'][s],candidate_id=cid,source=candidate['source'],arm=arm,scope=meta['scope_name'],target=gid,parent_iou=old,added=len(added),deleted=len(deleted),delete_opportunity=delete_opportunity,add_opportunity=add_opportunity,changed=changed)
                    if known:
                        aa=int((gt[added]==gid).sum());ao=int(((gt[added]>=1000)&(gt[added]!=gid)).sum());du=int((gt[deleted]==gid).sum());do=int(((gt[deleted]>=1000)&(gt[deleted]!=gid)).sum());unadd=len(added)-aa-ao;undel=len(deleted)-du-do
                        score=float(after[k]);delta=score-old;pool_delta=score-baseline[k];newk=int(np.argmax(after)) if after.max(initial=0)>0 else None;drift=bool(changed and newk is not None and newk!=k and (gt[p]==ids[newk]).sum()>=20)
                        joint=delta>1e-12 and (pool_delta>=.02-1e-12 or any(baseline[k]<=t<score for t in (.75,.9,.95)))
                        if joint:value[arm].add((s,gid))
                        counts.update(added_target=aa,added_other=ao,deleted_target=du,deleted_other=do,unannotated_add=unadd,unannotated_delete=undel,unknown_edits=unadd+undel,changed_parent_target=hit if changed else 0,drift=drift,parent_improved_02=delta>=.02-1e-12,parent_degraded_02=delta<-.02-1e-12,good_parents=old>=.9,good_degraded=old>=.9 and delta<-.02-1e-12)
                        for tail in (.1,.25,.5):counts[f'good_loss_gt_{tail}']+=old>=.9 and delta<-tail-1e-12
                        record.update(repaired_iou=score,parent_delta=delta,original_NT_best=float(baseline[k]),pool_delta=float(pool_delta),joint_value=joint,drift=drift,added_target=aa,added_other=ao,deleted_target=du,deleted_other=do,unannotated_add=unadd,unannotated_delete=undel,unknown_edits=unadd+undel)
                    else:
                        counts.update(unknown_identity_add=len(added),unknown_identity_delete=len(deleted),unknown_identity_actions=changed,unknown_edits=len(added)+len(deleted));record.update(unknown_identity_add=len(added),unknown_identity_delete=len(deleted),unknown_edits=len(added)+len(deleted))
                    if arm=='C':
                        cp=prediction[cid+'_C_candidate_probability'];predicted_scope=int(cp.argmax());confusion[meta['scope'],predicted_scope]+=1;record.update(candidate_predicted_scope=run.learn.SCOPE[predicted_scope],candidate_edit_probability=float(cp[0]))
                    local=Counter(parents=1,actions=changed,added=len(added),deleted=len(deleted),delete_opportunity=delete_opportunity,add_opportunity=add_opportunity,added_target=record.get('added_target',0),added_other=record.get('added_other',0),deleted_target=record.get('deleted_target',0),deleted_other=record.get('deleted_other',0),unknown_edits=record['unknown_edits'],parent_improved_02=record.get('parent_delta',0)>=.02-1e-12,parent_degraded_02=record.get('parent_delta',0)<-.02-1e-12,joint_value=record.get('joint_value',False),drift=record.get('drift',False))
                    for category in (candidate['source'],meta['scope_name'],'fold'+str(c['folds'][s]),'fixed10' if s in c['fixed10'] else 'remaining190','historical90' if s in c['groups']['historical90'] else 'new110'):breakdown[arm].setdefault(category,Counter()).update(local)
                    rows.append(record)
        ss=dict(scene=s,targets=len(ids),arms={})
        for arm in run.ARMS:
            counts=scene_counts[arm];counts['unique_new']=len(newkeys[arm]-original_keys)
            for mode in ('full','add_only','delete_only'):
                modebest=best[arm][mode];counts[mode+'_pool_gain02']=int((modebest-baseline>=.02-1e-12).sum())
                for k,gid in enumerate(ids):
                    valuable=modebest[k]-baseline[k]>=.02-1e-12 or any(baseline[k]<=t<modebest[k] for t in (.75,.9,.95))
                    if valuable:mode_value[arm][mode].add((s,int(gid)))
                    targets.append(dict(scene=s,fold=c['folds'][s],arm=arm,mode=mode,gt_id=int(gid),original_NT_best=float(baseline[k]),expanded_NT_best=float(modebest[k]),delta=float(modebest[k]-baseline[k]),value=valuable))
            for t in THRESHOLDS:counts[f'coverage_{t}']=int((best[arm]['full']>t).sum());counts[f'new_cross_{t}']=int(((baseline<=t)&(best[arm]['full']>t)).sum())
            totals[arm].update(counts);ss['arms'][arm]=dict(counts)
        scene_rows.append(ss)
    summary=dict(complete=True,scenes=200,parents=1599,targets=sum(ss['targets'] for ss in scene_rows),baseline_coverage=dict(basecoverage),arms={},breakdown={a:{name:dict(cnt) for name,cnt in groups.items()} for a,groups in breakdown.items()},candidate_scope_confusion=confusion.tolist(),candidate_scope_names=list(run.learn.SCOPE),A_reused_frozen=True,SL_modified=False,AP_run=False)
    for arm in run.ARMS:
        m=metrics(totals[arm]);m.update(action_scenes=len(actions[arm]),joint_value_targets=len(value[arm]),joint_value_scenes=len({s for s,g in value[arm]}),delete_recall=m.get('deleted_other',0)/m['delete_opportunity'],add_recall=m.get('added_target',0)/m['add_opportunity'])
        for mode in ('full','add_only','delete_only'):m[mode+'_pool_value_targets']=len(mode_value[arm][mode])
        gate=c['gate'];g=dict(value_targets=m['joint_value_targets']>=gate['min_value_targets'],value_scenes=m['joint_value_scenes']>=gate['min_value_scenes'],actions=m['actions']>=gate['min_actions'],action_scenes=m['action_scenes']>=gate['min_action_scenes'],add_precision=m['add_target_precision'] is None or m['add_target_precision']>=gate['min_add_target_precision'],delete_precision=m['delete_pollution_precision'] is None or m['delete_pollution_precision']>=gate['min_delete_pollution_precision'],target_loss=m['target_loss'] is None or m['target_loss']<=gate['max_target_loss'],good_preservation=m['good_degradation_rate'] is None or m['good_degradation_rate']<=gate['max_good_degradation_rate'],identity=m.get('drift',0)<=gate['max_drift'],unknown=m['unknown_action_fraction'] is None or m['unknown_action_fraction']<=gate['max_unknown_action_fraction'])
        m.update(original_gate=g,original_geometry_passed=all(g.values()),deletion_first_passed=m['delete_only_pool_gain02']>=1)
        summary['arms'][arm]=m
    for arm,reference in (('B','A0'),('C','B')):
        m=summary['arms'][arm];r=summary['arms'][reference];comparison=dict(reference=reference,addition_pool_value_retained=m['add_only_pool_value_targets']>=r['add_only_pool_value_targets'],neighbor_wrong_add_not_increased=m['added_other']<=r['added_other'],severe_good_loss_not_increased=m.get('good_loss_gt_0.1',0)<=r.get('good_loss_gt_0.1',0),drift_not_increased=m.get('drift',0)<=r.get('drift',0))
        m.update(paired_acceptance=comparison,complete_geometry_passed=m['original_geometry_passed'] and m['deletion_first_passed'] and all(v for k,v in comparison.items() if k!='reference'))
    # Reused A must independently reproduce the original accepted geometry statistics.
    old=run.read(run.SOURCE/'summary.json')['arms'][run.OBS];a=summary['arms']['A']
    for key in ('parents','actions','added','deleted','added_target','added_other','deleted_target','deleted_other','unknown_edits','drift','good_parents','good_degraded','parent_improved_02','parent_degraded_02','joint_value_targets'):assert a[key]==old[key],(key,a[key],old[key])
    assert a['full_pool_gain02']==old['pool_gain_02'] and a['delete_opportunity']==153126 and a['add_opportunity']==195153
    d=out/'evaluation';d.mkdir(exist_ok=True)
    for name,rr in [('candidates.csv',rows),('targets.csv',targets)]:
        with (d/name).open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=sorted({k for r in rr for k in r}));writer.writeheader();writer.writerows(rr)
    freeze_json(d/'scenes.json',scene_rows);freeze_json(out/'summary.json',summary)
    lines=['# 固定GD10六帧的纠错监督与身份消融','','开发200原物理五折、1599父候选、冻结六帧/0.15米/骨干/SL。A复用封存预测；A0/B/C独立训练GD10-only，各5模型固定12轮。A原训练双观测臂，监督归因比较B-A0，身份表示比较C-B。','',
           '|臂|父提升≥.02|全NT增益≥.02|池价值|只删除池增益≥.02|可见污染删除召回|误加邻居点|好父损失>.10|漂移|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in run.ARMS:
        m=summary['arms'][arm];lines.append(f"|{arm}|{m['parent_improved_02']}|{m['full_pool_gain02']}|{m['joint_value_targets']}|{m['delete_only_pool_gain02']}|{m['delete_recall']:.4%}|{m['added_other']}|{m.get('good_loss_gt_0.1',0)}|{m.get('drift',0)}|")
    lines+=['','这里只删除/只补入是对已封存预测的后置拆解，不是可选输出策略；所有完整NT原候选保留。几何覆盖不等于AP。未知编辑按范围外/未标注/混合/小实例分别报告，GT范围分类不进入推理。','',
            '所有编辑门固定，GT分层和候选/关系监督仅训练折使用；pilot只训练内诊断，完整训练从头，未按外层错误清单筛动作或挑门限。详细误删误加、N/T、好父>.02/.10/.25/.50尾部、四臂/六档覆盖、候选scope混淆矩阵见summary与逐候选/目标CSV。','',
            '几何门：'+str({arm:summary['arms'][arm].get('complete_geometry_passed',False) for arm in ('B','C')})+'。本轮不接自身语义/SL/AP，不增加帧、不重跑SAM或DINO。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    freeze_json(out/'evaluation_frozen.json',dict(complete=True,outputs={str(p):sha(p) for p in list(d.iterdir())+[out/'summary.json',out/'report.md']}));run.stage('four_arm_geometry_evaluated')

if __name__=='__main__':evaluate()
