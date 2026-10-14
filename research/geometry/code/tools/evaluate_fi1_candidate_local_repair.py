"""Post-seal geometry evaluation; no scores, classes or AP are inherited."""
from pathlib import Path
from collections import Counter
import csv,json,sys
import numpy as np
from tools import run_fi1_candidate_local_repair as run
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
read=run.read;OUT=run.OUT;ARMS=run.core.ARMS
THRESHOLDS=(.50,.75,.78,.90,.95,.98)


def metrics(counts):
    m=dict(counts)
    for name,numerator,denominator in [('add_target_precision','added_target','added'),('delete_pollution_precision','deleted_other','deleted'),('target_loss','deleted_target','changed_parent_target'),('good_degradation_rate','good_degraded','good_parents'),('unknown_action_fraction','unknown_edits','edits')]:
        m[name]=m.get(numerator,0)/m[denominator] if m.get(denominator,0) else None
    return m


def evaluate():
    c=read(OUT/'contract.json');sys.addaudithook(run.Guard(c,'evaluate').hook)
    run.audit.prior.old.verify_files(c['code']);run.audit.prior.old.verify_files(c['frozen_SL_models'])
    run.audit.prior.old.verify_files(read(OUT/'predictions_frozen.json')['outputs'])
    totals={a:Counter() for a in ARMS};baseline_coverage=Counter();rows=[];target_rows=[];scenes=[]
    actions={a:set() for a in ARMS};valuable={a:set() for a in ARMS};unique_valuable={a:set() for a in ARMS}
    breakdown={a:{} for a in ARMS}
    for s in c['scenes']:
        run.stage('post_seal_geometry_evaluation',scene=s,completed=len(scenes))
        job=c['jobs'][s];assert sha(job['GT'])==c['GT_inputs'][job['GT']];gt=np.loadtxt(job['GT'],dtype=np.int64)
        ids,sizes=run.audit.prior.valid_targets(gt,c['class_ids']);geometry,points=run.audit.prior.old.load_nt(job)
        baseline=np.zeros(len(ids))
        original_keys={p.astype('<i8').tobytes() for p in points.values()}
        for p in points.values():baseline=np.maximum(baseline,run.audit.prior.ious(p,gt,ids,sizes))
        for t in THRESHOLDS:baseline_coverage[str(t)]+=int((baseline>t).sum())
        scene_counts={a:Counter(targets=len(ids),parents=0) for a in ARMS};newkeys={a:set() for a in ARMS};joint={a:set() for a in ARMS}
        best={a:baseline.copy() for a in ARMS}
        with np.load(OUT/'features'/s/'data.npz') as features,np.load(OUT/'predictions'/s/'predictions.npz') as prediction:
            for r in read(OUT/'plan'/s/'candidates.json'):
                cid=r['candidate_id'];parent=features[cid+'_parent'];before=run.audit.prior.ious(parent,gt,ids,sizes)
                k=int(np.argmax(before)) if len(ids) and before.max()>0 else None
                gid=int(ids[k]) if k is not None else None;hit=int((gt[parent]==gid).sum()) if gid is not None else 0
                known=gid is not None and hit>=20;old=float(before[k]) if k is not None else 0.
                for a in ARMS:
                    p=prediction[cid+'_'+a+'_points'];after=run.audit.prior.ious(p,gt,ids,sizes);best[a]=np.maximum(best[a],after)
                    deleted=np.setdiff1d(parent,p,assume_unique=True);added=np.setdiff1d(p,parent,assume_unique=True);changed=bool(len(deleted)+len(added))
                    counts=scene_counts[a];counts['parents']+=1;counts['actions']+=changed;counts['added']+=len(added);counts['deleted']+=len(deleted);counts['edits']+=len(added)+len(deleted)
                    counts['unknown_identity_actions']+=changed and not known
                    if changed:actions[a].add(s)
                    key=p.astype('<i8').tobytes()
                    counts['duplicate_original']+=changed and key in original_keys;counts['duplicate_repair']+=changed and key in newkeys[a];newkeys[a].add(key)
                    record=dict(scene=s,fold=c['folds'][s],source=r['source'],candidate_id=cid,arm=a,parent_target=gid if known else None,parent_iou=old,added=len(added),deleted=len(deleted),changed=changed,
                                good_parent=old>=.90,minimum_mask_abstain=next(v for v in read(OUT/'predictions'/s/'decisions.json') if v['candidate_id']==cid)['arms'][a]['reason']=='minimum_mask_abstain')
                    if known:
                        added_target=int((gt[added]==gid).sum());deleted_target=int((gt[deleted]==gid).sum())
                        added_other=int(((gt[added]>=1000)&(gt[added]!=gid)).sum());deleted_other=int(((gt[deleted]>=1000)&(gt[deleted]!=gid)).sum())
                        unknown=int((gt[added]<1000).sum()+(gt[deleted]<1000).sum())
                        score=float(after[k]);delta=score-old;pool_delta=score-float(baseline[k]);newk=int(np.argmax(after)) if len(ids) and after.max()>0 else None
                        drift=bool(changed and newk is not None and newk!=k and (gt[p]==ids[newk]).sum()>=20)
                        counts.update(added_target=added_target,added_other=added_other,deleted_target=deleted_target,deleted_other=deleted_other,unknown_edits=unknown,changed_parent_target=hit if changed else 0,drift=drift)
                        counts['parent_improved_02']+=delta>=.02-1e-12;counts['parent_degraded_02']+=delta<-.02-1e-12;counts['good_parents']+=old>=.90;counts['good_degraded']+=old>=.90 and delta<-.02-1e-12
                        for t in (.75,.90,.95):counts[f'parent_cross_{t}']+=old<=t and score>t
                        pool_value=pool_delta>=.02-1e-12 or any(baseline[k]<=t<score for t in (.75,.90,.95))
                        joint_value=delta>1e-12 and pool_value
                        if joint_value:joint[a].add(gid);valuable[a].add((s,gid))
                        record.update(repaired_iou=score,parent_delta=delta,original_pool_iou=float(baseline[k]),pool_delta=pool_delta,joint_value=joint_value,drift=drift,added_target=added_target,added_other=added_other,deleted_target=deleted_target,deleted_other=deleted_other,unknown_edits=unknown)
                    else:counts['unknown_edits']+=len(added)+len(deleted)
                    rows.append(record)
                    categories=[r['source'],'fold'+str(c['folds'][s]),'fixed10' if s in c['fixed10'] else 'remaining190','historical90' if s in c['groups']['historical90'] else 'new110']
                    local=Counter(parents=1,actions=changed,parent_improved_02=record.get('parent_delta',0)>=.02-1e-12,parent_degraded_02=record.get('parent_delta',0)<-.02-1e-12,joint_value=record.get('joint_value',False),added=len(added),deleted=len(deleted),deleted_target=record.get('deleted_target',0),added_other=record.get('added_other',0))
                    for category in categories:breakdown[a].setdefault(category,Counter()).update(local)
        ss=dict(scene=s,targets=len(ids),original_NT=len(points),baseline_coverage={str(t):int((baseline>t).sum()) for t in THRESHOLDS},arms={})
        for a in ARMS:
            counts=scene_counts[a];counts['unique_new']=len(newkeys[a]-original_keys);counts['pool_gain_02']=int((best[a]-baseline>=.02-1e-12).sum());counts['joint_value_targets']=len(joint[a])
            for t in THRESHOLDS:counts[f'coverage_{t}']=int((best[a]>t).sum());counts[f'new_cross_{t}']=int(((baseline<=t)&(best[a]>t)).sum())
            totals[a].update(counts);ss['arms'][a]=metrics(counts)
            for i,gid in enumerate(ids):
                val=best[a][i]-baseline[i]>=.02-1e-12 or any(baseline[i]<=t<best[a][i] for t in (.75,.90,.95))
                if val:unique_valuable[a].add((s,int(gid)))
                target_rows.append(dict(scene=s,fold=c['folds'][s],arm=a,gt_id=int(gid),original_NT_best=float(baseline[i]),expanded_NT_best=float(best[a][i]),delta=float(best[a][i]-baseline[i]),value=val))
        scenes.append(ss)
    summary=dict(complete=True,scenes=200,parents=sum(s['arms'][ARMS[0]]['parents'] for s in scenes),original_NT_targets=sum(s['targets'] for s in scenes),baseline_coverage=dict(baseline_coverage),arms={},breakdown={a:{k:dict(v) for k,v in b.items()} for a,b in breakdown.items()},geometry_only=True,SL_modified=False,AP_measured=False)
    for a in ARMS:
        m=metrics(totals[a]);m.update(action_scenes=len(actions[a]),joint_value_targets=len(valuable[a]),joint_value_scenes=len({s for s,g in valuable[a]}),pool_value_targets=len(unique_valuable[a]))
        gate=c['gate'];conditions=dict(value_targets=m['joint_value_targets']>=gate['min_value_targets'],value_scenes=m['joint_value_scenes']>=gate['min_value_scenes'],actions=m['actions']>=gate['min_actions'],action_scenes=m['action_scenes']>=gate['min_action_scenes'],add_precision=m['add_target_precision'] is None or m['add_target_precision']>=gate['min_add_target_precision'],delete_precision=m['delete_pollution_precision'] is None or m['delete_pollution_precision']>=gate['min_delete_pollution_precision'],target_loss=m['target_loss'] is None or m['target_loss']<=gate['max_target_loss'],good_preservation=m['good_degradation_rate'] is None or m['good_degradation_rate']<=gate['max_good_degradation_rate'],identity=m.get('drift',0)<=gate['max_drift'],unknown=m['unknown_action_fraction'] is None or m['unknown_action_fraction']<=gate['max_unknown_action_fraction'])
        m.update(gate=conditions,geometry_passed=all(conditions.values()));summary['arms'][a]=m
    dest=OUT/'evaluation';dest.mkdir(exist_ok=True)
    for name,records in [('candidates.csv',rows),('targets.csv',target_rows)]:
        columns=sorted({k for r in records for k in r})
        with (dest/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(records)
    freeze_json(dest/'scenes.json',scenes);freeze_json(OUT/'summary.json',summary)
    table=['|组别|真实动作|父提升≥.02|父退化>.02|池提升≥.02目标|联合价值目标|好候选退化|增点精度|删污染精度|几何门|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for a in ARMS:
        m=summary['arms'][a];table.append(f"|{a}|{m.get('actions',0)}|{m.get('parent_improved_02',0)}|{m.get('parent_degraded_02',0)}|{m.get('pool_gain_02',0)}|{m['joint_value_targets']}|{m.get('good_degraded',0)}/{m.get('good_parents',0)}|{m['add_target_precision']}|{m['delete_pollution_precision']}|{m['geometry_passed']}|")
    text='# 真实局部增删修复：开发200原物理五折\n\n'+f"200场景、{summary['parents']}父候选、{summary['original_NT_targets']}有效GT目标；五个共享双臂模型固定12轮，全部折外点集先封存再评测。原始NT全部保留。\n\n"+'\n'.join(table)+'\n\n详见summary.json与evaluation逐父/逐目标CSV，包括六档覆盖、误删/误加/未知、漂移和源/折/190场景细分。池级增加覆盖是追加原NT的几何上界，不是最终AP。旧30/GD10权重相同，选帧最多6且无GT。DINO冻结，只训练小型点图网络。\n\n几何门未通过的臂不进入自身语义/SL。本轮不改生产、SL、轨迹或新增SAM观测；上游开发暴露，不称整系统盲测。\n'
    (OUT/'report.md').write_text(text)
    (OUT/'decision.md').write_text('# 决策\n\n'+('\n'.join(a+': '+('通过预注册几何门；后续另行获取自身语义并重算上下文。' if summary['arms'][a]['geometry_passed'] else '未通过预注册几何门；不接SL，不据此宣称几何没有提升空间。') for a in ARMS))+'\n')
    freeze_json(OUT/'evaluation_frozen.json',dict(complete=True,outputs={str(p):sha(p) for p in list(dest.iterdir())+[OUT/'summary.json',OUT/'report.md',OUT/'decision.md']}))
    run.stage('real_geometry_evaluated',arms={a:summary['arms'][a]['geometry_passed'] for a in ARMS})

if __name__=='__main__':evaluate()
