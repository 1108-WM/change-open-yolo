#!/usr/bin/env python3
"""Actual pointsets, complete NT pool, safety and feature attribution; never instance AP."""
from pathlib import Path
from collections import Counter
import csv,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_mask3d_feature_ab as run
from tools.evaluate_fi1_deletion_blocks import iou
T=(.5,.75,.78,.9,.95,.98)

def metricize(count,scenes,value_targets,value_scenes):
    m=dict(count);m.update(action_scenes=len(scenes),pool_value_targets=len(value_targets),pool_value_scenes=len(value_scenes))
    for name,num,den in (('add_target_precision','added_target','added'),('delete_pollution_precision','deleted_other','deleted'),('target_loss','deleted_target','changed_parent_target'),('unknown_fraction','unknown_edits','edits'),('good_rate','good_degraded','good_parents')):m[name]=m.get(num,0)/m[den] if m.get(den,0) else None
    return m

def evaluate_scenes(c,scenes,modes,calibration=False,accepted=False):
    results={f'{a}:{t}':dict(count=Counter(),scenes=set(),values=set(),value_scenes=set(),candidates=[],targets=[],regions=[]) for a,t in modes}
    for s in scenes:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);assert run.sha(c['jobs'][s]['GT'])==c['GT_inputs'][c['jobs'][s]['GT']]
        ids,sizes=run.original.audit.prior.valid_targets(gt,c['class_ids']);_,nt=run.original.audit.prior.old.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids));originalkeys={p.astype('<i8').tobytes() for p in nt.values()}
        for p in nt.values():baseline=np.maximum(baseline,iou(p,gt,ids,sizes))
        best={k:baseline.copy() for k in results};newkeys={k:set() for k in results};rules=run.read(run.OUT/'rules_frozen.json')['chosen'] if accepted else None
        with np.load(run.SOURCE/'features'/s/'data.npz') as z,np.load(run.OUT/'probabilities'/s/'data.npz') as pr:
            out=np.load(run.OUT/'predictions'/s/'data.npz') if accepted else None
            bank=np.load(run.REGIONS/'regions'/s/'regions.npz') if not calibration else None
            for r in run.read(run.SOURCE/'plan'/s/'candidates.json'):
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];before=iou(parent,gt,ids,sizes)
                index=int(np.argmax(before)) if len(ids) and before.max()>0 else None;gid=int(ids[index]) if index is not None else None;hits=int((gt[parent]==gid).sum()) if gid is not None else 0;stable=gid is not None and hits>=20 and hits/len(parent)>=.6
                dominant,counts=np.unique(gt[parent],return_counts=True);dominant=int(dominant[counts.argmax()]);fraction=float(counts.max()/len(parent))
                scope='stable' if stable else 'out_of_scope' if dominant>=1000 and dominant//1000 not in c['class_ids'] and fraction>=.6 else 'unannotated' if (gt[parent]<1000).mean()>=.5 else 'ambiguous'
                for a,t in modes:
                    key=f'{a}:{t}';res=results[key];m=res['count'];p=out[cid+'_'+a+'_accepted'] if accepted else run.core.repair(parent,halo,pr[cid+'_'+a+'_probability'],z[cid+'_'+run.OBS+'_groups'],t)
                    if not accepted and t==.8:assert np.array_equal(p,pr[cid+'_'+a+'_generated'])
                    after=iou(p,gt,ids,sizes);best[key]=np.maximum(best[key],after);add=np.setdiff1d(p,parent);deleted=np.setdiff1d(parent,p);changed=bool(len(add)+len(deleted));old=float(before[index]) if index is not None else 0
                    m.update(parents=1,actions=changed,added=len(add),deleted=len(deleted),edits=len(add)+len(deleted),targets=0)
                    if changed:res['scenes'].add(s)
                    code=p.astype('<i8').tobytes();m['duplicate_original']+=changed and code in originalkeys;m['duplicate_generated']+=changed and code in newkeys[key];newkeys[key].add(code)
                    row=dict(scene=s,fold=c['folds'][s],source=r['source'],candidate_id=cid,arm=a,mode='accepted' if accepted else 'generated',threshold=t,scope=scope,stable=stable,parent_iou=old,added=len(add),deleted=len(deleted),changed=changed)
                    if stable:
                        score=float(after[index]);delta=score-old;targetadd=int((gt[add]==gid).sum());targetdelete=int((gt[deleted]==gid).sum());otheradd=int(((gt[add]>=1000)&(gt[add]!=gid)).sum());otherdelete=int(((gt[deleted]>=1000)&(gt[deleted]!=gid)).sum());unknown=len(add)+len(deleted)-targetadd-targetdelete-otheradd-otherdelete
                        drift=changed and len(ids)>0 and int(ids[np.argmax(after)])!=gid and int((gt[p]==ids[np.argmax(after)]).sum())>=20
                        m.update(added_target=targetadd,added_other=otheradd,deleted_target=targetdelete,deleted_other=otherdelete,unknown_edits=unknown,unknown_annotation_edits=unknown,changed_parent_target=hits if changed else 0,drift=drift,parent_iou_sum=old,repaired_iou_sum=score,parent_delta_sum=delta,stable_parents=1,parent_improved_02=delta>=.02-1e-12,parent_degraded_02=delta<-.02-1e-12,good_parents=old>=.9,good_degraded=old>=.9 and delta<-.02-1e-12)
                        for loss in (.1,.25,.5):m[f'good_loss_gt_{loss}']+=old>=.9 and delta<-loss-1e-12
                        m['worst_good_loss']=max(m.get('worst_good_loss',0),-delta if old>=.9 else 0)
                        row.update(target=gid,repaired_iou=score,parent_delta=delta,pool_delta=score-baseline[index],added_target=targetadd,added_other=otheradd,deleted_target=targetdelete,deleted_other=otherdelete,unknown_edits=unknown,drift=drift)
                    else:
                        m['unknown_edits']+=len(add)+len(deleted);m['unknown_identity_edits']+=len(add)+len(deleted);m['unknown_identity_actions']+=changed;m[scope+'_edits']+=len(add)+len(deleted)
                        row['unknown_edits']=len(add)+len(deleted)
                    res['candidates'].append(row)
                    if bank is not None and not accepted:
                        flat=bank[cid+'_block_indices'];offset=bank[cid+'_block_offsets'];allblocks=[flat[lo:hi] for lo,hi in zip(offset[:-1],offset[1:])]
                        for rid in bank[cid+'_budget96']:
                            b=allblocks[int(rid)];points=halo[b];removed=np.isin(points,deleted);pred=pr[cid+'_'+a+'_probability'][b].astype(np.float32).argmax(1)
                            truth=np.where(gt[points]==gid,0,np.where(gt[points]>=1000,1,2)) if stable else np.full(len(b),2)
                            res['regions'].append(dict(scene=s,candidate_id=cid,arm=a,region=int(rid),points=len(b),removed=int(removed.sum()),known_points=int((truth!=2).sum()),known_correct=int(((truth!=2)&(truth==pred)).sum()),unknown_points=int((truth==2).sum())))
            if out is not None:out.close()
            if bank is not None:bank.close()
        for key,res in results.items():
            expanded=best[key];m=res['count'];m['targets']+=len(ids);m['pool_gain02']+=int((expanded-baseline>=.02-1e-12).sum());m['pool_delta_sum']+=float((expanded-baseline).sum());m['parent_net_improved']=m.get('parent_improved_02',0)-m.get('parent_degraded_02',0)
            for threshold in T:m[f'coverage_{threshold}']+=int((expanded>threshold).sum());m[f'baseline_{threshold}']+=int((baseline>threshold).sum());m[f'new_cross_{threshold}']+=int(((baseline<=threshold)&(expanded>threshold)).sum())
            for i,g in enumerate(ids):
                value=expanded[i]-baseline[i]>=.02-1e-12 or any(baseline[i]<=threshold<expanded[i] for threshold in (.75,.9,.95))
                if value:res['values'].add((s,int(g)));res['value_scenes'].add(s)
                res['targets'].append(dict(scene=s,arm=key.split(':')[0],mode='accepted' if accepted else 'generated',gt_id=int(g),baseline=float(baseline[i]),expanded=float(expanded[i]),delta=float(expanded[i]-baseline[i]),value=value))
        if not calibration:run.stage('actual_pointset_evaluation',scene=s,accepted=accepted)
    for res in results.values():res['metrics']=metricize(res['count'],res['scenes'],res['values'],res['value_scenes'])
    return results

def evaluate():
    c=run.read(run.OUT/'contract.json');sys.addaudithook(run.Guard(c,'evaluate').hook);assert run.read(run.OUT/'predictions_frozen.json')['complete'];run.integrity(c['frozen_SL_models'])
    summary=dict(complete=True,development_only=True,AP_run=False,SL_modified=False,splits={},rules=run.read(run.OUT/'rules_frozen.json')['chosen']);rows=[];targets=[];regions=[]
    for split in ('fit','selection','calibration','holdout'):
        summary['splits'][split]={}
        for accepted in (False,True):
            rr=evaluate_scenes(c,c['splits'][split],[(a,.8) for a in ('A','B')],accepted=accepted)
            for a in ('A','B'):
                r=rr[a+':0.8'];m=r['metrics'];m['protection']=run.core.protection(m);m['acceptable_geometry']=all(m['protection'].values()) and m['pool_value_targets']>=1
                summary['splits'][split][a+('_accepted' if accepted else '_generated')]=m;rows.extend(dict(v,split=split) for v in r['candidates']);targets.extend(dict(v,split=split) for v in r['targets']);regions.extend(dict(v,split=split) for v in r['regions'])
    # Explicit original train112 aggregate from disjoint fit and selection, no rerun/pooling of identities.
    summary['splits']['train112']={}
    for mode in ('A_generated','B_generated','A_accepted','B_accepted'):
        a,b=[summary['splits'][k][mode] for k in ('fit','selection')];m={k:a.get(k,0)+b.get(k,0) for k in set(a)|set(b) if isinstance(a.get(k,0),(int,float)) and isinstance(b.get(k,0),(int,float)) and k not in ('worst_good_loss','acceptable_geometry') and 'precision' not in k and k not in ('target_loss','unknown_fraction','good_rate')}
        m['worst_good_loss']=max(a['worst_good_loss'],b['worst_good_loss']);m=metricize(m,range(int(m['action_scenes'])),range(int(m['pool_value_targets'])),range(int(m['pool_value_scenes'])));m['protection']=run.core.protection(m);m['acceptable_geometry']=all(m['protection'].values()) and m['pool_value_targets']>=1;summary['splits']['train112'][mode]=m
    h=summary['splits']['holdout'];a=h['A_generated'];b=h['B_generated']
    fields=('pool_value_targets','pool_gain02','parent_net_improved');improved=all(b.get(k,0)>=a.get(k,0) for k in fields) and any(b.get(k,0)>a.get(k,0) for k in fields) and b.get('parent_delta_sum',0)>=a.get('parent_delta_sum',0)-1e-10
    summary['B_generation_improved']=improved;summary['B_acceptable']=h['B_accepted']['acceptable_geometry']
    summary['decision']='retain_instance_features_then_separate_interaction_test' if improved and summary['B_acceptable'] else 'generation_improved_study_specific_repair_quality' if improved else 'no_generation_improvement_inspect_prior_and_target_condition'
    d=run.OUT/'evaluation';d.mkdir(exist_ok=True)
    for name,data in (('candidates.csv',rows),('targets.csv',targets),('regions.csv',regions)):
        with (d/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in data for k in r}));w.writeheader();w.writerows(data)
    summary['breakdown']={}
    for split in ('fit','selection','calibration','holdout'):
        for arm in ('A','B'):
            for mode in ('generated','accepted'):
                selected=[r for r in rows if r['split']==split and r['arm']==arm and r['mode']==mode]
                for group in ('native','track','historical90','new110'):
                    picked=[r for r in selected if r['source']==group or r['scene'] in c['groups'].get(group,[])];summary['breakdown'][f'{split}/{arm}/{mode}/{group}']={k:sum(r.get(k,0) for r in picked) for k in ('changed','added','deleted','added_target','added_other','deleted_target','deleted_other','unknown_edits','drift','parent_delta')}
    run.seal(run.OUT/'summary.json',summary)
    lines=['# 冻结Mask3D实例特征A/B：局部增删开发对照','','仅新增128维冻结mask head特征。A置零/B启用；原点输出、模型宽度、修正损失与目标配对一致。原112拆96拟合/16选择，24校准/24内部检查；留出历史暴露。无新帧/质量头/CliqueNet/SL/AP。','','接口导出前后全部场景Mask3D输出逐值一致；此为资产接入验证，不是几何提升。','']
    for split in ('fit','selection','train112','calibration','holdout'):
        lines += [f'## {split}','','|组/输出|动作|父改善/退化≥.02|NT≥.02|池价值|补邻居|误删目标|未知|好父>.02/>.10|漂移|几何合格|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for mode,m in summary['splits'][split].items():lines.append(f"|{mode}|{m.get('actions',0)}|{m.get('parent_improved_02',0)}/{m.get('parent_degraded_02',0)}|{m.get('pool_gain02',0)}|{m['pool_value_targets']}|{m.get('added_other',0)}|{m.get('deleted_target',0)}|{m.get('unknown_edits',0)}|{m.get('good_degraded',0)}/{m.get('good_loss_gt_0.1',0)}|{m.get('drift',0)}|{m['acceptable_geometry']}|")
        lines.append('')
    lines += ['锁定校准规则：'+str(summary['rules']),'','B生成改善：'+str(improved)+'；B接收几何合格：'+str(summary['B_acceptable'])+'。','决定：'+summary['decision'],'','原NT保留故全池覆盖不下降，不能由此推断父安全或未来K800/AP。生成与准入分开；全部弃权不是成功。当前实验结束，不自动启动后续模型。完整逐点损害、连续IoU、多档/源/历史分层及96区域诊断见summary/evaluation。']
    (run.OUT/'report.md').write_text('\n'.join(lines)+'\n');run.seal(run.OUT/'evaluation_frozen.json',dict(complete=True,outputs={str(p):run.sha(p) for p in [*d.iterdir(),run.OUT/'summary.json',run.OUT/'report.md']}));run.stage('AB_geometry_complete',decision=summary['decision'])

if __name__=='__main__':evaluate()
