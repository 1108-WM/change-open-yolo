#!/usr/bin/env python3
"""Independent point-action replay and post-seal multi-threshold diagnosis."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha
from tools import run_fi1_anchor_local_decontamination as run


def independent_crop(parent, anchor, local, observations, params):
    excluded=defaultdict(set);supported=set()
    for v in observations:
        if not v['valid']:
            continue
        known=set(parent[v['reliable']].tolist())
        excluded[v['group']].update(known.intersection(parent[v['excluded']].tolist()))
        supported.update(known.intersection(parent[v['inside']].tolist()))
    if len(excluded)<params['min_pose_groups']:
        return parent.copy()
    counts=Counter(p for ss in excluded.values() for p in ss)
    remove={p for p,n in counts.items() if n>=params['min_pose_groups']}
    remove.intersection_update(parent[local].tolist())
    remove.difference_update(supported);remove.difference_update(anchor.tolist())
    if len(remove)>params['max_deleted_fraction']*len(parent):
        return parent.copy()
    return np.array([p for p in parent if p not in remove],dtype=np.int64)


def overlap(points, gt, ids, sizes):
    """Independent dense counts; evaluator minimum applies to geometric cover."""
    counts=Counter(gt[points].tolist())
    if len(points)<100:
        return np.zeros(len(ids))
    return np.array([counts[g]/(len(points)+n-counts[g]) for g,n in zip(ids,sizes)])


def evaluate(c):
    seal=run.read(run.OUT/'geometry_frozen.json');assert seal['complete']
    sys.addaudithook(run.Guard(c,labels=True).hook)
    run.verify_files(c['code']);run.verify_files(c['inputs']);run.verify_files(c['frozen_SL_models'])
    run.verify_files(seal['outputs'])
    groups={n:dict(old=Counter(),reprompt=Counter()) for n in c['groups']}
    totals={a:Counter() for a in run.ARMS};targets=[];actions=[];boundary=[];scene_results={}
    replayed=0;pointsets=0;thresholds=c['thresholds'];gate=c['geometry_gate']
    for s in c['scenes']:
        run.stage('postseal_geometry_diagnosis',scene=s)
        j=c['jobs'][s];assert sha(j['GT'])==c['GT_inputs'][j['GT']]
        gt=np.loadtxt(j['GT'],dtype=np.int64);assert len(gt)==j['binding']['point_count']
        allids,allcounts=np.unique(gt,return_counts=True)
        valid=(allids>=1000)&(allcounts>=100)&np.isin(allids//1000,c['class_ids'])
        ids,sizes=allids[valid],allcounts[valid];validset=set(ids.tolist())
        gs,nt=run.load_nt(j)
        original=np.array([overlap(p,gt,ids,sizes) for p in nt.values()])
        base=original.max(axis=0,initial=0)
        pd=run.OUT/'plan'/s;gd=run.OUT/'geometry'/s
        plans=run.read(pd/'candidates.json');decisions=run.read(gd/'decisions.json');views=run.read(gd/'views.json')
        decision_by={r['candidate_id']:r for r in decisions};byview=defaultdict(list)
        for v in views:byview[v['candidate_id']].append(v)
        nt_bytes={p.astype('<i8').tobytes() for p in nt.values()}
        unique={a:{} for a in run.ARMS};st={a:Counter() for a in run.ARMS}
        with np.load(pd/'arrays.npz',allow_pickle=False) as pa,np.load(gd/'points.npz',allow_pickle=False) as pp,np.load(gd/'evidence.npz',allow_pickle=False) as ev:
            for plan in plans:
                cid=plan['candidate_id'];parent=pa[cid+'_parent'];anchor=pa[cid+'_core'];local=pa[cid+'_local']
                assert np.array_equal(parent,nt[plan['parent']])
                rawcore=Counter(gt[anchor].tolist());known=[(n,g) for g,n in rawcore.items() if g in validset]
                anchor_n,anchor_gid=max(known,default=(0,None))
                reliable=anchor_gid is not None and anchor_n>=c['parameters']['min_core_points'] and anchor_n/max(1,len(anchor))>=.75
                before=overlap(parent,gt,ids,sizes);parent_gid=int(ids[int(np.argmax(before))]) if len(ids) and before.max()>0 else None
                core_state=dict(reliable=bool(reliable),gt_id=anchor_gid,dominant_fraction=anchor_n/max(1,len(anchor)),
                                parent_best_gt=parent_gid,parent_best_differs=bool(reliable and parent_gid!=anchor_gid))
                detail=dict(scene=s,candidate_id=cid,source=plan['source'],parent=plan['parent'],anchor=core_state,arms={})
                for arm in run.ARMS:
                    observations=[]
                    for v in byview[cid]:
                        vi=v['view_index']
                        item={k:ev[f'{cid}_v{vi}_{arm}_{k}'] for k in ('reliable','inside','excluded')}
                        item.update(group=v['group'],valid=v[arm+'_valid']);observations.append(item)
                        quality=v['old_quality' if arm=='old' else 'new_quality']
                        stability=v['old_stability' if arm=='old' else 'new_stability']
                        valid_view=(v['core_reliable_points']>=c['parameters']['min_core_visible_points'] and
                                    v['core_reliable_points']/len(anchor)>=c['parameters']['min_core_visible_fraction'] and
                                    v[arm+'_core_support']>=c['parameters']['core_mask_coverage'] and
                                    quality>c['parameters']['sam_quality_min'] and stability>=c['parameters']['sam_stability_min'])
                        assert bool(valid_view)==item['valid']
                    expected=independent_crop(parent,anchor,local,observations,c['parameters'])
                    actual=pp[cid+'_'+arm];rec=decision_by[cid]['arms'][arm]
                    assert np.array_equal(expected,actual)
                    assert hashlib.sha256(actual.astype('<i8').tobytes()).hexdigest()==rec['points_sha256']
                    assert set(anchor.tolist())<=set(actual.tolist()) and set(actual.tolist())<=set(parent.tolist())
                    assert rec['deleted']==len(parent)-len(actual)
                    replayed+=1;pointsets+=1;st[arm]['parents']+=1
                    removed=np.setdiff1d(parent,actual,assume_unique=True);after=overlap(actual,gt,ids,sizes)
                    token=actual.astype('<i8').tobytes();duplicate=token in nt_bytes
                    if rec['accepted']:
                        st[arm]['accepted_actions']+=1;st[arm]['deleted_points']+=len(removed)
                        st[arm]['duplicate_original_NT']+=duplicate
                        if not duplicate:
                            unique[arm].setdefault(token,actual)
                        st[arm]['unknown_anchor_actions']+=not reliable
                        if reliable:
                            deleted_target=int(np.sum(gt[removed]==anchor_gid));deleted_pollution=len(removed)-deleted_target
                            parent_target=int(np.sum(gt[parent]==anchor_gid));fraction=deleted_target/max(1,parent_target)
                            st[arm]['reliable_anchor_actions']+=1
                            st[arm]['deleted_target_points']+=deleted_target
                            st[arm]['deleted_pollution_points']+=deleted_pollution
                            st[arm]['parent_target_points']+=parent_target
                            st[arm]['excess_target_loss_actions']+=fraction>gate['max_per_action_target_loss_fraction']
                            best_gid=int(ids[int(np.argmax(after))]) if len(ids) and after.max()>0 else None
                            drift=best_gid!=anchor_gid;st[arm]['identity_drift']+=drift
                            before_anchor=float(before[np.where(ids==anchor_gid)[0][0]])
                            after_anchor=float(after[np.where(ids==anchor_gid)[0][0]])
                            extra=dict(deleted_target_points=deleted_target,deleted_pollution_points=deleted_pollution,
                                       target_loss_fraction=fraction,before_anchor_iou=before_anchor,after_anchor_iou=after_anchor,
                                       refined_best_gt=best_gid,identity_drift=bool(drift))
                        else:
                            st[arm]['unknown_deleted_points']+=len(removed);extra={}
                        lost=ids[(before>.75)&(after<=.75)].tolist();st[arm]['parent_high_iou_loss_events']+=len(lost)
                    else:
                        lost=[];extra={}
                    detail['arms'][arm]={**rec,**extra, 'duplicate_original_NT':bool(duplicate),
                                         'parent_high_iou_lost_gt':lost}
                actions.append(detail)
                for v in byview[cid]:
                    vi=v['view_index']
                    row=dict(scene=s,candidate_id=cid,frame=v['frame'],anchor_reliable=bool(reliable),
                             old_valid=v['old_valid'],reprompt_valid=v['reprompt_valid'],
                             changed_pixel_fraction=v['changed_pixel_fraction'],mask_iou=v['mask_iou'],
                             old_SAM_replay_iou=v['old_SAM_replay_iou'])
                    if reliable:
                        with np.load(gd/'observations'/f"frame{v['frame']:06d}.npz",allow_pickle=False) as z:
                            visible_target=np.flatnonzero(z['reliable']&(gt==anchor_gid))
                        for arm in run.ARMS:
                            region=ev[f'{cid}_v{vi}_{arm}_region']
                            hit=len(np.intersect1d(region,visible_target,assume_unique=True))
                            row[arm+'_visible_region_iou']=hit/max(1,len(region)+len(visible_target)-hit)
                    boundary.append(row)
        best={}
        for arm in run.ARMS:
            q=np.array([overlap(p,gt,ids,sizes) for p in unique[arm].values()]).reshape(-1,len(ids))
            best[arm]=np.maximum(base,q.max(axis=0,initial=0))
            st[arm]['unique_new_geometries']=len(unique[arm])
            st[arm]['duplicate_new_actions']=st[arm]['accepted_actions']-st[arm]['duplicate_original_NT']-len(unique[arm])
            for t in thresholds:
                st[arm][f'pool_crossing_{t:.2f}']=int(np.sum((base<=t)&(best[arm]>t)))
                st[arm][f'already_covered_improved_{t:.2f}']=int(np.sum((base>t)&(best[arm]>=base+gate['pool_iou_improvement'])))
            totals[arm].update(st[arm])
            for n,ss in c['groups'].items():
                if s in ss:groups[n][arm].update(st[arm])
        scene_results[s]={a:dict(v) for a,v in st.items()}
        for gid,baseline,old,new in zip(ids,base,best['old'],best['reprompt']):
            value=(new>=baseline+gate['pool_iou_improvement'] or
                   any(baseline<=t<new for t in (.75,.90,.95)))
            paired=new>=old+gate['paired_iou_improvement']
            targets.append(dict(scene=s,gt_id=int(gid),NT_best=float(baseline),old_pool_best=float(old),
                                reprompt_pool_best=float(new),new_value=bool(value),paired_improved=bool(paired),
                                already075_to_above090=bool(.75<baseline<=.90<new),
                                crossing_thresholds=[t for t in thresholds if baseline<=t<new]))
    new=totals['reprompt'];accepted=new['accepted_actions'];known=new['reliable_anchor_actions']
    deleted_known=new['deleted_target_points']+new['deleted_pollution_points']
    pollution_precision=new['deleted_pollution_points']/max(1,deleted_known)
    target_loss=new['deleted_target_points']/max(1,new['parent_target_points'])
    value=[r for r in targets if r['new_value']];paired=[r for r in targets if r['paired_improved']]
    scene_actions={s for s,r in scene_results.items() if r['reprompt'].get('unique_new_geometries',0)>0}
    checks=dict(
        minimum_actions=new['unique_new_geometries']>=gate['min_new_actions'],
        action_scene_support=len(scene_actions)>=gate['min_action_scenes'],
        multi_threshold_or_continuous_value=len(value)>=gate['min_value_targets'] and len({r['scene'] for r in value})>=gate['min_value_scenes'],
        paired_old_mask_improvement=len(paired)>=gate['min_value_targets'] and len({r['scene'] for r in paired})>=gate['min_value_scenes'],
        positive_sum_paired_pool_iou=sum(r['reprompt_pool_best']-r['old_pool_best'] for r in targets)>0,
        reliable_pollution_precision=deleted_known>0 and pollution_precision>=gate['deletion_pollution_precision'],
        aggregate_target_loss=target_loss<=gate['max_target_loss_fraction'],
        per_action_target_loss=new['excess_target_loss_actions']==0,
        unknown_anchor_fraction=accepted>0 and new['unknown_anchor_actions']/accepted<=gate['max_unknown_action_fraction'],
        no_reliable_anchor_drift=new['identity_drift']<=gate['max_identity_drift'],
        no_parent_high_iou_loss=new['parent_high_iou_loss_events']<=gate['max_parent_high_iou_losses'])
    advance=all(checks.values())
    result=dict(complete=True,geometry_gate_passed=advance,gate_checks=checks,
                arms={a:dict(v) for a,v in totals.items()},scenes=scene_results,
                groups={n:{a:dict(v) for a,v in arms.items()} for n,arms in groups.items()},
                targets=targets,actions=actions,boundary_views=boundary,
                useful_targets=len(value),paired_improved_targets=len(paired),
                reliable_pollution_precision=pollution_precision,target_loss_fraction=target_loss,
                original_NT_retained=True,new_instance_AP=False,own_semantics_acquired=False,
                independent_action_replays=replayed,pointsets_checked=pointsets)
    freeze_json(run.OUT/'geometry_quality.json',result)
    report(result,c)
    run.verify_files(c['frozen_SL_models']);run.verify_files(c['code'])
    outputs={str(run.OUT/n):sha(run.OUT/n) for n in ('geometry_quality.json','report.md')}
    freeze_json(run.OUT/'independent_verified.json',dict(complete=True,geometry_outputs=len(seal['outputs']),
                independent_action_replays=replayed,pointsets_checked=pointsets,outputs=outputs,
                GT_post_geometry_seal=True,SL_weights_unchanged=True,new_training=False,new_instance_AP=False))
    freeze_json(run.OUT/'decision.json',dict(complete=True,geometry_gate_passed=advance,checks=checks,
                next_phase='freeze_own_semantics_SL_contract' if advance else 'stop_before_own_semantics_SL',
                new_training=False,SL_modified=False,production_modified=False))
    run.stage('geometry_passed_pending_SL' if advance else 'complete_geometry_gate_failed',complete=not advance,
              geometry_gate_passed=advance)


def report(r,c):
    lines=['# 锚点约束局部去污染：旧掩码与同模型重新提示','',
           '原开发200中固定10场景、原五折×旧90/新110；每场景最多4N/4T。核心、视角、原GD框、SAM ViT-B与局部删除算子相同，新臂只增加3个核心正点。全部计划及几何先封存，之后读取白名单GT。未训练，不改SL或生产。','',
           f"几何接入门：**{r['geometry_gate_passed']}**。有价值目标{r['useful_targets']}，相对旧mask臂改善目标{r['paired_improved_targets']}。覆盖与可见区域IoU不是实例AP。",'',
           '|证据臂|接受动作|独有新点集|未知锚动作|删除目标点|删除污染点|身份漂移|父>.75失守事件|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in run.ARMS:
        v=r['arms'][a]
        lines.append('|'+a+'|'+'|'.join(str(v.get(k,0)) for k in
            ('accepted_actions','unique_new_geometries','unknown_anchor_actions','deleted_target_points',
             'deleted_pollution_points','identity_drift','parent_high_iou_loss_events'))+'|')
    lines+=['','|严格IoU|旧mask新增全NT覆盖|重新提示新增全NT覆盖|旧mask已覆盖目标提升>=.02|重新提示已覆盖目标提升>=.02|',
            '|---|---:|---:|---:|---:|']
    for t in c['thresholds']:
        tag=f'{t:.2f}';lines.append('|>'+tag+'|'+'|'.join(str(r['arms'][a].get(k+tag,0)) for k,a in
                    [('pool_crossing_','old'),('pool_crossing_','reprompt'),
                     ('already_covered_improved_','old'),('already_covered_improved_','reprompt')])+'|')
    lines+=['',f"原NT已>.75且从<=.90提升到>.90的目标：{sum(q['already075_to_above090'] for q in r['targets'])}。",'',
            f"新臂可靠锚动作的删除污染比例{r['reliable_pollution_precision']:.6f}，合计目标误删比例{r['target_loss_fraction']:.6f}；未知锚动作/删除点另列，不能将可靠子集外推为全池安全概率。",'',
            '|预注册检查|通过|','|---|---|']
    lines.extend(f'|{k}|{v}|' for k,v in r['gate_checks'].items())
    paired=[v for v in r['boundary_views'] if v.get('anchor_reliable') and v['old_valid'] and v['reprompt_valid']]
    allknown=[v for v in r['boundary_views'] if v.get('anchor_reliable')]
    for name,rows in [('全部可靠锚视图（包含弃权）',allknown),('双臂有效视图（条件子集）',paired)]:
        if rows:
            old=np.array([v['old_visible_region_iou'] for v in rows]);new=np.array([v['reprompt_visible_region_iou'] for v in rows])
            lines+=['',f'{name}：{len(rows)}；实测可见区域平均IoU旧={old.mean():.6f}、新={new.mean():.6f}，改善/退化/相同={int((new>old).sum())}/{int((new<old).sum())}/{int((new==old).sum())}。区域IoU不包含不可见目标，不能当完整实例质量。']
    lines+=['','动作有效性与完整边界统计均保留，不能用双臂有效子集替代全候选结论。失守是同父替换假设的配对/目标事件，原NT实际保留；没有K800输出或实际SL覆盖损失统计。',
            '', '门未通过时停止于几何阶段，不获取自身语义，不降低阈值、增加视角或训练；门通过才进入另行冻结的自身语义/完整扩池上下文/SL三臂对照。',
            '', '局部域来自父几何与原场景点；core稳定支持仍不保证实例身份。限定旧30帧、3组与同原GD框，负结果只对应这一小规模实现，不否定所有锚点分割或污染裁剪方法。',
            '', '完整每目标IoU、每动作与弃权、逐视图、N/T源与旧90/新110/五折结果在geometry_quality.json；contract.json及code_snapshot绑定当前版本，independent_verified.json记录动作独立重放和输入/模型身份。']
    path=run.OUT/'report.md';text='\n'.join(lines)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):evaluate(run.read(run.OUT/'contract.json'))
