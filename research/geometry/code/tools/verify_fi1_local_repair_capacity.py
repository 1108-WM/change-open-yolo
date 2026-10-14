#!/usr/bin/env python3
"""Post-seal independent metric/ratio checks and a concrete route decision."""
from pathlib import Path
from collections import Counter,defaultdict
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from tools import diagnose_fi1_local_repair_capacity as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def score(points,gt,gid):
    if len(points)<100:return 0.
    hit=int(np.sum(gt[points]==gid))
    return hit/(len(points)+int(np.sum(gt==gid))-hit)


def repair(parent,editable,gt,gid,observable=None):
    allowed=set(editable.tolist())
    if observable is not None:allowed={p for p in allowed if observable[p]}
    current=set(parent.tolist())
    current.difference_update(allowed)
    current.update(p for p in allowed if gt[p]==gid)
    if len(current)<100:
        current.update(sorted(set(parent.tolist())-current)[:100-len(current)])
    return np.array(sorted(current),np.int64)


def ratio_ceiling(parent,editable,gt,gid,units):
    """Independent fractional optimization, without ratio sorting/prefixes."""
    inside=set(editable.tolist());outside=np.array([p for p in parent if p not in inside],np.int64)
    a=int(np.sum(gt[outside]==gid));b=len(outside)-a;n=int(np.sum(gt==gid))
    by=defaultdict(lambda:[0,0])
    for p in editable:
        key=tuple(np.atleast_1d(units[p]).tolist());by[key][gt[p]!=gid]+=1
    values=np.array(list(by.values()),np.int64).reshape(-1,2)
    fraction=a/(n+b)
    for _ in range(200):
        use=values[:,0]-fraction*values[:,1]>0
        t,f=values[use].sum(axis=0)
        new=(a+t)/(n+b+f)
        if abs(new-fraction)<1e-14:return float(new)
        fraction=new
    raise AssertionError('fractional solver did not converge')


def main():
    c=run.read(run.OUT/'contract.json');assert run.read(run.OUT/'verified.json')['complete']
    sys.addaudithook(run.LabelGuard(c,labels=True).hook)
    run.old.verify_files(c['code']);run.old.verify_files(c['frozen_SL_models'])
    run.old.verify_files(run.read(run.OUT/'plan_frozen.json')['outputs'])
    run.old.verify_files(run.read(run.OUT/'verified.json')['outputs'])
    checks=Counter();rows=[];global_sp={};group=defaultdict(lambda:Counter())
    for s in c['scenes']:
        j=c['jobs'][s];assert sha(j['GT'])==c['GT_inputs'][j['GT']]
        gt=np.loadtxt(j['GT'],dtype=np.int64);raw=np.load(j['paths']['processed'],mmap_mode='r')
        ids,sizes=run.valid_targets(gt,c['class_ids']);_,nt=run.old.load_nt(j)
        baseline={int(g):max(score(p,gt,int(g)) for p in nt.values()) for g in ids}
        rr=run.read(run.OUT/'diagnosis'/s/'rows.json');rows.extend(rr)
        inv=run.read(run.OUT/'plan'/s/'frames.json');hashes=run.read(run.OUT/'diagnosis'/s/'frame_hashes.json')
        base=Path(j['paths']['processed']).parent
        available=[{int(q.stem) for q in (base/n).iterdir() if q.stem.isdigit()} for n in ('color','depth','poses')]
        complete=set.intersection(*available)
        assert complete=={r['frame'] for r in inv['records']}|set(inv['invalid_poses'])
        assert len(hashes)==len(inv['records'])==c['inventory'][s]['valid']
        for r in inv['records']:
            for key in ('color','depth','pose_file'):run.assert_stat(r[key])
            assert r['depth']['path'] in hashes
        checks['valid_frame_bindings']+=len(inv['records']);checks['invalid_poses']+=len(inv['invalid_poses'])
        sp=raw[:,9].astype(np.int64);vox=np.floor(raw[:,:3]/c['voxel_size']).astype(np.int64)
        valid=set(ids.tolist());bysp=defaultdict(set)
        for segment,label in zip(sp,gt):
            if label in valid:bysp[int(segment)].add(int(label))
        global_sp[s]=dict(superpoints=len(np.unique(sp)),valid_instance_mixed_superpoints=sum(len(v)>1 for v in bysp.values()))
        with np.load(run.OUT/'plan'/s/'domains.npz') as d,np.load(run.OUT/'diagnosis'/s/'error_visibility.npz') as ev:
            domain=np.unique(np.concatenate([d[r['candidate_id']+'_parent_halo15'] for r in rr]))
            assert np.isin(ev['point_ids'],domain).all()
            visibility={}
            for bank,field in [('cached30','cached30_groups'),('all_frames','all_frames_groups')]:
                full=np.zeros(len(gt),np.uint8);full[ev['point_ids']]=ev[field];visibility[bank]=full
                assert np.isin(ev[field],[0,1,2]).all()
            for row in rr:
                if row['target'] is None:checks['unknown_parents']+=1;continue
                cid=row['candidate_id'];gid=row['target'];parent=d[cid+'_parent']
                assert abs(score(parent,gt,gid)-row['parent_iou'])<1e-12
                assert abs(baseline[gid]-row['NT_best'])<1e-12
                for name in run.DOMAINS:
                    editable=d[cid+'_'+name];q=row['domains'][name]
                    result=repair(parent,editable,gt,gid)
                    assert abs(score(result,gt,gid)-q['oracle_iou'])<1e-12
                    assert len(set(parent)-set(result))==q['removed_pollution']
                    assert len(set(result)-set(parent))==q['added_target']
                    checks['independent_oracle_pointsets']+=1
                    for bank,seen in visibility.items():
                        for minimum in (1,2):
                            result=repair(parent,editable,gt,gid,seen>=minimum)
                            assert abs(score(result,gt,gid)-q[bank][str(minimum)]['iou'])<1e-12
                            changed=set(result)^set(parent)
                            assert all(seen[p]>=minimum for p in changed)
                            checks['independent_oracle_pointsets']+=1
                    for unit,values in [('superpoint',sp),('voxel2cm',vox)]:
                        ceiling=ratio_ceiling(parent,editable,gt,gid,values)
                        assert abs(ceiling-q['units'][unit]['iou'])<1e-12
                        checks['independent_unit_ceilings']+=1
                halo=row['domains']['parent_halo15']
                for name in ('all',row['source'],'anchored' if row['original_anchor'] else 'anchor_abstain',
                             'fold'+str(c['folds'][s]),'historical90' if s in c['groups']['historical90'] else 'new110'):
                    group[name]['known_parents']+=1
                    group[name]['halo_unrestricted_pool_gain002']+=halo['oracle_iou']>=row['NT_best']+.02
                    group[name]['halo_allframes2_pool_gain002']+=halo['all_frames']['2']['iou']>=row['NT_best']+.02
    summary=run.read(run.OUT/'summary.json');f=summary['funnel']['all_non_target']
    assert [f[n] for n in ('reliable_anchors','parent_pollution','editable_outside_core','selected_visible_2groups','old_valid_2groups','reprompt_valid_2groups','old_after_positive_veto')]==[21,4461,2955,221,8,0,0]
    assert not any(r['value'] for r in summary['selected_view_oracles'])
    result=dict(complete=True,checks=dict(checks),global_superpoint_mixing=global_sp,
                groups={n:dict(v) for n,v in group.items()},
                code_sha256=sha(__file__),original_funnel_reproduced=True,
                original_selected_views_no_oracle_pool_value=True,
                diagnostic_only=True,new_training=False,new_instance_AP=False)
    freeze_json(run.OUT/'independent_check.json',result)
    lines=['# 修复能力检查后的路线决策','',
           '停止旧框正点提示与保守裁剪规则叠加，推进候选条件的局部实例归属学习准备。理由是已确认局部范围和可用观测仍包含池级修复机会；这不是已证明分割模块可学习或可接入SL。','',
           '原21可靠锚漏斗完整复现4461→2955→221→8（旧有效）／0（新有效）→0（旧删除）。补充细分发现旧有效mask之后的“两组带边距排除”已经为0，因此正支持否决在本批污染点上没有再削减点数。不能把其机制风险写成本轮已量化的直接主因。原两个所选视角可见区域的完美污染删除，无预算、完整父及旧边界两个口径均没有≥.02完整NT增益或跨>.75/.90/.95。','',
           '|范围|任意证据的理想池增益≥.02目标|旧30帧≥2组可见|全部有效帧≥2组可见|',
           '|---|---:|---:|---:|']
    for name in run.DOMAINS:
        nums=[summary['scenarios'][name+'/'+e]['counts']['unique_pool_gain_ge002_targets']
              for e in ('unrestricted','cached30_2','all_frames_2')]
        lines.append('|'+name+'|'+'|'.join(map(str,nums))+'|')
    lines+=['','候选与范围固定在GT前，不把上述GT理想点集作为预测或按其补选训练/推理动作。80父候选含40N/40T，73有有效目标，7未知；首批只覆盖10场景。全部14,321完整帧中13,248有效，1,073 pose无效；无效pose不能作为可见证据。全帧至少两组可见的可增删邻域理论增益覆盖26目标/9场景，跨>.75为1、>.90为13、>.95为16；所以边界细化是主要机会，不是大量新增>.75目标。','',
           '优先改观测获取：旧局部边界本身已有17目标理想增量，但旧30帧仅5、全帧两组可见14。父+.15m邻域可增删把理想增量扩展到32，旧30帧7、全帧26。扩大域可增加机会，但不能代替获取分歧区域可见证据；至少一个组的31目标与两组26目标之差也需保留为回退区域。下一步推理选视角只能依据候选分歧/邻居/几何可见性，不用GT理想点集排名。','',
           '当前原始ScanNet超点并未显示有效两实例混合：10场景全量有效实例混合超点为0。这个现象有数据构造原因：processed第9列来自原始segs.json，而实例GT在aggregation的segments集合上赋值；不能把0混合当成学习模块能分离真实边界的证据，也不能推成跨数据保证。处理源码：tools/prepare_scannet200_train_scene.py:154、models/Mask3D/mask3d/datasets/preprocessing/scannet_preprocessing.py:135。几何输入只允许列0:10；后两列语义/实例标签不能进入输入。','',
           '2cm整体素比原超点在本样本更粗：可增删域32个理想≥.02池增益目标降至24，23/73父候选有≥.02 IoU上界损失。优先保留点或原细超点的空间关系，不默认2cm整块决策足够。所有整单元数字仍是允许范围边界裁断的乐观上界，<100点最优解单列在summary；不能当执行成功。','',
           f"独立重算{checks['independent_oracle_pointsets']}个理想点集与{checks['independent_unit_ceilings']}个分式上界，核对13,248有效帧绑定、1,073无效pose和全池基线。原保存45帧在本次156,625个误差域点上的可靠可见性一致（0差异），不是对全场景全部点的重新投影复现。5项测试通过。",'',
           '下一阶段：在开发200原物理五折固定候选/邻域采样规则，用共享小型模块监督候选条件局部归属；覆盖邻居污染、混合、缺失和好候选保持。冻结骨干与现有SL，未知/遮挡区回退。应分别验收父收益与完整NT池收益、误删、漂移、重复；折外几何成立后才获取自身语义并完整重算扩池上下文及固定SL两种输出对照。此次尚未训练、没有自身语义、新实例AP或生产改动。']
    (run.OUT/'decision.md').write_text('\n'.join(lines)+'\n')
    freeze_json(run.OUT/'completion_verified.json',dict(complete=True,
                outputs={str(run.OUT/n):sha(run.OUT/n) for n in
                         ('contract.json','plan_frozen.json','verified.json','summary.json','report.md',
                          'independent_check.json','decision.md','tests_verified.json')},
                independent_code_sha256=sha(__file__),SL_weights_unchanged=True,
                new_training=False,new_instance_AP=False,production_modified=False))
    print(dict(checks),flush=True)


if __name__=='__main__':main()
