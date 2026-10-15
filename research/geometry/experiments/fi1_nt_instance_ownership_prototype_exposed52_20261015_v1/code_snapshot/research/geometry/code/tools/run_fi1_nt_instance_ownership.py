#!/usr/bin/env python3
"""Fixed N/T instance-reference ownership prototype; query reused for clipping only."""
from pathlib import Path
from collections import Counter
import argparse
import shutil
import sys
import time
import numpy as np
from scipy.sparse import coo_matrix
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_local_block_attribution_selection as previous
import fi1_nt_instance_ownership_core as core
from analyze_fi1_local_block_attribution_selection import census, union
from tools.fi1_r0_r1_protocol import original_aliases
D=previous.diag
MET=previous.met
BASE=previous.OUT
OUT=D.ROOT/'research/geometry/experiments/fi1_nt_instance_ownership_prototype_exposed52_20261015_v1'
ARMS=('all_C','relation_C')


def stage(name,**kw):
    D.write_json(OUT/'execution.json',dict(stage=name,time=time.time(),**kw))
    print(name,kw,flush=True)


def prepare():
    assert not OUT.exists();D.reserve();previous.prior.guard()
    old=D.read(BASE/'contract.json')
    inputs={str(BASE/'contract.json'):D.sha(BASE/'contract.json'),
            str(BASE/'predictions_frozen.json'):D.sha(BASE/'predictions_frozen.json')}
    inputs.update(D.read(BASE/'predictions_frozen.json')['outputs'])
    for scene,job in old['jobs'].items():
        for p in (Path(job['binding']['geometry']),Path(job['binding']['points']),
                  Path(job['asset_root'])/'structure'/scene/'data.npz',
                  Path(job['asset_root'])/'structure'/scene/'candidates.json',
                  previous.SOURCE/'relations'/scene/'parents.json'):
            inputs[str(p)]=old['inputs'][str(p)]
    files=[Path(__file__).resolve(),Path(core.__file__),
           Path(__file__).with_name('verify_fi1_nt_instance_ownership.py'),
           D.ROOT/'research/geometry/code/tests/test_fi1_nt_instance_ownership.py',
           Path(previous.__file__),Path(MET.__file__)]
    D.verify(inputs)
    OUT.mkdir(parents=True)
    contract=dict(version=OUT.name,scenes=old['scenes'],cohorts=old['cohorts'],jobs=old['jobs'],
        class_ids=old['class_ids'],GT_inputs=old['GT_inputs'],inputs=inputs,
        code={str(p):D.sha(p) for p in files},parameters=core.PARAMETERS,
        scope='207 sampled T parents/52 exposed physical scenes/original halo and editable blocks; internal development only.',
        corrected_interpretation='Previous round had no attribution selector; it tested all blocks/clipping only. Mixed-block clipping lost 49.09% of target additions. This round implements actual accept/reject/unknown ownership.',
        relation='All deduplicated original N/T pointsets, full-scene pair intersections and directional coverage. Query and semantic scores never enter ownership. Local graph nodes intersect original halo or belong to original <=.15m neighbors, without original top4 truncation. Complete-link duplicate groups computed scene-wise, geometry-key order, reciprocal coverage>=.5; no transitive bridge grouping.',
        instance_reference='A duplicate group is a provisional credible instance reference iff >=100-point native and track members have distinct pointsets. A single geometry with N/T aliases is not independent support. N and cross-source agreement are hypotheses, never GT correctness. Groups may be mixed or split one object.',
        ownership='Group covers parent/block if ANY member covers >=.5 of its points. Exactly one parent group and exactly one block group: same -> accept, different -> reject. Missing groups or any multiple group competition -> unknown. No majority vote, ranking, tie breaking into ownership, query or common-frame gate.',
        broad_N='Complete-link prevents one large bridging N from connecting disjoint seeds transitively. Multiple plausible instance groups still abstain. Postseal audit measures accepted supporting N cross-object composition and false bridges; no assumption N is correct.',
        clipping='Reuse exactly previous GT-free highest-block-mean query and >=.5 clipped added points. Query is not used to choose a group/block. Only accepted blocks enter relation_C; all_C exactly replays previous union. Preserve every original parent point and original NT. At most one union per sampled T parent per arm.',
        fixed_comparisons=['all_C: previous all-block clipped baseline','relation_C: tri-state N/T ownership then identical clipped readout'],
        independent_validation_gate='Before any labels: provisional development pass requires >=2 physical scenes with same-parent completeNT IoU gain>=.02 and added-target fraction>=.9; no good(parentIoU>=.9) copy loss>.02; aggregate stable-target addition precision>=.9 with annotation unknown in denominator; no accepted unresolved-identity parent for a safety claim. These are evaluation criteria, NOT inference gates. Abstaining everywhere fails. Failure stops this fixed prototype, not all N/T/query methods.',
        followup='Only if fixed development rule passes all risks, freeze physically isolated unused development scenes excluding exposed52/variants and final holdout; do not use GT for scene selection. After isolated validation only, downstream own-semantics/frozen SL/AP. No AP or extra observations/training automatically on development failure.',
        evaluation='All207 parents and1308 blocks including mixed/unknown. Ownership and clipping separate; actual per-parent union; fullNT continuous delta and gain02; neighbor/unknown/good-copy loss; supporter-N dominance/redundancy; low-overlap opportunities and all negatives. GT never chooses references, queries, groups, blocks, output points.',
        training=False,new_observations=False,threshold_search=False,production_modified=False,
        github_push=False,reading_entry_modified=False)
    D.seal(OUT/'contract.json',contract)
    for p in files:
        dest=OUT/'code_snapshot'/p.relative_to(D.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    stage('contract_frozen',scenes=52,parents=207)


def generate():
    c=D.read(OUT/'contract.json');previous.prior.guard();D.verify(c['code']);D.verify(c['inputs'])
    outputs={};total=Counter()
    for si,(scene,job) in enumerate(c['jobs'].items()):
        stage('GT_free_relation_and_ownership',scene=scene,completed=si,total=52)
        geometry,nt=D.load_nt(job);bykey={g['geometry_key']:g for g in geometry};keys=sorted(nt)
        pointsets=[nt[k] for k in keys];sizes=np.array([len(p) for p in pointsets],np.int64)
        sources=[sorted({a['source'] for a in original_aliases(bykey[k])}) for k in keys]
        rows=np.concatenate(pointsets);cols=np.repeat(np.arange(len(keys)),sizes)
        incidence=coo_matrix((np.ones(len(rows),np.int32),(rows,cols)),shape=(job['binding']['point_count'],len(keys))).tocsr()
        intersections=(incidence.T@incidence).toarray().astype(np.int64)
        assert np.array_equal(np.diag(intersections),sizes)
        groups=core.duplicate_groups(intersections,sizes,keys);credible=core.credible_groups(groups,sizes,sources)
        keyindex={k:i for i,k in enumerate(keys)}
        nodes=[dict(node=i,key=k,points=int(sizes[i]),sources=sources[i]) for i,k in enumerate(keys)]
        grouprows=[dict(group=g,members=members,credible=g in credible) for g,members in enumerate(groups)]
        parents=D.read(previous.SOURCE/'relations'/scene/'parents.json')
        baseblocks=D.read(BASE/'predictions'/scene/'blocks.json')
        metadata={r['candidate_id']:r for r in D.read(Path(job['asset_root'])/'structure'/scene/'candidates.json')}
        blockrows=[];parentrows=[];unionrows=[];arrays=dict(intersections=intersections,sizes=sizes)
        with np.load(BASE/'predictions'/scene/'data.npz') as prev,np.load(Path(job['asset_root'])/'structure'/scene/'data.npz') as st:
            for p in parents:
                cid=p['candidate_id'];parent=st[cid+'_parent'];halo=st[cid+'_halo']
                assert np.array_equal(parent,nt[p['parent_geometry']])
                parenthits=intersections[keyindex[p['parent_geometry']]]
                halohits=np.asarray(incidence[halo].sum(axis=0)).ravel()
                local=set(np.flatnonzero(halohits))
                for n in metadata[cid]['neighbors']+metadata[cid]['omitted']:local.add(keyindex[n['key']])
                arrays[cid+'_parent_hits']=parenthits;arrays[cid+'_local_nodes']=np.array(sorted(local),np.int32)
                selected=[];accepted=[];rejected=[];unknown=[]
                for b in [b for b in baseblocks if b['candidate_id']==cid]:
                    bid=b['block_id'];block=prev[bid+'_block'];clip=prev[bid+'_clipped_proposal']
                    hits=np.asarray(incidence[block].sum(axis=0)).ravel().astype(np.int64)
                    decision=core.ownership(parenthits,len(parent),hits,len(block),groups,credible)
                    sponsors=groups[decision['parent_groups'][0]] if decision['state']=='accept' else []
                    bridge=[n for n in sponsors if parenthits[n]>=.5*len(parent) and hits[n]>=.5*len(block)]
                    arrays[bid+'_hits']=hits
                    row=dict(scene=scene,cohort=job['cohort'],candidate_id=cid,region=b['region'],block_id=bid,
                        points=len(block),clipped_points=len(clip),query=b['supporter'],anchor=b['anchor'],
                        query_parent_hard_fraction=b['query_parent_hard_fraction'],
                        anchor_supporter_hard_intersection=b['anchor_supporter_hard_intersection'],
                        anchor_supporter_hard_union=b['anchor_supporter_hard_union'],
                        frame_states=b['frame_states'],minimum_distance=b['minimum_distance'],
                        **decision,sponsor_nodes=sponsors,bridge_nodes=bridge,
                        native_bridge_nodes=[n for n in bridge if 'native' in sources[n]],
                        original_parent_points_preserved=True)
                    blockrows.append(row);total[decision['state']]+=1
                    if decision['state']=='accept':selected.append(clip);accepted.append(b['region'])
                    elif decision['state']=='reject':rejected.append(b['region'])
                    else:unknown.append(b['region'])
                chosen=union(selected)
                for arm,points in [('all_C',prev[cid+'_C_points']),('relation_C',np.union1d(parent,chosen))]:
                    assert not len(np.setdiff1d(parent,points));arrays[cid+'_'+arm]=points
                    added=np.setdiff1d(points,parent)
                    unionrows.append(dict(scene=scene,cohort=job['cohort'],candidate_id=cid,arm=arm,points=len(points),added=len(added),
                        accepted_regions=accepted if arm=='relation_C' else p['regions'],
                        extra_candidate=bool(len(added)),points_sha256=D.hashlib.sha256(points.astype('<i8').tobytes()).hexdigest()))
                    total[arm+'_actions']+=bool(len(added));total[arm+'_added']+=len(added)
                parentrows.append(dict(scene=scene,candidate_id=cid,parent_geometry=p['parent_geometry'],
                    local_nodes=len(local),accepted=accepted,rejected=rejected,unknown=unknown,
                    parent_groups=core.group_support(parenthits,len(parent),groups,credible)))
        directory=OUT/'predictions'/scene;directory.mkdir(parents=True)
        for name,data in [('nodes',nodes),('groups',grouprows),('parents',parentrows),('blocks',blockrows),('unions',unionrows)]:D.seal(directory/(name+'.json'),data)
        np.savez_compressed(directory/'data.npz',**arrays)
        D.seal(directory/'receipt.json',dict(GT_used=False,outputs={str(p):D.sha(p) for p in directory.iterdir()}))
        outputs.update({str(p):D.sha(p) for p in directory.iterdir()})
        total.update(parents=len(parents),blocks=len(blockrows),NT_nodes=len(nodes),groups=len(groups),credible_groups=len(credible));D.reserve()
    assert total['parents']==207 and total['blocks']==1308
    D.seal(OUT/'predictions_frozen.json',dict(complete=True,GT_used=False,counts=dict(total),outputs=outputs))
    stage('all_predictions_sealed',counts=dict(total))


def evaluate():
    c=D.read(OUT/'contract.json');D.verify(c['code']);frozen=D.read(OUT/'predictions_frozen.json');assert frozen['complete'];D.verify(frozen['outputs']);D.verify(c['GT_inputs'])
    allrows=[];allblocks=[];allpool=[];scopes=[];opportunities=[];outputs={};bridge_audit=[]
    for si,(scene,job) in enumerate(c['jobs'].items()):
        stage('postseal_GT',scene=scene,completed=si,total=52)
        gt=np.loadtxt(job['GT'],dtype=np.int64);ids,sizes=MET.valid_targets(gt,c['class_ids'])
        geometry,nt=D.load_nt(job);baseline=MET.pool_base(nt,gt,ids,sizes)
        nodes=D.read(OUT/'predictions'/scene/'nodes.json');parents=D.read(previous.SOURCE/'relations'/scene/'parents.json')
        blockrefs=D.read(OUT/'predictions'/scene/'blocks.json');unions=D.read(OUT/'predictions'/scene/'unions.json')
        oldrows={r['candidate_id']:r for r in D.read(BASE/'evaluation'/scene/'unions.json') if r['mode']=='oracle_target_points'}
        before_scope={r['candidate_id']:r for r in D.read(BASE/'evaluation/parent_scopes.json') if r['scene']==scene}
        rows=[];blocks=[];bridges=[];best={a:baseline.copy() for a in ARMS}
        with np.load(OUT/'predictions'/scene/'data.npz') as z,np.load(BASE/'predictions'/scene/'data.npz') as prev,np.load(Path(job['asset_root'])/'structure'/scene/'data.npz') as st:
            for p in parents:
                cid=p['candidate_id'];parent=st[cid+'_parent'];identity=MET.identity(parent,gt,ids,sizes)
                target=identity['target'] if identity['stable'] else None
                assert identity['scope']==before_scope[cid]['scope']
                scopes.append(dict(scene=scene,candidate_id=cid,**identity));accepted_nodes=set()
                for b in [b for b in blockrefs if b['candidate_id']==cid]:
                    bid=b['block_id'];whole=prev[bid+'_block'];clip=prev[bid+'_clipped_proposal']
                    bc=census(whole,gt,target);cc=census(clip,gt,target)
                    r=dict(b,parent_scope=identity['scope'],stable=identity['stable'],target=target,
                        block_GT=bc,clipped_GT=cc,dropped_GT=census(np.setdiff1d(whole,clip),gt,target),
                        possible_pool_opportunity=oldrows[cid].get('pool_delta',-1)>=.02-1e-12)
                    if identity['stable']:
                        single,_=MET.outcome(parent,np.union1d(parent,clip),gt,ids,sizes,identity,baseline)
                        r['single_C']=single
                    blocks.append(r)
                    if b['state']=='accept':accepted_nodes.update(b['native_bridge_nodes'])
                scores_by_arm={}
                best_native=0.;best_bridging_native=0.
                if identity['stable']:
                    ix=identity['index'];g=target
                    best_native=max((float(MET.ious(nt[n['key']],gt,ids,sizes)[ix]) for n in nodes if 'native' in n['sources']),default=0.)
                    for n in sorted(accepted_nodes):
                        points=nt[nodes[n]['key']];value=float(MET.ious(points,gt,ids,sizes)[ix]);best_bridging_native=max(best_bridging_native,value)
                        dist=census(points,gt,target)
                        bridges.append(dict(scene=scene,candidate_id=cid,node=n,key=nodes[n]['key'],
                            target=target,full_scene_GT=dist,full_scene_target_iou=value,
                            multiple_known_instances=dist['known_instances']>1,
                            substantial_other=dist['other']/len(points)>.1))
                for arm in ARMS:
                    points=z[cid+'_'+arm];r,score=MET.outcome(parent,points,gt,ids,sizes,identity,baseline)
                    selected=[b for b in blocks if b['candidate_id']==cid and (arm=='all_C' or b['state']=='accept')]
                    r.update(scene=scene,cohort=job['cohort'],candidate_id=cid,arm=arm,
                        selected_blocks=len(selected),accepted_blocks=sum(b['state']=='accept' for b in selected),
                        safe_gain02=bool(r.get('pool_delta',-1)>=.02-1e-12 and r['added'] and r.get('added_target',0)/r['added']>=.9),
                        same_parent_gain02=r.get('pool_delta',-1)>=.02-1e-12,
                        added_GT=census(np.setdiff1d(points,parent),gt,target),
                        best_native_iou=best_native if identity['stable'] else None,
                        best_bridge_native_iou=best_bridging_native if identity['stable'] else None,
                        duplicate_original=any(np.array_equal(points,pts) for pts in nt.values()))
                    if identity['stable']:
                        ix=identity['index'];best[arm][ix]=max(best[arm][ix],score[ix])
                        r['native_already_at_least_as_good']=best_native>=r['repaired_iou']-1e-12
                        r['bridging_native_already_at_least_as_good']=bool(accepted_nodes and best_bridging_native>=r['repaired_iou']-1e-12)
                    rows.append(r);scores_by_arm[arm]=r
                if oldrows[cid].get('pool_delta',-1)>=.02-1e-12:
                    opportunities.append(dict(scene=scene,candidate_id=cid,target=target,
                        point_oracle_gain=oldrows[cid]['pool_delta'],
                        arms={a:{k:scores_by_arm[a].get(k) for k in ('pool_delta','parent_delta','added','added_target','added_other','unknown_edits','safe_gain02','native_already_at_least_as_good')} for a in ARMS},
                        ownership=dict(Counter(b['state'] for b in blocks if b['candidate_id']==cid)),
                        blocks=[dict(region=b['region'],state=b['state'],reason=b['reason'],GT=b['block_GT'],
                            parent_groups=b['parent_groups'],block_groups=b['block_groups']) for b in blocks if b['candidate_id']==cid]))
        directory=OUT/'evaluation'/scene;directory.mkdir(parents=True)
        for name,value in [('blocks',blocks),('unions',rows),('native_bridges',bridges)]:
            path=directory/(name+'.json');D.seal(path,value);outputs[str(path)]=D.sha(path)
        allrows+=rows;allblocks+=blocks;bridge_audit+=bridges
        for arm in ARMS:allpool+=MET.pool_rows(scene,arm,baseline,best[arm],ids)
    modes={}
    for arm in ARMS:
        rr=[r for r in allrows if r['arm']==arm];stable=[r for r in rr if r['stable']];changed=[r for r in stable if r['changed']]
        added=sum(r['added'] for r in stable);target=sum(r.get('added_target',0) for r in stable)
        modes[arm]=dict(unions=MET.summarize_rows(rr),same_parent_pool=MET.pool_summary([r for r in allpool if r['mode']==arm]),
            stable_added=added,stable_added_target=target,stable_add_precision=target/added if added else None,
            annotation_unknown=sum(r['added_GT']['unknown'] for r in rr),
            unresolved_actions=sum(r['changed'] and not r['stable'] for r in rr),
            safe_gain02_scenes=len({r['scene'] for r in rr if r['safe_gain02']}),
            same_parent_gain02_parents=sum(r['same_parent_gain02'] for r in rr),
            safe_gain02_parents=sum(r['safe_gain02'] for r in rr),
            native_redundant_actions=sum(r['native_already_at_least_as_good'] for r in changed),
            bridge_native_redundant_actions=sum(r['bridging_native_already_at_least_as_good'] for r in changed),
            by_scope={scope:dict(parents=len(sub),actions=sum(r['changed'] for r in sub),added=sum(r['added'] for r in sub),
                annotated=sum(r['added_GT']['annotated'] for r in sub),annotation_unknown=sum(r['added_GT']['unknown'] for r in sub))
                for scope in sorted({r['scope'] for r in rr}) for sub in [[r for r in rr if r['scope']==scope]]},
            by_cohort={cohort:dict(parents=len(sub),actions=sum(r['changed'] for r in sub),same_parent_gain02_parents=sum(r['same_parent_gain02'] for r in sub),
                safe_gain02_scenes=len({r['scene'] for r in sub if r['safe_gain02']}),good_copy_loss_gt02=sum(r.get('good_loss',0)>.02+1e-12 for r in sub))
                for cohort in c['cohorts'] for sub in [[r for r in rr if r['cohort']==cohort]]})
    selected=modes['relation_C'];u=selected['unions']
    gates=dict(multiscene_safe_gain=selected['safe_gain02_scenes']>=2,
        good_copy_protection=u['good_loss_gt_02']==0,
        stable_add_precision=selected['stable_add_precision'] is not None and selected['stable_add_precision']>=.9,
        no_unresolved_actions=selected['unresolved_actions']==0,
        annotation_unknown=u['added']>0 and selected['annotation_unknown']/u['added']<=.1)
    # The prior all_C failure baseline must reproduce, including adverse cases.
    old=D.read(BASE/'evaluation_summary.json')['modes']['C']
    assert modes['all_C']['same_parent_pool']==old['same_parent_pool']
    for key in ('added','added_target','added_other','good_loss_gt_02','parent_delta'):
        assert abs(modes['all_C']['unions'][key]-old['unions'][key])<1e-12
    ownership_by_state={state:dict(blocks=len(rs),parent_scopes=dict(Counter(r['parent_scope'] for r in rs)),
        ownership=dict(Counter(r['block_GT']['ownership'] for r in rs)),
        stable_W={k:sum(r['block_GT'][k] for r in rs if r['stable']) for k in ('points','target','other','unknown')},
        stable_C={k:sum(r['clipped_GT'][k] for r in rs if r['stable']) for k in ('points','target','other','unknown')})
        for state in ('accept','reject','unknown') for rs in [[r for r in allblocks if r['state']==state]]}
    summary=dict(complete=True,scenes=52,parents=len(scopes),blocks=len(allblocks),
        scopes=dict(Counter(r['scope'] for r in scopes)),ownership=ownership_by_state,modes=modes,
        development_gates=gates,development_pass=all(gates.values()),
        native_bridge_audit=dict(rows=len(bridge_audit),multiple_known_instances=sum(r['multiple_known_instances'] for r in bridge_audit),
            substantial_other=sum(r['substantial_other'] for r in bridge_audit)),
        original_points_deleted=0,training=False,AP_run=False,independent_validation_run=False)
    directory=OUT/'evaluation'
    for name,value in [('pool',allpool),('parent_scopes',scopes),('opportunities',opportunities)]:
        path=directory/(name+'.json');D.seal(path,value);outputs[str(path)]=D.sha(path)
    D.seal(OUT/'summary.json',summary);outputs[str(OUT/'summary.json')]=D.sha(OUT/'summary.json')
    D.seal(OUT/'evaluation_frozen.json',dict(complete=True,outputs=outputs,GT_after_all_predictions=True,
        predictions_sha256=D.sha(OUT/'predictions_frozen.json')))
    stage('evaluation_complete',gates=gates,development_pass=summary['development_pass'])


def report():
    s=D.read(OUT/'summary.json');v=D.read(OUT/'verification.json');assert v['complete']
    assert not s['development_pass'],'Passed prototype requires independent validation before completion.'
    decision=dict(complete=True,status='stop_fixed_prototype_before_isolated_validation_and_AP',
        development_pass=False,gates=s['development_gates'],
        interpretation='This round actually implemented tri-state N/T ownership. Failure applies to this complete-link/cross-source/unique-reference construction, not proof of insufficient information or of all instance-relation methods. Query clipping remains imperfect on mixed blocks.',
        next_priority='Targeted observation or original T per-frame provenance for unresolved valuable blocks; separately freeze proposed observation/association tests. Do not automatically tune same thresholds, expand training, or merge an instance tree.',
        training=False,new_observations=False,independent_validation_run=False,AP_run=False,
        production_modified=False,github_push=False,reading_entry_modified=False)
    D.seal(OUT/'decision.json',decision)
    def pct(value):return 'NA' if value is None else f'{value:.2%}'
    lines=[]
    for arm,label in [('all_C','全部块原query裁剪'),('relation_C','N/T归属后同样裁剪')]:
        r=s['modes'][arm];u=r['unions'];p=r['same_parent_pool']
        lines.append(f"| {label} | {u['actions']} | {p['pool_gain02']} | {r['safe_gain02_scenes']} | {u['good_loss_gt_02']}/12 | {pct(r['stable_add_precision'])} | {r['unresolved_actions']} |")
    statelines=[]
    for state,r in s['ownership'].items():statelines.append(f"| {state} | {r['blocks']} | {r['ownership']} | {r['stable_C']} |")
    cases=[]
    for r in D.read(OUT/'evaluation/opportunities.json'):
        a=r['arms']['all_C'];b=r['arms']['relation_C']
        cases.append(f"| {r['scene']}/{r['candidate_id']} | {a['pool_delta']:+.6f} | {b['pool_delta']:+.6f} | {r['ownership']} |")
    text=f'''# N/T局部实例关系与三态块归属原型

2026-10-15；已完成开发测试，未通过预先固定验收，停止本原型，不进入独立场景/AP。范围仅52个已暴露物理场景、207个采样T父、原1,308块。未训练，原N/T保留。

## 本轮具体改变

承认前一轮只测试全部块和裁剪，没有实现归属选择器，不能据此认定现有证据不可学习。纯目标块98.40%保留也不能掩盖23个混合块中49.09%目标补点被丢掉；本轮不把裁剪视为已解决。

本轮已实现真实accept/reject/unknown，代码[fi1_nt_instance_ownership_core.py](../../code/tools/fi1_nt_instance_ownership_core.py)。query完全不进入身份图、组或接收条件，只复用前轮每块固定query及相同≥0.5新增点裁剪。拒绝和无法确定均不编辑，原父点从不删除。

关系图使用完整原N/T点集，含所有原候选源，无类别/分数过滤。保存全场景交集矩阵、方向覆盖、重复组以及每父/块的全部候选覆盖。局部节点为与原halo相交或属于原≤0.15m邻居的候选，不再仅取前4个邻居；原halo、分块、query集合、六帧观测本身均不改变。

固定最小规则（GT前封存于[contract.json](contract.json)）：几何key顺序进行complete-link分组，同组每对候选互相覆盖≥一半；不使用传递连通分量把两个不重叠物体经大N粘在一起。只有组内存在点数≥100的不同N和T点集时才作为暂定实例参照；同点集的两个来源别名不算独立证据。“可靠”只是待检验的跨来源几何假设。

某组任一成员覆盖父/块≥一半，该组就对其提出归属。父和块均恰有一个组且相同→接受；均恰有一个但不同→拒绝；缺参照或多组竞争→无法确定。同一T合并所有接收块的原裁剪点，最多产生一个副本。没有query父覆盖、query重叠或共同帧硬门槛，没有GT选取实例参照、父、块或点。

## 固定输出结果

所有规则/证据/完整52场景输出封存后才读取本轮GT。全部143稳定父、14含糊父、33超评估类别范围父、17未标注父保留。非稳定父不强行分配GT目标，其目标/邻物标签为null，已知/未知点另记。

| 对照 | 动作父数 | 完整NT新增目标≥.02 | 安全增量场景 | 好副本损失>.02 | 稳定父新增目标精度 | 非稳定父动作 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

“安全增量场景”是后置诊断：同父相对完整NT增量≥.02且新增目标占比≥.9，未知仍计入分母。不是推理条件，更不是GT回选输出。“好副本”指原IoU≥.9的T产生的修补副本；原T仍保留，未声称发生下游损害。完整NT池始终保留原候选，几何上限不会因追加差副本下降。

| 三态输出 | 块数 | GT归属构成 | 稳定父裁剪点（目标/邻物/未知分别统计） |
|---|---:|---|---|
{chr(10).join(statelines)}

全量每父连续IoU、pool_delta、误补、原点删除、未知、非稳定父动作及cohort分层见[summary.json](summary.json)、evaluation/{{scene}}/unions.json。当前原型完全固定，没有查看GT后挑父或只保留获益块。

## 机会与冗余必须一起检查

下表覆盖点级oracle有价值的所有父；其余负例同样完整评测。数值为实际union对同父目标IoU减完整NT最佳IoU。无动作也可能显示负数，因为原父已不如完整NT；负数不自动表示本次损坏，须同时查parent_delta。

| 场景/父 | 失败基线pool_delta | 新原型pool_delta | 原型块状态 |
|---|---:|---:|---|
{chr(10).join(cases)}

N只提供参照，不能作为正确答案。对每个实际接收块的“同一原N同时多数覆盖父与块”连接，后置统计该N完整点集的目标/其他/未知组成，并比较修补副本、连接N、全部N及完整NT的IoU。native_bridge_audit共{s['native_bridge_audit']['rows']}个父/N连接，其中{s['native_bridge_audit']['multiple_known_instances']}个包含多个已知实例，{s['native_bridge_audit']['substantial_other']}个含>10%已知邻物；这不是推理筛选条件。稳定父实际动作中，{s['modes']['relation_C']['native_redundant_actions']}个副本不如或等于已有最佳N，{s['modes']['relation_C']['bridge_native_redundant_actions']}个不如或等于直接连接N。身份解释不等于新增池价值。

## 预先验收与停止边界

[预定验收结果](summary.json)为{s['development_gates']}。继续条件为多物理场景安全池增量、保护全部好副本、稳定父新增精度≥.9、未知比例≤.1及无未解决父身份的动作；这些是保守开发验收口径，不是GT推理门槛。少操作或零操作不能通过。

本轮未通过，停止当前“complete-link重复分组＋跨源参照＋唯一归属”固定构造。不继续调同一阈值，不开展独立验证/AP，也不扩大原修复器训练。失败不证明N/T关系无用或原始信息不能训练归属；本轮分组可能将同物体拆成多个竞争组，N/T一致也可能共同错误。下一优先级按用户建议是针对争议区域设计新增观测或追查原T逐帧来源，须另行明确封存；本轮没有擅自增加帧或运行新观测。

只有以后明确规则先通过开发风险验收，才冻结未参与设计的物理隔离开发场景（排除52场景及变体，不动holdout），验证通过后再计算自身语义与冻结下游AP。

## 代码与核验

4项针对性测试通过，独立核验：[verification.json](verification.json)。GT前代码快照在code_snapshot/，证据/预测在predictions/，GT后标签在evaluation/。baseline all_C与上一轮所有关键池/损害/点数逐项一致。执行顺序：prepare → generate → evaluate → verify_fi1_nt_instance_ownership.py → report；使用.venvs/yoloe_gpu/bin/python，OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4。已有seal禁止覆盖重跑；从头复现另建目录。

本轮未修改生产、阅读入口或GitHub，无训练、GPU语义或AP。全部原父点保留。
'''
    (OUT/'report.md').write_text(text)
    D.seal(OUT/'completion_verified.json',dict(complete=True,development_pass=False,tests='4 passed',
        outputs={str(p):D.sha(p) for p in [OUT/'report.md',OUT/'decision.json',OUT/'verification.json',OUT/'contract.json',OUT/'predictions_frozen.json',OUT/'evaluation_frozen.json']},
        original_parent_points_deleted=0,training=False,independent_validation_run=False,AP_run=False,github_push=False,reading_entry_modified=False))
    stage('complete_fixed_prototype_stopped')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','generate','evaluate','report'))
    globals()[parser.parse_args().mode]()
