#!/usr/bin/env python3
"""Posthoc failure explanation only. Never changes fixed groups or outputs."""
from pathlib import Path
from collections import Counter
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_nt_instance_ownership as run
D=run.D;OUT=run.OUT


def main():
    c=D.read(OUT/'contract.json');D.verify(D.read(OUT/'evaluation_frozen.json')['outputs'])
    previous_evidence=D.read(run.BASE/'analysis/valuable_block_evidence.json')
    valuable={(r['scene'],r['block_id']):r for r in previous_evidence}
    rows=[];competition=Counter();state_census=Counter();false_bridges=[];unresolved=[];redundant=[]
    for scene,job in c['jobs'].items():
        nodes=D.read(OUT/'predictions'/scene/'nodes.json');groups=D.read(OUT/'predictions'/scene/'groups.json')
        blocks=D.read(OUT/'evaluation'/scene/'blocks.json')
        unions=D.read(OUT/'evaluation'/scene/'unions.json')
        _,nt=D.load_nt(job);gt=np.loadtxt(job['GT'],dtype=np.int64)
        for b in blocks:
            if b['state']=='unknown':competition[b['reason']]+=1
        with np.load(OUT/'predictions'/scene/'data.npz') as z:
            for b in blocks:
                key=(scene,b['block_id'])
                if key not in valuable:continue
                e=valuable[key];cid=b['candidate_id'];hits=z[b['block_id']+'_hits'];ph=z[cid+'_parent_hits']
                parent_size=next(r['parent_points'] for r in unions if r['candidate_id']==cid)
                parent_supported=np.flatnonzero(ph>=.5*parent_size);block_supported=np.flatnonzero(hits>=.5*b['points'])
                raw_bridges=sorted(set(map(int,parent_supported))&set(map(int,block_supported)))
                raw_native=[i for i in raw_bridges if 'native' in nodes[i]['sources']]
                explanation='accepted' if b['state']=='accept' else 'parent_reference_missing' if not b['parent_groups'] else 'block_reference_missing' if not b['block_groups'] else 'competing_references' if b['state']=='unknown' else 'different_reference'
                def ns(i):return dict(node=i,key=nodes[i]['key'],sources=nodes[i]['sources'],points=nodes[i]['points'],
                    parent_coverage=float(ph[i]/parent_size),block_coverage=float(hits[i]/b['points']),
                    groups=[g['group'] for g in groups if i in g['members']],
                    credible_groups=[g['group'] for g in groups if g['credible'] and i in g['members']])
                rows.append(dict(scene=scene,candidate_id=cid,region=b['region'],state=b['state'],reason=b['reason'],
                    explanation=explanation,parent_groups=b['parent_groups'],block_groups=b['block_groups'],
                    GT=b['block_GT'],clipped_GT=b['clipped_GT'],point_oracle_contribution=e['point_oracle_contribution'],
                    pure_oracle_contribution=e['pure_oracle_contribution'],
                    parent_query_hard_fraction=b['query_parent_hard_fraction'],
                    anchor_query_hard_iou=b['anchor_supporter_hard_intersection']/b['anchor_supporter_hard_union'] if b['anchor_supporter_hard_union'] else None,
                    raw_parent_support_nodes=len(parent_supported),raw_block_support_nodes=len(block_supported),
                    raw_bridge_nodes=len(raw_bridges),raw_native_bridges=len(raw_native),
                    bridge_examples=[ns(i) for i in raw_bridges[:5]],
                    parent_support_examples=[ns(int(i)) for i in parent_supported[:5]],
                    block_support_examples=[ns(int(i)) for i in block_supported[:5]]))
                state_census[(b['state'],explanation)]+=1
            for b in blocks:
                if b['state']!='accept' or not b['stable'] or not b['clipped_GT']['other']:continue
                related=[]
                for n in b['native_bridge_nodes']:
                    pts=nt[nodes[n]['key']];t=b['target'];known=gt[pts]>=1000
                    related.append(dict(node=n,key=nodes[n]['key'],points=len(pts),target=int((gt[pts]==t).sum()),
                        other=int((known&(gt[pts]!=t)).sum()),unknown=int((~known).sum())))
                false_bridges.append(dict(scene=scene,candidate_id=b['candidate_id'],region=b['region'],
                    clipped_GT=b['clipped_GT'],native_bridge_nodes=related))
        unresolved += [dict(scene=scene,candidate_id=r['candidate_id'],scope=r['scope'],added_GT=r['added_GT']) for r in unions if r['arm']=='relation_C' and r['changed'] and not r['stable']]
        redundant += [dict(scene=scene,candidate_id=r['candidate_id'],parent_iou=r['parent_iou'],
            repaired_iou=r['repaired_iou'],pool_delta=r['pool_delta'],best_native_iou=r['best_native_iou'],
            best_bridge_native_iou=r['best_bridge_native_iou'],bridge_native_redundant=r['bridging_native_already_at_least_as_good'])
            for r in unions if r['arm']=='relation_C' and r['changed'] and r['stable']]
    point=rows;pure=[r for r in rows if r['pure_oracle_contribution']>0]
    summary=dict(posthoc_explanation_only=True,predictions_changed=False,
        valuable_point_blocks=len(point),valuable_point_scenes=len({r['scene'] for r in point}),
        point_states=dict(Counter(r['state'] for r in point)),point_causes=dict(Counter(r['explanation'] for r in point)),
        valuable_pure_blocks=len(pure),pure_states=dict(Counter(r['state'] for r in pure)),
        pure_causes=dict(Counter(r['explanation'] for r in pure)),
        unknown_all_causes=dict(competition),
        valuable_point_raw_native_bridge=sum(r['raw_native_bridges']>0 for r in point),
        valuable_point_raw_bridge_but_not_accepted=sum(r['raw_bridge_nodes']>0 and r['state']!='accept' for r in point),
        accepted_stable_blocks_with_neighbor_points=len(false_bridges),
        unresolved_changed_parents=len(unresolved),stable_changed_parents=len(redundant),
        limits='Prototype lacks references is not absence of raw N/T identity information. Greedy complete-link may overfragment one instance or be order-sensitive; cross-source agreement can be wrong. No GT-informed repair of groups, rules or outputs.',
        next_observation_targets='Use fixed output unresolved blocks to define next-round candidate list, not just GT positives. Pair valuable examples with same-reason neighbor negatives; first inspect original T per-frame provenance and available frame visibility before adding observations. No new observations performed here.')
    dest=OUT/'failure_diagnosis';dest.mkdir()
    for name,val in [('summary',summary),('valuable_blocks',rows),('accepted_neighbor_pollution',false_bridges),('unresolved_actions',unresolved),('native_redundancy',redundant)]:D.seal(dest/(name+'.json'),val)
    text=f'''# 原型失败定位（后置诊断，不改规则）

本轮有明确归属选择器，其结果是减少污染而未保住完整NT增量。当前判断只针对本轮complete-link/跨来源参照/唯一归属构造。

全部无法确定块：{dict(competition)}。点级oracle机会贡献32块中：{summary['point_states']}；具体原因{summary['point_causes']}。纯整块oracle机会贡献21块中：{summary['pure_states']}；具体原因{summary['pure_causes']}。

特别是，“缺少合格参照”与“原N/T根本没有联系”不同：32个机会贡献块里，{summary['valuable_point_raw_native_bridge']}个本来存在原N同时多数覆盖父和块；{summary['valuable_point_raw_bridge_but_not_accepted']}个虽有原N/T连接却未被接收。逐块保存原始支持节点、跨源资格、组和连续覆盖，不能把门槛/分组失败概括成证据不足。

- scene0002/c006/r1（真碎片）：父有1个参照组，块没有，输出unknown；没有因为query父覆盖0而直接拒绝，但N/T参照构造仍未找回它。
- scene0085/c007及scene0301/c004,c007：父缺少合格跨源参照组，关键块无法接收，原baseline的收益也被一并过滤。
- scene0078/c004：接收9块仍无池收益，关键纯目标r11的177点因缺块参照而遗漏。须保留所有接收块联合评测，不能GT补回这一块。
- scene0369/c006/r4：块有参照组而父无，unknown；其余小目标块也没有合格联系。
- scene0688/c006/r1：父/块均缺合格组，unknown。前轮单块裁剪确有+.042515增量但47/147未知；仍不能把该块视作已安全解决。

大N风险仍存在：接受集合有{len(false_bridges)}个稳定父块裁剪后带邻物点，明细accepted_neighbor_pollution.json记录实际连接N的完整GT组成。不要把N/T一致作为GT真值。11个非稳定父产生动作也完整保留，不能GT后将它们删除来报安全。

55个稳定父有实际动作，其中52个修补副本不如或等于已有最佳N，且52个不如或等于实际连接N。关系更容易解释已有N已经覆盖的部分，这就是当前完整NT收益消失的直接限制。并非所有身份正确的补点都值得追加。

本轮停止；不事后改组、放松参照或重新选阈值。若开展下一项观测研究，先以本轮所有争议/缺参照块固定目标表，包含邻物负例与混杂/身份不明父；优先追查原T逐帧关联来源，区分缺帧、遮挡、自动mask碎片化与真实冲突，再单独封存新增帧查找/观测协议。此处未启动新帧或新训练。

量化文件：[summary.json](summary.json)、[机会块原始联系](valuable_blocks.json)、[误补连接N](accepted_neighbor_pollution.json)、[N冗余对照](native_redundancy.json)。
'''
    (dest/'report.md').write_text(text)
    D.seal(OUT/'failure_diagnosis_frozen.json',dict(complete=True,new_rules=False,new_outputs=False,
        outputs={str(p):D.sha(p) for p in dest.iterdir()},code_sha256=D.sha(Path(__file__))))
    print(summary,flush=True)


if __name__=='__main__':main()
