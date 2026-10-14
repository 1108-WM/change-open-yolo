#!/usr/bin/env python3
"""Explain fixed evidence gates; no threshold scan or alternate output selection."""
from pathlib import Path
from collections import Counter,defaultdict
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_complementary_fragment_diagnostic as run

def main():
    out=run.OUT;c=run.read(out/'contract.json');assert run.read(out/'independent_verified.json')['complete']
    summary=run.read(out/'summary.json');opps=summary['opportunity_rows'];oracle=Counter();disjoint=Counter();accepted=defaultdict(list);negative=Counter();examples=defaultdict(list)
    for s in c['scenes']:
        features=run.read(out/'features'/s/'pairs.json');labels=run.read(out/'labels'/s/'pairs.json')
        for x,y in zip(features,labels):
            v=x['views']
            if y['identity_label']=='different_instances' and x['retrieval']['combined']:
                negative['retrieved']+=1;negative['joint_at_least2']+=len(v['joint_groups'])>=2;negative['compact_at_least2']+=len(v['compact_groups'])>=2;negative['any_separation']+=bool(v['separated_groups']);negative['compact_no_separation']+=x['signatures']['compact_no_separation'];negative['strong']+=x['signatures']['compact_boundary_depth']
            for sig,ok in x['signatures'].items():
                if not ok or not x['retrieval']['combined']:continue
                for z in y['directions']:
                    if z['anchor']['reliable']:accepted[sig].append(dict(scene=s,left=x['left'],right=x['right'],**z))
                name=sig+'_'+y['identity_label']
                if len(examples[name])<3:examples[name].append(dict(scene=s,feature=x,label=y))
    for r in opps:
        counts=Counter(total=1)
        if not r['pair_in_domain']:counts['not_in_complementary_domain']=1
        else:
            rr=r['retrieval'];v=r['views'];bd=r['boundary'];counts.update(old_overlap=int(rr['legacy_overlap']),spatial=int(rr['spatial']),projection=int(rr['projection']),combined=int(rr['combined']))
            if not rr['combined']:counts['new_retrieval_missed']=1
            if v is None:counts['no_retrieved_evidence']=1
            else:
                counts.update(no_joint=int(not v['joint_groups']),joint_only_one_group=int(len(v['joint_groups'])==1),joint_at_least2=int(len(v['joint_groups'])>=2),compact_at_least2=int(len(v['compact_groups'])>=2),separation_veto=int(bool(v['separated_groups'])),compact_depth_at_least2=int(len(v['joint_and_compact_depth_groups'])>=2),contacts05_at_least3=int(bd['contacts05']>=3))
                for sig,vv in r['signatures'].items():counts[sig]=int(vv)
        oracle.update(counts)
        if r['pair_iou']==0:disjoint.update(counts)
    composition={}
    for sig in c['signatures']:
        rows=accepted[sig];target=np.array([r['added_target_points'] for r in rows]);pollution=np.array([r['added_pollution_points'] for r in rows]);fraction=target/(target+pollution) if len(rows) else np.array([])
        composition[sig]=dict(reliable_anchor_directions=len(rows),target_majority=int((target>pollution).sum()),pollution_majority=int((target<pollution).sum()),ties=int((target==pollution).sum()),median_added_target=int(np.median(target)) if len(rows) else None,median_added_pollution=int(np.median(pollution)) if len(rows) else None,median_target_fraction=float(np.median(fraction)) if len(rows) else None)
    data=dict(complete=True,oracle60=dict(oracle),disjoint26=dict(disjoint),retrieved_reliable_neighbor_negatives=dict(negative),added_region_composition=composition,examples=dict(examples),no_method_changes=True,no_new_AP=True)
    run.freeze_json(out/'supplement.json',data)
    lines=['# 固定归属证据的失败位置','',
           '本补充只解释已封存的检索和归属规则，不改变阈值、选择弱规则或接入评分。60个GT挑选机会是诊断参照，不能当全部可修复缺口的上界。','',
           '|检查项|60个并集机会|其中26个不交配对|','|---|---:|---:|']
    for key in sorted(set(oracle)|set(disjoint)):lines.append(f'|{key}|{oracle[key]}|{disjoint[key]}|')
    lines+=['','各证据项可以同时发生，计数不能相加解释总损失。完全不交配对的分母由已封存机会清单核验；不满足检索域的配对单列。','',
            '|固定归属规则|可靠锚方向|新增目标较多|新增污染较多|相同|新增区域目标占比中位数|','|---|---:|---:|---:|---:|---:|']
    for sig,d in composition.items():lines.append(f"|{sig}|{d['reliable_anchor_directions']}|{d['target_majority']}|{d['pollution_majority']}|{d['ties']}|{d['median_target_fraction']}|")
    lines+=['','归属负例为双方独有区域可靠但属于不同有效实例；混合／身份不足保留在主报告，不能从可靠子集precision推断所有候选安全。',f'邻物负例检查：`{dict(negative)}`。','',
            '紧致联合掩码是一个保守证据假设：原始分割即使只覆盖同一目标的一部分，也可能提供各自独有区域的分离掩码。因此分离反证可能包含真互补碎片，不能将本规则的拒绝都解释成“确实不同实例”。边界近邻和由3D点得到的相机深度连续性同样不足以独立证明身份。','',
            '当前30帧属于既有姿态组观测；没有共同支持只能说明这些观测未提供固定规则需要的证据，不能证明全部原始帧或真实世界都不存在同实例证据。纯追加仍不移除已有污染；后续裁剪属于独立动作。','',
            '是否进入自身语义／冻结SL双AP仍严格采用原合同 strongest 签名的门，见decision.json；没有根据这些诊断更换动作或调分。']
    (out/'supplemental_report.md').write_text('\n'.join(lines)+'\n')
    print(dict(oracle),dict(disjoint),flush=True)

if __name__=='__main__':main()
