#!/usr/bin/env python3
"""Independent set-based reference groups, tri-state decisions, union/GT/pool audit."""
from pathlib import Path
from collections import Counter
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_nt_instance_ownership as run
D=run.D;OUT=run.OUT


def iou(points,gt,g,size):
    if len(points)<100:return 0.
    hits=np.count_nonzero(gt[points]==g)
    return float(hits/(len(points)+size-hits))


def main():
    c=D.read(OUT/'contract.json');D.verify(c['code'])
    for fn in ('predictions_frozen.json','evaluation_frozen.json'):D.verify(D.read(OUT/fn)['outputs'])
    checks=Counter();pools=D.read(OUT/'evaluation/pool.json')
    for scene,job in c['jobs'].items():
        print('independent audit',scene,flush=True)
        _,nt=D.load_nt(job);keys=sorted(nt);sets=[set(map(int,nt[k])) for k in keys]
        nodes=D.read(OUT/'predictions'/scene/'nodes.json');savedgroups=D.read(OUT/'predictions'/scene/'groups.json')
        blocks=D.read(OUT/'predictions'/scene/'blocks.json');rows=D.read(OUT/'evaluation'/scene/'unions.json')
        gt=np.loadtxt(job['GT'],dtype=np.int64);hist=Counter(map(int,gt))
        ids=sorted(g for g,n in hist.items() if g>=1000 and n>=100 and g//1000 in c['class_ids'])
        baseline={g:max((iou(pts,gt,g,hist[g]) for pts in nt.values()),default=0.) for g in ids}
        best={arm:dict(baseline) for arm in run.ARMS}
        with np.load(OUT/'predictions'/scene/'data.npz') as z,np.load(run.BASE/'predictions'/scene/'data.npz') as prev,np.load(Path(job['asset_root'])/'structure'/scene/'data.npz') as st:
            intersection=z['intersections'];assert [n['key'] for n in nodes]==keys
            # Every matrix pair independently reconstructed using Python sets.
            for i,a in enumerate(sets):
                for j in range(i,len(sets)):
                    assert intersection[i,j]==len(a&sets[j])==intersection[j,i]
                    checks['independent_NT_pairs']+=1
            groups=[]
            for node in range(len(keys)):
                compatible=[]
                for gi,group in enumerate(groups):
                    if all(2*len(sets[node]&sets[m])>=max(len(sets[node]),len(sets[m])) for m in group):compatible.append(gi)
                if compatible:groups[compatible[0]].append(node)
                else:groups.append([node])
            assert groups==[g['members'] for g in savedgroups]
            credible=[]
            for gi,group in enumerate(groups):
                yes=any(n!=t and len(sets[n])>=100 and len(sets[t])>=100 and
                    'native' in nodes[n]['sources'] and 'track' in nodes[t]['sources'] for n in group for t in group)
                if yes:credible.append(gi)
                assert yes==savedgroups[gi]['credible'];checks['independent_groups']+=1
            def owners(points):
                domain=set(map(int,points));return [gi for gi in credible if any(2*len(domain&sets[n])>=len(domain) for n in groups[gi])]
            parents=D.read(run.previous.SOURCE/'relations'/scene/'parents.json')
            evalblocks={b['block_id']:b for b in D.read(OUT/'evaluation'/scene/'blocks.json')}
            for p in parents:
                cid=p['candidate_id'];parent=st[cid+'_parent'];pg=owners(parent);accepted=[]
                scores=[iou(parent,gt,g,hist[g]) for g in ids];gid=ids[int(np.argmax(scores))] if scores and max(scores)>0 else None
                stable=gid is not None and np.count_nonzero(gt[parent]==gid)>=20 and np.mean(gt[parent]==gid)>=.6
                for b in [b for b in blocks if b['candidate_id']==cid]:
                    bid=b['block_id'];block=prev[bid+'_block'];clip=prev[bid+'_clipped_proposal'];bg=owners(block)
                    state=('accept' if pg==bg else 'reject') if len(pg)==len(bg)==1 else 'unknown'
                    assert b['state']==state and b['parent_groups']==pg and b['block_groups']==bg
                    assert np.array_equal(z[bid+'_hits'],[len(set(map(int,block))&pts) for pts in sets])
                    assert np.setdiff1d(clip,block).size==0 and np.intersect1d(parent,clip).size==0
                    if state=='accept':accepted.extend(map(int,clip))
                    for name,pts in [('block_GT',block),('clipped_GT',clip)]:
                        cr=evalblocks[bid][name]
                        assert cr['unknown']==np.count_nonzero(gt[pts]<1000)
                        assert cr['target']==(int(np.count_nonzero(gt[pts]==gid)) if stable else None)
                        assert cr['other']==(int(np.count_nonzero((gt[pts]>=1000)&(gt[pts]!=gid))) if stable else None)
                        checks['independent_point_censuses']+=1
                    checks[state]+=1;checks['independent_block_decisions']+=1
                for arm in run.ARMS:
                    points=z[cid+'_'+arm]
                    expected=np.union1d(parent,accepted) if arm=='relation_C' else prev[cid+'_C_points']
                    assert np.array_equal(points,expected)
                    assert not len(np.setdiff1d(parent,points))
                    r=next(r for r in rows if r['candidate_id']==cid and r['arm']==arm)
                    add=np.setdiff1d(points,parent);assert r['added']==len(add) and r['deleted']==0 and r['stable']==stable
                    if stable:
                        value=iou(points,gt,gid,hist[gid]);assert abs(value-r['repaired_iou'])<1e-12
                        assert abs(value-baseline[gid]-r['pool_delta'])<1e-12
                        assert r['added_target']==np.count_nonzero(gt[add]==gid)
                        assert r['added_other']==np.count_nonzero((gt[add]>=1000)&(gt[add]!=gid))
                        best[arm][gid]=max(best[arm][gid],value)
                    checks['independent_unions']+=1
        for r in [r for r in pools if r['scene']==scene]:
            assert abs(r['baseline']-baseline[r['gt_id']])<1e-12
            assert abs(r['expanded']-best[r['mode']][r['gt_id']])<1e-12
            checks['independent_pool_target_modes']+=1
    assert checks['independent_block_decisions']==1308 and checks['independent_unions']==414
    D.seal(OUT/'verification.json',dict(complete=True,checks=dict(checks),code_sha256=D.sha(Path(__file__)),
        query_not_in_ownership=True,GT_never_selects_output=True,original_parent_points_deleted=0))
    print(dict(checks),flush=True)


if __name__=='__main__':main()
