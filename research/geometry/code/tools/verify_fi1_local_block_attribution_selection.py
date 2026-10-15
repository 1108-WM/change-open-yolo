#!/usr/bin/env python3
"""Independent sets, counts, single-target IoUs, references and frame-state checks."""
from pathlib import Path
from collections import Counter
import json
import sys
import numpy as np
from scipy.spatial.distance import cdist
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_local_block_attribution_selection as run
D=run.diag;OUT=run.OUT


def iou(points,gt,gid,size):
    if len(points)<100:return 0.
    hit=int(np.count_nonzero(gt[points]==gid))
    return hit/(len(points)+size-hit)


def frame_state(edit,ref,visible,masks):
    e=edit[visible[edit]];r=ref[visible[ref]]
    if not len(e) or not len(r) or not len(masks):return 'missing'
    a=np.array([np.count_nonzero(np.isin(e,m))/len(e) for m in masks])
    b=np.array([np.count_nonzero(np.isin(r,m))/len(r) for m in masks])
    if a.max()==0 or b.max()==0:return 'observed_no_support'
    if np.count_nonzero(a==a.max())!=1 or np.count_nonzero(b==b.max())!=1:return 'ambiguous_tie'
    if a.max()<.5 or b.max()<.5:return 'observed_weak'
    return 'common' if a.argmax()==b.argmax() else 'conflict'


def main():
    c=D.read(OUT/'contract.json');D.verify(c['code'])
    for fn in ('predictions_frozen.json','evaluation_frozen.json','analysis_frozen.json'):D.verify(D.read(OUT/fn)['outputs'])
    checks=Counter();pool_expected={};allrows=[]
    for scene,job in c['jobs'].items():
        print('independent verification',scene,flush=True)
        root=Path(job['asset_root']);gt=np.loadtxt(job['GT'],dtype=np.int64)
        hist=Counter(map(int,gt));valid=sorted(g for g,n in hist.items() if g>=1000 and n>=100 and g//1000 in c['class_ids'])
        _,nt=D.load_nt(job)
        baseline={g:max((iou(pts,gt,g,hist[g]) for pts in nt.values()),default=0.) for g in valid}
        parents=D.read(run.SOURCE/'relations'/scene/'parents.json')
        evalrows=D.read(OUT/'evaluation'/scene/'unions.json');allrows+=evalrows
        evalblocks={r['block_id']:r for r in D.read(OUT/'evaluation'/scene/'blocks.json')}
        blockrefs=D.read(OUT/'predictions'/scene/'blocks.json')
        frames=D.read(OUT/'predictions'/scene/'frames.json')
        xyz=D.core.safe_input(np.load(job['paths']['processed'],mmap_mode='r'))[:,:3]
        obs=[json.loads(l) for l in Path(job['paths']['observations']).read_text().splitlines()]
        bankroot=Path(obs[0]['point_indices_path']).parent.parent;banks={}
        for f in sorted({f for p in parents for f in p['frames']}):
            banks[f]=[]
            for entry in D.read(bankroot/f'frame{f:06d}'/'observations.json'):
                with np.load(entry['point_indices_path']) as zz:banks[f].append(zz['point_indices'].copy())
        with np.load(root/'structure'/scene/'data.npz') as st,np.load(root/'priors'/scene/'data.npz') as pr,np.load(OUT/'predictions'/scene/'data.npz') as z,np.load(OUT/'evaluation'/scene/'oracle_diagnostic_points.npz') as oz,np.load(D.SOURCE/'plan'/scene/'evidence.npz') as ev:
            reliable=np.unpackbits(ev['reliable'],axis=1,bitorder='little')[:,:len(ev['roi'])].astype(bool)
            for p in parents:
                cid=p['candidate_id'];parent=st[cid+'_parent'];halo=st[cid+'_halo'];labels=st[cid+'_labels'];visible=st[cid+'_visible'];prob=pr['P1_probability'][np.searchsorted(pr['point_ids'],halo)]
                assert all(g in valid for g in [])
                before=[iou(parent,gt,g,hist[g]) for g in valid]
                g=valid[int(np.argmax(before))] if before and max(before)>0 else None
                stable=g is not None and np.count_nonzero(gt[parent]==g)>=20 and np.mean(gt[parent]==g)>=.6
                target=g if stable else None
                rows=[r for r in evalrows if r['candidate_id']==cid]
                refs=[r for r in blockrefs if r['candidate_id']==cid]
                expected={'W':[],'C':[],'oracle_pure_W':[],'oracle_pure_C':[],'oracle_target_points':[]}
                for bi,b in enumerate(refs):
                    bid=b['block_id'];block=halo[(labels==b['region'])&(~np.isin(halo,parent))&(visible>=2)]
                    assert np.array_equal(block,z[bid+'_block'])
                    ix=np.searchsorted(halo,block)
                    q=min(p['queries'],key=lambda qq:(-float(np.sum(prob[ix,qq],dtype=np.float64)/len(ix)),qq))
                    assert q==b['supporter']
                    clip=block[prob[ix,q]>=.5];assert np.array_equal(clip,z[bid+'_clipped_proposal'])
                    assert not np.intersect1d(parent,block).size
                    expected['W'].append(block);expected['C'].append(clip)
                    br=evalblocks[bid]
                    for key,pts in [('block_GT',block),('clipped_GT',clip),('dropped_GT',np.setdiff1d(block,clip))]:
                        cr=br[key]
                        assert cr['unknown']==np.count_nonzero(gt[pts]<1000)
                        assert cr['annotated']==np.count_nonzero(gt[pts]>=1000)
                        assert cr['target']==(int(np.count_nonzero(gt[pts]==target)) if stable else None)
                        assert cr['other']==(int(np.count_nonzero((gt[pts]>=1000)&(gt[pts]!=target))) if stable else None)
                        checks['independent_point_censuses']+=1
                    if stable:
                        expected['oracle_target_points'].append(block[gt[block]==target])
                        if np.all(gt[block]==target):expected['oracle_pure_W'].append(block);expected['oracle_pure_C'].append(clip)
                        for arm,pts in [('W',block),('C',clip)]:
                            value=iou(np.union1d(parent,pts),gt,g,hist[g])
                            assert abs(br[arm+'_single']['repaired_iou']-value)<1e-12
                            assert abs(br[arm+'_single']['pool_delta']-(value-baseline[g]))<1e-12
                            checks['independent_single_block_ious']+=1
                    # Exhaustive point distances for first supplied block of every parent.
                    if bi==0:
                        alternatives=[]
                        for rid in sorted(set(labels[np.isin(halo,parent)])):
                            ref=halo[(labels==rid)&np.isin(halo,parent)]
                            mind=min(float(cdist(xyz[block[k:k+64]],xyz[ref]).min()) for k in range(0,len(block),64))
                            alternatives.append((mind,int(rid)))
                        best=min(alternatives)
                        assert best[1]==b['reference_region'] and abs(best[0]-b['minimum_distance'])<1e-7
                        checks['exhaustive_reference_parent_checks']+=1
                    reference=halo[(labels==b['reference_region'])&np.isin(halo,parent)]
                    wf=[r for r in frames if r['candidate_id']==cid and r['region']==b['region']]
                    for f in wf:
                        vi=reliable[list(ev['frames']).index(f['frame'])]
                        vispoint=ev['roi'][vi]
                        # Global indices, unlike generator's local halo-index matrix.
                        both=np.unique(np.concatenate([block,reference]));seen=np.isin(both,vispoint)
                        state=frame_state(np.searchsorted(both,block),np.searchsorted(both,reference),seen,[np.searchsorted(both,np.intersect1d(both,m)) for m in banks[f['frame']]])
                        assert state==f['state']
                        checks['independent_frame_states']+=1
                    states=Counter(f['state'] for f in wf);assert dict(states)==b['frame_states']
                    assert b['accepted'] is False
                    checks['independent_blocks']+=1
                for r in rows:
                    mode=r['mode'];parts=expected[mode]
                    add=np.unique(np.concatenate(parts)) if parts else np.empty(0,np.int64)
                    points=np.union1d(parent,add)
                    assert np.array_equal(points,oz[cid+'_'+mode])
                    if mode in ('W','C'):assert np.array_equal(points,z[cid+'_'+mode+'_points'])
                    assert not np.setdiff1d(parent,points).size and r['deleted']==0
                    assert r['stable']==stable and r['added']==len(add)
                    if stable:
                        value=iou(points,gt,g,hist[g]);assert abs(r['repaired_iou']-value)<1e-12
                        assert abs(r['pool_delta']-(value-baseline[g]))<1e-12
                        checks['independent_union_same_parent_ious']+=1
                        pool_expected[(scene,mode,g)]=max(pool_expected.get((scene,mode,g),baseline[g]),value)
                    checks['independent_unions']+=1
        for mode in expected:
            for g in valid:pool_expected.setdefault((scene,mode,g),baseline[g])
        for row in D.read(OUT/'evaluation/pool_same_parent.json'):
            if row['scene']!=scene:continue
            assert abs(row['baseline']-baseline[row['gt_id']])<1e-12
            assert abs(row['expanded']-pool_expected[scene,row['mode'],row['gt_id']])<1e-12
            checks['independent_pool_target_modes']+=1
    assert checks['independent_blocks']==1308 and checks['independent_unions']==207*5
    D.seal(OUT/'verification.json',dict(complete=True,checks=dict(checks),code_sha256=D.sha(Path(__file__)),
        original_parent_points_deleted=0,GT_outputs_never_production=True))
    print(dict(checks),flush=True)


if __name__=='__main__':main()
