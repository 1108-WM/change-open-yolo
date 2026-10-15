"""Read-only replay plus independent supply, redundancy and query identity checks."""
from pathlib import Path
import csv
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'research/geometry/code/tools'))
import verify_fi1_t_add_union as replay
DEST=Path(__file__).resolve().parent

def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def iou(points,gt,ids,sizes):
    if len(points)<100:return np.zeros(len(ids))
    g,n=np.unique(gt[points],return_counts=True)
    counts=dict(zip(g.tolist(),n.tolist()))
    hits=np.array([counts.get(int(x),0) for x in ids])
    return hits/(len(points)+sizes-hits)

def main():
    run=replay.run
    captured=[]
    original=run.seal
    def capture(path,value):
        assert Path(path)==run.OUT/'verification.json'
        captured.append(value)
    run.seal=capture
    try:replay.verify()
    finally:run.seal=original
    assert len(captured)==1
    base=run.OUT;c=run.read(base/'contract.json')
    run.diag.verify(run.read(base/'effective_runtime_frozen.json')['runtime_code'])
    run.diag.verify(run.read(base/'failure_diagnosis_frozen.json')['outputs'])
    sl={(r['scene'],r['candidate_id']):r for r in table(base/'failure_diagnosis/supply_opportunity.csv')}
    qrows=table(base/'supplement/query_identity_evidence.csv')
    rr=run.read(base/'evaluation/rows.json');rl={(r['scene'],r['candidate_id'],r['arm']):r for r in rr}
    result=dict(complete=True,reused_verifier_replay=captured[0],supply_stable_parents=0,point_mapping=0,query_identity_rows=0,point_gain_targets=[],pure_region_gain_targets=[],point_continuous_gain=0.,pure_region_continuous_gain=0.,improved_covered_by_N={'A':0,'B':0},opportunity_details=[],duplicates={a:sum(r['duplicate_original'] for r in rr if r['arm']==a) for a in ('A','B')})
    for s,j in c['jobs'].items():
        gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True)
        ok=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[ok],sizes[ok]
        geom,nt=run.diag.load_nt(j);bn=np.zeros(len(ids));bt=bn.copy()
        for g in geom:
            score=iou(nt[g['geometry_key']],gt,ids,sizes)
            sources={a['source'] for a in run.diag.original_aliases(g)}
            if 'native' in sources:bn=np.maximum(bn,score)
            if 'track' in sources:bt=np.maximum(bt,score)
        baseline=np.maximum(bn,bt);bp=baseline.copy();br=baseline.copy()
        blocks=run.read(base/'predictions'/s/'blocks.json')
        with np.load(base/'structure'/s/'data.npz') as st,np.load(base/'predictions'/s/'data.npz') as pred,np.load(base/'priors'/s/'data.npz') as prior:
            raw=np.load(j['paths']['processed'],mmap_mode='r');xyz=np.floor(raw[:,:3]/.02).astype(np.int32)
            assert np.array_equal(prior['input_sparse_coordinates'][prior['original_to_sparse_row'],1:],xyz)
            result['point_mapping']+=len(xyz)
            probs=prior['P1_probability'];roi=prior['point_ids']
            for meta in c['parents'][s]:
                cid=meta['candidate_id'];parent=st[cid+'_parent'];halo=st[cid+'_halo']
                for r in [r for r in qrows if r['scene']==s and r['candidate_id']==cid]:
                    ix=np.searchsorted(roi,halo);mask=probs[ix,int(r['query_id'])]>=.5;g=int(r['parent_target'])
                    counts={k:int(v) for k,v in zip(*np.unique(gt[halo[mask]],return_counts=True))}
                    assert sum(counts.values())==int(r['local_predicted_points'])
                    assert counts.get(g,0)==int(r['local_target_hits'])
                    assert sum(n for k,n in counts.items() if k>=1000 and k!=g)==int(r['local_other_hits'])
                    assert sum(n for k,n in counts.items() if k<1000)==int(r['local_unknown_hits'])
                    result['query_identity_rows']+=1
                if (s,cid) not in sl:continue
                r=sl[s,cid];g=int(r['target']);idx=int(np.flatnonzero(ids==g)[0]);correct=set();pure=set()
                for b in [b for b in blocks if b['candidate_id']==cid]:
                    pts=pred[b['block_id']+'_block'];good=pts[gt[pts]==g]
                    correct.update(map(int,good))
                    if len(good)==len(pts):pure.update(map(int,pts))
                point=np.union1d(parent,np.array(sorted(correct),dtype=np.int64));region=np.union1d(parent,np.array(sorted(pure),dtype=np.int64))
                vp=iou(point,gt,ids,sizes)[idx];vr=iou(region,gt,ids,sizes)[idx]
                assert abs(float(r['point_add_only_optimal_iou'])-vp)<1e-12
                assert abs(float(r['pure_region_union_iou'])-vr)<1e-12
                assert abs(float(r['original_N_best'])-bn[idx])<1e-12
                assert abs(float(r['original_T_best'])-bt[idx])<1e-12
                bp[idx]=max(bp[idx],vp);br[idx]=max(br[idx],vr)
                result['supply_stable_parents']+=1
                for arm in ('A','B'):
                    ar=rl[s,cid,arm];out=pred[cid+'_'+arm+'_points'];after=iou(out,gt,ids,sizes)[idx]
                    if ar['parent_delta']>=.02-1e-12:
                        assert bn[idx]>=after-1e-12
                        result['improved_covered_by_N'][arm]+=1
                if vp-baseline[idx]>=.02-1e-12:
                    result['opportunity_details'].append(dict(scene=s,parent=cid,target=g,baseline=float(baseline[idx]),pure=float(vr),point=float(vp),query_rejected_target_points=sum(int((gt[pred[b['block_id']+'_block']]==g).sum()) for b in blocks if b['candidate_id']==cid and not b['A'])))
        for name,v in [('point',bp),('pure_region',br)]:
            result[name+'_gain_targets'] += [[s,int(ids[k])] for k in np.flatnonzero(v-baseline>=.02-1e-12)]
            result[name+'_continuous_gain']+=float((v-baseline).sum())
    result['limitations']=['Existing reviewed verifier rerun with final seal intercepted; original artifacts unchanged.','Supply/IoU, N/T redundancy and query mask composition independently recomputed; input loader reused.','No GPU re-export or full depth/mask pipeline rerun.','Query identity check verifies local mask composition, not globally identical instances.']
    (DEST/'audit_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
