#!/usr/bin/env python3
"""Independent query selection, mask/census, relation and frame cross-check."""
from pathlib import Path
from collections import Counter
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial.distance import cdist
import run_fi1_query_relation_diagnostic as run
OUT=run.OUT;read=run.read;seal=run.seal;sha=run.sha

def independent_frame(edit,ref,visible,masks):
    a=edit[visible[edit]];b=ref[visible[ref]];r=dict(edit_visible=len(a),reference_visible=len(b),missing=True,common=False,conflict=False,edit_reference_mask_coverage=None,edit_best_mask_coverage=None,reference_best_mask_coverage=None,common_best_mask=False)
    if not len(a) or not len(b) or not masks.shape[1]:return r
    eh=np.count_nonzero(masks[a],axis=0)/len(a);rh=np.count_nonzero(masks[b],axis=0)/len(b);ei=int(np.argmax(eh));ri=int(np.argmax(rh))
    if eh[ei]==0 or rh[ri]==0:return r
    r.update(missing=False,common=bool(eh[ri]>=.5 and rh[ri]>=.5),conflict=bool(ei!=ri and eh[ei]>=.5 and rh[ri]>=.5 and eh[ri]<.5),edit_reference_mask_coverage=float(eh[ri]),edit_best_mask_coverage=float(eh[ei]),reference_best_mask_coverage=float(rh[ri]),common_best_mask=ei==ri);return r

def independent_counts(points,gt,target):
    rr=gt[points];hist=Counter(map(int,rr));targetn=hist.get(target,0) if target is not None else 0;other=sum(n for gid,n in hist.items() if gid>=1000 and gid!=target);unknown=sum(n for gid,n in hist.items() if gid<1000);return dict(points=len(points),target=targetn,other=other,unknown=unknown,distribution={str(k):v for k,v in sorted(hist.items())})
def checkcounts(expected,computed):
    for k,v in computed.items():assert expected[k]==v,(k,expected[k],v)
def brute_min(x,y):
    if not len(x) or not len(y):return None
    return min(float(cdist(x[i:i+128],y).min()) for i in range(0,len(x),128))

def main():
    c=read(OUT/'contract.json');run.diag.verify(c['code']);run.diag.verify(read(OUT/'relations_frozen.json')['outputs']);run.diag.verify(read(OUT/'evaluation_frozen.json')['outputs']);run.diag.verify(read(OUT/'analysis_frozen.json')['outputs']);totals=Counter();dist_error=0.
    for s,j in c['jobs'].items():
        print('independent verify',s,flush=True);root=Path(j['asset_root']);d=OUT/'relations'/s;parents=read(d/'parents.json');edges=read(d/'query_regions.json');pairs=read(d/'query_pairs.json');frames=read(d/'frames.json');GTparents={p['candidate_id']:p for p in read(OUT/'evaluation'/s/'parents.json')};GTpairs={(p['candidate_id'],p['query_a'],p['query_b']):p for p in read(OUT/'evaluation'/s/'pairs.json')};GTregions={(p['candidate_id'],p['region'],p['query']):p for p in read(OUT/'evaluation'/s/'query_regions.json')};gt=np.loadtxt(j['GT'],dtype=np.int64);metas={p['candidate_id']:p for p in read(root/'structure'/s/'candidates.json') if p['source']=='track'}
        run.diag.verify(read(d/'receipt.json')['inputs']);obs=[json.loads(l) for l in Path(j['paths']['observations']).read_text().splitlines()];bankroot=Path(obs[0]['point_indices_path']).parent.parent;banks={}
        for f in {r['frame'] for r in frames}:
            banks[f]=[]
            for row in read(bankroot/f'frame{f:06d}'/'observations.json'):
                with np.load(row['point_indices_path']) as zz:banks[f].append(zz['point_indices'].copy())
        xyz=np.load(j['paths']['processed'],mmap_mode='r')[:,:3]
        with np.load(d/'data.npz') as z,np.load(root/'structure'/s/'data.npz') as st,np.load(root/'priors'/s/'data.npz') as pr,np.load(run.diag.SOURCE/'plan'/s/'evidence.npz') as ev:
            visibility=np.unpackbits(ev['reliable'],axis=1,bitorder='little')[:,:len(ev['roi'])].astype(bool);fidx={int(f):i for i,f in enumerate(ev['frames'])}
            for p in parents:
                cid=p['candidate_id'];meta=metas[cid];halo=st[cid+'_halo'];parent=st[cid+'_parent'];member=np.isin(halo,parent);labels=st[cid+'_labels'];groups=st[cid+'_visible'];pi=np.searchsorted(pr['point_ids'],halo);prob=pr['P1_probability'][pi];hit=prob[member].astype(float).sum(0);siou=hit/(len(parent)+pr['P1_full_volume']-hit);ptop=np.lexsort((np.arange(150),-siou))[:2].tolist();assert p['parent_top2']==ptop and p['anchor']==ptop[0]
                candidate=set(ptop);blocks=[];means=[]
                for rid in sorted(set(map(int,labels))):
                    mask=(labels==rid)&(~member)&(groups>=2)
                    if not mask.any():continue
                    blocks.append(rid);mean=prob[mask].astype(float).sum(0)/mask.sum();means.append(mean);candidate.update(np.lexsort((np.arange(150),-mean))[:2].tolist());assert np.array_equal(z[f'{cid}_r{rid:04d}_block'],halo[mask])
                assert p['queries']==sorted(candidate) and p['regions']==blocks;assert np.array_equal(z[cid+'_queries'],sorted(candidate));assert np.array_equal(z[cid+'_region_means_all150'],np.asarray(means).reshape(-1,150));hard=prob>=.5;qs=z[cid+'_queries'];stored=np.unpackbits(z[cid+'_hard_masks'],axis=0,bitorder='little')[:len(halo)];assert np.array_equal(stored,hard[:,qs]);assert np.array_equal(z[cid+'_probability'],prob[:,qs]);totals['parent_candidate_sets']+=1;totals['bipartite_edges']+=len(candidate)*len(blocks)
                target=GTparents[cid]['target'] if GTparents[cid]['stable'] else None
                for r in [r for r in pairs if r['candidate_id']==cid]:
                    a,b=hard[:,r['query_a']],hard[:,r['query_b']];inter=np.count_nonzero(a&b);union=np.count_nonzero(a|b);assert r['intersection']==inter and r['hard_iou']==(inter/union if union else None);assert r['a_exclusive']==np.count_nonzero(a&~b) and r['b_exclusive']==np.count_nonzero(b&~a);g=GTpairs[cid,r['query_a'],r['query_b']]
                    for key,mask in [('a_halo',a),('b_halo',b),('a_exclusive_GT',a&~b),('b_exclusive_GT',b&~a),('union_GT',a|b)]:checkcounts(g[key],independent_counts(halo[mask],gt,target))
                    for key,q in [('a_ROI',r['query_a']),('b_ROI',r['query_b'])]:checkcounts(g[key],independent_counts(pr['point_ids'][pr['P1_probability'][:,q]>=.5],gt,target))
                    totals['query_pairs_census']+=1
                no=[r for r in pairs if r['candidate_id']==cid and r['no_overlap'] and r['minimum_distance'] is not None]
                if no:
                    r=no[0];value=brute_min(xyz[halo[hard[:,r['query_a']]]],xyz[halo[hard[:,r['query_b']]]]);err=abs(value-r['minimum_distance']);assert err<=1e-7;dist_error=max(dist_error,err);totals['independent_brute_distances']+=1
                parentsubs=[(int(rid),np.flatnonzero((labels==rid)&member)) for rid in np.unique(labels[member])];fm={}
                for f in p['frames']:
                    assert f in meta['views'][run.diag.OBS]['frames'];fm[f]=(visibility[fidx[f],np.searchsorted(ev['roi'],halo)],np.column_stack([np.isin(halo,b) for b in banks[f]]) if banks[f] else np.zeros((len(halo),0),bool))
                assert len(p['frames'])<=6
                for r in [r for r in frames if r['candidate_id']==cid]:
                    vis,masks=fm[r['frame']]
                    if r['kind']=='region_local_vs_whole':
                        dom=np.flatnonzero((labels==r['region'])&(~member)&(groups>=2));centroid=xyz[halo[dom]].mean(0);nearest=min(parentsubs,key=lambda rr:(float(np.linalg.norm(xyz[halo[rr[1]]].mean(0)-centroid)),rr[0]));ref=nearest[1][prob[nearest[1],p['anchor']]>=.5];assert nearest[0]==r['local_reference_region'];assert r['local']==independent_frame(dom,ref,vis,masks);assert r['whole']==independent_frame(dom,np.flatnonzero(member),vis,masks)
                    else:
                        a,b=hard[:,r['query_a']],hard[:,r['query_b']];expect=independent_frame(np.flatnonzero(b&~a),np.flatnonzero(a&~b),vis,masks)
                        for k,v in expect.items():assert r[k]==v
                    totals['independent_frames']+=1
                for r in [r for r in edges if r['candidate_id']==cid]:
                    dom=z[f"{cid}_r{r['region']:04d}_block"];g=GTregions[cid,r['region'],r['query']];checkcounts(g['block_GT'],independent_counts(dom,gt,target));assert abs(r['query_mean']-float(prob[np.isin(halo,dom),r['query']].astype(float).sum()/len(dom)))<=1e-12;totals['query_region_census']+=1
    assert totals['parent_candidate_sets']==207 and totals['query_pairs_census']==5707 and totals['query_region_census']==11004 and totals['independent_frames']==41450
    seal(OUT/'verification.json',dict(complete=True,counts=dict(totals),independent_distance_max_error=dist_error,GT_free_candidate_sets_recomputed=True,independent_frame_evidence=True,postseal_GT_census_recomputed=True,code_sha256=sha(Path(__file__)),scope='Full selection/masks/frames/census; local ROI identities only; upstream Mask3D or full projection pipeline not rerun. Posthoc pool checked separately.'))
    print(dict(totals),flush=True)
if __name__=='__main__':main()
