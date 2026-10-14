#!/usr/bin/env python3
"""Independent feasible-set/aggregate/raw-mask replay; leaves original seals intact."""
from pathlib import Path
import ast, csv, sys, time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_nt_region_identity as run
from verify_fi1_nt_region_identity import vector


def table(p):return list(csv.DictReader(p.open()))
def truth(v):return ast.literal_eval(v)


def main():
    c=run.read(run.OUT/'contract.json');run.verify(run.read(run.OUT/'supplement_frozen.json')['outputs'])
    feasible_rows=table(run.OUT/'structure_diagnosis/parents.csv')
    feasible_target_rows=table(run.OUT/'structure_diagnosis/targets.csv')
    feasible_lookup={(r['scene'],r['candidate_id'],r['mode']):r for r in feasible_rows}
    targetlookup={(r['scene'],r['mode'],int(r['gt_id'])):r for r in feasible_target_rows}
    rawlookup={(r['scene'],r['candidate_id'],r['prior'],int(r['query'])):r for r in table(run.OUT/'supplement/all_raw_queries.csv')}
    modes=['point_feasible','region_feasible','point_delete_feasible','region_delete_feasible','old96_delete_feasible']
    count=targets=rawcount=edges_checked=oldheadcount=0
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True)
        keep=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[keep],sizes[keep]
        _,nt=run.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,vector(set(map(int,p)),gt,ids,sizes))
        best={m:baseline.copy() for m in modes}
        with np.load(run.OUT/'structure'/s/'data.npz') as z,np.load(run.OUT/'structure_diagnosis'/(s+'_feasible.npz')) as fs,\
             np.load(run.REGIONS/'regions'/s/'regions.npz') as bank,np.load(run.SOURCE/'features'/s/'data.npz') as original,\
             np.load(run.OUT/'priors'/s/'data.npz') as ps,np.load(run.AB/'instance_features'/s/'data.npz') as old:
            raw=np.load(c['jobs'][s]['paths']['processed'],mmap_mode='r');xyz=raw[:,:3]
            roi=ps['point_ids'];oldix=np.searchsorted(old['point_ids'],roi);assert np.array_equal(old['point_ids'][oldix],roi)
            assert np.array_equal(old['features'][oldix],ps['P0_head_roi']);oldheadcount+=len(roi)
            for r in c['parents'][s]:
                cid=r['candidate_id'];parent=set(map(int,z[cid+'_parent']));halo=z[cid+'_halo'];visible=z[cid+'_visible'];labels=z[cid+'_labels'];members=z[cid+'_membership']
                assert np.array_equal(original[cid+'_parent'],z[cid+'_parent']) and np.array_equal(original[cid+'_halo'],halo)
                assert np.array_equal(original[cid+'_'+run.OBS+'_groups'],visible)
                nn=original[cid+'_neighbors'];src=np.repeat(np.arange(len(halo)),nn.shape[1]);dst=nn.ravel()
                close=np.linalg.norm(xyz[halo[src]]-xyz[halo[dst]],axis=1)<=.05
                same=np.all(members[src]==members[dst],axis=1);allowed=close&same
                assert np.all(labels[src[allowed]]==labels[dst[allowed]])
                # Independent union-find proves neither merged nor split components.
                root=np.arange(len(halo))
                def find(i):
                    while root[i]!=i:root[i]=root[root[i]];i=int(root[i])
                    return i
                for u,v in zip(src[allowed],dst[allowed]):
                    ru,rv=find(int(u)),find(int(v))
                    if ru!=rv:root[rv]=ru
                component={};reverse={}
                for i in range(len(halo)):
                    a=find(i);b=int(labels[i]);component.setdefault(a,b);reverse.setdefault(b,a)
                    assert component[a]==b and reverse[b]==a
                edges_checked+=int(allowed.sum())
                before=vector(parent,gt,ids,sizes);idx=int(before.argmax()) if len(ids) and before.max()>0 else None
                gid=int(ids[idx]) if idx is not None else None;hits=sum(gt[i]==gid for i in parent) if gid is not None else 0
                stable=gid is not None and hits>=20 and hits/len(parent)>=.6
                order=z[cid+'_order'];off=z[cid+'_offsets'];regions=[order[a:b] for a,b in zip(off[:-1],off[1:])]
                flat=bank[cid+'_block_indices'];offold=bank[cid+'_block_offsets'];oldregions=[flat[offold[i]:offold[i+1]] for i in bank[cid+'_budget96']]
                for mode in modes:
                    add=set();remove=set();deletion_only='delete' in mode
                    if stable:
                        regionlist=regions if mode.startswith('region') else oldregions if mode.startswith('old96') else None
                        if regionlist is None:
                            for i,point in enumerate(halo):
                                if visible[i]<2:continue
                                point=int(point);g=int(gt[point])
                                if point not in parent and g==gid and not deletion_only:add.add(point)
                                if point in parent and g>=1000 and g!=gid:remove.add(point)
                        else:
                            for b in regionlist:
                                domain=[int(halo[i]) for i in b if not deletion_only or int(halo[i]) in parent and visible[i]>=2]
                                if not domain:continue
                                pure_target=all(gt[i]==gid for i in domain)
                                pure_other=all(gt[i]>=1000 and gt[i]!=gid for i in domain)
                                for i in b:
                                    point=int(halo[i])
                                    if visible[i]<2:continue
                                    if pure_target and point not in parent and gt[point]==gid and not deletion_only:add.add(point)
                                    if pure_other and point in parent and gt[point]>=1000 and gt[point]!=gid:remove.add(point)
                    expected=(parent-remove)|add
                    if len(expected)<100:expected=parent.copy()
                    actual=set(map(int,fs[cid+'_'+mode]));assert expected==actual,(s,cid,mode)
                    row=feasible_lookup[s,cid,mode];assert int(row['added'])==len(actual-parent) and int(row['deleted'])==len(parent-actual)
                    after=vector(actual,gt,ids,sizes);best[mode]=np.maximum(best[mode],after)
                    if stable:assert abs(float(row['repaired_iou'])-after[idx])<1e-10
                    count+=1
                ix=np.searchsorted(roi,halo);target=gt[halo]
                for family in ('P0','P1'):
                    for q in range(ps[family+'_probability'].shape[1]):
                        mask=ps[family+'_probability'][ix,q]>=.5;row=rawlookup[s,cid,family,q]
                        assert int(row['raw_halo_mask_points'])==int(mask.sum())
                        assert int(row['raw_halo_unknown_points'])==int((mask&(target<1000)).sum())
                        if stable:
                            intersection=int((mask&(target==gid)).sum());union=int(mask.sum()+(target==gid).sum()-intersection)
                            assert abs(float(row['raw_target_iou'])-intersection/max(1,union))<1e-10
                        rawcount+=1
        for mode in modes:
            for i,g in enumerate(ids):
                row=targetlookup[s,mode,int(g)];assert abs(float(row['expanded'])-best[mode][i])<1e-10
                assert truth(row['gain02'])==bool(best[mode][i]-baseline[i]>=.02-1e-12);targets+=1
        print('supplement verified',s,count,rawcount,flush=True)
    prior=run.read(run.OUT/'prior_diagnosis/summary.json')
    parents=table(run.OUT/'prior_diagnosis/parents.csv');pool=table(run.OUT/'prior_diagnosis/targets.csv')
    for mode in c['modes']:
        rr=[r for r in parents if r['mode']==mode];tt=[r for r in pool if r['mode']==mode];m=prior['modes'][mode]
        for field in ('added','deleted','edits','added_target','added_other','deleted_target','deleted_other','unknown_edits'):
            assert m[field]==sum(int(r[field] or 0) for r in rr)
        assert m['actions']==sum(truth(r['changed']) for r in rr)
        assert m['pool_gain02']==sum(truth(r['gain02']) for r in tt)
        assert m['pool_value_targets']==sum(truth(r['value']) for r in tt)
        assert abs(m['pool_delta_sum']-sum(float(r['delta']) for r in tt))<1e-10
    timeline={name:run.read(run.OUT/(name+'_frozen.json'))['time'] for name in ('structure','structure_diagnosis','priors','associations','prior_diagnosis','supplement')}
    assert timeline['structure']<timeline['structure_diagnosis']<timeline['priors']<timeline['associations']<timeline['prior_diagnosis']<timeline['supplement']
    receipt=dict(complete=True,feasible_pointsets=count,feasible_pool_targets=targets,raw_query_readouts=rawcount,
        graph_edges_checked=edges_checked,P0_head_matches_old_prior_ROI_points=oldheadcount,stage_timeline=timeline,
        independent_aggregate_metrics=True,old_parents_halos_visibility_exact=True,time=time.time(),
        code={str(Path(__file__).resolve()):run.sha(Path(__file__).resolve())})
    run.seal(run.OUT/'supplement_verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':main()
