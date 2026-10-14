#!/usr/bin/env python3
"""Post-seal independent histogram reconstruction and evidence spot checks."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from tools import run_fi1_complementary_fragment_diagnostic as run

def close(a,b):
    assert np.isclose(a,b,rtol=0,atol=1e-12),(a,b)

def independent_views(a,b,frames,boundary,parameters):
    names=('eligible','joint','compact','separated','projection','depth')
    groups={n:set() for n in names};witnesses=[]
    for f in frames:
        av=a[f['visible'][a]];bv=b[f['visible'][b]];na,nb=len(av),len(bv)
        if min(na,nb)<3 or na/len(a)<.1 or nb/len(b)<.1:continue
        group=f['group'];groups['eligible'].add(group)
        px,py=f['pixels'][av],f['pixels'][bv]
        delta=np.maximum(0,np.maximum(px.min(0)-py.max(0),py.min(0)-px.max(0)));gap=float(np.sqrt((delta*delta).sum()))
        if gap<=3:groups['projection'].add(group)
        aset=set(av.tolist());bset=set(bv.tolist());joint=False;compact=False;sepa=False;sepb=False
        for m in f['masks']:
            hit_a=len(aset.intersection(m));hit_b=len(bset.intersection(m))
            ca,cb=hit_a/na,hit_b/nb
            both=ca>=.5 and cb>=.5;joint|=both;compact|=both and (hit_a+hit_b)>=.8*len(m)
            sepa|=ca>=.5 and cb<=.1;sepb|=cb>=.5 and ca<=.1
        separated=sepa and sepb
        for n,v in (('joint',joint),('compact',compact),('separated',separated)):
            if v:groups[n].add(group)
        cp=np.array(boundary['contact_pairs'],int).reshape(-1,2);known=False;continuous=False
        if len(cp):
            cp=cp[f['visible'][cp].all(1)]
            if len(cp)>=3:
                known=True;continuous=float(np.median(np.abs(f['depth'][cp[:,0]]-f['depth'][cp[:,1]])))<=.05
                if continuous:groups['depth'].add(group)
        witnesses.append(dict(frame=f['frame'],group=group,visible_a=na,visible_b=nb,projection_bbox_gap=gap,joint=joint,compact=compact,separated=separated,depth_known=known,depth_continuous=continuous))
    result={n+'_groups':sorted(v) for n,v in groups.items()}
    result['joint_and_compact_depth_groups']=sorted(groups['compact']&groups['depth'])
    result.update(witnesses=witnesses,unknown_identity=not groups['joint'] and not groups['separated'])
    return result

def main():
    out=run.OUT;c=run.read(out/'contract.json');sealed=run.read(out/'features_frozen.json');labels=run.read(out/'labels_frozen.json');summary=run.read(out/'summary.json')
    assert sealed['complete'] and labels['complete'] and sealed['GT_used'] is False
    sys.addaudithook(run.Guard(c,True).hook)
    checks=Counter();bound=dict(c['inputs']);bound.update(c['label_inputs']);bound.update(c['code']);bound.update(c['GT_inputs']);bound.update(sealed['outputs']);bound.update(labels['outputs'])
    run.parent.diag.prior.src.ctl.base.verify_files(bound);checks['bound_file_hashes']=len(bound)
    for q,h in c['code'].items():assert run.sha(out/'code_snapshot'/Path(q).relative_to(ROOT))==h
    prior=[json.loads(x) for x in Path(next(k for k in c['label_inputs'] if k.endswith('targets.jsonl'))).read_text().splitlines()]
    per_scene={};diagnostics=Counter();examples={};opportunity_detail=[]
    for index,s in enumerate(c['scenes']):
        print('verify',index,s,flush=True)
        nt,pts=run.load_nt(c['jobs'][s]);keys=list(pts);keyidx={k:i for i,k in enumerate(keys)}
        pairs=run.read(out/'features'/s/'pairs.json');pairlabels=run.read(out/'labels'/s/'pairs.json')
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);allids,inverse,gt_sizes=np.unique(gt,return_inverse=True,return_counts=True)
        valid=(allids>=1000)&(gt_sizes>=100)&np.isin(allids//1000,c['class_ids']);ids=allids[valid];sizes=gt_sizes[valid];validset=set(ids.tolist())
        hist=np.array([np.bincount(inverse[pts[k]],minlength=len(allids)) for k in keys]);geomsize=np.array([len(pts[k]) for k in keys]);base=hist[:,valid]/(geomsize[:,None]+sizes[None,:]-hist[:,valid]);base[geomsize<100]=0;bestNT=base.max(0)
        def identity(h):
            # Ignore zero-count bins; np.unique's tie chooses the lowest GT id.
            ii=int(np.argmax(h));gid=int(allids[ii]);hit=int(h[ii]);purity=hit/int(h.sum())
            return dict(gt_id=gid,purity=purity,target_points=hit,reliable=gid in validset and hit>=20 and purity>=.75)
        stats={mode:Counter() for mode in ('legacy_overlap','spatial','projection','combined')};signals={sig:Counter() for sig in c['signatures']};quality={sig:{} for sig in c['signatures']}
        expected_domain=set();expected_rows={sig:{} for sig in c['signatures']};recordlookup={};sample=set()
        for r,y in zip(pairs,pairlabels):
            assert (r['left'],r['right'])==(y['left'],y['right'])
            a,b=map(keyidx.get,(r['left'],r['right']));pa,pb=pts[keys[a]],pts[keys[b]];common=np.intersect1d(pa,pb,assume_unique=True);ncommon=len(common)
            ch=np.bincount(inverse[common],minlength=len(allids));ha,hb=hist[a]-ch,hist[b]-ch;uh=hist[a]+hist[b]-ch;usize=int(uh.sum())
            assert r['shared_points']==ncommon and r['left_exclusive_points']==int(ha.sum()) and r['right_exclusive_points']==int(hb.sum())
            close(r['pair_iou'],ncommon/usize);assert r['retrieval']['legacy_overlap']==(.05<ncommon/usize<.9)
            assert r['retrieval']['combined']==(r['retrieval']['spatial'] or r['retrieval']['projection'])
            expected_domain.add((a,b));left,right=identity(ha),identity(hb)
            assert left==y['left_identity'] and right==y['right_identity']
            label='same_instance' if left['reliable'] and right['reliable'] and left['gt_id']==right['gt_id'] else ('different_instances' if left['reliable'] and right['reliable'] else 'mixed_or_unreliable')
            assert label==y['identity_label'];ui=uh[valid]/(usize+sizes-uh[valid]);ui=ui if usize>=100 else np.zeros_like(ui)
            close(y['best_union_iou'],ui.max(initial=0));assert y['best_union_gt']==int(ids[int(ui.argmax())])
            for t in c['thresholds']:assert y['new_hits'][f'{t:.2f}']==ids[(bestNT<=t)&(ui>t)].tolist()
            for h,extra,z in zip((hist[a],hist[b]),(hb,ha),y['directions']):
                anchor=identity(h);target=int(extra[allids==anchor['gt_id']].sum()) if anchor['gt_id'] in validset else 0;pollution=int(extra.sum())-target
                assert z==dict(anchor=anchor,added_target_points=target,added_pollution_points=pollution,target_majority=target>pollution)
            for mode,selected in r['retrieval'].items():
                if selected:stats[mode].update(pairs=1,**{label:1})
            if r['views']:
                v=r['views'];bd=r['boundary'];joint=len(v['joint_groups'])>=2;compact=len(v['compact_groups'])>=2 and not v['separated_groups'];surface=bd['contacts05']>=3 and bd['normal_difference'] is not None and bd['color_difference'] is not None and bd['normal_valid_fraction']>=.5 and bd['normal_difference']<=.5 and bd['color_difference']<=.4
                expected=dict(joint_only=joint,compact_no_separation=bool(compact),compact_boundary_depth=bool(compact and surface and len(v['joint_and_compact_depth_groups'])>=2))
                assert expected==r['signatures'];diagnostics['retrieved_pairs']+=1
                diagnostics['no_joint_no_separation']+=v['unknown_identity'];diagnostics['separation_counterevidence']+=bool(v['separated_groups']);diagnostics['joint_two_groups']+=joint;diagnostics['compact_two_groups']+=len(v['compact_groups'])>=2
                for sig,selected in expected.items():
                    if selected and r['retrieval']['combined']:
                        cc=signals[sig];cc.update(pairs=1,**{label:1});cc['disjoint_pairs']+=ncommon==0;cc['new75_pairs']+=bool(y['new_hits']['0.75']);cc['anchor_reliable_directions']+=sum(z['anchor']['reliable'] for z in y['directions']);cc['target_majority_directions']+=sum(z['anchor']['reliable'] and z['target_majority'] for z in y['directions']);cc['unreliable_anchor_directions']+=sum(not z['anchor']['reliable'] for z in y['directions'])
                        pp=np.union1d(pa,pb);digest=hashlib.sha256(pp.astype('<i8').tobytes()).hexdigest();key=s+':geometry:'+digest
                        if key not in pts:expected_rows[sig].setdefault(key,pp)
                if expected['compact_no_separation'] or expected['compact_boundary_depth']:sample.add((a,b))
            elif index<5 and len(sample)<2:sample.add((a,b))
            recordlookup[tuple(sorted((r['left'],r['right'])))]=r;checks['all_pair_labels']+=1
        # Full domain rebuilt from point membership, independently of records.
        for k in keys:assert np.all(np.diff(pts[k])>0)
        membership=sparse.csr_matrix((np.ones(int(geomsize.sum()),np.int64),np.concatenate([pts[k] for k in keys]),np.r_[0,np.cumsum(geomsize)]),shape=(len(keys),len(gt)))
        intersections=(membership@membership.T).toarray()
        for a in range(len(keys)):
            for b in range(a+1,len(keys)):
                overlap=int(intersections[a,b]);usize=geomsize[a]+geomsize[b]-overlap
                assert ((a,b) in expected_domain)==(geomsize[a]-overlap>=3 and geomsize[b]-overlap>=3 and overlap/usize<.9)
                checks['NT_domain_pairs']+=1
        opps=run.read(out/'labels'/s/'opportunities.json');expected_op=[]
        for target in prior:
            p=target.get('NT_top12_pair_union_oracle')
            if target['scene']==s and target['missing075'] and p and p['iou']>.75:
                edge=tuple(sorted((p['left'],p['right'])));r=recordlookup.get(edge);sample.add(tuple(sorted((keyidx[edge[0]],keyidx[edge[1]]))))
                expected_op.append(dict(gt_id=target['gt_id'],left=edge[0],right=edge[1],oracle_iou=p['iou'],pair_in_domain=r is not None,retrieval=r['retrieval'] if r else None,pair_iou=r['pair_iou'] if r else None,signatures=r['signatures'] if r else None,views=r['views'] if r else None,boundary=r['boundary'] if r else None))
        assert opps==expected_op
        for r in opps:opportunity_detail.append(dict(scene=s,**r));checks['oracle_pairs']+=1
        for mode,cc in stats.items():
            cc['recovered_oracle60']=sum(r['retrieval'] is not None and r['retrieval'][mode] for r in opps);cc['recovered_disjoint26']=sum(r['pair_iou']==0 and r['retrieval'][mode] for r in opps if r['retrieval'] is not None)
        qold=run.read(out/'labels'/s/'geometry_quality.json')
        for sig in c['signatures']:
            rows=run.read(out/'features'/s/(sig+'_geometry.json'));assert len(rows)==len(expected_rows[sig]);newhist=[];newsize=[]
            with np.load(out/'features'/s/(sig+'_points.npz')) as z:
                assert set(z.files)=={r['array_key'] for r in rows}
                for r in rows:
                    p=z[r['array_key']];assert np.array_equal(p,expected_rows[sig][r['geometry_key']]);assert r['point_count']==len(p);assert r['points_sha256']==hashlib.sha256(p.astype('<i8').tobytes()).hexdigest();newhist.append(np.bincount(inverse[p],minlength=len(allids))[valid]);newsize.append(len(p));checks['hypothesis_geometries']+=1
            nh=np.asarray(newhist).reshape(-1,len(ids));ns=np.array(newsize);iou=nh/(ns[:,None]+sizes[None,:]-nh);iou[ns<100]=0;nbest=iou.max(axis=0,initial=0)
            for t in c['thresholds']:
                qt=dict(new_over_complete_NT=ids[(bestNT<=t)&(nbest>t)].tolist(),already_NT_covered=ids[(bestNT>t)&(nbest>t)].tolist(),qualified_geometry=int(np.sum(iou.max(1,initial=0)>t)),nonqualified_geometry=int(np.sum(iou.max(1,initial=0)<=t)));assert qt==qold[sig][f'{t:.2f}'];quality[sig][f'{t:.2f}']={k:len(v) if isinstance(v,list) else v for k,v in qt.items()};checks['geometry_coverage_checks']+=1
        # Physical evidence: all compact-accepted pairs + all60 oracle + deterministic controls.
        if sample:
            raw=np.load(c['jobs'][s]['paths']['processed'],mmap_mode='r')[:,:9].copy();frames,_,_=run.load_frames(c,s,raw,[pts[k] for k in keys])
            for a,b in sorted(sample):
                r=recordlookup.get(tuple(sorted((keys[a],keys[b]))))
                if r is None:continue
                common=np.intersect1d(pts[keys[a]],pts[keys[b]],assume_unique=True);aa=np.setdiff1d(pts[keys[a]],common);bb=np.setdiff1d(pts[keys[b]],common)
                dist,indexes=cKDTree(raw[aa,:3]).query(raw[bb,:3],distance_upper_bound=.1);near=np.isfinite(dist)&(dist<=.05);left=aa[indexes[near]];right=bb[near]
                assert r['retrieval']['spatial']==(np.isfinite(dist).sum()>=3)
                if len(left):
                    rgb=np.clip(raw[:,3:6]/255.,0,1);color=np.linalg.norm(rgb[left]-rgb[right],axis=1)/np.sqrt(3)
                    normals=raw[:,6:9];norm=np.linalg.norm(normals,axis=1);good=(norm[left]>1e-12)&(norm[right]>1e-12)
                    nd=1-np.abs(np.sum(normals[left[good]]/norm[left[good],None]*normals[right[good]]/norm[right[good],None],axis=1))
                    close(r['boundary']['color_difference'],float(np.median(color)));close(r['boundary']['normal_valid_fraction'],float(good.mean()))
                    if len(nd):close(r['boundary']['normal_difference'],float(np.median(nd)))
                    else:assert r['boundary']['normal_difference'] is None
                assert r['boundary']['contacts05']==int(near.sum()) and r['boundary']['contacts10']==int(np.isfinite(dist).sum())
                cp=np.column_stack((left,right));choose=np.linspace(0,len(cp)-1,min(len(cp),128),dtype=int) if len(cp) else np.empty(0,int);bd=dict(contact_pairs=cp[choose].tolist())
                if r['views']:
                    v=independent_views(aa,bb,frames,bd,c['parameters']);assert v==r['views'];assert r['retrieval']['projection']==(len(v['projection_groups'])>=2)
                checks['physical_pair_replays']+=1
        per_scene[s]=dict(retrieval=stats,signatures=signals,geometry_quality=quality,oracle_opportunities=len(opps))
    for name,scenes in c['groups'].items():
        expected=summary['groups'][name]
        for section in ('retrieval','signatures'):
            for sub in per_scene[scenes[0]][section]:
                cc=Counter()
                for s in scenes:cc.update(per_scene[s][section][sub])
                if section=='signatures':
                    den=cc['same_instance']+cc['different_instances'];cc['reliable_pairs']=den;cc['reliable_precision']=cc['same_instance']/den if den else None
                assert dict(cc)==expected[section][sub],(name,section,sub)
        for sig in c['signatures']:
            for t in c['thresholds']:
                cc=Counter()
                for s in scenes:cc.update(per_scene[s]['geometry_quality'][sig][f'{t:.2f}'])
                assert dict(cc)==expected['geometry_quality'][sig][f'{t:.2f}'];checks['group_coverage_aggregates']+=1
    assert opportunity_detail==summary['opportunity_rows'];assert len(opportunity_detail)==60
    strong=summary['groups']['development200']['signatures']['compact_boundary_depth'];new75=summary['groups']['development200']['geometry_quality']['compact_boundary_depth']['0.75']['new_over_complete_NT'];rate=strong['target_majority_directions']/strong['anchor_reliable_directions'] if strong['anchor_reliable_directions'] else 0
    foldpass=all(v['signatures']['compact_boundary_depth']['reliable_pairs']<10 or (v['signatures']['compact_boundary_depth']['reliable_precision'] or 0)>=.9 for k,v in summary['groups'].items() if k.startswith('fold'));advance=new75>1 and (strong['reliable_precision'] or 0)>=.9 and rate>.5 and foldpass
    assert advance==summary['advance_to_own_semantics']==run.read(out/'decision.json')['advance_to_own_semantics'];close(rate,summary['addition_target_majority_rate']);assert foldpass==summary['fold_gate']
    run.freeze_json(out/'independent_verified.json',dict(complete=True,checks=dict(checks),evidence_diagnostics=dict(diagnostics),gate_rebuilt=advance,own_semantic_or_AP_performed=False,GT_read_only_post_seal=True,source_sha256=run.sha(__file__),limitations='Identity precision is conditional on reliable diagnostic subset. Physical replay covers all compact accepted and oracle pairs plus deterministic controls, not all retrieved edges. Depth is derived from visible3D contact points; not independent identity proof. No blind test.'))
    run.write_json(out/'execution.json',dict(status='complete',stage='independently_verified',updated=time.time(),advance_to_own_semantics=advance))
    print('verified',dict(checks),flush=True)

if __name__=='__main__':
    with threadpool_limits(4):main()
