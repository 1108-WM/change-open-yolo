#!/usr/bin/env python3
"""Sealed GT-free query/region graph on exposed52; labels strictly post-seal.
Only descriptions and post-hoc references, no new production geometry/rules.
"""
from pathlib import Path
from collections import Counter
from itertools import combinations
import sys,json,argparse,shutil,hashlib,time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial import cKDTree
import run_fi1_nt_region_identity as diag
from fi1_local_edit_evidence_core import frame_evidence
OUT=ROOT/'research/geometry/experiments/fi1_query_relation_diagnostic_exposed52_20261014_v1'
SOURCES={'fit12':diag.OUT,'exposed16':ROOT/'research/geometry/experiments/fi1_query_transfer_eval_20261014_v1','internal24':ROOT/'research/geometry/experiments/fi1_t_add_union_internal24_20261014_v1'}
read=diag.read;seal=diag.seal;sha=diag.sha

def guard():
    def hook(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
            p=str(args[0])
            if any(t in p for t in ('ground_truth','/labels/','/diagnosis/','/evaluation/','failure_diagnosis','structure_diagnosis','prior_diagnosis')):raise PermissionError('GT/results prohibited during relation generation: '+p)
    sys.addaudithook(hook)
def stage(n,**kw):
    diag.write_json(OUT/'execution.json',dict(stage=n,time=time.time(),**kw));print(n,kw,flush=True)
def top2(v):return np.lexsort((np.arange(len(v)),-v))[:2].tolist()
def fraction(n,d):return float(n/d) if d else None

def prepare():
    assert not OUT.exists();diag.reserve();assert read(ROOT/'research/geometry/experiments/fi1_t_add_union_downstream_internal24_20261014_v1/completion_verified.json')['complete']
    jobs={};cohorts={};gt={};inputs={};parents=0
    for group,root in SOURCES.items():
        c=read(root/'contract.json');cohorts[group]=c['scenes'];inputs[str(root/'contract.json')]=sha(root/'contract.json')
        for s in c['scenes']:
            assert s not in jobs;j=c['jobs'][s];jobs[s]=dict(j,cohort=group,asset_root=str(root));gt[j['GT']]=c['GT_inputs'][j['GT']]
            for p in (root/'structure'/s/'data.npz',root/'structure'/s/'candidates.json',root/'priors'/s/'data.npz',diag.SOURCE/'features'/s/'data.npz',diag.SOURCE/'plan'/s/'evidence.npz',j['binding']['geometry'],j['binding']['points'],j['paths']['processed'],j['paths']['observations']):inputs[str(p)]=sha(p)
            parents+=sum(r['source']=='track' for r in read(root/'structure'/s/'candidates.json'))
    assert len(jobs)==52 and len({s.split('_')[0] for s in jobs})==52 and parents==207
    files=[Path(__file__).resolve(),Path(__file__).with_name('fi1_local_edit_evidence_core.py'),Path(__file__).with_name('fi1_nt_region_identity_core.py'),ROOT/'related papers/Clutt3R-Seg_2026_arXiv2602.11660.txt']
    OUT.mkdir(parents=True)
    c=dict(version=OUT.name,jobs=jobs,scenes=list(jobs),cohorts=cohorts,class_ids=c['class_ids'],GT_inputs=gt,inputs=inputs,code={str(p):sha(p) for p in files},parents=parents,
      candidate_queries='Original full-scene-softIoU parent top2 + mean top2 for every nonempty outside-parent member region at fixed visible>=2 domain; ID ascending ties; dedup; anchor original parent top1.',
      regions='Original member regions; editable domain visible>=2; no new split, size filter or gate; full region composition also recorded',
      local_parent_reference='For each outside editable region choose closest original parent member region by centroid (region ID tie). Local reference is that region intersect anchor P1>=.5; no fallback when empty. Pointset reference excludes all editable outside points. This is new descriptive support, not whole-parent reference.',
      query_pairs='All unordered candidate pairs, including zero overlap; no GT pruning. Hard threshold existing .5 for descriptions only. Region contact original .05 edges; min hard-support spatial distance retained including zero overlap.',
      multiview='Only each parents existing <=6 selected frames and original automatic masks/depth reliability; same .5 reference coverage definitions as prior diagnostics. Missing/no-common/conflict separate; local and whole-parent reference both saved.',
      pairing_frames='Exclusive hard-supported halo subsets per query pair, same visibility/mask bank; duplicate masks with no exclusive support are missing/uncomparable, never negative.',
      GT_scope='Query identities available only within stored halo/ROI, not full-scene mask. Global soft/hard volumes recorded; no claim of global same-instance identity from local purity. Small supports<20 retained unknown. Safety requires target>=.9 AND known annotation>=.9 for diagnostic bins only, not executed filters.',
      diagnostic='Post-seal all150 query reference plus candidate singles, anchor+pairs, supplied pure whole-region and point constructions against complete original NT. All GT constructs labeled posthoc, never appended or promoted to rule.',
      statistical='Macro physical-scene AUROC plus scene bootstrap200 seed20261014. Stratify complementary, duplicate, neighbor, small and no-overlap. No fitted grouping rule;52 previously exposed development scenes.',
      task1_completed_before_this_contract=True,training=False,new_observations=False,new_production_candidates=False,production_modified=False,github_push=False)
    diag.verify(inputs);seal(OUT/'contract.json',c)
    for p in files:q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    stage('prepared',scenes=52,T_parents=parents)

def framesummary(rr):
    valid=[r for r in rr if not r['missing']]
    return dict(frames=len(rr),valid_frames=len(valid),missing_frames=len(rr)-len(valid),common_frames=sum(r['common'] for r in valid),conflict_frames=sum(r['conflict'] for r in valid),no_common_no_conflict_frames=sum(not r['common'] and not r['conflict'] for r in valid))
def spatial(mask1,mask2,xyz):
    if not mask1.any() or not mask2.any():return None
    if (mask1&mask2).any():return 0.
    return float(cKDTree(xyz[mask1]).query(xyz[mask2],k=1)[0].min())

def generate():
    c=read(OUT/'contract.json');guard();diag.verify(c['inputs']);diag.verify(c['code']);outputs={};totals=Counter()
    for si,(s,j) in enumerate(c['jobs'].items()):
        stage('GT_free_relation_graph',scene=s,completed=si,total=52);root=Path(j['asset_root']);metas=[r for r in read(root/'structure'/s/'candidates.json') if r['source']=='track'];raw=diag.core.safe_input(np.load(j['paths']['processed'],mmap_mode='r'));xyz=raw[:,:3]
        obs=[json.loads(l) for l in Path(j['paths']['observations']).read_text().splitlines()];bankroot=Path(obs[0]['point_indices_path']).parent.parent;banks={};bank_inputs={}
        for f in sorted({f for r in metas for f in r['views'][diag.OBS]['frames']}):
            path=bankroot/f'frame{f:06d}'/'observations.json';bank_inputs[str(path)]=sha(path);banks[f]=[]
            for r in read(path):
                p=Path(r['point_indices_path']);bank_inputs[str(p)]=sha(p)
                with np.load(p) as z:banks[f].append(z['point_indices'].copy())
        parentrows=[];bipartite=[];pairrows=[];frame_rows=[];arrays={}
        with np.load(root/'structure'/s/'data.npz') as st,np.load(root/'priors'/s/'data.npz') as pr,np.load(diag.SOURCE/'plan'/s/'evidence.npz') as ev:
            visible=np.unpackbits(ev['reliable'],axis=1,bitorder='little')[:,:len(ev['roi'])].astype(bool);fidx={int(f):i for i,f in enumerate(ev['frames'])}
            for meta in metas:
                cid=meta['candidate_id'];halo=st[cid+'_halo'];parent=st[cid+'_parent'];member=np.isin(halo,parent);labels=st[cid+'_labels'];groups=st[cid+'_visible'];ix=np.searchsorted(pr['point_ids'],halo);assert np.array_equal(pr['point_ids'][ix],halo);p=pr['P1_probability'][ix];own=np.flatnonzero(member);localxyz=xyz[halo]
                assoc=diag.core.association(p,pr['P1_full_volume'],own);anchor=int(assoc['top'][0]);queries=set(assoc['top']);blocks=[];means=[];doms=[];parentsubs=[(rid,np.flatnonzero((labels==rid)&member)) for rid in np.unique(labels[member])]
                for rid in np.unique(labels):
                    dom=np.flatnonzero((labels==rid)&(~member)&(groups>=2))
                    if not len(dom):continue
                    m=p[dom].mean(0,dtype=np.float64);queries.update(top2(m));blocks.append(int(rid));means.append(m);doms.append(dom)
                queries=sorted(queries);hard=p[:,queries]>=.5;arrays[cid+'_queries']=np.asarray(queries,np.int32);arrays[cid+'_halo']=halo;arrays[cid+'_parent']=parent;arrays[cid+'_block_regions']=np.asarray(blocks,np.int32);arrays[cid+'_region_means_all150']=np.asarray(means).reshape(-1,150);arrays[cid+'_hard_masks']=np.packbits(hard,axis=0,bitorder='little');arrays[cid+'_probability']=p[:,queries];fm={}
                for f in meta['views'][diag.OBS]['frames']:
                    vis=visible[fidx[f],np.searchsorted(ev['roi'],halo)];masks=np.column_stack([np.isin(halo,b) for b in banks[f]]) if banks[f] else np.zeros((len(halo),0),bool);fm[f]=(vis,masks)
                edges={tuple(sorted(map(int,e))) for e in st[cid+'_edges']};nearref={};blockframes={}
                for rid,dom,m in zip(blocks,doms,means):
                    arrays[f'{cid}_r{rid:04d}_block']=halo[dom];center=localxyz[dom].mean(0);nearest=min(parentsubs,key=lambda r:(float(np.linalg.norm(localxyz[r[1]].mean(0)-center)),r[0]));reference=nearest[1][p[nearest[1],anchor]>=.5];nearref[rid]=reference;rr=[];whole=[]
                    for f,(vis,masks) in fm.items():
                        row=frame_evidence(dom,reference,vis,masks);wr=frame_evidence(dom,own,vis,masks);rr.append(row);whole.append(wr);frame_rows.append(dict(scene=s,candidate_id=cid,kind='region_local_vs_whole',region=rid,frame=f,local_reference_region=nearest[0],local=row,whole=wr))
                    blockframes[rid]=(framesummary(rr),framesummary(whole),nearest[0],len(reference))
                    for qi,q in enumerate(queries):
                        h=hard[:,qi];bh=h[dom];ph=h[own];allreg=np.flatnonzero(labels==rid);regionhard=float((p[allreg,q]>=.5).mean());local,wh,refid,refpoints=blockframes[rid]
                        bipartite.append(dict(scene=s,cohort=j['cohort'],candidate_id=cid,region=rid,query=q,anchor=anchor,is_anchor=q==anchor,points=len(dom),full_region_points=len(allreg),editable_fraction=len(dom)/len(allreg),query_mean=float(m[q]),region_top2=q in top2(m),query_parent_mean=float(p[own,q].mean()),hard_region_fraction=float(bh.mean()),hard_full_region_fraction=regionhard,hard_parent_fraction=float(ph.mean()),hard_halo_points=int(h.sum()),soft_full_scene_volume=float(pr['P1_full_volume'][q]),hard_full_scene_volume=int(pr['P1_hard_volume'][q]),hard_halo_fraction_of_full=fraction(h.sum(),pr['P1_hard_volume'][q]),parent_contact_region=tuple(sorted((rid,refid))) in edges,local_reference_region=refid,local_reference_points=refpoints,**{'local_'+k:v for k,v in local.items()},**{'whole_'+k:v for k,v in wh.items()}))
                for ia,ib in combinations(range(len(queries)),2):
                    qa,qb=queries[ia],queries[ib];a,b=hard[:,ia],hard[:,ib];inter=int((a&b).sum());union=int((a|b).sum());exclusivea=np.flatnonzero(a&~b);exclusiveb=np.flatnonzero(b&~a);rr=[]
                    for f,(vis,masks) in fm.items():
                        row=frame_evidence(exclusiveb,exclusivea,vis,masks);rr.append(row);frame_rows.append(dict(scene=s,candidate_id=cid,kind='query_pair_exclusive',query_a=qa,query_b=qb,frame=f,**row))
                    ra=set(map(int,np.unique(labels[a])));rb=set(map(int,np.unique(labels[b])));contact=sum((x in ra and y in rb) or (y in ra and x in rb) for x,y in edges);centerdist=float(np.linalg.norm(localxyz[a].mean(0)-localxyz[b].mean(0))) if a.any() and b.any() else None
                    pairrows.append(dict(scene=s,cohort=j['cohort'],candidate_id=cid,anchor=anchor,query_a=qa,query_b=qb,has_anchor=anchor in (qa,qb),a_hard_points=int(a.sum()),b_hard_points=int(b.sum()),intersection=inter,hard_iou=inter/union if union else None,a_containment=fraction(inter,a.sum()),b_containment=fraction(inter,b.sum()),a_exclusive=len(exclusivea),b_exclusive=len(exclusiveb),parent_a_coverage=float(a[own].mean()),parent_b_coverage=float(b[own].mean()),parent_union_coverage=float((a|b)[own].mean()),editable_a_coverage=float(a[(~member)&(groups>=2)].mean()) if ((~member)&(groups>=2)).any() else None,editable_b_coverage=float(b[(~member)&(groups>=2)].mean()) if ((~member)&(groups>=2)).any() else None,soft_intersection=float(np.minimum(p[:,qa],p[:,qb]).sum()),soft_iou=float(np.minimum(p[:,qa],p[:,qb]).sum()/np.maximum(np.maximum(p[:,qa],p[:,qb]).sum(),1e-12)),a_full_volume=float(pr['P1_full_volume'][qa]),b_full_volume=float(pr['P1_full_volume'][qb]),minimum_distance=spatial(a,b,localxyz),centroid_distance=centerdist,region_contact_edges=contact,shared_regions=len(ra&rb),no_overlap=inter==0,small_support=min(a.sum(),b.sum())<20,**framesummary(rr)))
                parentrows.append(dict(scene=s,cohort=j['cohort'],candidate_id=cid,parent_geometry=meta['parent'],parent_points=len(parent),halo_points=len(halo),anchor=anchor,parent_top2=assoc['top'],association=assoc,queries=queries,regions=blocks,editable_points=sum(map(len,doms)),frames=meta['views'][diag.OBS]['frames']))
        d=OUT/'relations'/s;d.mkdir(parents=True,exist_ok=True)
        for name,rr in [('parents',parentrows),('query_regions',bipartite),('query_pairs',pairrows),('frames',frame_rows)]:seal(d/(name+'.json'),rr)
        np.savez_compressed(d/'data.npz',**arrays);seal(d/'receipt.json',dict(complete=True,inputs=bank_inputs,GT_used=False,outputs={str(p):sha(p) for p in d.iterdir()}))
        outputs.update({str(p):sha(p) for p in d.iterdir()});totals.update(parents=len(parentrows),query_region_edges=len(bipartite),query_pairs=len(pairrows),frame_records=len(frame_rows),regions=sum(len(r['regions']) for r in parentrows),candidate_queries=sum(len(r['queries']) for r in parentrows))
        diag.reserve()
    seal(OUT/'relations_frozen.json',dict(complete=True,outputs=outputs,counts=dict(totals),GT_used=False));stage('relations_sealed',counts=dict(totals))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','generate'));a=p.parse_args();globals()[a.mode]()
