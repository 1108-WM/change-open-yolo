#!/usr/bin/env python3
"""Round2: sealed evidence and point readout references; no mandatory attribution gate."""
from pathlib import Path
from collections import Counter
import argparse,json,sys,shutil,time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial import cKDTree
import run_fi1_nt_region_identity as diag
import run_fi1_query_relation_diagnostic as prior
import fi1_rejectable_local_block_core as core
import evaluate_fi1_nt_region_identity as met
OUT=ROOT/'research/geometry/experiments/fi1_local_block_attribution_selection_diagnostic_exposed52_20261015_v1'
SOURCE=prior.OUT;read=diag.read;seal=diag.seal;sha=diag.sha

def stage(n,**kw):
    diag.write_json(OUT/'execution.json',dict(stage=n,time=time.time(),**kw));print(n,kw,flush=True)
def prepare():
    assert not OUT.exists();diag.reserve();old=read(SOURCE/'contract.json');inputs={str(SOURCE/'contract.json'):sha(SOURCE/'contract.json'),str(SOURCE/'relations_frozen.json'):sha(SOURCE/'relations_frozen.json')};inputs.update(old['inputs']);inputs.update(read(SOURCE/'relations_frozen.json')['outputs'])
    files=[Path(__file__).resolve(),Path(__file__).with_name('analyze_fi1_local_block_attribution_selection.py'),Path(__file__).with_name('fi1_rejectable_local_block_core.py'),Path(__file__).with_name('verify_fi1_local_block_attribution_selection.py'),ROOT/'research/geometry/code/tests/test_fi1_local_block_attribution_selection.py']
    OUT.mkdir(parents=True)
    c=dict(version=OUT.name,scenes=old['scenes'],cohorts=old['cohorts'],jobs=old['jobs'],GT_inputs=old['GT_inputs'],class_ids=old['class_ids'],inputs=inputs,code={str(p):sha(p) for p in files},scope='All previously exposed52 physical scenes/207 sampled T parents; developmental diagnostic only. No final holdout or independent validation claim.',
      target='Diagnostic only, original N/T unchanged; 207 sampled T parents with original halo/regions/query set/observations. No method asserted, training or new hard filter.',
      query='Choose highest whole editable block mean among sealed queries, ascending ID tie, independent of GT. No high parent support/overlap/common-frame prerequisite.',
      attribution_reference='For each editable block use original parent membership region at minimum point-to-point distance, region ID tie. Full region reference, no anchor-filter. Frame observations and evidence signatures are descriptors only.',
      observation='Original <=6 selected frames, same automatic masks and reliability; raw unique best-mask coverage and common/conflict/missing/tie/no-support/weak states retained. No state is a mandatory attribution gate.',
      point_selection='Two fixed UNATTRIBUTED comparison constructions: W adds the entire supplied outside block; C adds only chosen query>=.5 points INSIDE the same block, never another region. Both preserve all original parent points. Also per-parent union all blocks separately for W/C; not claimed as safe methods.',
      output_status='No accepted attributable output is defined in this round. Every block remains pending/abstain unless future independent rule authorized. W/C are geometry-only unattributed references, not deployed candidates or AP evidence.',
      evidence_signature='Descriptive tuple fixed preGT: best supporter==anchor, supporter hard parent coverage zero/nonzero, anchor/supporter halo masks overlap zero/nonzero, original region contact, common present, conflict present, missing present, tie present, any unique observed mask. Signatures are NOT selectors, scores or new thresholds.',
      factorial_diagnostic='After full seal, label block ownership and independently count W/C target/other/unknown precision/retention; including rejected/pending, mixed parents and unknown. GT-pure-block attribution oracle + W/C selection and point-target oracle explicitly posthoc and NEVER full method. Actual fixed-union supply against completeNT remains separate.',
      evidence_analysis='Fixed signature collision census, opposite-label cross-scene examples and continuous feature ranges are descriptive only, not a fitted selector or impossibility proof. Valuable-block subsets are GT-posthoc diagnostics with all negatives retained. Separate parent scopes and unknown annotation.',
      opportunity='CompleteNT same-parent IoU gain>=.02 plus continuous gain. Analyze every stable point/pure-region opportunity, all previous candidate-single opportunity parents, pure-block positive union contributions and individually valuable W/C blocks. Attribute low overlap/conflict/unknown/mixed constraints without GT output selection.',
      success='Proceed only if a reproducible GT-free ATTRIBUTION AND point choice survives all negatives and retains >=2 safe gain02 physical scenes. AUC, GT-oracle construction, average T improvement, or abstaining everywhere do not pass. If no such selector established stop this fixed-evidence round before AP, not all query approaches.',
      downstream='Only a qualified frozen actual output would receive own semantics/frozen SL and internal AP; no AP claim for GT-selected blocks or geometry-only reference. Geometry gain zero alone never implies no AP benefit. No method output is asserted here.',
      constraint_update='Separate development round following explicit user constraints, before any prior-round GT access. Earlier mandatory-shared-mask prototype sealed and retired without GT/AP. No overwrite of old seals.',
      training=False,new_observations=False,threshold_search=False,production_modified=False,github_push=False,reading_entry_modified=False)
    diag.verify(inputs);seal(OUT/'contract.json',c)
    for p in files:q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    stage('contract_frozen',scenes=len(c['scenes']))

def generate():
    c=read(OUT/'contract.json');prior.guard();diag.verify(c['code']);diag.verify(c['inputs']);outputs={};totals=Counter()
    for si,(s,j) in enumerate(c['jobs'].items()):
        stage('GT_free_attribution_and_selection',scene=s,completed=si,total=52);asset=Path(j['asset_root']);parents=read(SOURCE/'relations'/s/'parents.json');xyz=diag.core.safe_input(np.load(j['paths']['processed'],mmap_mode='r'))[:,:3];metas={r['candidate_id']:r for r in read(asset/'structure'/s/'candidates.json')};obs=[json.loads(l) for l in Path(j['paths']['observations']).read_text().splitlines()];bankroot=Path(obs[0]['point_indices_path']).parent.parent;banks={};bankinputs={}
        for f in sorted({f for p in parents for f in p['frames']}):
            path=bankroot/f'frame{f:06d}'/'observations.json';bankinputs[str(path)]=sha(path);banks[f]=[]
            for r in read(path):
                path=Path(r['point_indices_path']);bankinputs[str(path)]=sha(path)
                with np.load(path) as z:banks[f].append(z['point_indices'].copy())
        blocks=[];frames=[];unions=[];arrays={}
        with np.load(asset/'structure'/s/'data.npz') as st,np.load(asset/'priors'/s/'data.npz') as pr,np.load(diag.SOURCE/'plan'/s/'evidence.npz') as ev:
            reliable=np.unpackbits(ev['reliable'],axis=1,bitorder='little')[:,:len(ev['roi'])].astype(bool);fi={int(f):i for i,f in enumerate(ev['frames'])}
            for p in parents:
                cid=p['candidate_id'];parent=st[cid+'_parent'];halo=st[cid+'_halo'];labels=st[cid+'_labels'];member=np.isin(halo,parent);groups=st[cid+'_visible'];prob=pr['P1_probability'][np.searchsorted(pr['point_ids'],halo)];qs=p['queries'];anchor=p['anchor'];ownregions=[]
                for rid in np.unique(labels[member]):
                    idx=np.flatnonzero(member&(labels==rid));ownregions.append((int(rid),idx,cKDTree(xyz[halo[idx]])))
                fm={f:(reliable[fi[f],np.searchsorted(ev['roi'],halo)],np.column_stack([np.isin(halo,b) for b in banks[f]]) if banks[f] else np.zeros((len(halo),0),bool)) for f in p['frames']};selected={'W':[],'C':[]};accepted=[]
                for rid in p['regions']:
                    idx=np.flatnonzero((labels==rid)&(~member)&(groups>=2));assert len(idx);q,mean=core.supporter(prob,idx,qs);distances=sorted([(float(tree.query(xyz[halo[idx]],k=1)[0].min()),rr,ref) for rr,ref,tree in ownregions],key=lambda r:(r[0],r[1]));distance,refid,reference=distances[0];witness=[]
                    for f,(vis,masks) in fm.items():
                        row=core.frame_witness(idx,reference,vis,masks);witness.append(row);frames.append(dict(scene=s,candidate_id=cid,region=rid,frame=f,reference_region=refid,**row))
                    decision='pending_no_validated_attribution';accept=False;whole,clip=core.readout(idx,prob,q,True);block=halo[idx];clipproposal=halo[idx[prob[idx,q]>=.5]];arrays[f'{cid}_r{rid:04d}_block']=block;arrays[f'{cid}_r{rid:04d}_clipped_proposal']=clipproposal
                    row=dict(scene=s,cohort=j['cohort'],candidate_id=cid,region=rid,block_id=f'{cid}_r{rid:04d}',points=len(idx),supporter=q,supporter_mean=mean,anchor=anchor,supporter_halo_points=int((prob[:,q]>=.5).sum()),anchor_halo_points=int((prob[:,anchor]>=.5).sum()),anchor_supporter_hard_intersection=int(((prob[:,q]>=.5)&(prob[:,anchor]>=.5)).sum()),anchor_supporter_hard_union=int(((prob[:,q]>=.5)|(prob[:,anchor]>=.5)).sum()),supporter_full_soft_volume=float(pr['P1_full_volume'][q]),supporter_full_hard_volume=int(pr['P1_hard_volume'][q]),parent_region_contact=tuple(sorted((rid,refid))) in {tuple(sorted(map(int,e))) for e in st[cid+'_edges']},alternate_supporter=q!=anchor,query_parent_mean=float(prob[member,q].mean()),query_parent_hard_fraction=float((prob[member,q]>=.5).mean()),anchor_block_mean=float(prob[idx,anchor].mean()),reference_region=refid,reference_points=len(reference),minimum_distance=distance,reference_alternatives=[dict(region=rr,distance=d,points=len(ref)) for d,rr,ref in distances],state=decision,frame_states=dict(Counter(r['state'] for r in witness)),whole_reference_points=len(idx),clipped_reference_points=len(clipproposal),accepted=accept)
                    row['candidate_query_means']=[dict(query=int(qq),mean=float(prob[idx,qq].mean(dtype=np.float64)),hard_fraction=float((prob[idx,qq]>=.5).mean())) for qq in sorted(qs)];states=row['frame_states'];row['evidence_signature']=[q==anchor,row['query_parent_hard_fraction']==0,row['anchor_supporter_hard_intersection']==0,row['parent_region_contact'],states.get('common',0)>0,states.get('conflict',0)>0,states.get('missing',0)>0,states.get('ambiguous_tie',0)>0,any(x['edit_best_mask'] is not None and x['reference_best_mask'] is not None for x in witness)];blocks.append(row);totals[decision]+=1
                    selected['W'].append(halo[whole]);selected['C'].append(halo[clip])
                for arm in ('W','C'):
                    add=np.unique(np.concatenate(selected[arm])) if selected[arm] else np.empty(0,np.int64);points=np.union1d(parent,add);assert not np.intersect1d(parent,add).size
                    arrays[cid+'_'+arm+'_added']=add;arrays[cid+'_'+arm+'_points']=points;unions.append(dict(scene=s,cohort=j['cohort'],candidate_id=cid,parent_geometry=p['parent_geometry'],arm=arm,added=len(add),points=len(points),extra_candidate=False,reference_nonempty=bool(len(add)),output_status='unattributed_comparison_only',reference_regions=p['regions'],accepted_regions=[],points_sha256=diag.hashlib.sha256(points.astype('<i8').tobytes()).hexdigest()));totals[arm+'_nonempty']+=bool(len(add));totals[arm+'_added']+=len(add)
        d=OUT/'predictions'/s;d.mkdir(parents=True)
        for n,x in [('blocks',blocks),('frames',frames),('unions',unions)]:seal(d/(n+'.json'),x)
        np.savez_compressed(d/'data.npz',**arrays);seal(d/'receipt.json',dict(complete=True,inputs=bankinputs,GT_used=False,outputs={str(p):sha(p) for p in d.iterdir()}));outputs.update({str(p):sha(p) for p in d.iterdir()});totals.update(parents=len(parents),blocks=len(blocks),frames=len(frames));diag.reserve()
    seal(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,counts=dict(totals),GT_used=False));stage('all_evidence_and_reference_constructions_sealed',counts=dict(totals))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','generate'));a=p.parse_args();globals()[a.mode]()
