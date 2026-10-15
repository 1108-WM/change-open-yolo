#!/usr/bin/env python3
"""Fixed rejectable local block attribution, with separated whole/point readout."""
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
OUT=ROOT/'research/geometry/experiments/fi1_rejectable_local_block_exposed52_20261015_v1'
SOURCE=prior.OUT;read=diag.read;seal=diag.seal;sha=diag.sha

def stage(n,**kw):
    diag.write_json(OUT/'execution.json',dict(stage=n,time=time.time(),**kw));print(n,kw,flush=True)
def prepare():
    assert not OUT.exists();diag.reserve();old=read(SOURCE/'contract.json');inputs={str(SOURCE/'contract.json'):sha(SOURCE/'contract.json'),str(SOURCE/'relations_frozen.json'):sha(SOURCE/'relations_frozen.json')};inputs.update(old['inputs']);inputs.update(read(SOURCE/'relations_frozen.json')['outputs'])
    files=[Path(__file__).resolve(),Path(__file__).with_name('fi1_rejectable_local_block_core.py'),ROOT/'research/geometry/code/tests/test_fi1_rejectable_local_block.py']
    OUT.mkdir(parents=True)
    c=dict(version=OUT.name,scenes=old['scenes'],cohorts=old['cohorts'],jobs=old['jobs'],GT_inputs=old['GT_inputs'],class_ids=old['class_ids'],inputs=inputs,code={str(p):sha(p) for p in files},scope='All previously exposed52 physical scenes/207 sampled T parents; developmental diagnostic only. No final holdout or independent validation claim.',
      target='Keep every original N/T. Append at most one candidate per original T and per readout. No N edit, T deletion, replacement, new region or new query generation.',
      query='Original sealed candidate query set; choose maximum block mean, ascending query ID ties; mean>=.5 defines support, not GT identity. No parent overlap/coverage or competitor-query veto.',
      attribution_reference='For each whole editable block choose original parent membership region with minimum point-to-point distance to block, region ID breaks ties. Full original region is reference, never anchor-filtered, never edited/expanded. All alternatives and distances recorded. Query cannot expand reference.',
      observation='Only original selected <=6 frames/depth reliability/automatic masks. On each visible block and reference choose unique best mask; both coverage>=.5 and same ID common; differing ID conflict; tied maxima ambiguous. Empty observations missing; observed zero/weak separate. Conflict is observed mask conflict, not proof of different objects.',
      rejectable='Query support mean<.5 -> abstain; any observed conflict -> reject; otherwise >=2 common frames -> accept; otherwise abstain. Fixed certificate hypothesis using existing .5 support/two-view convention; NOT AUC-chosen threshold or previously established safe rule. No threshold scan or post-GT modification.',
      point_selection='Both readouts share exactly the same attribution decision. W=entire original editable whole block. C=only selected supporter query probability>=.5 points inside THAT block; never other regions. Per-parent union all accepted blocks for each readout, no GT block picking. Empty additions no new output. Exact pointset dedup against original NT and within arms at downstream conversion.',
      evaluation='After all attribution, frame evidence and W/C unions frozen: stable-parent target/other/unknown, reject missed target, accept contamination, good T harm, same-parent completeNT and pool. All parents retained, unstable reported separately. W/C fixed outputs get own semantics and frozen downstream NT/context/append AP even if geometric pool gain zero; if all outputs empty use exact NT equivalence instead.',
      acceptance=dict(min_strict_same_parent_gain02_scenes=2,strict_added_target_fraction=.9,strict_added_known_fraction=.9,overall_known_target_precision_min=.9,overall_unknown_fraction_max=.1,good_parent_iou=.9,good_loss_gt02_allowed=0,AP='semantic append>NT AND append>corresponding context; CA and original TP exits/gains/losses disclosed. Developmental support only, not independent success.'),
      stop='If key opportunity blocks fail attribution or accepted neighbors/unknown violate safety and no safe value, do not tune filters or expand training. Diagnose observation/reference/region-expression limits. No universal impossibility claim for all query combinations, trees, observations or classifiers.',
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
                    decision=core.attribution(witness,mean);accept=decision.startswith('accept');whole,clip=core.readout(idx,prob,q,accept);block=halo[idx];clipproposal=halo[idx[prob[idx,q]>=.5]];arrays[f'{cid}_r{rid:04d}_block']=block;arrays[f'{cid}_r{rid:04d}_clipped_proposal']=clipproposal
                    row=dict(scene=s,cohort=j['cohort'],candidate_id=cid,region=rid,block_id=f'{cid}_r{rid:04d}',points=len(idx),supporter=q,supporter_mean=mean,anchor=anchor,alternate_supporter=q!=anchor,query_parent_mean=float(prob[member,q].mean()),query_parent_hard_fraction=float((prob[member,q]>=.5).mean()),anchor_block_mean=float(prob[idx,anchor].mean()),reference_region=refid,reference_points=len(reference),minimum_distance=distance,reference_alternatives=[dict(region=rr,distance=d,points=len(ref)) for d,rr,ref in distances],state=decision,frame_states=dict(Counter(r['state'] for r in witness)),whole_points_if_accepted=len(idx),clipped_points_if_accepted=len(clipproposal),accepted=accept)
                    blocks.append(row);totals[decision]+=1
                    if accept:
                        selected['W'].append(halo[whole]);selected['C'].append(halo[clip]);accepted.append(rid)
                for arm in ('W','C'):
                    add=np.unique(np.concatenate(selected[arm])) if selected[arm] else np.empty(0,np.int64);points=np.union1d(parent,add);assert not np.intersect1d(parent,add).size
                    arrays[cid+'_'+arm+'_added']=add;arrays[cid+'_'+arm+'_points']=points;unions.append(dict(scene=s,cohort=j['cohort'],candidate_id=cid,parent_geometry=p['parent_geometry'],arm=arm,added=len(add),points=len(points),extra_candidate=bool(len(add)),accepted_regions=accepted,points_sha256=diag.hashlib.sha256(points.astype('<i8').tobytes()).hexdigest()));totals[arm+'_nonempty']+=bool(len(add));totals[arm+'_added']+=len(add)
        d=OUT/'predictions'/s;d.mkdir(parents=True)
        for n,x in [('blocks',blocks),('frames',frames),('unions',unions)]:seal(d/(n+'.json'),x)
        np.savez_compressed(d/'data.npz',**arrays);seal(d/'receipt.json',dict(complete=True,inputs=bankinputs,GT_used=False,outputs={str(p):sha(p) for p in d.iterdir()}));outputs.update({str(p):sha(p) for p in d.iterdir()});totals.update(parents=len(parents),blocks=len(blocks),frames=len(frames));diag.reserve()
    seal(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,counts=dict(totals),GT_used=False));stage('all_local_predictions_sealed',counts=dict(totals))

def evaluate():
    c=read(OUT/'contract.json');diag.verify(read(OUT/'predictions_frozen.json')['outputs']);diag.verify(c['GT_inputs']);blocks=[];rows=[];pool=[];strict=[];outputs={}
    for si,(s,j) in enumerate(c['jobs'].items()):
        stage('postseal_attribution_GT',scene=s,completed=si,total=52);gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=met.valid_targets(gt,c['class_ids']);_,nt=diag.load_nt(j);baseline=met.pool_base(nt,gt,ids,sizes);asset=Path(j['asset_root']);parents=read(SOURCE/'relations'/s/'parents.json');metas={};best={a:baseline.copy() for a in ('W','C')};scene=[];br=[]
        with np.load(OUT/'predictions'/s/'data.npz') as z,np.load(asset/'structure'/s/'data.npz') as st:
            for p in parents:metas[p['candidate_id']]=met.identity(st[p['candidate_id']+'_parent'],gt,ids,sizes)
            for r in read(OUT/'predictions'/s/'blocks.json'):
                cid=r['candidate_id'];im=metas[cid];target=im['target'] if im['stable'] else None;b=z[r['block_id']+'_block'];cl=z[r['block_id']+'_clipped_proposal'];row=dict(r,stable=im['stable'],target=target,parent_scope=im['scope'])
                from evaluate_fi1_query_relation_diagnostic import counts
                row['block_GT']=counts(b,gt,target);row['clipped_GT']=counts(cl,gt,target);br.append(row)
            for u in read(OUT/'predictions'/s/'unions.json'):
                cid=u['candidate_id'];a=u['arm'];parent=st[cid+'_parent'];points=z[cid+'_'+a+'_points'];r,v=met.outcome(parent,points,gt,ids,sizes,metas[cid],baseline);r.update(u);scene.append(r)
                if r['stable']:
                    idx=metas[cid]['index'];best[a][idx]=max(best[a][idx],v[idx]);strict.append(dict(scene=s,cohort=j['cohort'],candidate_id=cid,arm=a,target=r['target'],extra_candidate=u['extra_candidate'],gain_over_NT=r['pool_delta'],strict=bool(r['pool_delta']>=.02-1e-12 and r['added'] and r['added_target']/r['added']>=.9 and (r['added_target']+r['added_other'])/r['added']>=.9)))
        d=OUT/'evaluation'/s;d.mkdir(parents=True)
        for name,val in [('blocks',br),('unions',scene)]:seal(d/(name+'.json'),val);outputs[str(d/(name+'.json'))]=sha(d/(name+'.json'))
        blocks+=br;rows+=scene
        for a in best:pool+=met.pool_rows(s,a,baseline,best[a],ids)
    d=OUT/'evaluation';seal(d/'pool.json',pool);seal(d/'strict.json',strict)
    summary=dict(blocks=len(blocks),state_counts=dict(Counter(r['state'] for r in blocks)),accepted_blocks=sum(r['accepted'] for r in blocks),stable_accepted_blocks=sum(r['accepted'] and r['stable'] for r in blocks),unknown_accepted_blocks=sum(r['accepted'] and not r['stable'] for r in blocks),arms={a:dict(unions=met.summarize_rows([r for r in rows if r['arm']==a]),pool=met.pool_summary([r for r in pool if r['mode']==a]),strict_same_parent_gain02_scenes=len({r['scene'] for r in strict if r['arm']==a and r['strict']})) for a in ('W','C')})
    seal(OUT/'evaluation_summary.json',summary)
    for p in [d/'pool.json',d/'strict.json',OUT/'evaluation_summary.json']:outputs[str(p)]=sha(p)
    seal(OUT/'evaluation_frozen.json',dict(complete=True,outputs=outputs,GT_after_all_predictions=True));stage('local_GT_diagnosis_completed',summary=summary)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','generate','evaluate'));a=p.parse_args();globals()[a.mode]()
