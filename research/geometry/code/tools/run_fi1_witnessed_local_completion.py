#!/usr/bin/env python3
"""Isolated one-action geometry experiment; frozen SL score head remains unchanged."""
from pathlib import Path
from collections import Counter
import json,hashlib,sys,argparse,shutil,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from tools import diagnose_fi1_nt_geometry_gaps as diag
from tools.fi1_witnessed_local_completion import witnessed_addition,PARAMETERS
from tools.fi1_r3_core import pair_overlap,point_matrix
from tools.fi1_core_ablation import source_pool
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
OUT=ROOT/'.local/fi1_witnessed_local_completion_development200_20261010_v1'
read=diag.read


def stage(mode,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=mode,updated=time.time(),**kw));print(mode,kw,flush=True)


def prepare():
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>=50*1024**3
    p=read(diag.OUT/'contract.json');assert read(diag.OUT/'summary.json')['complete']
    inputs=dict(p['inputs']);inputs[str(diag.OUT/'summary.json')]=sha(diag.OUT/'summary.json')
    inputs[str(diag.OUT/'targets.jsonl')]=sha(diag.OUT/'targets.jsonl')
    code=dict(p['code'])
    for n in ('tools/fi1_witnessed_local_completion.py','tools/run_fi1_witnessed_local_completion.py','tests/test_fi1_witnessed_local_completion.py'):
        code[str(ROOT/n)]=sha(ROOT/n)
    c=dict(version=OUT.name,scenes=p['scenes'],folds=p['folds'],groups=p['groups'],class_ids=p['class_ids'],thresholds=p['thresholds'],jobs=p['jobs'],
        inputs=inputs,GT_inputs=p['GT_inputs'],code=code,parameters=PARAMETERS,
        motivation='Geometry extent diagnosis:60 GT-pickedNT pair opportunities vs2 conservative observation crop opportunities. Investigate bounded witnessed local completion, not another semantic head.',
        action='All unorderedNT pairs .05<overlap<.9, both directions; add only donor-minus-parent points jointly supported by>=2 independent pose groups. Retain alloriginalNT; no transitive growth, no parent score/class inheritance. Separation veto, physical point contact,<=100%parent additions.',
        selection='Per-parent unique greatest independent joint-group count; tied extents abstain. Exact identical output masks deduplicate;NTduplicates omitted. No GT/sourceclass/score used.',
        observations='Reuse sealed30frame physical masks, distinct maskaliases collapsed. Pose groups independent; parent-source views not excluded, so this is multiview corroboration, not novel evidence acquisition.',
        scientific_scope='Exposed development200, method chosen after geometry diagnostic; not blind whole-system validation. FreezeoriginalSLfolds/models/K800, own semantics fornewpointsets before complete-context scoring.',
        acceptance='Geometry stage reports新增 overcompleteNT and outputquality contamination. If no new>.50/.75/.90 coverage, retainfailedgeometricprototype and do notspend visual/AP inference. Otherwise acquireownsemantics thenappend/scoring/contextcontrols withfrozenSL, allpredictionssealed beforeAP.',
        model_training=False,SL_weights_modified=False,dev55_read=False,production_modified=False)
    diag.prior.src.ctl.base.verify_files(inputs);diag.prior.src.ctl.base.verify_files(code)
    OUT.mkdir();freeze_json(OUT/'contract.json',c)
    for q in code:
        d=OUT/'code_snapshot'/Path(q).relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,d)
    stage('prepared')


def generate(c):
    gc=dict(c);guard=diag.Guard(gc);sys.addaudithook(guard.hook)
    outputs={};total=Counter()
    for i,s in enumerate(c['scenes']):
        stage('GT_free_geometry',scene=s,completed=i,total=200)
        job=c['jobs'][s];gs,pts,_=diag.prior.src.ctl.base.load_geometry(job['binding']);nt=source_pool(gs);keys=[g['geometry_key'] for g in nt]
        fs,bound=diag.load_frames(c,s);assert bound==read(diag.OUT/'scenes'/s/'evidence.json')['inputs']
        raw=np.load(job['paths']['processed'],mmap_mode='r',allow_pickle=False);xyz=np.asarray(raw[:,:3]).copy();del raw
        assert xyz.shape==(job['binding']['point_count'],3) and np.isfinite(xyz).all()
        ov,_=pair_overlap([pts[k] for k in keys],len(xyz));pairs=list(zip(*np.nonzero(np.triu((ov>PARAMETERS['min_pair_iou'])&(ov<PARAMETERS['max_pair_iou']),k=1))))
        # Cheap raw-mask fraction prefilter; identical to final region conditions.
        masks=point_matrix([pts[k] for k in keys],len(xyz));prepared=[]
        for f in fs:
            mm=point_matrix(f['masks'],len(xyz));inter=(masks@mm.T).toarray();vis=np.array([f['visible'][pts[k]].sum() for k in keys])
            prepared.append(dict(frame=f,intersection=inter,visible=vis))
        proposals={};decisions=[];stats=Counter(pairs=len(pairs))
        for a,b in pairs:
            common=np.intersect1d(pts[keys[a]],pts[keys[b]],assume_unique=True)
            for parent,donor in ((a,b),(b,a)):
                extra=np.setdiff1d(pts[keys[donor]],pts[keys[parent]],assume_unique=True)
                if not len(extra) or len(extra)>len(pts[keys[parent]]):stats['early_identity_or_empty']+=1;continue
                possible=set()
                for pf in prepared:
                    f=pf['frame'];pa=int(pf['visible'][parent]);ex=int(pf['visible'][donor]-f['visible'][common].sum())
                    if pa<3 or ex<3 or pa/len(pts[keys[parent]])<.1 or ex/len(extra)<.1:continue
                    shared=np.array([len(np.intersect1d(common,m,assume_unique=True)) for m in f['masks']])
                    lc=pf['intersection'][parent]/pa;rc=(pf['intersection'][donor]-shared)/ex
                    if np.any(np.minimum(lc,rc)>=.5):possible.add(f['group'])
                if len(possible)<2:stats['early_no_joint']+=1;continue
                pp,rec=witnessed_addition(pts[keys[parent]],pts[keys[donor]],fs,xyz,c['parameters']);stats[rec['reason']]+=1
                record=dict(parent=keys[parent],donor=keys[donor],**rec);decisions.append(record)
                if rec['accepted']:
                    digest=hashlib.sha256(np.asarray(pp,dtype='<i8').tobytes()).hexdigest();gkey=s+':geometry:'+digest
                    if gkey in keys:stats['identical_NT']+=1;continue
                    proposals.setdefault(keys[parent],[]).append((len(rec['joint_groups']),gkey,pp,record))
        arrays={};rows=[]
        for parent,candidates in sorted(proposals.items()):
            best=max(x[0] for x in candidates);top={x[1]:x for x in candidates if x[0]==best}
            if len(top)!=1:stats['ambiguous_extent_abstain']+=1;continue
            _,gkey,pp,rec=next(iter(top.values()))
            if any(r['geometry_key']==gkey for r in rows):continue
            arr='g'+str(len(rows)).zfill(5);arrays[arr]=pp
            rows.append(dict(geometry_key=gkey,points_sha256=gkey.rsplit(':',1)[-1],point_count=len(pp),array_key=arr,
                action='witnessed_local_completion',parent=parent,donor=rec['donor'],joint_groups=rec['joint_groups'],added_points=rec['added_points']))
        d=OUT/'geometry'/s;d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'points.npz',**arrays)
        freeze_json(d/'geometry.json',rows);freeze_json(d/'decisions.json',decisions);stats['new_geometry']=len(rows);total.update(stats)
        freeze_json(d/'receipt.json',dict(complete=True,GT_used=False,inputs=bound,counts=dict(stats),outputs={str(d/n):sha(d/n) for n in ('points.npz','geometry.json','decisions.json')}))
        outputs.update(read(d/'receipt.json')['outputs'])
    freeze_json(OUT/'geometry_frozen.json',dict(complete=True,scenes=200,counts=dict(total),outputs=outputs,GT_used=False))
    stage('geometry_sealed',counts=dict(total))


def quality(c):
    assert read(OUT/'geometry_frozen.json')['complete'];gc=dict(c);guard=diag.Guard(gc);guard.labels=True;sys.addaudithook(guard.hook)
    groups={name:{format(t,'.2f'):Counter() for t in c['thresholds']} for name in c['groups']};per_scene={}
    oldrows=[json.loads(x) for x in (diag.OUT/'targets.jsonl').read_text().splitlines()];allwitness=[]
    from tools.fi1_r3_core import gt_iou
    for s in c['scenes']:
        rows=read(OUT/'geometry'/s/'geometry.json');gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64)
        with np.load(OUT/'geometry'/s/'points.npz') as z:points=[z[r['array_key']] for r in rows]
        io,ids=gt_iou(points,gt,c['class_ids']);by={r['gt_id']:r for r in oldrows if r['scene']==s};before=np.array([by[int(g)]['best_iou'] for g in ids]);best=io.max(axis=0,initial=0)
        st={}
        for t in c['thresholds']:
            tag=format(t,'.2f');gain=(before<=t)&(best>t);stat=dict(new_over_complete_NT=int(gain.sum()),already_NT_covered=int(np.sum((before>t)&(best>t))),
                qualified_new_geometries=int(np.sum(io.max(axis=1,initial=0)>t)),nonqualified_new_geometries=int(np.sum(io.max(axis=1,initial=0)<=t)))
            st[tag]=stat
            for name,ss in c['groups'].items():
                if s in ss:groups[name][tag].update(stat)
            for j in np.flatnonzero(gain):
                k=int(io[:,j].argmax());allwitness.append(dict(scene=s,gt_id=int(ids[j]),threshold=t,new_geometry=rows[k],new_iou=float(io[k,j]),NT_best=float(before[j])))
        per_scene[s]=st
    freeze_json(OUT/'geometry_quality.json',dict(complete=True,groups={n:{t:dict(v) for t,v in g.items()} for n,g in groups.items()},scenes=per_scene,new_cover_witnesses=allwitness,
        original_NT_retained=True,geometry_pool_lost_targets=0,raw_GT_diagnostic_only=True,new_AP=False))
    stage('geometry_quality_completed',development=groups['development200'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','generate','quality'));args=p.parse_args()
    with threadpool_limits(4):
        if args.mode=='prepare':prepare()
        elif args.mode=='generate':generate(read(OUT/'contract.json'))
        else:quality(read(OUT/'contract.json'))
