#!/usr/bin/env python3
"""Fixed five-arm own-semantic downstream test of sealed T-add unions.
No geometry generation, fitting, threshold search or historical OUT writes.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import argparse, json, sys, hashlib, shutil, time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_witnessed_completion_sl as ref
from tools.fi1_frontend_protocol import freeze_json, write_json, sha256_file as sha
OUT=ROOT/'research/geometry/experiments/fi1_t_add_union_downstream_internal24_20261014_v1'
GEOM=ROOT/'research/geometry/experiments/fi1_t_add_union_internal24_20261014_v1'
ARMS=('NT','A_context','A_append','B_context','B_append');VIEWS=ref.VIEWS
read=ref.read;sl=ref.sl;src=ref.src;ctl=ref.ctl
ref.OUT=OUT;ref.geom=SimpleNamespace(OUT=OUT)
def stage(n,**kw):
    write_json(OUT/'execution.json',dict(stage=n,updated=time.time(),**kw));print(n,kw,flush=True)
ref.stage=stage

def prepare():
    assert not (OUT/'contract.json').exists();assert shutil.disk_usage(ROOT).free>=50*2**30
    old=read(GEOM/'contract.json');seal=read(GEOM/'predictions_frozen.json');assert seal['complete']
    ctl.base.verify_files(seal['outputs'])
    lc=read(ref.L/'contract.json');sc=read(ref.S/'contract.json')
    planpath=Path(old['jobs'][old['scenes'][0]]['binding']['evidence']).parents[1]/'plan.json';plan=read(planpath);model=plan['model']
    inputs={str(GEOM/'contract.json'):sha(GEOM/'contract.json'),str(GEOM/'predictions_frozen.json'):sha(GEOM/'predictions_frozen.json'),str(planpath):sha(planpath),**seal['outputs']}
    inputs.update(old['frozen_SL_models']);inputs[str(ref.L/'contract.json')]=sha(ref.L/'contract.json')
    inputs[sc['B_model']]=sc['B_model_sha256']
    for k in ('alpha_clip_base','alpha_clip_checkpoint','sam_checkpoint'):inputs[model[k]]=model[k+'_sha256']
    config=ROOT/'pretrained/config_scannet200.yaml';inputs[str(config)]=sha(config)
    OUT.mkdir(parents=True,exist_ok=True);text=Path(plan['text_features']);assert sha(text)==model['text_features_sha256'];shutil.copyfile(text,OUT/'text_features.npy');inputs[str(OUT/'text_features.npy')]=sha(OUT/'text_features.npy')
    jobs={};foldaudit={};conversions=[];counts=Counter();geometry_outputs={}
    for s in old['scenes']:
        j=old['jobs'][s];b=j['binding'];inputs.update({b[k]:sha(b[k]) for k in ('geometry','points','evidence')})
        v=Path(b['geometry']).parents[2]/'views'/s/'receipt.json';vr=read(v);inputs[str(v)]=sha(v)
        caches=[p for p in vr['input_files'] if p.endswith('/'+s+'.pt') and 'yoloworld_bboxes_2d' in p];assert len(caches)==1;inputs[caches[0]]=vr['input_files'][caches[0]]
        f=lc['folds'][s];recpath=ref.L/'models'/('fold'+str(f))/'receipt.json';rec=read(recpath);inputs[str(recpath)]=sha(recpath)
        assert s in rec['test_scenes'] and s not in rec['train_scenes'];assert not {x.split('_')[0] for x in rec['train_scenes']} & {s.split('_')[0]}
        foldaudit[s]=dict(fold=f,L_train_excludes_physical_scene=True,train_scenes=rec['train_scenes'],test_scenes=rec['test_scenes'],upstream_B26_exposed=True)
        jobs[s]=dict(binding=b,GT=j['GT'],yolo_cache=caches[0],old_view_receipt=str(v),prepared_root=str(Path(j['paths']['processed']).parents[1]))
        for p in (ref.S/'predictions'/s/'pool.npz',ref.S/'predictions'/s/'members.jsonl',ref.SL/'predictions'/s/'semantic_SL.json',ref.SL/'predictions'/s/'class_agnostic_SL.json'):inputs[str(p)]=sha(p)
        gs,pts,_=ctl.base.load_geometry(b);nt=src.source_pool(gs);by={g['points_sha256']:g['geometry_key'] for g in nt};fresh={};arrays={};rows=[]
        with np.load(GEOM/'predictions'/s/'data.npz') as z:
            for u in read(GEOM/'predictions'/s/'unions.json'):
                if not u['extra_candidate']:counts[u['arm']+'_empty']+=1;continue
                p=z[u['candidate_id']+'_'+u['arm']+'_points'].astype(np.int64);h=hashlib.sha256(p.astype('<i8').tobytes()).hexdigest();assert h==u['points_sha256']
                counts[u['arm']+'_nonempty']+=1
                original=h in by;duplicate=h in fresh
                k=by.get(h,s+':geometry:'+h)
                conversions.append(dict(scene=s,arm=u['arm'],candidate_id=u['candidate_id'],parent=u['parent_geometry'],geometry_key=k,points_sha256=h,original_NT_duplicate=original,earlier_new_duplicate=duplicate))
                if original:counts[u['arm']+'_NT_duplicate']+=1;continue
                if duplicate:fresh[h]['arms']=sorted(set(fresh[h]['arms']+[u['arm']]));counts[u['arm']+'_fresh_duplicate']+=1;continue
                g=dict(geometry_key=k,points_sha256=h,point_count=len(p),array_key='g'+str(len(rows)),arms=[u['arm']],legacy_members=[],dv3_plans=[dict(append_only=True,plan_key=s+':t_add:'+h,frozen_class_index=-1)])
                rows.append(g);fresh[h]=g;arrays[g['array_key']]=p
        d=OUT/'geometry'/s;d.mkdir(parents=True);freeze_json(d/'geometry.json',rows);np.savez_compressed(d/'points.npz',**arrays)
        for p in d.iterdir():geometry_outputs[str(p)]=sha(p)
        counts['unique_own_masks']+=len(rows)
    files=[Path(__file__).resolve(),Path(ref.__file__).resolve(),ROOT/'tools/fi1_sl_inference.py',ROOT/'tools/fi1_r2_core.py',ROOT/'tools/fi1_r3_core.py',ROOT/'tools/fi1_s_semantic_rank_residual.py',ROOT/'tools/complete_fi1_r0_r1_alpha.py',ROOT/'tools/bind_fi1_r0_r1_alpha_cache.py']
    files += [ref.SNAP/'tools'/n for n in ('export_fi1_refined_semantic_evidence.py','fi1_full_alpha_adapter.py','fi1_unified_evidence_core.py','build_dm_sms1_alpha_embedding_ledger.py')]
    code={str(p):sha(p) for p in files};ctl.base.verify_files(inputs)
    c=dict(version=OUT.name,scenes=old['scenes'],folds={s:lc['folds'][s] for s in old['scenes']},groups={'internal24':old['scenes']},class_ids=old['class_ids'],thresholds=[.25,.5,.55,.6,.65,.7,.75,.8,.85,.9],jobs=jobs,GT_inputs=old['GT_inputs'],inputs=inputs,code=code,model=model,prompts=plan['prompts'],config=str(config),b26=sc['B_model'],budget=800,arms=ARMS,
      baseline_tolerance=0.,full_score_check_tolerance=1e-7,
      own_semantics='Fresh own-mask YOLO40/Alpha20 SAMViTB scales0/.2/.4; no copied parent category/score, no top1/SMS filtering',source_adapter='Fixed existing refined_union compatibility one-hot; sentinel class=-1 alias; explicit SL input-distribution extension',
      policy='Original exact NT; extended full-pool context per A/B, context excludes only fresh masks before S800; append includes fresh. Frozen L on selected S800; CA actual selected geometry max no refill. Exact duplicates no extra vote.',
      scope=old['dataset_usage'],new_training=False,geometry_regenerated=False,thresholds_changed=False,production_modified=False,github_push=False)
    freeze_json(OUT/'contract.json',c);freeze_json(OUT/'fold_audit.json',foldaudit);freeze_json(OUT/'conversion.json',dict(rows=conversions,counts=dict(counts)))
    freeze_json(OUT/'geometry_frozen.json',dict(complete=True,outputs=geometry_outputs,GT_used=False))
    for p in files:
        q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    stage('prepared',counts=dict(counts))

def acquire(c):ref.acquire(c)

def expanded(c,s,arm=None):
    gs,pts,ev=ctl.base.load_geometry(c['jobs'][s]['binding']);nt=src.source_pool(gs);evidence={g['geometry_key']:ev[g['geometry_key']] for g in nt}
    own={r['geometry_key']:r for r in read(OUT/'evidence'/s/'evidence.json')}
    with np.load(OUT/'geometry'/s/'points.npz') as z:
        for g in read(OUT/'geometry'/s/'geometry.json'):
            if arm is not None and arm not in g['arms']:continue
            k=g['geometry_key'];assert k not in evidence;nt.append(g);pts[k]=z[g['array_key']].copy();evidence[k]=own[k]
    return nt,pts,evidence

def predict(c):
    sys.addaudithook(ref.Guard(c).hook);ctl.base.verify_files(c['code']);ctl.base.verify_files(read(OUT/'geometry_frozen.json')['outputs']);ctl.base.verify_files(read(OUT/'evidence_frozen.json')['outputs'])
    import joblib,torch
    torch.set_num_threads(4);b26=joblib.load(c['b26']);models={f:sl.load_L(ref.L/'models'/('fold'+str(f))/'model.pt',ref.L/'models'/('fold'+str(f))/'normalization.npz') for f in set(c['folds'].values())}
    outputs={};counts=Counter();verification=[]
    for i,s in enumerate(c['scenes']):
        stage('five_arm_predictions',scene=s,completed=i,total=len(c['scenes']));d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True)
        gs,pts,ev=ctl.base.load_geometry(c['jobs'][s]['binding']);nt=src.source_pool(gs);mm,xx=ref.features(nt,pts,{g['geometry_key']:ev[g['geometry_key']] for g in nt},c['jobs'][s]['binding']['point_count']);ss=np.clip(b26.predict(xx),0,1)
        with np.load(ref.S/'predictions'/s/'pool.npz') as z:assert np.array_equal(xx,z['x26']) and np.array_equal(ss,z['scores'])
        assert mm==ctl.base.rows(ref.S/'predictions'/s/'members.jsonl')
        rr,cc,_,_=sl.rescore_pool(mm,xx,ss,*models[c['folds'][s]])
        assert rr==read(ref.SL/'predictions'/s/'semantic_SL.json') and cc==read(ref.SL/'predictions'/s/'class_agnostic_SL.json')
        results={'NT':(rr,cc)};newall=set()
        for a in ('A','B'):
            gs,pts,ev=expanded(c,s,a);new={g['geometry_key'] for g in gs if g.get('arms')};newall|=new
            em,ex=ref.features(gs,pts,ev,c['jobs'][s]['binding']['point_count']);es=np.clip(b26.predict(ex),0,1);x28=np.column_stack((ex,es,sl.core.base_logits(es))).astype(np.float32);lookup={ctl.shared.key(m):j for j,m in enumerate(em)}
            model,mean,std=models[c['folds'][s]]
            with torch.no_grad():
                full=np.concatenate([model(torch.from_numpy(((x28[start:start+8192]-mean)/std).astype(np.float32))).numpy() for start in range(0,len(x28),8192)])
            for kind in ('context','append'):
                allowed=[{**m,'score':float(es[j])} for j,m in enumerate(em) if kind=='append' or m['geometry_key'] not in new];sel=src.rank(allowed)[:800];ix=np.array([lookup[ctl.shared.key(m)] for m in sel]);sc,delta=sl.score_selected(model,x28,es,ix,mean,std)
                error=float(np.max(np.abs(delta-full[ix])));assert error<=c['full_score_check_tolerance']
                rows=src.rank([{**m,'score':float(v)} for m,v in zip(sel,sc)]);results[a+'_'+kind]=(rows,ctl.collapse_members(rows));verification.append(dict(scene=s,arm=a+'_'+kind,full_score_max_error=error));counts[a+'_'+kind+'_new_members']+=sum(r['geometry_key'] in new for r in rows)
            np.savez_compressed(d/(a+'_pool.npz'),x26=ex,scores=es,L_delta=full);freeze_json(d/(a+'_members.json'),em)
            for n in (a+'_pool.npz',a+'_members.json'):outputs[str(d/n)]=sha(d/n)
        for arm,(rows,ca) in results.items():
            assert len(rows)==800 and len({ctl.shared.key(m) for m in rows})==800
            for view,val in zip(VIEWS,(rows,ca)):
                p=d/(view+'_'+arm+'.json');freeze_json(p,val);outputs[str(p)]=sha(p)
        counts['fresh_masks']+=len(newall)
    freeze_json(OUT/'inference_verification.json',dict(baseline_members_features_scores_order_bitwise_exact=True,full_vs_padded=verification))
    freeze_json(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,counts=dict(counts),original_SL_bitwise_exact=True,GT_used=False));stage('predictions_sealed',counts=dict(counts))

def evaluate(c):
    sys.addaudithook(ref.Guard(c,True).hook);ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs']);ctl.base.verify_files(c['GT_inputs'])
    ev=ctl.base.evaluator();ca,valid,uid=ctl.class_agnostic_evaluator();matches={v:{a:{} for a in ARMS} for v in VIEWS};outputs={};ap={};arrays={}
    for i,s in enumerate(c['scenes']):
        stage('sealed_prediction_matching',scene=s,completed=i,total=len(c['scenes']));_,pts,_=expanded(c,s);gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);mapped=ctl.remap_gt(gt,valid,uid);oldload=ca.util_3d.load_ids
        d=OUT/'evaluation'/s;d.mkdir(parents=True,exist_ok=True)
        for arm in ARMS:
            for view,evaluator,truth in (('semantic',ev,gt),('class_agnostic',ca,mapped)):
                rows=read(OUT/'predictions'/s/(view+'_'+arm+'.json'))
                if view=='class_agnostic':ca.util_3d.load_ids=lambda p:mapped.copy()
                try:match=ctl.shared.assign(evaluator,rows,pts,c['jobs'][s]['GT'],len(gt),view=='class_agnostic')
                finally:ca.util_3d.load_ids=oldload
                ctl.shared.check_match(match,rows,pts,truth,evaluator,view=='class_agnostic');match=ctl.base.serializable_matches(match);p=d/(view+'_'+arm+'_matches.json');freeze_json(p,match);outputs[str(p)]=sha(p);matches[view][arm][s]=match
    freeze_json(OUT/'matches_frozen.json',dict(complete=True,outputs=outputs,independent_bidirectional_point_census=True))
    for view,evaluator in (('semantic',ev),('class_agnostic',ca)):
        ap[view]={}
        for arm in ARMS:
            x,*_=evaluator.evaluate_matches(matches[view][arm]);ap[view][arm]=ctl.base.metrics(evaluator,x);p=OUT/'evaluation'/(view+'_'+arm+'_AP.npy');np.save(p,x);arrays[str(p)]=sha(p);print('RESULT',view,arm,ap[view][arm],flush=True)
    freeze_json(OUT/'evaluation_summary.json',dict(complete=True,AP=ap,AP_arrays=arrays,aggregation='Official pooled internal24, not mean scene AP'));stage('AP_completed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','acquire','predict','evaluate'));a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):
        if a.mode=='prepare':prepare()
        else:globals()[a.mode](read(OUT/'contract.json'))
