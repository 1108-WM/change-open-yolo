#!/usr/bin/env python3
"""Own visual evidence and frozen OOF SL controls for one sealed geometry action."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import argparse,json,sys,shutil,time,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_witnessed_local_completion as geom
from tools import fi1_sl_inference as sl
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
src=sl.source;ctl=src.ctl;read=geom.read
OUT=ROOT/'.local/fi1_witnessed_completion_SL_development200_20261010_v1'
S=ROOT/'.local/fi1_native_symmetric_eligibility_development200_20261009_v1'
L=ROOT/'.local/fi1_s_semantic_rank_residual_development200_20261010_v2'
SL=ROOT/'.local/fi1_s_fixed_L_rescore_development200_20261010_v1'
SNAP=ROOT/'full_bundle/code_sources/evidence_snapshot'
ARMS=('SL','context_NT','append');VIEWS=('semantic','class_agnostic')

def stage(n,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=n,updated=time.time(),**kw));print(n,kw,flush=True)

class Guard:
    def __init__(self,c,labels=False):self.c=c;self.labels=labels
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(Path(args[0]).resolve())
        if any(t in p for t in ('fi1_sl_single200_dev55','reserve100','val312')):raise PermissionError('excluded evaluation dataset/model')
        if any(t.startswith('ground_truth') for t in Path(p).parts):
            if not self.labels or p not in self.c['GT_inputs'] or not (OUT/'predictions_frozen.json').exists():raise PermissionError('GT before predictions seal/outside200')
        if not self.labels and (p.endswith('overlap.npz') or p.endswith('target_census.json') or 'geometry_quality.json' in p or 'targets.jsonl' in p):raise PermissionError('diagnostic labels forbidden in inference')
        import re
        if any(s not in self.c['scenes'] for s in re.findall(r'scene\d{4}_\d{2}',p)):raise PermissionError('scene outside development200')

def prepare():
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>=50*1024**3
    c=read(geom.OUT/'contract.json');sealed=read(geom.OUT/'geometry_frozen.json');assert sealed['complete']
    # Read only immutable text model vectors from legacy location, no legacy scene/results.
    planpath=Path(c['jobs'][c['scenes'][0]]['binding']['evidence']).parents[1]/'plan.json'
    plan=read(planpath);model=plan['model'];text=Path(plan['text_features']);assert sha(text)==model['text_features_sha256']
    inputs=dict(sealed['outputs']);inputs[str(geom.OUT/'geometry_frozen.json')]=sha(geom.OUT/'geometry_frozen.json')
    inputs[str(planpath)]=sha(planpath)
    b=read(S/'contract.json');inputs[b['B_model']]=b['B_model_sha256']
    for k in ('alpha_clip_base','alpha_clip_checkpoint','sam_checkpoint'):inputs[model[k]]=model[k+'_sha256']
    config=ROOT/'pretrained/config_scannet200.yaml';inputs[str(config)]=sha(config)
    jobs={}
    for s in c['scenes']:
        j=c['jobs'][s];bb=j['binding'];inputs.update({bb[n]:c['inputs'][bb[n]] for n in ('geometry','points','evidence')})
        v=Path(bb['geometry']).parents[2]/'views'/s/'receipt.json';vr=read(v);inputs[str(v)]=sha(v)
        caches=[p for p in vr['input_files'] if p.endswith('/'+s+'.pt') and 'yoloworld_bboxes_2d' in p];assert len(caches)==1
        inputs[caches[0]]=vr['input_files'][caches[0]]
        jobs[s]=dict(binding=bb,GT=j['GT'],yolo_cache=caches[0],old_view_receipt=str(v),prepared_root=str(Path(j['paths']['processed']).parents[1]))
        for p in (L/'models'/f"fold{c['folds'][s]}"/'model.pt',L/'models'/f"fold{c['folds'][s]}"/'normalization.npz',S/'predictions'/s/'pool.npz',S/'predictions'/s/'members.jsonl',SL/'predictions'/s/'semantic_SL.json',SL/'predictions'/s/'class_agnostic_SL.json'):inputs[str(p)]=sha(p)
    codefiles=[Path(__file__).resolve(),ROOT/'tools/fi1_sl_inference.py',ROOT/'tools/fi1_r2_core.py',ROOT/'tools/fi1_r3_core.py',ROOT/'tools/fi1_s_semantic_rank_residual.py',ROOT/'tools/complete_fi1_r0_r1_alpha.py',ROOT/'tools/bind_fi1_r0_r1_alpha_cache.py']
    codefiles += [SNAP/'tools'/n for n in ('export_fi1_refined_semantic_evidence.py','fi1_full_alpha_adapter.py','fi1_unified_evidence_core.py','build_dm_sms1_alpha_embedding_ledger.py')]
    code={str(p):sha(p) for p in codefiles};ctl.base.verify_files(inputs)
    OUT.mkdir();shutil.copyfile(text,OUT/'text_features.npy');inputs[str(OUT/'text_features.npy')]=sha(OUT/'text_features.npy')
    contract=dict(version=OUT.name,scenes=c['scenes'],folds=c['folds'],groups=c['groups'],class_ids=c['class_ids'],thresholds=c['thresholds'],jobs=jobs,GT_inputs=c['GT_inputs'],inputs=inputs,code=code,model=model,prompts=plan['prompts'],config=str(config),b26=b['B_model'],budget=800,arms=ARMS,
        own_semantics='Fresh own-mask YOLO votes40 views + Alpha20views SAMViTB scales0/.2/.4. No parent category/score inheritance, no top1/SMS filtering.',
        source_adapter='New witnessed local completion uses refined_union one-hot as existing26D compatibility slot; synthetic alias has class=-1, no foreground inherited class. This is an explicit input-distribution extension, not previous NT SL validation.',
        policy='SL exact original fullNT; context_NT recomputes26D over extended symmetric pool but S chooses800 amongNT; append same extended scores select800 including new. Frozen corresponding heldoutL rescores selected800, complete-pool context retained. Exactpointset max from actual selected members for CA, no refill.',
        text_asset=dict(original_path=str(text),sha256=model['text_features_sha256'],scope='Immutable198 text embeddings only; copied here before guard. No val312 scene or evaluation result read.'),
        new_training=False,dev55_read=False,production_modified=False,parameters_changed_after_geometry_quality=False)
    freeze_json(OUT/'contract.json',contract)
    for p in codefiles:
        q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    stage('prepared')

def acquire(c):
    sys.addaudithook(Guard(c).hook);ctl.base.verify_files(c['code'])
    import tools;tools.__path__.append(str(SNAP/'tools'))
    import torch,yaml
    from tools.export_fi1_refined_semantic_evidence import load_scene_context
    from tools.fi1_unified_evidence_core import baseline_maps,yolo_evidence
    from tools.fi1_full_alpha_adapter import make_views
    from tools.build_dm_sms1_alpha_embedding_ledger import _load_alpha_clip,_load_sam,_process_scene
    from tools.complete_fi1_r0_r1_alpha import manifest_row
    from tools.bind_fi1_r0_r1_alpha_cache import recover_alpha,jsonl
    assert torch.cuda.is_available();torch.set_num_threads(1)
    a=SimpleNamespace(**{k:Path(v) if k.endswith(('_source','_base','_checkpoint')) else v for k,v in c['model'].items()})
    a.output_root=OUT/'fresh_cache';a.sam_batch_size=8;a.alpha_batch_size=32;a.sms_threshold=0.
    config=yaml.safe_load(Path(c['config']).read_text());labels=config['network2d']['text_prompts'];assert len(labels)==198
    state=_load_alpha_clip(a,c['prompts']);predictor=_load_sam(a);tf=np.load(OUT/'text_features.npy');diff=float(np.max(np.abs(tf-state['text_features'])));assert diff<2e-5
    freeze_json(OUT/'visual_runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),text_max_abs_difference=diff,new_training=False))
    outputs={};counts=Counter()
    for i,s in enumerate(c['scenes']):
        stage('own_visual_evidence',scene=s,completed=i,total=200);d=OUT/'evidence'/s
        if (d/'receipt.json').exists():
            r=read(d/'receipt.json');ctl.base.verify_files(r['outputs']);outputs.update(r['outputs']);counts.update(r['counts']);continue
        rows=read(geom.OUT/'geometry'/s/'geometry.json');records=[];files={}
        if rows:
            j=c['jobs'][s];args=SimpleNamespace(prepared_root=Path(j['prepared_root']),yolo_cache_template=j['yolo_cache'])
            context=load_scene_context(s,args,config,labels);assert context['visibility'].shape[1]==j['binding']['point_count']
            old=read(j['old_view_receipt'])['input_files'];files=context['projection_inputs'];assert all(old.get(p)==h for p,h in files.items())
            assert context['cache_hash']==c['inputs'][j['yolo_cache']];files={**files,j['yolo_cache']:context['cache_hash']};maps=baseline_maps(context,198)
            with np.load(geom.OUT/'geometry'/s/'points.npz') as z:
                for g in rows:
                    p=z[g['array_key']];assert hashlib.sha256(p.astype('<i8').tobytes()).hexdigest()==g['points_sha256']
                    yr=yolo_evidence(p,context,maps,198,40);v=make_views(p,context)
                    records.append(dict(scene_name=s,geometry_key=g['geometry_key'],points_sha256=g['points_sha256'],point_count=len(p),yolo=dict(acquisition_complete=True,available=yr['label_map']['available'],votes=yr['label_map']['counts'],missing_reason=yr['label_map']['missing_reason']),yolo_raw=yr,alpha=dict(acquisition_complete=False,view_requests=v),ground_truth_read=False,ap_computed=False))
                    for view in v:files[view['image_path']]=view['image_sha256']
            del context,maps;torch.cuda.empty_cache()
            requests=[manifest_row(r) for r in records];token=hashlib.sha256(json.dumps(requests,sort_keys=True).encode()).hexdigest()
            _process_scene(s,requests,a,state,predictor,tf,token);cache=a.output_root/'scenes'/s
            scales=np.load(cache/'scale_features.npy');ag=np.load(cache/'aggregate_features.npy');sim=np.load(cache/'similarities.npy')
            by={r['geometry_key']:r for r in jsonl(cache/'records.jsonl')}
            for r in records:r['alpha']={**recover_alpha(by[r['geometry_key']],scales,ag,sim,tf), 'fresh_visual_inference':True,'sms_filter_applied':False,'alpha_top1_applied':False}
            files.update({str(cache/n):sha(cache/n) for n in ('records.jsonl','scale_features.npy','aggregate_features.npy','similarities.npy')})
        d.mkdir(parents=True,exist_ok=False);freeze_json(d/'evidence.json',records);stat=dict(geometries=len(records),alpha_available=sum(r['alpha']['available'] for r in records),yolo_available=sum(r['yolo']['available'] for r in records))
        r=dict(complete=True,inputs=files,outputs={str(d/'evidence.json'):sha(d/'evidence.json')},counts=stat,GT_used=False);freeze_json(d/'receipt.json',r);outputs.update(r['outputs']);counts.update(stat)
        assert shutil.disk_usage(ROOT).free>=50*1024**3
    freeze_json(OUT/'evidence_frozen.json',dict(complete=True,outputs=outputs,counts=dict(counts),GT_used=False));stage('evidence_sealed',counts=dict(counts))

def expanded(c,s):
    j=c['jobs'][s];gs,pts,ev=ctl.base.load_geometry(j['binding']);nt=src.source_pool(gs);rows=read(geom.OUT/'geometry'/s/'geometry.json')
    evidence={g['geometry_key']:ev[g['geometry_key']] for g in nt}
    with np.load(geom.OUT/'geometry'/s/'points.npz') as z:
        for n,g in enumerate(rows):
            k=g['geometry_key'];assert k not in evidence
            nt.append({**g,'legacy_members':[],'dv3_plans':[dict(append_only=True,plan_key=s+':witnessed_completion:'+str(n),frozen_class_index=-1)]});pts[k]=z[g['array_key']].copy()
    evidence.update({r['geometry_key']:r for r in read(OUT/'evidence'/s/'evidence.json')});return nt,pts,evidence

def features(gs,pts,ev,N):
    members=src.symmetric_members(gs,ev);keys=[g['geometry_key'] for g in gs];own={g['geometry_key']:src.geometry_features(g,ev[g['geometry_key']]) for g in gs}
    io,co=src.pair_overlap([pts[k] for k in keys],N)
    x=np.column_stack((np.asarray([own[m['geometry_key']][m['class_index']] for m in members]),src.context_features(members,keys,io,co)));assert np.isfinite(x).all();return members,x

def predict(c):
    sys.addaudithook(Guard(c).hook);ctl.base.verify_files(c['code']);ctl.base.verify_files(read(OUT/'evidence_frozen.json')['outputs'])
    import joblib,torch
    torch.set_num_threads(4);b26=joblib.load(c['b26']);models={f:sl.load_L(L/'models'/f'fold{f}'/'model.pt',L/'models'/f'fold{f}'/'normalization.npz') for f in range(5)}
    outputs={};counts=Counter()
    for i,s in enumerate(c['scenes']):
        stage('frozen_SL_inference',scene=s,completed=i,total=200);d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True)
        gs,pts,ev=expanded(c,s);new={g['geometry_key'] for g in read(geom.OUT/'geometry'/s/'geometry.json')};nt=[g for g in gs if g['geometry_key'] not in new];N=c['jobs'][s]['binding']['point_count']
        mm,xx=features(nt,pts,{g['geometry_key']:ev[g['geometry_key']] for g in nt},N);ss=np.clip(b26.predict(xx),0,1)
        with np.load(S/'predictions'/s/'pool.npz') as z:assert np.array_equal(xx,z['x26']) and np.array_equal(ss,z['scores'])
        assert mm==ctl.base.rows(S/'predictions'/s/'members.jsonl')
        ref,ca,_,_=sl.rescore_pool(mm,xx,ss,*models[c['folds'][s]])
        assert ref==read(SL/'predictions'/s/'semantic_SL.json') and ca==read(SL/'predictions'/s/'class_agnostic_SL.json')
        em,ex=features(gs,pts,ev,N);es=np.clip(b26.predict(ex),0,1);x28=np.column_stack((ex,es,sl.core.base_logits(es))).astype(np.float32)
        lookup={ctl.shared.key(m):j for j,m in enumerate(em)}
        results={'SL':(ref,ca)}
        for arm in ('context_NT','append'):
            allowed=[{**m,'score':float(es[j])} for j,m in enumerate(em) if arm=='append' or m['geometry_key'] not in new]
            selected=src.rank(allowed)[:800];ix=np.array([lookup[ctl.shared.key(m)] for m in selected]);sc,delta=sl.score_selected(*[models[c['folds'][s]][0],x28,es,ix,models[c['folds'][s]][1],models[c['folds'][s]][2]])
            rr=src.rank([{**m,'score':float(v)} for m,v in zip(selected,sc)]);results[arm]=(rr,ctl.collapse_members(rr));counts[arm+'_new_members']+=sum(r['geometry_key'] in new for r in rr)
        for arm,(rr,cc) in results.items():
            assert len(rr)==800 and len({ctl.shared.key(m) for m in rr})==800
            for view,value in zip(VIEWS,(rr,cc)):
                p=d/f'{view}_{arm}.json';freeze_json(p,value);outputs[str(p)]=sha(p)
        np.savez_compressed(d/'extended_pool.npz',x26=ex,scores=es);freeze_json(d/'members.json',em)
        for n in ('extended_pool.npz','members.json'):outputs[str(d/n)]=sha(d/n)
        counts.update(original_members=len(mm),extended_members=len(em),new_geometries=len(new))
    freeze_json(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,counts=dict(counts),original_SL_bitwise_exact=True,GT_used=False));stage('predictions_sealed',counts=dict(counts))

def evaluate(c):
    sys.addaudithook(Guard(c,True).hook);ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs']);ctl.base.verify_files(c['GT_inputs'])
    ev=ctl.base.evaluator();ca,valid,uid=ctl.class_agnostic_evaluator();ap={g:{v:{} for v in VIEWS} for g in c['groups']};arrays={};coverage={};matches={v:{a:{} for a in ARMS} for v in VIEWS};receipts={}
    from tools.fi1_r3_core import gt_iou
    for i,s in enumerate(c['scenes']):
        stage('original_matching',scene=s,completed=i,total=200);gs,pts,_=expanded(c,s);gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);mapped=ctl.remap_gt(gt,valid,uid);oldload=ca.util_3d.load_ids
        keys=[g['geometry_key'] for g in gs];io,ids=gt_iou([pts[k] for k in keys],gt,c['class_ids']);by={k:j for j,k in enumerate(keys)};cov={}
        d=OUT/'evaluation'/s;d.mkdir(parents=True,exist_ok=True)
        for arm in ARMS:
            rr=read(OUT/'predictions'/s/f'semantic_{arm}.json');cc=read(OUT/'predictions'/s/f'class_agnostic_{arm}.json')
            for view,rows,evaluator,truth in (('semantic',rr,ev,gt),('class_agnostic',cc,ca,mapped)):
                if view=='class_agnostic':ca.util_3d.load_ids=lambda p: mapped.copy()
                try:match=ctl.shared.assign(evaluator,rows,pts,c['jobs'][s]['GT'],len(gt),view=='class_agnostic')
                finally:ca.util_3d.load_ids=oldload
                edges=ctl.shared.check_match(match,rows,pts,truth,evaluator,view=='class_agnostic');match=ctl.base.serializable_matches(match)
                q=d/f'{view}_{arm}_matches.json';freeze_json(q,match);receipts[str(q)]=sha(q);matches[view][arm][s]=match
            ca_best=np.max(io[[by[r['geometry_key']] for r in cc]],axis=0,initial=0);sem=np.zeros(len(ids))
            for r in rr:sem=np.maximum(sem,np.where(ids//1000==c['class_ids'][r['class_index']],io[by[r['geometry_key']]],0))
            cov[arm]=dict(semantic=sem.tolist(),class_agnostic=ca_best.tolist())
        coverage[s]=dict(gt_ids=ids.tolist(),arms=cov)
    freeze_json(OUT/'matches_frozen.json',dict(complete=True,outputs=receipts,independent_bidirectional_point_census=True));freeze_json(OUT/'coverage.json',coverage)
    refs=read(SL/'evaluation_summary.json')['AP']
    for view,evaluator in (('semantic',ev),('class_agnostic',ca)):
        for group,scenes in c['groups'].items():
            for arm in ARMS:
                stage('AP',group=group,view=view,arm=arm);x,*_=evaluator.evaluate_matches({s:matches[view][arm][s] for s in scenes});ap[group][view][arm]=ctl.base.metrics(evaluator,x)
                if arm=='SL':
                    old=np.load(SL/'evaluation'/f'{group}_{view}_SL_AP.npy');assert np.array_equal(x,old,equal_nan=True)
                q=OUT/'evaluation'/f'{group}_{view}_{arm}_AP.npy';np.save(q,x);arrays[str(q)]=sha(q)
                print('RESULT',group,view,arm,ap[group][view][arm],flush=True)
    freeze_json(OUT/'evaluation_summary.json',dict(complete=True,AP=ap,AP_arrays=arrays,original_SL_AP_exact=True))
    stage('evaluation_completed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','acquire','predict','evaluate'));a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):
        if a.mode=='prepare':prepare()
        else:globals()[a.mode](read(OUT/'contract.json'))
