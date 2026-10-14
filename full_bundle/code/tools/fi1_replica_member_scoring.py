#!/usr/bin/env python3
"""Own-mask semantics, registered R1/B26 selection and post-seal Replica AP."""
from __future__ import annotations
import argparse, gc, importlib.util, inspect, json, os, pickle, sys, types
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from tools.run_replica_registered_full import OUT, DATA, SCENES, WORK, BMODEL, bootstrap, read, reserve
from tools.fi1_replica_full_adapter import binding,jsonl,rows,function_adapter
from tools.fi1_frontend_protocol import sha256_file as sha, digest, write_json as write, freeze_json as freeze

def semantic_functions(count=48):
 """Only vocabulary dimensions/background and normalized class-rank denominator."""
 from tools import fi1_r0_r1_protocol as r0,fi1_r2_core as r2
 namespace={**r0.__dict__}
 for fn in [r0.direct_score,r0.original_aliases,r0.build_pair]:
  source=inspect.getsource(fn)
  source=source.replace('198',str(count))
  exec(compile(source,'<Replica semantic '+fn.__name__+'>','exec'),namespace)
 featns={**r2.__dict__,'direct_score':namespace['direct_score'],'original_aliases':namespace['original_aliases']}
 source=inspect.getsource(r2.geometry_features).replace('198',str(count)).replace('197.',str(count-1)+'.')
 exec(compile(source,'<Replica B26 own features>','exec'),featns)
 return namespace['build_pair'],featns['geometry_features']

def evidence(scenes):
 import torch,yaml
 from tools.prepare_fi1_r3b_val312 import no_gt_io
 from tools.export_fi1_refined_semantic_evidence import load_scene_context
 from tools.fi1_unified_evidence_core import baseline_maps,yolo_evidence
 from tools.fi1_full_alpha_adapter import make_views
 from tools.complete_fi1_r0_r1_alpha import manifest_row
 from tools.bind_fi1_r0_r1_alpha_cache import recover_alpha
 from tools.build_dm_sms1_alpha_embedding_ledger import _load_alpha_clip,_load_sam,_process_scene
 no_gt_io(); torch.set_num_threads(4)
 config=yaml.safe_load((OUT/'full.yaml').read_text()); labels=config['network2d']['text_prompts']
 provenance_path=Path('/media/jia/软件1/scannet_validation_stream/fi1_d_v3_dm_sms1_val312_local_20260902/dm_sms1_4090_repro_20260902/05_alpha_embedding_ledger/model_provenance.json')
 provenance=read(provenance_path)
 for k in ['alpha_clip_base','alpha_clip_checkpoint','sam_checkpoint']: assert sha(provenance[k])==provenance[k+'_sha256']
 args=NS(**{k:Path(provenance[k]) for k in ['alpha_clip_source','alpha_clip_base','alpha_clip_checkpoint','sam_source','sam_checkpoint']},sam_model_type=provenance['sam_model_type'],output_root=OUT/'fresh_alpha',sam_batch_size=8,alpha_batch_size=32,sms_threshold=0.)
 prompts=['a blurry photo of '+label+' in a room.' for label in labels]
 alpha=_load_alpha_clip(args,prompts); predictor=_load_sam(args); text=np.asarray(alpha['text_features'])
 np.save(OUT/'replica_alpha_text_features.npy',text)
 write(OUT/'alpha_identity.json',{'model_provenance':{str(provenance_path):sha(provenance_path)},'model_files':{str(provenance[k]):sha(provenance[k]) for k in ['alpha_clip_base','alpha_clip_checkpoint','sam_checkpoint']},'prompts':prompts,'text_features_sha256':sha(OUT/'replica_alpha_text_features.npy'),'GT_read':False})
 recover=function_adapter(recover_alpha,[('198','48')])
 for s in scenes:
  reserve(); dest=OUT/'own'/s
  if (dest/'sealed.json').exists(): continue
  geometries,packed=binding(s)
  cargs=NS(prepared_root=OUT/'prepared',yolo_cache_template=str(OUT/'records'/'{scene}'/'yoloworld_bboxes_2d'/'{scene}.pt'))
  context=load_scene_context(s,cargs,config,labels); maps=baseline_maps(context,48)
  records=[]
  for g in geometries:
   points=packed[g['array_key']]; yolo=yolo_evidence(points,context,maps,48,40); y=yolo['label_map']
   record={'geometry_key':g['geometry_key'],'points_sha256':g['points_sha256'],'point_count':g['point_count'],'scene_name':s,'yolo':{'acquisition_complete':True,'available':y['available'],'votes':y['counts'],'missing_reason':y['missing_reason']},'yolo_raw':yolo,'alpha':{'acquisition_complete':False,'available':False,'missing_reason':'not_acquired','view_requests':make_views(points,context)},'ground_truth_read':False,'ap_computed':False}
   records.append(record)
  rows(dest/'yolo_and_alpha_requests.jsonl',records)
  print('own Alpha',s,len(records),sum(len(r['alpha']['view_requests']) for r in records),flush=True)
  requests=[manifest_row(r) for r in records]
  _process_scene(s,requests,args,alpha,predictor,text,digest(requests))
  cache=args.output_root/'scenes'/s
  scale=np.load(cache/'scale_features.npy',mmap_mode='r'); agg=np.load(cache/'aggregate_features.npy'); sim=np.load(cache/'similarities.npy'); cached={r['geometry_key']:r for r in jsonl(cache/'records.jsonl')}
  for record in records: record['alpha']=recover(cached[record['geometry_key']],scale,agg,sim,text)
  rows(dest/'evidence.jsonl',records)
  write(dest/'sealed.json',{'GT_read':False,'geometries':len(records),'inputs':{str(OUT/'backend'/s/'sealed.json'):sha(OUT/'backend'/s/'sealed.json')},'outputs':{str(p):sha(p) for p in [dest/'geometry.json',dest/'points.npz',dest/'evidence.jsonl',dest/'yolo_and_alpha_requests.jsonl']}})
  del context,maps,packed,scale,agg,sim; gc.collect(); torch.cuda.empty_cache()

def score(scenes):
 import joblib
 from tools.fi1_r3_core import pair_overlap,context_features
 from tools.prepare_fi1_r3b_val312 import no_gt_io
 no_gt_io(); model=joblib.load(BMODEL); build,features=semantic_functions(48)
 for s in scenes:
  dest=OUT/'full'/s
  if (dest/'sealed.json').exists(): continue
  own=OUT/'own'/s; seal=read(own/'sealed.json')
  for p,h in seal['outputs'].items(): assert sha(p)==h
  gs=read(own/'geometry.json'); es={r['geometry_key']:r for r in jsonl(own/'evidence.jsonl')}
  pool=build(gs,es,budget=800)['R1']['eligible']; keys=[g['geometry_key'] for g in gs]
  points=np.load(own/'points.npz'); pp=[points[g['array_key']] for g in gs]
  domain=len(np.load(OUT/'prepared'/s/(s+'.npy'),mmap_mode='r'))
  iou,coverage=pair_overlap(pp,domain)
  xx={g['geometry_key']:features(g,es[g['geometry_key']]) for g in gs}
  x=np.column_stack([np.stack([xx[r['geometry_key']][r['class_index']] for r in pool]),context_features(pool,keys,iou,coverage)])
  scores=np.clip(model.predict(x),0,1); assert x.shape==(len(pool),26) and np.isfinite(scores).all()
  ranked=sorted([{**r,'score':float(v)} for r,v in zip(pool,scores)],key=lambda r:(-r['score'],not r['has_original_native_alias'],r['geometry_key'],r['class_index']))
  selected=ranked[:800]
  dest.mkdir(parents=True,exist_ok=True); np.save(dest/'features26.npy',x); np.save(dest/'scores.npy',scores)
  rows(dest/'pool.jsonl',pool); rows(dest/'selected.jsonl',selected)
  write(dest/'sealed.json',{'GT_read':False,'model_sha256':sha(BMODEL),'eligible':len(pool),'selected':len(selected),'selected_geometry':len(set(r['geometry_key'] for r in selected)),'outputs':{str(p):sha(p) for p in [dest/'features26.npy',dest/'scores.npy',dest/'pool.jsonl',dest/'selected.jsonl']}})
  print('Full K800 sealed',s,len(pool),len(selected),flush=True)
 if set(scenes)==set(SCENES):
  seals=[OUT/d/s/'sealed.json' for d in ['baseline','full'] for s in SCENES]
  freeze(OUT/'predictions_sealed.json',{'GT_read':False,'scenes':SCENES,'receipts':{str(p):sha(p) for p in seals},'new_training':False})

def full_prediction(scene):
 gs=read(OUT/'own'/scene/'geometry.json'); keys={g['geometry_key']:g['array_key'] for g in gs}
 points=np.load(OUT/'own'/scene/'points.npz'); rs=jsonl(OUT/'full'/scene/'selected.jsonl')
 domain=len(np.load(OUT/'prepared'/scene/(scene+'.npy'),mmap_mode='r'))
 masks=np.zeros((domain,len(rs)),dtype=bool)
 for i,r in enumerate(rs): masks[points[keys[r['geometry_key']]],i]=True
 return {'pred_masks':masks,'pred_classes':np.asarray([r['class_index'] for r in rs]),'pred_scores':np.asarray([r['score'] for r in rs])}

def prediction(scene,method):
 if method=='Full': return full_prediction(scene)
 root=OUT/'baseline'/scene
 masks=np.load(root/(scene+'_pred_masks.npy'),mmap_mode='r'); classes=np.load(root/(scene+'_pred_classes.npy')); scores=np.load(root/(scene+'_pred_scores.npy'))
 return {'pred_masks':masks,'pred_classes':classes,'pred_scores':np.ones_like(scores) if method=='baseline_official' else scores}

def dedup_agnostic(pred):
 """Actual selected valid members; exact point equality; maximum selected score."""
 by_hash={}
 for i,c in enumerate(pred['pred_classes']):
  if not 0<=c<48: continue
  mask=np.asarray(pred['pred_masks'][:,i],dtype=bool); key=np.packbits(mask,bitorder='little').tobytes()
  if key not in by_hash or by_hash[key][1]<pred['pred_scores'][i]: by_hash[key]=(i,pred['pred_scores'][i])
 ii=[v[0] for v in by_hash.values()]
 return {'pred_masks':pred['pred_masks'][:,ii],'pred_classes':np.zeros(len(ii),dtype=np.int64),'pred_scores':np.asarray([pred['pred_scores'][i] for i in ii])}

def clean(value):
 if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)): return [clean(v) for v in value]
 if isinstance(value,(np.floating,float)): return float(value) if np.isfinite(value) else None
 if isinstance(value,(np.integer,)): return int(value)
 return value

def evaluate():
 from evaluate.replica import eval_semantic_instance as e
 seal=read(OUT/'predictions_sealed.json')
 for p,h in seal['receipts'].items(): assert sha(p)==h
 for d in ['baseline','full']:
  for s in SCENES:
   for p,h in read(OUT/d/s/'sealed.json').get('outputs',read(OUT/d/s/'sealed.json').get('files',{})).items(): assert sha(p)==h
 evaldir=OUT/'evaluation'; evaldir.mkdir(exist_ok=True)
 # Clone original evaluator globals for actual-output class-agnostic protocol.
 ag=types.ModuleType('replica_agnostic'); exec(compile(Path(e.__file__).read_text(),e.__file__,'exec'),ag.__dict__)
 ag.CLASS_LABELS=['object']; ag.VALID_CLASS_IDS=np.asarray([1]); ag.ID_TO_LABEL={1:'object'}; ag.LABEL_TO_ID={'object':1}; ag.PRED_ID_TO_ID={0:1}
 gtroot=evaldir/'agnostic_gt'; gtroot.mkdir(exist_ok=True); gt_hashes={}; ids=read(OUT/'contract.json')['vocabulary']
 for s in SCENES:
  path=DATA/'ground_truth'/(s+'.txt'); gt=np.loadtxt(path,dtype=np.int64); gt_hashes[str(path)]=sha(path)
  assert len(gt)==len(np.load(OUT/'prepared'/s/(s+'.npy'),mmap_mode='r'))
  out=np.zeros_like(gt); valid=[int(v) for v in np.unique(gt) if v>=1000 and v//1000 in e.VALID_CLASS_IDS]
  assert len(valid)<1000
  for i,v in enumerate(valid): out[gt==v]=1001+i
  np.savetxt(gtroot/(s+'.txt'),out,fmt='%d')
 methods=['baseline_official','baseline_native','Full']; result={}
 for method in methods:
  dest=evaldir/method; dest.mkdir(exist_ok=True); semantic={}; agnostic={}
  for s in SCENES:
   pred=prediction(s,method)
   g,p=e.assign_instances_for_scan(pred,str(DATA/'ground_truth'/(s+'.txt'))); semantic[s]={'gt':g,'pred':p}
   cp=dedup_agnostic(pred); g,p=ag.assign_instances_for_scan(cp,str(gtroot/(s+'.txt'))); agnostic[s]={'gt':g,'pred':p}
   print('matched',method,s,pred['pred_masks'].shape[1],cp['pred_masks'].shape[1],flush=True)
   del pred,cp; gc.collect()
  result[method]={}
  for name,module,matches in [('semantic',e,semantic),('actual_class_agnostic',ag,agnostic)]:
   with (dest/(name+'_matches.pkl')).open('wb') as f: pickle.dump(matches,f,protocol=4)
   ap=module.evaluate_matches(matches); np.save(dest/(name+'_ap.npy'),ap)
   avg=module.compute_averages(ap); module.write_result_file(avg,str(dest/(name+'.csv')))
   result[method][name]=clean(avg)
   per_scene={}
   for s in SCENES:
    pa=module.evaluate_matches({s:matches[s]}); np.save(dest/(name+'_'+s+'_ap.npy'),pa); per_scene[s]=clean(module.compute_averages(pa))
   write(dest/(name+'_per_scene.json'),per_scene)
   print(method,name,avg['all_ap'],avg['all_ap_50%'],avg['all_ap_25%'],flush=True)
 write(evaldir/'results.json',result)
 write(evaldir/'receipt.json',{'prediction_seal_sha256':sha(OUT/'predictions_sealed.json'),'GT_after_prediction_seal':True,'GT':gt_hashes,'evaluator':{str(Path(e.__file__).resolve()):sha(e.__file__)},'methods':methods,'outputs':{str(p):sha(p) for p in evaldir.rglob('*') if p.is_file() and p.name!='receipt.json'}})

def main():
 p=argparse.ArgumentParser(); p.add_argument('command',choices=['evidence','score','evaluate']); p.add_argument('--scenes',nargs='+',choices=SCENES,default=SCENES); a=p.parse_args(); bootstrap(); {'evidence':lambda:evidence(a.scenes),'score':lambda:score(a.scenes),'evaluate':evaluate}[a.command]()
if __name__=='__main__': main()
