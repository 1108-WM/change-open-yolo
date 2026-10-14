#!/usr/bin/env python3
"""Replica-only adapter. Uses registered geometry kernels and frozen weights.

Only input domain/depth/vocabulary/scene-name adapters differ from ScanNet.
No GT, fitting, output gates, NMS or S/SL substitutions are introduced.
"""
from __future__ import annotations
import argparse, copy, gc, inspect, json, os, sys, types
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from tools.run_replica_registered_full import OUT, DATA, SCENES, WORK, BMODEL, DROOT, bootstrap, read, reserve
from tools.fi1_frontend_protocol import sha256_file as sha, digest, write_json as write, freeze_json as freeze

def jsonl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def rows(p,rs):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rs))
def function_adapter(fn,replacements):
 """Freeze explicit source deltas, leaving the production module untouched."""
 source=inspect.getsource(fn)
 for before,after in replacements:
  if before not in source: raise ValueError('adapter source changed: '+before)
  source=source.replace(before,after)
 scope=dict(fn.__globals__); exec(compile(source,'<Replica '+fn.__name__+'>','exec'),scope)
 return scope[fn.__name__]

def manifest(scene):
 root=OUT/'prepared'/scene
 frames=[]
 for i,f in enumerate(range(0,200,10)):
  r={'frame_id':str(f),'frame_index':i,'pose_valid':True}
  for n,d,e in [('color','color','.jpg'),('depth','depth','.png'),('pose','poses','.txt')]:
   p=root/d/(str(f)+e); r[n]={'path':str(p),'sha256':sha(p)}
  frames.append(r)
 files={n:{'path':str(p),'sha256':sha(p)} for n,p in [('mesh',root/(scene+'_mesh.ply')),('intrinsics',root/'intrinsics.txt'),('processed',root/(scene+'.npy'))]}
 return {'scene_name':scene,'original_frame_count':200,'arms':{'A':list(range(0,200,10)),'D':list(range(0,200,10))},'frames':frames,'files':files,'frame_index_domain':'manifest.frames','dataset':'Replica'}

def frontends(scenes):
 import torch,yaml
 from tools import run_fi1_frontend_comparison as impl
 from tools.export_groundingdino_f30_boxes import _build_prompt_groups
 from utils import OpenYolo3D,WORLD_2_CAM
 from tools.export_dense_frame_instance_observations import export_scene_dense_observations
 import tools.export_dense_frame_instance_observations as dense
 from tools.fi1_shared import export_sam_fused_proposals as sam_fused
 for name in ['_clamp_box','_frame_key','_prepare_image','_safe_label','_sam_mask_to_indices','_to_numpy']:
  setattr(dense,name,getattr(sam_fused,name))
 torch.set_num_threads(4)
 config=yaml.safe_load((OUT/'full.yaml').read_text()); labels=config['network2d']['text_prompts']
 adapter_project=function_adapter(impl.project_frame,[('1000.0, 0.05','6553.5, 0.05')])
 frame_scope=dict(impl.export_frame.__globals__); frame_scope['project_frame']=adapter_project
 export=types.FunctionType(impl.export_frame.__code__,frame_scope,'export_frame',impl.export_frame.__defaults__)
 front=OUT/'frontend'
 for s in scenes: freeze(front/'manifests'/(s+'.json'),manifest(s))
 # Common validation consumes exact frame identities and YOLO box-prompted SAM.
 # It no longer needs automatic-SAM candidates to establish those identities.
 for s in scenes:
  freeze(OUT/'common/verification_frames'/s/'verification_frames.json',
         {'frames':[{'frame_id':f['frame_id'],'frame_index':f['frame_index']} for f in manifest(s)['frames']]})
 for kind,arm in [('grounded_sam','D')]:
  if all(((front/'arms/A/hierarchy'/s/'summary.json') if arm=='A' else (front/'arms/D/stage_receipts'/(s+'_merged.json'))).exists() for s in scenes): continue
  model=impl.Frontends(kind,front/(kind+'_initial_model_receipt.json'))
  if kind=='grounded_sam':
   model.labels=labels; model.groups=_build_prompt_groups(labels,model.gd.tokenizer,int(model.gd.max_text_len))
  write(front/(kind+'_replica_model_receipt.json'),{'labels':labels if arm=='D' else None,'depth_scale':6553.5,'parameters':impl.PARAMETERS,'source':str(Path(impl.__file__).resolve()),'source_sha256':sha(impl.__file__),'GT_read':False})
  for s in scenes:
   reserve(); m=read(front/'manifests'/(s+'.json')); identity=digest({'contract':sha(OUT/'contract.json'),'manifest':digest(m),'arm':arm,'adapter':sha(__file__)})
   cache=front/'frame_cache'/kind/s
   for f in m['frames']:
    r=export(model,m,f,cache/('frame'+f['frame_id'].zfill(6)),digest({'identity':identity,'frame':f}),len(np.load(OUT/'prepared'/s/(s+'.npy'),mmap_mode='r')))
    print(kind,s,f['frame_id'],r['kept_observation_count'],flush=True)
   impl.assemble_arm(front,arm,m,cache,identity)
   impl.track_arm(front,arm,m,cache,len(np.load(OUT/'prepared'/s/(s+'.npy'),mmap_mode='r')),identity)
  del model; gc.collect(); torch.cuda.empty_cache()
 # Reuse original YOLO-SAM exporter, with its unchanged fixed parameters.
 pending=[s for s in scenes if not (OUT/'common/yolosam'/s/'summary.json').exists()]
 if pending:
  sys.path.insert(0,str(impl.SAM_SOURCE))
  from segment_anything import SamPredictor,sam_model_registry
  model=NS(predictor=SamPredictor(sam_model_registry['vit_b'](checkpoint=str(impl.SAM_WEIGHT)).eval().cuda()))
  for s in pending:
   world=WORLD_2_CAM(str(OUT/'prepared'/s),6553.5,config)
   obj=OpenYolo3D.__new__(OpenYolo3D); obj.world2cam=world; obj.openyolo3d_config=config
   obj.mesh_projections=world.get_mesh_projections(); obj.scaling_params=[world.depth_resolution[j]/world.image_resolution[j] for j in range(2)]
   payload=torch.load(OUT/'records'/s/'yoloworld_bboxes_2d'/(s+'.pt'),map_location='cpu'); obj.preds_2d=payload['predictions']
   args=NS(overwrite=False,max_frames=30,frame_stride=1,frame_selection='uniform',max_detections_per_frame=20,detection_score_th=.08,max_box_area_ratio=.85,sam_multimask_topk=1,min_mask_area=64,min_visible_points=8,duplicate_iou=.92,duplicate_containment=.985,max_masks_per_frame=20)
   export_scene_dense_observations(obj,model.predictor,s,OUT/'common/yolosam',args)
   print('common YOLO-SAM',s,flush=True); del obj,world; gc.collect(); torch.cuda.empty_cache()
  del model
 write(OUT/'frontend/sealed.json',{'GT_read':False,'scenes':scenes,'code':sha(__file__)})

def backend(scenes):
 import yaml,sklearn
 from threadpoolctl import threadpool_limits
 from tools.evaluate_fi1_frontend_comparison import full_backend_modules, bind_gvc_track_frame_counts, raw_frame_observations, checked_points
 from tools.prepare_fi1_r3b_val312 import no_gt_io
 no_gt_io(); modules=full_backend_modules()
 assert sklearn.__version__=='1.3.0',sklearn.__version__
 champion=ROOT/'output/train_candidate_champion_pair_union_combined_full_official100_v2_prior_corrected'
 if not (champion/'model_package.pkl').exists():
  # The registered CLI default is the authoritative package path.
  import tools.evaluate_fi1_frontend_comparison as e
  source=Path(e.__file__).read_text()
  import re
  match=re.search(r"parser.add_argument\('--champion-models'.*default=Path\('([^']+)'",source)
  if not match: raise FileNotFoundError('registered champion package')
  champion=Path(match.group(1))
 assert sha(champion/'model_package.pkl')=='b5f91148eb9cf310bee25d1ee6237699965cd368b04a3bb3a3d143aa395652a8'
 config=yaml.safe_load((OUT/'full.yaml').read_text())
 for s in scenes:
  reserve(); dest=OUT/'backend'/s
  if (dest/'sealed.json').exists(): continue
  dest.mkdir(parents=True,exist_ok=True); (dest/'scene.txt').write_text(s+'\n')
  native=dest/'native'; native.mkdir(exist_ok=True)
  for p in (OUT/'records'/s/'native_cache').glob('*.npy'):
   q=native/p.name
   if not q.exists(): q.symlink_to(p)
  write(native/'native_cache_no_gt_manifest.json',{'scene_count':1,'candidate_inputs':{'mode':'mask3d_yoloworld_only'},'decision_state':'Replica frozen raw native; no oracle GT'})
  front=OUT/'frontend'; m=read(front/'manifests'/(s+'.json'))
  tracks_payload=read(front/'arms/D/merged'/s/'automatic_tracks.json'); source_tracks=tracks_payload['tracks']
  obs=jsonl(front/'arms/D/hierarchy'/s/'automatic_observations.jsonl')
  converted=raw_frame_observations(m,obs,source_tracks,'D')
  rows(dest/'automatic_raw_frames'/s/'automatic_observations.jsonl',converted)
  common=OUT/'common/verification_frames'; yolosam=OUT/'common/yolosam'
  # Existing frozen runs predate the independent frame contracts. Read their
  # registered frame mapping without altering the original outputs/contracts.
  if not (common/s/'verification_frames.json').exists():
   common=front/'arms/A/hierarchy'
  _,fmap=modules['gvc']._frame_contract(common/s)
  masks=np.load(native/(s+'_pred_masks.npy'),mmap_mode='r'); scores=np.load(native/(s+'_pred_scores.npy')); sizes=np.count_nonzero(masks,axis=0)
  assert masks.shape[0]==len(np.load(OUT/'prepared'/s/(s+'.npy'),mmap_mode='r'))
  with threadpool_limits(limits=4):
   relations=[]; plans=[]
   for t in source_tracks:
    rr,ss=modules['competition'].track_native_relations(checked_points(t['points_path'],masks.shape[0]),masks,sizes,scores,t['proposal_id'],t['mean_node_quality'],'hierarchy_safe_d2b')
    relations.extend(rr); plans.append(modules['duplicate'].plan_proposal(ss,rr,'hierarchy_safe_d2b'))
   retained,actions,ledger=modules['filter'].filter_tracks(source_tracks,plans,geometry_variant='hierarchy_safe_d2b')
   trackroot=dest/'filtered_tracks'; write(trackroot/s/'automatic_tracks.json',{**tracks_payload,'tracks':retained})
   rows(dest/'track_native_relations.jsonl',relations); rows(dest/'duplicate_plan.jsonl',plans); rows(dest/'filter_ledger.jsonl',ledger); rows(dest/'filter_actions.jsonl',actions)
   kw=dict(dataset_root=OUT/'prepared',config=config,depth_scale=6553.5,track_root=trackroot)
   semantics=modules['semantics']._scene_records(s,NS(**kw,bboxes_2d_root=OUT/'records'/s/'yoloworld_bboxes_2d'))
   write(dest/'semantics'/s/'automatic_track_yoloworld_semantics.json',semantics)
   gr,gs=modules['gvc']._scene_records(s,NS(**kw,automatic_root=common,yoloworld_sam_root=yolosam,native_prediction_cache=native,max_views=10,min_visible_points=30))
   gr=bind_gvc_track_frame_counts(gr,retained,fmap)
   write(dest/'gvc'/s/'c1_gvc_quality_ledger.json',gr); write(dest/'gvc'/s/'summary.json',gs)
   write(dest/'gvc/summary.json',{'scene_count':1,'ground_truth_usage':'none','ground_truth_read':False,'ap_evaluation_run':False})
   shared=dict(scene_list=dest/'scene.txt',native_cache=native,track_root=trackroot,quality_ledger_root=dest/'gvc',expected_scene_count=1,max_scenes=None)
   modules['relation'].run(NS(**shared,automatic_root=common,yoloworld_sam_root=yolosam,dataset_root=OUT/'prepared',config_path=OUT/'full.yaml',output_root=dest/'relations',resume=False))
   modules['champion'].run(NS(**shared,relation_root=dest/'relations',model_root=champion,output_root=dest/'champion'))
   modules['unique'].run(NS(scene_list=dest/'scene.txt',native_cache=native,track_root=trackroot,records_root=OUT/'records',plan_root=dest/'champion',semantic_root=dest/'semantics',output_dir=dest/'unique',preregistration_path=ROOT/'docs/FI1_VAL48_FRONTEND_FULL_PIPELINE_PROTOCOL_20260914.md',expected_scene_count=1,max_scenes=None,class_count=48))
   cc=copy.deepcopy(config); cc['openyolo3d']['frequency']=1
   cp=dest/'stage_c_raw_frames.yaml'; cp.write_text(yaml.safe_dump(cc))
   summary=modules['dv3'].run(NS(scene_list=dest/'scene.txt',prepared_root=OUT/'prepared',unique_geometry_root=dest/'unique',gvc_root=dest/'gvc',relation_root=dest/'relations',champion_plan_root=dest/'champion',track_root=trackroot,automatic_root=dest/'automatic_raw_frames',config_path=cp,model_root=DROOT,output_root=dest/'dv3',min_points=100,dataset_name='Replica-frozen-transfer'))
  write(dest/'sealed.json',{'GT_read':False,'source_tracks':len(source_tracks),'retained_tracks':len(retained),'Dv3':summary,'files':{str(p):sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='sealed.json'}})
  print('D-v3 backend sealed',s,flush=True)

def binding(scene):
 from tools.audit_fi1_r0_r1_geometry import audit_scene
 adapted=function_adapter(audit_scene,[("f'{scene[5:]}.npy'","f'{scene}.npy'"),("f'{scene}_vh_clean_2.ply'","f'{scene}_mesh.ply'")])
 dest=OUT/'backend'/scene
 gs,packed,files,summary=adapted(scene,{'prepared_root':str(OUT/'prepared'),'records_root':str(OUT/'records')},jsonl(dest/'unique/unique_geometry_ledger.jsonl'),jsonl(dest/'dv3/complete_plan/complete_inference_ap_plan.jsonl'))
 out=OUT/'own'/scene; out.mkdir(parents=True,exist_ok=True)
 write(out/'geometry.json',gs); np.savez_compressed(out/'points.npz',**packed)
 write(out/'binding.json',{'inputs':files,'summary':summary,'GT_read':False})
 return gs,packed

def main():
 p=argparse.ArgumentParser(); p.add_argument('command',choices=['frontends','backend']); p.add_argument('--scenes',nargs='+',choices=SCENES,default=SCENES); a=p.parse_args(); bootstrap(); {'frontends':frontends,'backend':backend}[a.command](a.scenes)
if __name__=='__main__': main()
