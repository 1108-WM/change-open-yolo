#!/usr/bin/env python3
"""Isolated Replica transfer of registered Full; never changes ScanNet contracts."""
from __future__ import annotations
import argparse, gc, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.fi1_frontend_protocol import sha256_file as sha, write_json as write, freeze_json as freeze
OUT=ROOT/'.local/replica_registered_full_20261011_v1'
DATA=ROOT/'data/replica'
SCENES=['office0','office1','office2','office3','office4','room0','room1','room2']
WORK=Path('/home/jia/Wm/wm_open-yolo/OpenYOLO3D')
BMODEL=Path('/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921/learning/models/B.joblib')
DROOT=WORK/'pretrained/fi1_d_v3_official100_full_models_20260822'

def read(p): return json.loads(Path(p).read_text())
def reserve():
 if shutil.disk_usage(ROOT).free<50*1024**3: raise RuntimeError('50GiB disk reserve reached')
def bootstrap():
 os.environ.setdefault('OMP_NUM_THREADS','4')
 os.environ.setdefault('HF_HUB_OFFLINE','1')
 os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
 for p in [ROOT/'models/Mask3D',ROOT/'models/YOLO-World',WORK]:
  if str(p) not in sys.path: sys.path.append(str(p))
 import tools
 for p in [WORK/'tools',ROOT/'full_bundle/code_sources/evidence_snapshot/tools']:
  if str(p) not in tools.__path__: tools.__path__.append(str(p))

def prepare():
 import yaml
 from plyfile import PlyData
 OUT.mkdir(parents=True,exist_ok=True)
 c=yaml.safe_load((ROOT/'pretrained/config_replica.yaml').read_text())
 full=yaml.safe_load((ROOT/'pretrained/config_scannet200.yaml').read_text())
 full['network2d']['text_prompts']=c['network2d']['text_prompts']
 full['openyolo3d']['depth_scale']=6553.5
 for name,config in [('baseline',c),('full',full)]:
  path=OUT/(name+'.yaml'); value=yaml.safe_dump(config,sort_keys=False)
  if path.exists() and path.read_text()!=value: raise ValueError('config changed')
  path.write_text(value)
 models=[ROOT/'pretrained/checkpoints/scannet200_val.ckpt',ROOT/'pretrained/checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth',ROOT/'pretrained/checkpoints/sam_vit_b_01ec64.pth',ROOT/'pretrained/groundingdino/groundingdino_swint_ogc.pth',BMODEL,*sorted(DROOT.glob('*.pkl')),*sorted(DROOT.glob('*.joblib')),DROOT/'summary.json']
 assert sha(BMODEL)=='48f1d79df57bd9e2ba70802ed18494d6390c5bbdfec6f990fc259449d84ea43a'
 inputs={str(p):sha(p) for p in models}
 inventory=[]
 for scene in SCENES:
  src=(DATA/scene).resolve(); dest=OUT/'prepared'/scene; dest.mkdir(parents=True,exist_ok=True)
  for name in ['color','depth','poses','intrinsics.txt',scene+'_mesh.ply']:
   p=dest/name
   if not p.exists(): p.symlink_to(src/name,target_is_directory=(src/name).is_dir())
   assert p.resolve()==src/name
  ply=PlyData.read(src/(scene+'_mesh.ply')); n=len(ply['vertex'].data)
  counts={d:len(list((src/d).iterdir())) for d in ['color','depth','poses']}
  assert len(set(counts.values()))==1 and counts['poses']>0
  assert (DATA/'ground_truth'/(scene+'.txt')).is_file()
  files=[src/'intrinsics.txt',src/(scene+'_mesh.ply')]
  files += [src/d/(str(i)+ext) for i in range(counts['poses']) for d,ext in [('color','.jpg'),('depth','.png'),('poses','.txt')]]
  inputs.update({str(p):sha(p) for p in files})
  inventory.append({'scene':scene,'points':n,'frames':counts['poses'],'GT_exists_not_read':True})
 freeze(OUT/'contract.json',{'method':'GD10 + official100 D-v3 + matched597 B26 / NTUUprime / K800','scenes':SCENES,'vocabulary':c['network2d']['text_prompts'],'depth_scale':6553.5,'models_and_inputs':inputs,'configs':{n:sha(OUT/(n+'.yaml')) for n in ['baseline','full']},'baseline':'original Replica config: top1 per Mask3D geometry, official uniform scores; additionally native confidence control','native_proposals':'regenerated frozen Mask3D on full original Replica mesh; no oracle masks','superpoints':'ScanNet Segmentator 0.01 / min20 on original full mesh; no half-domain Open3DIS superpoints','transfer':'frozen ScanNet-trained weights with Replica 48-class vocabulary; no training or test tuning','GT_policy':'predict and seal both methods before evaluator reads raw GT','new_training':False,'code':{str(Path(__file__).resolve()):sha(__file__)}})
 write(OUT/'inventory.json',inventory)
 print(json.dumps(inventory),flush=True)

def superpoints():
 from plyfile import PlyData, PlyElement
 binary=ROOT/'models/Mask3D/third_party/ScanNet/Segmentator/segmentator'
 for s in SCENES:
  reserve(); dest=OUT/'prepared'/s; path=dest/(s+'.npy')
  if path.exists(): continue
  # Segmentator writes next to input; copy rather than resolve a source symlink.
  local=OUT/'superpoint_work'/s; local.mkdir(parents=True,exist_ok=True)
  mesh=local/(s+'_mesh.ply')
  original=PlyData.read(DATA/s/mesh.name)
  v=original['vertex'].data
  # Replica faces are quads; the ScanNet executable requires triangles.
  # Fan triangulation preserves every original vertex and its exact index.
  triangles=[]
  for face in original['face'].data['vertex_indices']:
   triangles.extend((int(face[0]),int(face[i]),int(face[i+1])) for i in range(1,len(face)-1))
  xyz=np.empty(len(v),dtype=[('x','f4'),('y','f4'),('z','f4')])
  for k in ['x','y','z']: xyz[k]=v[k]
  faces=np.empty(len(triangles),dtype=[('vertex_indices','i4',(3,))]); faces['vertex_indices']=triangles
  PlyData([PlyElement.describe(xyz,'vertex'),PlyElement.describe(faces,'face')],text=False,byte_order='<').write(mesh)
  subprocess.run([str(binary),str(mesh),'0.01','20'],check=True)
  segfile=mesh.with_suffix('.0.010000.segs.json'); seg=np.asarray(read(segfile)['segIndices'])
  assert len(v)==len(seg)
  columns=[v[k] for k in ['x','y','z','red','green','blue']]
  if all(k in v.dtype.names for k in ['nx','ny','nz']):
   normals=np.column_stack([v[k] for k in ['nx','ny','nz']])
  else:
   positions=np.column_stack([v[k] for k in ['x','y','z']]); tri=np.asarray(triangles)
   face_normals=np.cross(positions[tri[:,1]]-positions[tri[:,0]],positions[tri[:,2]]-positions[tri[:,0]])
   normals=np.zeros_like(positions)
   for i in range(3): np.add.at(normals,tri[:,i],face_normals)
   normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
  columns.extend(normals[:,i] for i in range(3))
  columns+=[np.unique(seg,return_inverse=True)[1],np.full(len(v),-1),np.full(len(v),-1)]
  values=np.column_stack(columns).astype(np.float32); np.save(path,values)
  assert np.array_equal(values[:,:3],np.column_stack([v[k] for k in ['x','y','z']]))
  write(dest/'superpoint_receipt.json',{'input':{str(DATA/s/mesh.name):sha(DATA/s/mesh.name),str(binary):sha(binary)},'triangulated_mesh_sha256':sha(mesh),'vertex_order_unchanged':True,'output':{str(path):sha(path)},'points':len(v),'superpoints':int(len(np.unique(seg))),'GT_read':False})
  print('superpoints',s,len(np.unique(seg)),flush=True)

def native(scenes):
 import torch, yaml
 from utils import OpenYolo3D
 torch.set_num_threads(4)
 model=OpenYolo3D(str(OUT/'baseline.yaml'))
 masksroot=OUT/'mask3d'; masksroot.mkdir(exist_ok=True)
 for s in scenes:
  reserve(); dest=OUT/'baseline'/s; dest.mkdir(parents=True,exist_ok=True)
  if (dest/'sealed.json').exists(): continue
  result=model.predict(str(DATA/s),6553.5,datatype='mesh',path_to_3d_masks=str(masksroot) if (masksroot/(s+'.pt')).exists() else None,path_to_2d_preds=str(OUT/'yolo_baseline'),save_2d_preds=True)
  torch.save(tuple(x.detach().cpu() for x in model.preds_3d),masksroot/(s+'.pt'))
  for name,x in zip(['masks','classes','scores'],result[s]): np.save(dest/(s+'_pred_'+name+'.npy'),x.detach().cpu().numpy())
  write(dest/'sealed.json',{'GT_read':False,'files':{str(p):sha(p) for p in dest.glob('*.npy')},'mask3d_sha256':sha(masksroot/(s+'.pt'))})
  print('baseline sealed',s,flush=True); gc.collect(); torch.cuda.empty_cache()
 del model; gc.collect(); torch.cuda.empty_cache()
 full=OpenYolo3D(str(OUT/'full.yaml'))
 for s in scenes:
  dest=OUT/'records'/s/'native_cache'; dest.mkdir(parents=True,exist_ok=True)
  if (dest/'sealed.json').exists(): continue
  result=full.predict(str(OUT/'prepared'/s),6553.5,datatype='mesh',path_to_3d_masks=str(masksroot),path_to_2d_preds=str(OUT/'records'/s/'yoloworld_bboxes_2d'),save_2d_preds=True)
  for name,x in zip(['masks','classes','scores'],result[s]): np.save(dest/(s+'_pred_'+name+'.npy'),x.detach().cpu().numpy())
  write(dest/'sealed.json',{'GT_read':False,'files':{str(p):sha(p) for p in dest.glob('*.npy')}})
  print('full native sealed',s,flush=True); gc.collect(); torch.cuda.empty_cache()

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','superpoints','native']); parser.add_argument('--scenes',nargs='+',choices=SCENES,default=SCENES)
 args=parser.parse_args(); bootstrap(); {'prepare':prepare,'superpoints':superpoints,'native':lambda:native(args.scenes)}[args.command]()
if __name__=='__main__': main()
