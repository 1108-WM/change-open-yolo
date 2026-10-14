#!/usr/bin/env python3
"""Verify frozen Replica point identities, B26 inference, selection and AP arrays."""
import gc,json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from tools.run_replica_registered_full import OUT,DATA,SCENES,BMODEL,DROOT,bootstrap,read
from tools.fi1_frontend_protocol import sha256_file as sha,write_json as write
from tools.fi1_replica_full_adapter import jsonl
from tools.fi1_replica_member_scoring import semantic_functions

def main():
 bootstrap()
 import joblib
 from plyfile import PlyData
 from evaluate.replica import eval_semantic_instance as e
 from tools.fi1_r3_core import pair_overlap,independent_context
 contract=read(OUT/'contract.json'); checked=0
 for p,h in {**contract['models_and_inputs'],**read(OUT/'backend_model_code_binding.json')['inputs']}.items(): assert sha(p)==h,p; checked+=1
 for row in read(DROOT/'summary.json')['models']: assert sha(DROOT/row['model_file'])==row['model_sha256']
 seal=read(OUT/'predictions_sealed.json')
 for p,h in seal['receipts'].items(): assert sha(p)==h; checked+=1
 model=joblib.load(BMODEL); build,features=semantic_functions(); stats=[]
 assert contract['vocabulary']==e.CLASS_LABELS
 for s in SCENES:
  prepared=np.load(OUT/'prepared'/s/(s+'.npy'),mmap_mode='r'); vertex=PlyData.read(DATA/s/(s+'_mesh.ply'))['vertex'].data
  np.testing.assert_array_equal(prepared[:,:3],np.column_stack([vertex[k] for k in ['x','y','z']]))
  gs=read(OUT/'own'/s/'geometry.json'); pts=np.load(OUT/'own'/s/'points.npz'); es={r['geometry_key']:r for r in jsonl(OUT/'own'/s/'evidence.jsonl')}
  pool=build(gs,es,800)['R1']['eligible']; actual=jsonl(OUT/'full'/s/'pool.jsonl'); assert pool==actual
  pp=[pts[g['array_key']] for g in gs]; keys=[g['geometry_key'] for g in gs]
  for g,p in zip(gs,pp):
   import hashlib
   assert hashlib.sha256(np.asarray(p,dtype='<i8').tobytes()).hexdigest()==g['points_sha256']
   assert len(p)==g['point_count'] and np.array_equal(p,np.unique(p)) and p[-1]<len(prepared)
   for modality in ['yolo','alpha']:
    z=es[g['geometry_key']][modality]; assert z['acquisition_complete']
    if z['available']: assert len(z['votes' if modality=='yolo' else 'cosine'])==48
  own={g['geometry_key']:features(g,es[g['geometry_key']]) for g in gs}
  iou,cov=pair_overlap(pp,len(prepared)); context=independent_context(pool,keys,iou,cov)
  x=np.column_stack([np.stack([own[r['geometry_key']][r['class_index']] for r in pool]),context])
  np.testing.assert_allclose(x,np.load(OUT/'full'/s/'features26.npy'),rtol=0,atol=1e-15)
  score=np.clip(model.predict(x),0,1); np.testing.assert_array_equal(score,np.load(OUT/'full'/s/'scores.npy'))
  order=sorted(range(len(pool)),key=lambda i:(-score[i],not pool[i]['has_original_native_alias'],pool[i]['geometry_key'],pool[i]['class_index']))[:800]
  expected=[{**pool[i],'score':float(score[i])} for i in order]; assert expected==jsonl(OUT/'full'/s/'selected.jsonl')
  stats.append({'scene':s,'points':len(prepared),'geometries':len(gs),'members':len(pool),'selected':len(expected),'max_score':float(score.max())})
  del prepared,pts,gs,es,pool,actual,pp,x,context; gc.collect()
 result=read(OUT/'evaluation/results.json'); arrays=0
 for method in ['baseline_official','baseline_native','Full']:
  for name in ['semantic','actual_class_agnostic']:
   d=OUT/'evaluation'/method; ap=np.load(d/(name+'_ap.npy'))
   expected=[float(np.nanmean(ap[:,:,~np.isclose(e.opt['overlaps'],.25)])),float(np.nanmean(ap[:,:,np.isclose(e.opt['overlaps'],.5)])),float(np.nanmean(ap[:,:,np.isclose(e.opt['overlaps'],.25)]))]
   for key,v in zip(['all_ap','all_ap_50%','all_ap_25%'],expected): assert abs(v-result[method][name][key])<1e-15
   arrays+=1
   with (d/(name+'_matches.pkl')).open('rb') as f: matches=pickle.load(f)
   if name=='semantic': np.testing.assert_array_equal(e.evaluate_matches(matches),ap)
   assert set(matches)==set(SCENES)
   for s in SCENES: assert np.load(d/(name+'_'+s+'_ap.npy')).shape==ap.shape; arrays+=1
 write(OUT/'verified.json',{'complete':True,'input_hashes_checked':checked,'scenes':stats,'AP_arrays_checked':arrays,'original_XYZ_order':True,'own_evidence_complete':True,'frozen_B26_exact_replay':True,'independent_context':True,'K800_exact_replay':True,'semantic_AP_cache_replay':True,'GT_read_in_verification':False})
 print('Replica verification complete',checked,arrays)
if __name__=='__main__': main()
