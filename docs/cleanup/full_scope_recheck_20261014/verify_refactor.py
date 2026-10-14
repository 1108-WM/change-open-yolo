"""Replay archived AP formulas, current Full/SL scores and frame identities."""
from pathlib import Path
import ast, copy, importlib, json, os, tarfile
import joblib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(ROOT))
from tools import fi1_instance_eval_shared as shared
from tools.evaluate_fi1_frontend_comparison import full_backend_modules

checks=[]
def check(condition,label):
    assert condition,label
    checks.append(label)

archive=ROOT/'archive/retired_20261014/full_shared_formula_extraction.tar.gz'
reference={'np':np,'instance_eval':shared.instance_eval}
with tarfile.open(archive) as tar:
    for stem,names in [
        ('diagnose_mv3dis_global_feasible_action_oracle_gt',['_scene_ap_records','_average_precision']),
        ('diagnose_d2b_native_track_ranking_oracle_gt',['_empty_record','_append']),
    ]:
        text=tar.extractfile('full_bundle/code/tools/'+stem+'.py').read().decode()
        for node in ast.parse(text).body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node],type_ignores=[]),'<archived formula>','exec'),reference)

shared._configure_scannet200_instance_eval()
# Duplicate predictions, strict IoU ties, small/void regions and missed GT.
rng=np.random.default_rng(597)
for case in range(12):
    gt={'chair':[dict(instance_id=2001+i,vert_count=100+i*40,matched_pred=[]) for i in range(4)]}
    gt['chair'] += [dict(instance_id=1,vert_count=40,matched_pred=[]),dict(instance_id=2009,vert_count=50,matched_pred=[])]
    pred={'chair':[]}
    for i in range(case):
        p=dict(uuid=str(i),vert_count=100+i*10,confidence=float((i%3)/3),void_intersection=i*2,matched_gt=[])
        for g in gt['chair']:
            intersection=int(rng.integers(0,min(g['vert_count'],p['vert_count'])+1))
            p['matched_gt'].append(dict(g,intersection=intersection))
            g['matched_pred'].append(dict(p,intersection=intersection))
        pred['chair'].append(p)
    for t in shared.DIAGNOSTIC:
        old=reference['_scene_ap_records'](gt,pred,t)
        new=shared._record_from_matches(gt,pred,t)
        check(all(np.array_equal(a,b) for a,b in zip(old,new)),f'matching case {case} threshold {t}')
        old_record=reference['_empty_record']();new_record=shared._empty_record()
        reference['_append'](old_record,old);shared._append(new_record,new)
        args=lambda r:(r['true'],r['score'],r['fn'],r['has_gt'],r['has_pred'])
        a=reference['_average_precision'](*args(old_record));b=shared._average_precision(*args(new_record))
        check(a==b or np.isnan(a) and np.isnan(b),f'AP case {case} threshold {t}')

mods=full_backend_modules()
check(len(mods)==9,'all 9 registered backend modules import')
from tools.run_fi1_frontend_comparison import export_scope,Frontends
check(export_scope({'scenes':['fixture']})==(['fixture'],['D']),'active export defaults to D')
try:Frontends('automatic_sam',Path('/tmp/unused_receipt'))
except ValueError:checks.append('retired automatic SAM rejected before GPU initialization')
else:raise AssertionError('automatic SAM still accepted')
from tools.fi1_replica_full_adapter import manifest,SCENES
for scene in SCENES:
    current=[{'frame_id':f['frame_id'],'frame_index':f['frame_index']} for f in manifest(scene)['frames']]
    original=json.loads((ROOT/'full_bundle/artifacts/replica'/scene/'verification_frames.json').read_text())['frames']
    check(current==original,'Replica independent frame contract identity: '+scene)

torch.set_num_threads(4)
from tools.fi1_core_ablation import features_and_members,learned_selection
b26=joblib.load(ROOT/'full_bundle/models/b26/B.joblib')
routes={r['scene']:r for r in json.loads((ROOT/'full_bundle/metadata/scene_routes.json').read_text())}
for scene in ['scene0011_00','scene0153_00']:
    r=routes[scene];gs=json.loads(Path(r['bound_geometry']).read_text())
    evidence={e['geometry_key']:e for e in [json.loads(l) for l in Path(r['own_evidence']).read_text().splitlines() if l.strip()]}
    with np.load(r['bound_points']) as z: points={g['geometry_key']:z[g['array_key']] for g in gs}
    domain=json.loads(Path(r['bound_geometry']).with_name('receipt.json').read_text())['summary']['point_count']
    # Published project API returns X26 and eligible rows; no GT is read.
    members,x=features_and_members(gs,points,evidence,domain)
    selected,scores=learned_selection(members,x,b26)
    original=ROOT/'full_bundle/results/fi1_core_ablation_val312_20261006_v1/scenes'/scene
    check(np.array_equal(scores,np.load(original/'full_scores.npy')),'Full full-pool bitwise scores: '+scene)
    check(selected==json.loads((original/'full_selected.json').read_text()),'Full K800 exact rows: '+scene)

from tools.fi1_sl_inference import build_pool,load_L,rescore_pool
scene='scene0005_00';r=routes[scene]
domain=json.loads(Path(r['bound_geometry']).with_name('receipt.json').read_text())['summary']['point_count']
binding={'geometry':r['bound_geometry'],'points':r['bound_points'],'evidence':r['own_evidence'],'point_count':domain}
members,x26,scores=build_pool(binding,b26)
sl=ROOT/'sl_bundle/protected_sources/fi1_sl_single200_dev55_20261010_v1'
rows,ca,_,_=rescore_pool(members,x26,scores,*load_L(sl/'model/model.pt',sl/'model/normalization.npz'))
for name,value in [('semantic_SL.json',rows),('class_agnostic_SL.json',ca)]:
    check(value==json.loads((sl/'predictions'/scene/name).read_text()),'SL exact rows '+name)

receipt=dict(complete=True,checks=len(checks),details=checks,Full_scenes=['scene0011_00','scene0153_00'],SL_scene=scene,
             Full_scores_bitwise=True,Full_K800_rows_exact=True,SL_rows_exact=True,GT_read_by_scoring=False,
             GPU_frontend_recomputed=False,train597_retrained=False)
(OUT/'refactor_verified.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='details'}))
