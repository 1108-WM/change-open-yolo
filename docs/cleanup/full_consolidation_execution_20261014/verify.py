"""Verify consolidated routes, models, frames and frozen cached replay."""
from consolidate import ROOT, OUT, rows, sha, write
from pathlib import Path
import json, csv, sys
import numpy as np

checks=[]
def check(condition,label):
    assert condition,label
    checks.append(label)

routes=json.loads((ROOT/'full_bundle/metadata/scene_routes.json').read_text())
fields={'data':'prepared_root','records_resolved':'native_records','actual_front':'gd10_frontend',
        'actual_backend':'backend','actual_own_root':'own','yolosam':'yolosam',
        'bound_geometry':'geometry.json','bound_points':'points.npz','own_evidence':'evidence.jsonl'}
for r in routes:
    base=ROOT/'full_bundle/artifacts/scannet'/r['group']/r['scene']
    for field,name in fields.items():
        target=base/name
        check(target.exists(),str(target))
        check(target.resolve()==Path(r[field]).resolve(),'route identity: '+str(target))
check(len(routes)==964,'964 ScanNet routes')
for r in json.loads((ROOT/'docs/cleanup/full_sl_asset_map_20261014/key_model_identity_verified.json').read_text())['files']:
    p=Path(r['path'])
    if '/worktrees/' in str(p):p=ROOT/'full_bundle/models/dv3_official100/models'/p.name
    elif r['role']=='Full B26':p=ROOT/'full_bundle/models/b26/B.joblib'
    check(sha(p)==r['sha256'],'model hash: '+str(p))
for scene in ['scene0011_00','scene0153_00']:
    predicted=OUT/'replay/full'/scene
    original=ROOT/'full_bundle/results/fi1_core_ablation_val312_20261006_v1/scenes'/scene
    check(json.loads((predicted/'selected.json').read_text())==json.loads((original/'full_selected.json').read_text()),'Full K800 exact rows: '+scene)
    with np.load(predicted/'features_scores.npz') as z:
        check(np.array_equal(z['scores'],np.load(original/'full_scores.npy')),'Full full-pool scores bitwise: '+scene)
for name in ['semantic_SL.json','class_agnostic_SL.json']:
    check(json.loads((OUT/'replay/sl/scene0005_00'/name).read_text())==json.loads((ROOT/'sl_bundle/protected_sources/fi1_sl_single200_dev55_20261010_v1/predictions/scene0005_00'/name).read_text()),'SL exact rows: '+name)
check(json.loads((OUT/'frame_interface_verified.json').read_text())['count']==972,'972 frame contracts replay')
for r in json.loads((OUT/'deletion_executed.json').read_text())['deleted']:
    if r['path']=='_external/ESAM':
        check((ROOT/'_external/ESAM/ESAM-main/data/scannet200/meta_data').is_symlink(),'ESAM metadata-only alias')
        check(not (ROOT/'_external/ESAM/ESAM-main/oneformer3d').exists(),'ESAM implementation retired')
    else:check(not (ROOT/r['path']).exists(),'retired absent: '+r['path'])
check(not (ROOT/'pretrained/yoloe/mobileclip_blt.pt').exists(),'MobileCLIP retired')
check((ROOT/'full_bundle/provenance/legacy_val48_GD_source/frontend/arms/D').is_dir(),'GD5 D lineage preserved')
write(OUT/'verification.json',dict(complete=True,checks=len(checks),scannet_scenes=len(routes),
      Full_replayed_scenes=['scene0011_00','scene0153_00'],SL_replayed_scenes=['scene0005_00'],
      bitwise_full_scores=True,bitwise_selected_scores=True,full_GPU_recompute=False,train597_rerun=False))
print(json.dumps(dict(complete=True,checks=len(checks)),ensure_ascii=False))
