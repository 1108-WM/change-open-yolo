#!/usr/bin/env python3
"""Evaluate frozen paired predictions, keeping approved roles separate.

Reconstruct cached labels from raw mesh/segments/aggregation before evaluation.
No predictions or semantic protocol are changed based on labels or AP.
"""
import argparse
from collections.abc import Mapping
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.prepare_fi1_r0_r1_inputs import load_partition, digest, read, sha, write
from tools.fi1_r0_r1_protocol import PROTOCOL
from tools.prepare_scannet200_train_scene import DEFAULT_LABEL_MAP, _ply_arrays, _raw_category_to_id


def verified(files):
    for p,h in files.items():
        if sha(p) != h:
            raise ValueError(f'frozen file changed: {p}')


def prediction_geometry_paths(receipt, binding, scene):
    """Bind verification and consumption to the same canonical scene files."""
    paths = {}
    for field,name in [('geometry_file','geometry.json'),('points_file','points.npz')]:
        path = Path(receipt[field]).resolve()
        if path != (binding/scene/name).resolve():
            raise ValueError(f'{scene}: verified and consumed {field} paths differ')
        paths[field] = path
    return paths


def reconstruct_gt(scene, raw_root, processed_path, label_map, cached_path):
    raw = raw_root/scene
    mesh = raw/f'{scene}_vh_clean_2.ply'
    labels = raw/f'{scene}_vh_clean_2.labels.ply'
    segments = raw/f'{scene}_vh_clean_2.0.010000.segs.json'
    aggregation = raw/f'{scene}.aggregation.json'
    xyz,_,_ = _ply_arrays(mesh,normals=False)
    label_xyz,_,sem = _ply_arrays(labels,normals=False)
    prepared = np.load(processed_path,mmap_mode='r',allow_pickle=False)
    if sem is None or not np.array_equal(xyz,label_xyz) or not np.array_equal(xyz,prepared[:,:3]):
        raise ValueError('raw GT/prepared XYZ content or order differs')
    seg = np.asarray(read(segments)['segIndices'],dtype=np.int64)
    if seg.shape != (len(xyz),):
        raise ValueError('segment point domain differs')
    categories = _raw_category_to_id(label_map)
    sem = np.asarray(sem,dtype=np.int64).copy()
    instances = np.full(len(xyz),-1,dtype=np.int64)
    missing = set()
    for group in read(aggregation)['segGroups']:
        occupied = np.isin(seg,np.asarray(group['segments'],dtype=np.int64))
        instances[occupied] = int(group['id'])
        category = str(group['label'])
        if category not in categories:
            missing.add(category)
        sem[occupied] = categories.get(category,0)
    expected = sem*1000+instances+1
    actual = np.loadtxt(cached_path,dtype=np.int64)
    if not np.array_equal(expected,actual):
        raise ValueError('cached evaluator labels differ from raw annotation reconstruction')
    return {'point_count':len(actual),'missing_categories':sorted(missing),
            'raw_gt_reconstruction_equal':True,
            'files':{str(p):sha(p) for p in [mesh,labels,segments,aggregation,processed_path,label_map,cached_path]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--gt-root',type=Path,required=True)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--label-map',type=Path,default=DEFAULT_LABEL_MAP)
    parser.add_argument('--roles',nargs='+',choices=['train','development_validation','diagnostic'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-gt-evaluation',action='store_true')
    args = parser.parse_args()
    if not args.allow_gt_evaluation:
        raise ValueError('explicit evaluation flag required')
    for k,v in vars(args).items():
        if isinstance(v,Path):
            setattr(args,k,v.resolve())
    from evaluate.scannet200 import eval_semantic_instance as evaluator
    partition = load_partition(args.split)
    config = read(args.sources)
    summary = read(args.predictions/'summary.json')
    if not summary['complete'] or summary['smoke'] or summary['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('full frozen paired prediction output required')
    if read(args.predictions/'protocol.json') != PROTOCOL:
        raise ValueError('prediction protocol differs')
    evidence_identity = read(Path(summary['evidence_root'])/'identity.json')
    if summary['evidence_identity_sha256'] != digest(evidence_identity):
        raise ValueError('prediction evidence identity differs')
    view_identity = read(Path(evidence_identity['views'])/'identity.json')
    if evidence_identity['views_identity_sha256'] != digest(view_identity):
        raise ValueError('view identity no longer matches the prediction evidence')
    from evaluate.scannet200.scannet_constants import CLASS_LABELS_200, VALID_CLASS_IDS_200
    id_to_name = dict(zip(VALID_CLASS_IDS_200, CLASS_LABELS_200))
    evaluator_names = [id_to_name[evaluator.PRED_ID_TO_ID[i]] for i in range(198)]
    if evaluator_names != view_identity['class_names']:
        raise ValueError('semantic class indexes differ from evaluator label mapping')
    binding = Path(view_identity['binding'])
    bi = read(binding/'identity.json')
    if view_identity['binding_identity_sha256'] != digest(bi):
        raise ValueError('binding identity no longer matches the view export')
    if bi['partition_id'] != digest(partition) or bi['sources_sha256'] != sha(args.sources):
        raise ValueError('prediction split/source identity differs')
    source = bi['source']
    available = {r['scene_name'] for r in summary['rows']}
    roles = {role:partition['roles'][role] for role in args.roles}
    if len(args.roles) != len(set(args.roles)) or any(not set(scenes)<=available for scenes in roles.values()):
        raise ValueError('complete role coverage required; do not silently subset a role')
    selected_scenes = sorted({s for scans in roles.values() for s in scans})
    if not set(selected_scenes)<=set(partition['source_scans'][source]):
        raise ValueError('this evaluator requires a single source GT domain')
    args.output.mkdir(parents=True,exist_ok=False)
    source_gt = read(args.gt_root/'summary.json')
    if source_gt['label_map_sha256'] != sha(args.label_map):
        raise ValueError('GT label map identity differs')
    gt_files = {r['scene_name']:r for r in source_gt['scenes']}
    manifests, geometry, domains, identities, packed_points = {}, {}, {}, [], {}
    for scene in selected_scenes:
        receipt = read(args.predictions/scene/'receipt.json')
        if receipt['protocol_sha256'] != digest(PROTOCOL):
            raise ValueError('scene prediction protocol differs')
        evidence_receipt = Path(summary['evidence_root'])/scene/'receipt.json'
        if receipt['evidence_receipt_sha256'] != sha(evidence_receipt):
            raise ValueError('prediction scene evidence receipt link differs')
        geometry_paths = prediction_geometry_paths(receipt,binding,scene)
        verified({str(args.predictions/scene/k):h for k,h in receipt['files'].items()})
        verified({receipt['geometry_file']:receipt['geometry_sha256'],receipt['points_file']:receipt['points_sha256']})
        geometry[scene] = {r['geometry_key']:r for r in read(geometry_paths['geometry_file'])}
        packed_points[scene] = geometry_paths['points_file']
        processed = Path(config['sources'][source]['prepared_root'])/scene/f'{scene[5:]}.npy'
        gt_path = args.gt_root/f'{scene}.txt'
        if sha(gt_path) != gt_files[scene]['sha256']:
            raise ValueError('historical GT file changed')
        bound_gt = reconstruct_gt(scene,args.raw_root,processed,args.label_map,gt_path)
        domains[scene] = bound_gt['point_count']
        identities.append({'scene_name':scene,**bound_gt,'prediction_receipt_sha256':sha(args.predictions/scene/'receipt.json')})
        for arm in ['R0','R1']:
            with (args.predictions/scene/f'{arm}_selected.jsonl').open() as f:
                rows = [json.loads(line) for line in f]
            if len(rows)>600 or any(r['class_index'] not in range(198) or not np.isfinite(r['score']) for r in rows):
                raise ValueError('invalid selected foreground predictions')
            manifests[scene,arm] = rows
        print(f'GT reconstructed and bound: {scene}',flush=True)
    write(args.output/'input_identity.json',{'partition_id':digest(partition),'protocol_sha256':digest(PROTOCOL),
        'prediction_summary_sha256':sha(args.predictions/'summary.json'),'gt_summary_sha256':sha(args.gt_root/'summary.json'),
        'rows':identities,'roles':roles,'evaluator_sha256':sha(evaluator.__file__),
        'class_index_to_dataset_id':{str(i):int(evaluator.PRED_ID_TO_ID[i]) for i in range(198)},
        'code_sha256':sha(__file__),'upstream_interpretation':'fixed upstream development; not whole-system OOF',
        'historical_rgb_content_identity_verified':False,'new_training':False})

    class Predictions(Mapping):
        def __init__(self,scenes,arm):
            self.scenes,self.arm = scenes,arm
        def __len__(self):
            return len(self.scenes)
        def __iter__(self):
            return iter(self.scenes)
        def __getitem__(self,scene):
            if scene not in self.scenes:
                raise KeyError(scene)
            rows = manifests[scene,self.arm]
            masks = np.zeros((domains[scene],len(rows)),dtype=bool)
            with np.load(packed_points[scene],allow_pickle=False) as arrays:
                for i,row in enumerate(rows):
                    masks[arrays[geometry[scene][row['geometry_key']]['array_key']],i] = True
            return {'pred_masks':masks,'pred_classes':np.array([r['class_index'] for r in rows],dtype=np.int64),
                    'pred_scores':np.array([r['score'] for r in rows],dtype=np.float64)}

    results = {}
    for role,scenes in roles.items():
        results[role] = {}
        for arm in ['R0','R1']:
            started = time.monotonic()
            with (args.output/f'{role}_{arm}.log').open('x') as log, contextlib.redirect_stdout(log):
                averages,_,_,_ = evaluator.evaluate(Predictions(scenes,arm),str(args.gt_root),str(args.output/f'{role}_{arm}.csv'),dataset='scannet200')
            result = {k:float(averages[v]) for k,v in {'ap':'all_ap','ap50':'all_ap_50%','ap25':'all_ap_25%'}.items()}
            result.update(seconds=time.monotonic()-started,scene_count=len(scenes))
            results[role][arm] = result
            write(args.output/'progress.json',results)
            print(role,arm,result,flush=True)
    write(args.output/'summary.json',{'complete':True,'AP':results,'ground_truth_read':True,'new_training':False,
        'protocol_sha256':digest(PROTOCOL),'input_identity_sha256':sha(args.output/'input_identity.json'),
        'fixed_upstream_development':True,'whole_system_oof':False,'validation_is_blind':False,
        'diagnostic_is_independent_validation':False,'R2_R3_R4_completed':False})


if __name__ == '__main__':
    main()
