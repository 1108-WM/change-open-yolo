#!/usr/bin/env python3
"""Audit frozen four-arm geometry, export common semantics, then evaluate.

GT is read only by `evaluate`. The no-GT audit freezes inputs and the common
30-view, label-map, multi-label top-600 protocol before any performance result.
The legacy commands evaluate frontend candidates, not the full learned stack.
`full-controls` reuses the frozen full200 scoring kernel for val48 N/H controls.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fi1_frontend_protocol import ARMS, digest, freeze_json, sha256_file, write_json

PROTOCOL = {
    "version": 1,
    "semantic_views": "same frozen A/C uniform30 frames for native and every arm",
    "semantic_evidence": "cached YOLO-World; original overwrite label maps; no GD labels",
    "score": "foreground point votes divided by maximum foreground class vote per geometry",
    "unknown": "no foreground votes gives background index198 score1, as original topk branch",
    "prediction_budget": 600,
    "ties": "descending score, then native-first geometry insertion index, then class index",
    "geometry_dedup": "exact point-set equality only; preserve first native/track member",
    "semantic_members": "199-way distributions, including background; global top600, no top1 collapse",
    "geometry_thresholds": [0.25, 0.5, 0.75, 0.9],
    "comparison": "strict IoU > threshold; GT valid198 classes and predictions both >=100 points",
    "fragment": "candidate precision>=0.8 and 0.1<=GT coverage<0.5, at least two candidates per GT",
    "redundancy": "fraction of tracks with point IoU>0.5 to another track; no removal",
    "interpretation": "20-scene development frontend diagnostic, not full FI1 learned pipeline or val312",
    "learning": "none; no harm/union/refinement heads, Alpha or Qwen; no GT-selected rules",
}


def read(path):
    return json.loads(Path(path).read_text())


def raw_frame_observations(manifest, observations, tracks, arm):
    """Adapt only C-stage indices; preserve masks, point paths, IDs and lineage.

    With C's frequency=1, WORLD_2_CAM[index] addresses that raw video frame.
    This does not sample additional observations or change semantic view pools.
    """
    frames = manifest['frames']
    by_index = {int(f['frame_index']): f for f in frames}
    if len(by_index) != len(frames) or len({f['frame_id'] for f in frames}) != len(frames):
        raise ValueError('duplicate manifest frame identity')
    allowed = set(map(str, manifest['arms'][arm]))
    converted, lookup = [], {}
    for observation in observations:
        frame = by_index.get(int(observation['frame_index']))
        if frame is None or str(observation['frame_id']) != str(frame['frame_id']):
            raise ValueError('observation raw frame and manifest index disagree')
        if frame['frame_id'] not in allowed or not frame['pose_valid']:
            raise ValueError('observation outside arm or from invalid pose')
        oid = int(observation['observation_id'])
        if oid in lookup:
            raise ValueError('duplicate observation identity')
        lookup[oid] = observation
        converted.append({**observation, 'frame_index': int(frame['frame_id'])})
    for track in tracks:
        ids = list(map(int, track['observation_ids']))
        if len(ids) != len(set(ids)) or any(i not in lookup for i in ids):
            raise ValueError('track has duplicate or missing parent observations')
        if {str(lookup[i]['frame_id']) for i in ids} != set(map(str, track['frame_ids'])):
            raise ValueError('track source frames differ from actual parents')
    return converted


def full_backend_modules():
    """Use existing implementations from the complete original deployment worktree."""
    import importlib
    import tools
    workspace = ROOT / 'tools'
    if str(workspace) not in tools.__path__:
        tools.__path__.append(str(workspace))
    names = {
        'competition': 'build_track_native_competition_ledger',
        'duplicate': 'build_track_native_mutual_duplicate_plan',
        'filter': 'materialize_track_native_mutual_duplicate_filter',
        'semantics': 'annotate_automatic_mask_tracks_yoloworld',
        'gvc': 'build_c1_gvc_paper_reference_quality_ledger',
        'relation': 'build_dm_sms1_no_gt_relation_ledger',
        'champion': 'build_dm_sms1_frozen_champion_plan',
        'unique': 'build_dm_sms1_unique_geometry_ledger',
        'dv3': 'build_fi1_d_v3_frozen_inference_plan',
    }
    return {key: importlib.import_module('tools.'+name) for key, name in names.items()}


def bind_gvc_track_frame_counts(records, tracks, frame_map):
    """Separate actual support cardinality from the fixed GVC verification pool.

    The original quality model's source_frame_count is len(track.frame_ids).
    Legacy GVC counted only the intersection, equivalent on its uniform30 data.
    Preserve that intersection explicitly without changing any GVC measurement.
    """
    by_id = {int(t['track_id']):t for t in tracks}
    if len(by_id) != len(tracks):
        raise ValueError('duplicate GVC track identity')
    output, seen = [], set()
    for row in records:
        result = dict(row)
        if row['candidate_source'] == 'd2b_track':
            tid = int(row['candidate_id'])
            if tid not in by_id or tid in seen:
                raise ValueError('GVC track identity mismatch')
            seen.add(tid)
            frames = list(map(str,by_id[tid]['frame_ids']))
            if len(frames) != len(set(frames)):
                raise ValueError('duplicate raw support frame')
            inside = len(set(frames) & set(frame_map))
            outside = len(frames)-inside
            if row['source_frame_count'] != inside or row['unresolved_source_frame_id_count'] != outside:
                raise ValueError('GVC source intersection disagrees with raw support')
            result.update(source_frame_count=len(frames), source_frames_in_gvc_pool_count=inside,
                          source_frames_outside_gvc_pool_count=outside)
        output.append(result)
    if seen != set(by_id):
        raise ValueError('GVC track coverage incomplete')
    return output


def full_geometry(args):
    """Rebuild candidate-dependent legacy and D-v3 geometry; no GT/AP or R3b yet."""
    import copy
    import yaml
    from types import SimpleNamespace as NS
    from threadpoolctl import threadpool_limits
    from tools.run_fi1_frontend_comparison import check_point_order
    from tools.fi1_g10_closure_selection import fixed_selection, NATIVE_BINDING_ROLES
    from tools.prepare_fi1_r3b_val312 import no_gt_io, verify, require
    from tools.run_fi1_r3b_val312 import jsonl, save_rows
    import utils
    modules = full_backend_modules()
    import sklearn
    import torch
    require(sklearn.__version__ == '1.3.0',
            'legacy geometry requires the existing py310/sklearn1.3.0 inference environment')
    no_gt_io()
    selection = fixed_selection(args.selection_root)
    require(args.scenes and set(args.scenes) <= set(selection['scenes']), 'explicit fixed scene subset required')
    require(len(set(args.scenes)) == len(args.scenes), 'duplicate requested scenes')
    scenes = [s for s in selection['scenes'] if s in args.scenes]
    require(args.arms and len(set(args.arms)) == len(args.arms) and set(args.arms) <= set(ARMS), 'invalid arms')
    arms = [a for a in ARMS if a in args.arms]
    front, out = args.run.resolve(), args.output.resolve()
    require(out != front and not front.is_relative_to(out) and not out.is_relative_to(front), 'backend needs separate output')
    historical = args.legacy_root.resolve()
    native = historical/'01_native_cache'
    common = historical/'03_d1_hierarchy_safe'
    yolosam = historical/'11_yoloworld_sam_uniform30'
    config_path = args.backend_config.resolve()
    config = yaml.safe_load(config_path.read_text())
    require(config['openyolo3d']['frequency'] == 10, 'shared semantic pool must retain frequency10')
    dv3_models, champion_models = args.dv3_models.resolve(), args.champion_models.resolve()
    require(sha256_file(dv3_models/'summary.json') == '757ebcaa5744f5a765be533c732a0f6eaa821a9bdff22ff9669d322c44fa92f0', 'D-v3 package differs')
    require(sha256_file(champion_models/'model_package.pkl') == 'b5f91148eb9cf310bee25d1ee6237699965cd368b04a3bb3a3d143aa395652a8', 'legacy package differs')
    source_files = {str(Path(__file__).resolve()): sha256_file(__file__)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and (name.startswith('tools.') or name == 'utils'):
            source_files[str(Path(path).resolve())] = sha256_file(path)
    for p in [config_path, front/'contract.json', front/'selection.json',
              args.selection_root/'selection_receipt.json',
              modules['dv3'].PROJECT_ROOT/'docs/NCS_FI1_STAGE_B_PREREGISTRATION_20260822.md',
              ROOT/'docs/FI1_VAL48_FRONTEND_FULL_PIPELINE_PROTOCOL_20260914.md']:
        source_files[str(p.resolve())] = sha256_file(p)
    for directory in [dv3_models, champion_models]:
        for p in directory.iterdir():
            if p.is_file() and p.suffix in ('.json', '.pkl', '.joblib'):
                source_files[str(p)] = sha256_file(p)
    contract = {
        'version': 1, 'selection': selection, 'scenes': scenes, 'arms': arms,
        'runtime': {'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
                    'sklearn':sklearn.__version__,'torch':torch.__version__},
        'sources': source_files, 'modules': {k: str(Path(v.__file__).resolve()) for k,v in modules.items()},
        'common_gvc_frames': str(common), 'common_yoloworld_sam': str(yolosam),
        'native': str(native), 'dataset': str(args.dataset_root.resolve()),
        'track_semantics': 'original kernel; source raw IDs intersect frequency10 YOLO pool; bookkeeping only',
        'C_frames': 'original raw frame index with frequency1 path addressing; actual arm observations only',
        'GVC': 'same frozen30 YOLO-SAM observations; exclude actual source raw frames',
        'GVC_source_frame_count': 'total actual track raw frames; retain separate in/out verification-pool counts',
        'new_candidate_quality_and_relations': True, 'GT_read': False, 'AP_run': False,
        'new_training': False, 'Qwen': False, 'R3b_scoring_complete': False,
    }
    freeze_json(out/'identity.json', contract)
    # Keep a reviewable historical implementation when the original file evolves.
    import shutil
    (out/'source_snapshot').mkdir(exist_ok=True)
    for p,h in source_files.items():
        if Path(p).suffix == '.py':
            snapshot = out/'source_snapshot'/(h+'_'+Path(p).name)
            if not snapshot.exists():
                shutil.copyfile(p,snapshot)
            require(sha256_file(snapshot) == h, 'source snapshot differs')
    records = []
    for scene in scenes:
        manifest_path = front/'manifests'/f'{scene}.json'
        manifest = read(manifest_path)
        require(manifest['scene_name'] == scene, 'manifest scene mismatch')
        require(Path(manifest['files']['processed']['path']).parent.parent == args.dataset_root.resolve(), 'processed domain differs')
        point_count = check_point_order(manifest)
        for arm in arms:
            dest = out/arm/scene
            if (dest/'receipt.json').exists():
                receipt = read(dest/'receipt.json')
                require(receipt['identity_sha256'] == digest(contract), 'backend identity changed')
                verify(receipt['inputs']); verify({str(dest/n):h for n,h in receipt['outputs'].items()})
                records.append(receipt['summary']); continue
            inputs = {str(manifest_path): sha256_file(manifest_path)}
            for item in manifest['files'].values():
                inputs[item['path']] = item['sha256']
            for frame in manifest['frames']:
                for name in ['color', 'depth', 'pose']:
                    inputs[frame[name]['path']] = frame[name]['sha256']
                for name,folder,suffix in [('color','color','.jpg'),('depth','depth','.png'),('pose','poses','.txt')]:
                    require(Path(frame[name]['path']) == args.dataset_root.resolve()/scene/folder/(frame['frame_id']+suffix),
                            'manifest path does not address declared raw frame')
            arm_identity = digest({'contract': digest(read(front/'contract.json')), 'manifest': digest(manifest),
                                   'frontend': ARMS[arm]['frontend'], 'arm': arm})
            for stage in ['hierarchy', 'tracks', 'consensus', 'merged']:
                rp = front/'arms'/arm/'stage_receipts'/f'{scene}_{stage}.json'
                r = read(rp)
                require(r['identity'] == arm_identity, 'frontend stage identity mismatch')
                inputs[str(rp)] = sha256_file(rp)
                inputs.update({str(front/'arms'/arm/stage/scene/n): h for n,h in r['files'].items()})
            merged_path = front/'arms'/arm/'merged'/scene/'automatic_tracks.json'
            track_payload = read(merged_path)
            source_tracks = track_payload['tracks']
            obs_path = front/'arms'/arm/'hierarchy'/scene/'automatic_observations.jsonl'
            observations = jsonl(obs_path)
            raw_observations = raw_frame_observations(manifest, observations, source_tracks, arm)
            for row in [*observations, *source_tracks]:
                p = Path(row.get('point_indices_path', row.get('points_path')))
                inputs[str(p)] = sha256_file(p)
                checked_points(p, point_count)
            # Bind shared verification evidence, not historical candidate-specific ledgers.
            for p in [common/scene/'summary.json', common/scene/'automatic_observations.jsonl',
                      yolosam/scene/'observations.jsonl']:
                inputs[str(p)] = sha256_file(p)
            for row in jsonl(yolosam/scene/'observations.jsonl'):
                p = Path(row['point_indices_path']); inputs[str(p)] = sha256_file(p)
                require(int(row['frame_id']) == 10*int(row['frame_index']), 'common YOLO-SAM frame mismatch')
            anchors, frame_map = modules['gvc']._frame_contract(common/scene)
            require(all(int(raw) == 10*idx for raw,idx in frame_map.items()), 'common GVC index mismatch')
            for idx in anchors:
                for folder, suffix in [('poses','.txt'), ('depth','.png'), ('color','.jpg')]:
                    p = args.dataset_root/scene/folder/f'{10*idx}{suffix}'; inputs[str(p)] = sha256_file(p)
            cache = args.yolo_cache_root/scene/f'{scene}.pt'
            inputs[str(cache)] = sha256_file(cache)
            cache_receipt = cache.parent/'receipt.json'
            cache_record = read(cache_receipt)
            require(cache_record['summary']['complete'] and cache_record['summary']['scene_name'] == scene,
                    'YOLO source receipt incomplete/mismatched')
            require(cache_record['outputs'][str(cache.resolve())] == sha256_file(cache), 'YOLO receipt hash differs')
            inputs[str(cache_receipt)] = sha256_file(cache_receipt)
            additional = selection.get('dataset_role') in NATIVE_BINDING_ROLES
            old_binding_receipt = (args.full_preparation/'native_bindings'/scene/'receipt.json' if additional
                                   else args.full_preparation/'geometry_binding'/scene/'receipt.json')
            old_binding = read(old_binding_receipt)
            if additional:
                require(old_binding.get('dataset_role') == selection['dataset_role']
                        and old_binding.get('scene') == scene, 'additional native receipt mismatch')
                verify(old_binding['input_files'])
            inputs[str(old_binding_receipt)] = sha256_file(old_binding_receipt)
            for name in ['masks','classes','scores']:
                p = native/f'{scene}_pred_{name}.npy'
                bound_native = args.full_preparation.resolve()/'records_adapter'/scene/'native_cache'/p.name
                require(old_binding['input_files'][str(bound_native)] == sha256_file(p), 'native inputs differ from frozen H/N')
                inputs[str(p)] = sha256_file(p)
            verify(inputs)
            if dest.exists():
                raise FileExistsError(f'incomplete backend retained for audit: {dest}')
            dest.mkdir(parents=True)
            write_json(dest/'input_identity.json', inputs)
            (dest/'scene.txt').write_text(scene+'\n')
            cconfig = copy.deepcopy(config); cconfig['openyolo3d']['frequency'] = 1
            (dest/'stage_c_raw_frames.yaml').write_text(yaml.safe_dump(cconfig))
            automatic = dest/'automatic_raw_frames'/scene
            automatic.mkdir(parents=True)
            save_rows(automatic/'automatic_observations.jsonl',raw_observations)
            write_json(automatic/'frame_mapping.json', {
                'rows': [{'manifest_index':f['frame_index'], 'raw_frame_id':f['frame_id'],
                          'C_index':int(f['frame_id']), 'pose_valid':f['pose_valid']} for f in manifest['frames']],
                'observation_count':len(observations), 'observation_payload_changed_fields':['frame_index'],
                'new_frames_sampled':False})
            try:
                with threadpool_limits(limits=4):
                    masks = np.load(native/f'{scene}_pred_masks.npy', mmap_mode='r')
                    scores = np.load(native/f'{scene}_pred_scores.npy')
                    require(masks.shape[0] == point_count, 'native point domain mismatch')
                    sizes = np.count_nonzero(masks,axis=0)
                    relations, plans = [], []
                    for track in source_tracks:
                        rows, summary = modules['competition'].track_native_relations(
                            checked_points(track['points_path'], point_count), masks, sizes, scores,
                            track['proposal_id'], track['mean_node_quality'], 'hierarchy_safe_d2b')
                        relations.extend(rows)
                        plans.append(modules['duplicate'].plan_proposal(summary,rows,'hierarchy_safe_d2b'))
                    retained, actions, ledger = modules['filter'].filter_tracks(
                        source_tracks,plans,geometry_variant='hierarchy_safe_d2b')
                    trackroot = dest/'filtered_tracks'
                    write_json(trackroot/scene/'automatic_tracks.json',{**track_payload,'tracks':retained})
                    save_rows(dest/'track_native_relations.jsonl',relations)
                    save_rows(dest/'duplicate_plan.jsonl',plans)
                    save_rows(dest/'filter_ledger.jsonl',ledger)
                    save_rows(dest/'filter_actions.jsonl',actions)
                    print('full backend filtered',arm,scene,len(source_tracks),len(retained),flush=True)
                    common_args = dict(dataset_root=args.dataset_root.resolve(),config=config,
                        depth_scale=float(config['openyolo3d']['depth_scale']),track_root=trackroot)
                    semantic = modules['semantics']._scene_records(scene,NS(**common_args,bboxes_2d_root=cache.parent))
                    write_json(dest/'semantics'/scene/'automatic_track_yoloworld_semantics.json',semantic)
                    gvc_rows, gvc_summary = modules['gvc']._scene_records(scene,NS(**common_args,
                        automatic_root=common,yoloworld_sam_root=yolosam,native_prediction_cache=native,
                        max_views=10,min_visible_points=30))
                    gvc_rows = bind_gvc_track_frame_counts(gvc_rows,retained,frame_map)
                    write_json(dest/'gvc'/scene/'c1_gvc_quality_ledger.json',gvc_rows)
                    write_json(dest/'gvc'/scene/'summary.json',gvc_summary)
                    write_json(dest/'gvc'/'summary.json',{'scene_count':1,'ground_truth_usage':'none','ground_truth_read':False,'ap_evaluation_run':False})
                    print('full backend GVC',arm,scene,len(gvc_rows),flush=True)
                    shared = dict(scene_list=dest/'scene.txt',native_cache=native,track_root=trackroot,
                        quality_ledger_root=dest/'gvc',expected_scene_count=1,max_scenes=None)
                    modules['relation'].run(NS(**shared,automatic_root=common,yoloworld_sam_root=yolosam,
                        dataset_root=args.dataset_root.resolve(),config_path=config_path,output_root=dest/'relations',resume=False))
                    modules['champion'].run(NS(**shared,relation_root=dest/'relations',model_root=champion_models,
                        output_root=dest/'champion'))
                    modules['unique'].run(NS(scene_list=dest/'scene.txt',native_cache=native,track_root=trackroot,
                        records_root=args.full_preparation/'records_adapter',plan_root=dest/'champion',
                        semantic_root=dest/'semantics',output_dir=dest/'unique',
                        preregistration_path=ROOT/'docs/FI1_VAL48_FRONTEND_FULL_PIPELINE_PROTOCOL_20260914.md',
                        expected_scene_count=1,max_scenes=None,class_count=198))
                    dv3_summary = modules['dv3'].run(NS(scene_list=dest/'scene.txt',prepared_root=args.dataset_root.resolve(),
                        unique_geometry_root=dest/'unique',gvc_root=dest/'gvc',relation_root=dest/'relations',
                        champion_plan_root=dest/'champion',track_root=trackroot,automatic_root=automatic.parent,
                        config_path=dest/'stage_c_raw_frames.yaml',model_root=dv3_models,output_root=dest/'dv3',
                        min_points=100,dataset_name=('ScanNet200-' + selection['dataset_role'] + '-' if additional else 'ScanNet200-fixed-val48-')+arm))
                verify(inputs); verify(source_files)
                record = {'scene':scene,'arm':arm,'source_tracks':len(source_tracks),'retained_tracks':len(retained),
                          'D_v3':dv3_summary,'new_AP':False,'own_semantics_and_R3b_complete':False}
                write_json(dest/'receipt.json',{'identity_sha256':digest(contract),'inputs':inputs,
                    'outputs':{str(p.relative_to(dest)):sha256_file(p) for p in dest.rglob('*') if p.is_file()},'summary':record})
                records.append(record)
                print('full backend geometry complete',arm,scene,flush=True)
            except Exception as error:
                write_json(dest/'failure.json',{'error':repr(error),'identity_sha256':digest(contract),'complete':False})
                raise
    write_json(out/'summary.json',{'requested_geometry_complete':True,'records':records,'new_AP':False,
        'full48_ABCD_geometry_complete':len(records)==192,'own_semantics_and_R3b_complete':False,
        'identity_sha256':digest(contract)})


def full_bind(args):
    """Audit actual new pointsets and expose the existing own-evidence CLI contract."""
    from tools.audit_fi1_r0_r1_geometry import audit_scene
    from tools.prepare_fi1_r3b_val312 import no_gt_io, verify, require
    from tools.fi1_g10_closure_selection import fixed_selection, NATIVE_BINDING_ROLES
    from tools.run_fi1_r3b_val312 import jsonl
    no_gt_io()
    backend, out = args.backend_root.resolve(), args.output.resolve()
    old = read(backend/'identity.json')
    require(read(backend/'summary.json')['requested_geometry_complete'], 'backend incomplete')
    require(old['selection'] == fixed_selection(args.selection_root), 'backend selection differs')
    require(old['dataset'] == str(args.dataset_root.resolve()), 'binding dataset differs from backend')
    # Verify recorded historical code, without silently re-signing it as current code.
    for path,h in old['sources'].items():
        p = backend/'source_snapshot'/(h+'_'+Path(path).name) if Path(path).suffix == '.py' else Path(path)
        require(sha256_file(p) == h, 'backend source snapshot/input changed')
    base_sources = read(args.full_preparation/'inventory/input_sources.json')
    outputs = []
    for arm in old['arms']:
        dest = out/arm
        records = dest/'records'
        for scene in old['scenes']:
            folder = records/scene
            folder.mkdir(parents=True,exist_ok=True)
            for link,target in [(folder/'native_cache',args.full_preparation.resolve()/'records_adapter'/scene/'native_cache'),
                                (folder/'yoloworld_bboxes_2d',args.yolo_cache_root.resolve()/scene)]:
                if not link.exists():
                    link.symlink_to(target, target_is_directory=True)
                require(link.resolve() == target.resolve(), 'records adapter target differs')
        additional = old['selection'].get('dataset_role') in NATIVE_BINDING_ROLES
        source = (old['selection']['dataset_role'] + '_' if additional else 'val48_')+arm
        base_source = old['selection']['dataset_role'] if additional else 'val312'
        spec = {'prepared_root':str(args.dataset_root.resolve()), 'records_root':str(records),
                'legacy_geometry_root':str(backend/arm), 'alpha_cache':base_sources['sources'][base_source]['alpha_cache']}
        sources = {**base_sources,'sources':{source:spec}}
        freeze_json(dest/'sources.json',sources)
        binding = dest/'geometry_binding'
        identity = {'version':'val48_actual_geometry_binding_v1','source':source,'scenes':old['scenes'],
                    'sources_sha256':sha256_file(dest/'sources.json'), 'backend_identity_sha256':digest(old),
                    'code':{str(Path(__file__).resolve()):sha256_file(__file__),
                            str(ROOT/'tools/audit_fi1_r0_r1_geometry.py'):sha256_file(ROOT/'tools/audit_fi1_r0_r1_geometry.py')}}
        freeze_json(binding/'identity.json',identity)
        summaries = []
        for scene in old['scenes']:
            src = backend/arm/scene
            receipt = read(src/'receipt.json')
            require(receipt['identity_sha256'] == digest(old),'backend receipt identity differs')
            verify(receipt['inputs']); verify({str(src/n):h for n,h in receipt['outputs'].items()})
            target = binding/scene
            if (target/'receipt.json').exists():
                prior = read(target/'receipt.json')
                require(prior['identity_sha256'] == digest(identity),'binding resume differs')
                verify(prior['input_files']); verify({str(target/n):h for n,h in prior['output_files'].items()})
                summaries.append(prior['summary']); continue
            legacy_path = src/'unique/unique_geometry_ledger.jsonl'
            plan_path = src/'dv3/complete_plan/complete_inference_ap_plan.jsonl'
            rows, packed, files, summary = audit_scene(scene,spec,jsonl(legacy_path),jsonl(plan_path))
            files.update({str(p):sha256_file(p) for p in [src/'receipt.json',legacy_path,plan_path]})
            target.mkdir(parents=True,exist_ok=False)
            write_json(target/'geometry.json',rows)
            np.savez_compressed(target/'points.npz',**packed)
            write_json(target/'receipt.json',{'identity_sha256':digest(identity),'input_files':files,
                'output_files':{p.name:sha256_file(p) for p in target.iterdir()},'summary':summary})
            summaries.append(summary)
        write_json(binding/'summary.json',{'complete':True,'identity':identity,'rows':summaries,
                   'scene_count':len(summaries),'ground_truth_read':False,'ap_computed':False})
        outputs.append({'arm':arm,'binding':str(binding),'sources':str(dest/'sources.json'),
                        'scenes':old['scenes'],'alpha_cache':spec['alpha_cache']})
        print('full actual geometry binding complete',arm,len(summaries),flush=True)
    write_json(out/'summary.json',{'complete':True,'arms':outputs,'new_AP':False,'own_evidence_complete':False})


def validate_full_evidence_identity(binding_identity, evidence_identity, view_identity, reference_evidence, reference_views):
    """Reject same-shaped evidence made for another pointset binding or model."""
    if view_identity['binding_identity_sha256'] != digest(binding_identity):
        raise ValueError('evidence views belong to another geometry binding')
    if evidence_identity['views_identity_sha256'] != digest(view_identity):
        raise ValueError('evidence view identity changed')
    if (evidence_identity['cache'] != reference_evidence['cache']
            or evidence_identity['cache_inputs'] != reference_evidence['cache_inputs']):
        raise ValueError('Alpha evidence model/cache differs from frozen H/N')
    if (view_identity['class_names'] != reference_views['class_names']
            or view_identity['protocol'] != reference_views['protocol']
            or view_identity['config_sha256'] != reference_views['config_sha256']):
        raise ValueError('semantic vocabulary/view protocol differs from frozen H/N')


def execution_groups(selection, batches):
    if (not batches or len(set(batches)) != len(batches)
            or not set(batches) <= set(range(1,7)) or len(selection['scenes']) != 48):
        raise ValueError('execution requires unique fixed batches 1..6 and the frozen48')
    return [(i,selection['scenes'][(i-1)*8:i*8]) for i in range(1,7) if i in batches]


def full_workflow(args):
    """Run the original CLI tools in six fixed batches; stop before GT/AP."""
    import os
    import shutil
    import subprocess
    from tools.run_fi1_frontend_comparison import fixed_selection
    from tools.prepare_fi1_r3b_val312 import require, verify
    selection = fixed_selection(args.selection_root)
    groups = execution_groups(selection,args.batches)
    smoke_path = args.selection_root.resolve()/'full_chain_smoke_audit.json'
    smoke = read(smoke_path)
    require(smoke['complete'] and smoke['scene'] == selection['scenes'][0]
            and set(smoke['arms']) == set(ARMS) and not smoke['new_AP'],
            'all four arms must pass the fixed first-scene chain before batches')
    verify(smoke['files'])
    out = args.output.resolve()
    old_environment = ROOT/'.local/fi1_r0_r4_preparation_20260913/dv3_generation_v2/jobs.json'
    old = read(old_environment)['environment']
    cpu = {**os.environ,**old['environment']}
    cpu['PYTHONPATH'] = str(ROOT)+':'+':'.join(cpu['PYTHONPATH'].split(':')[1:])
    current_cpu = {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    python = str(ROOT/'.venvs/yoloe_gpu/bin/python')
    script = str(Path(__file__).resolve())
    sources = {str(old_environment):sha256_file(old_environment),script:sha256_file(script),
               str(smoke_path):sha256_file(smoke_path)}
    for name in ['run_fi1_frontend_comparison.py','export_fi1_r0_r1_yolo_views.py',
                 'bind_fi1_r0_r1_alpha_cache.py','complete_fi1_r0_r1_alpha.py']:
        p=ROOT/'tools'/name;sources[str(p)]=sha256_file(p)
    identity = {'selection':selection,'groups':[{'batch':i,'scenes':ss} for i,ss in groups],
                'code':sources,'front':str(args.run.resolve()),
                'new_AP':False,'new_training':False,'Qwen':False,'scoring':'frozen R3b full200',
                'geometry_environment':old,'own_semantic_python':python}
    freeze_json(out/'identity.json',identity)
    # Snapshot this adapter as well as leaving historical stage receipts untouched.
    if not (out/'adapter_source.py').exists():
        shutil.copyfile(script,out/'adapter_source.py')
    require(sha256_file(out/'adapter_source.py')==sources[script],'workflow snapshot changed')
    common = ['--selection-root',str(args.selection_root.resolve()),'--full-preparation',str(args.full_preparation.resolve()),
              '--full-execution',str(args.full_execution.resolve()),'--dataset-root',str(args.dataset_root.resolve()),
              '--yolo-cache-root',str(args.yolo_cache_root.resolve())]
    completed=[]
    def command(batch,label,argv,env=None):
        verify(sources)
        require(shutil.disk_usage(out).free > 8*1024**3, 'less than 8 GiB free; retain completed stages and stop')
        path=out/f'batch{batch:02d}'
        path.mkdir(exist_ok=True)
        state={'running':True,'pid':os.getpid(),'batch':batch,'stage':label,'argv':argv,
               'completed_batches':completed,'new_AP':False,'new_training':False,'Qwen':False}
        write_json(out/'execution.json',state)
        print('workflow stage',batch,label,flush=True)
        with (path/(label+'.log')).open('a') as log:
            child=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        if child.returncode:
            write_json(out/'execution.json',{**state,'running':False,'failed':True,'exit_code':child.returncode})
            raise RuntimeError(f'batch {batch}/{label} failed; retained log and all completed stages')
    try:
        for batch,scenes in groups:
            target=out/f'batch{batch:02d}'
            command(batch,'frontend',[python,str(ROOT/'tools/run_fi1_frontend_comparison.py'),'export',
                    '--output',str(args.run.resolve()),'--scenes',*scenes,'--arms','A','B','C','D'])
            command(batch,'geometry',[old['python'],script,'full-geometry',*common,'--run',str(args.run.resolve()),
                    '--output',str(target/'backend'),'--scenes',*scenes,'--arms','A','B','C','D',
                    '--legacy-root',str(args.legacy_root.resolve()),'--backend-config',str(args.backend_config.resolve()),
                    '--dv3-models',str(args.dv3_models.resolve()),'--champion-models',str(args.champion_models.resolve())],cpu)
            command(batch,'bind',[python,script,'full-bind',*common,'--backend-root',str(target/'backend'),
                    '--output',str(target/'own_evidence')],current_cpu)
            for job in read(target/'own_evidence/summary.json')['arms']:
                arm=job['arm'];folder=target/'own_evidence'/arm
                snapshot=ROOT/'.local/fi1_unified_evidence_20260910/workspace'
                command(batch,arm+'_views',[python,str(ROOT/'tools/export_fi1_r0_r1_yolo_views.py'),
                    '--binding',job['binding'],'--sources',job['sources'],'--snapshot',str(snapshot),'--output',str(folder/'views')],current_cpu)
                command(batch,arm+'_alpha_bind',[python,str(ROOT/'tools/bind_fi1_r0_r1_alpha_cache.py'),
                    '--views',str(folder/'views'),'--cache',job['alpha_cache'],'--output',str(folder/'alpha_bound')],current_cpu)
                evidence=folder/'alpha_bound'
                if read(evidence/'summary.json')['alpha_pending_count']:
                    plan=folder/'alpha_completed/plan.json'
                    if not plan.exists():
                        command(batch,arm+'_alpha_plan',[python,str(ROOT/'tools/complete_fi1_r0_r1_alpha.py'),'prepare',
                            '--bound',str(evidence),'--snapshot',str(snapshot),'--output',str(plan.parent)],current_cpu)
                    command(batch,arm+'_alpha_gpu',[python,str(ROOT/'tools/complete_fi1_r0_r1_alpha.py'),'run','--plan',str(plan)])
                    evidence=plan.parent
                command(batch,arm+'_R3b',[python,script,'full-score',*common,'--binding',job['binding'],
                    '--evidence',str(evidence),'--output',str(folder/'predictions')],current_cpu)
            completed.append(batch)
            write_json(target/'complete.json',{'complete':True,'scenes':scenes,'arms':list(ARMS),'new_AP':False})
        write_json(out/'execution.json',{'running':False,'complete':True,'completed_batches':completed,
                   'full48_predictions_complete':completed==list(range(1,7)),'new_AP':False,'new_training':False,'Qwen':False})
    except Exception as error:
        state=read(out/'execution.json') if (out/'execution.json').exists() else {}
        write_json(out/'execution.json',{**state,'running':False,'failed':True,'error':repr(error),'new_AP':False})
        raise


def full_score(args):
    """Use the same frozen 26-feature R3b kernel on complete own-mask evidence."""
    from tools.prepare_fi1_r3b_val312 import BASE, no_gt_io, load_models, score_scene, verify, require
    from tools.run_fi1_r3b_val312 import jsonl, save_rows, validate_selected, code_files
    from threadpoolctl import threadpool_limits
    no_gt_io()
    binding, evidence, out = args.binding.resolve(), args.evidence.resolve(), args.output.resolve()
    bi, ei = read(binding/'identity.json'), read(evidence/'identity.json')
    require(read(binding/'summary.json')['complete'] and read(evidence/'summary.json')['complete'],'scoring inputs incomplete')
    require(set(bi['scenes']) == set(ei['scenes']), 'evidence/binding scene mismatch')
    vi = read(Path(ei['views'])/'identity.json')
    reference_path = args.full_execution.resolve()/'full/alpha_bound/identity.json'
    reference = read(reference_path)
    reference_view_path = Path(reference['views'])/'identity.json'
    validate_full_evidence_identity(bi,ei,vi,reference,read(reference_view_path))
    models, files = load_models(BASE)
    code = {**code_files(),str(Path(__file__).resolve()):sha256_file(__file__)}
    identity = {'version':'val48_R3b_frozen_scoring_v1','binding_identity':digest(bi),
                'evidence_identity':digest(ei),'models':files,'code':code,
                'adapter_sha256':sha256_file(__file__),'new_training':False,'Qwen':False,'new_AP':False}
    freeze_json(out/'identity.json',identity)
    summaries = []
    with threadpool_limits(limits=4):
        for scene in bi['scenes']:
            br, er = read(binding/scene/'receipt.json'), read(evidence/scene/'receipt.json')
            require(br['identity_sha256']==digest(bi) and er['identity_sha256']==digest(ei),'input receipt differs')
            inputs = {**br['input_files'], **{str(binding/scene/n):h for n,h in br['output_files'].items()},
                      **er['inputs'], **{str(evidence/scene/n):h for n,h in er['outputs'].items()}, **ei['cache_inputs']}
            for p in [binding/'identity.json',evidence/'identity.json',binding/scene/'receipt.json',evidence/scene/'receipt.json',
                      Path(ei['views'])/'identity.json',reference_path,reference_view_path]:
                inputs[str(p)]=sha256_file(p)
            verify(inputs)
            target=out/scene
            if (target/'receipt.json').exists():
                prior=read(target/'receipt.json')
                require(prior['identity_sha256']==digest(identity),'score resume identity differs')
                verify(prior['inputs']);verify({str(target/n):h for n,h in prior['outputs'].items()})
                summaries.append(prior['summary']);continue
            geometry=read(binding/scene/'geometry.json')
            rows=jsonl(evidence/scene/'evidence.jsonl')
            own={r['geometry_key']:r for r in rows}
            require(len(own)==len(rows) and set(own)=={g['geometry_key'] for g in geometry},'own evidence key coverage differs')
            for g in geometry:
                r=own[g['geometry_key']]
                require(r['points_sha256']==g['points_sha256'] and r['point_count']==g['point_count'], 'evidence belongs to a different pointset')
                require(r['yolo']['acquisition_complete'] and r['alpha']['acquisition_complete'],'missing own evidence')
            with np.load(binding/scene/'points.npz',allow_pickle=False) as data:
                result=score_scene(geometry,[data[g['array_key']] for g in geometry],own,br['summary']['point_count'],models)
            eligible=result['R1']['eligible'];scores=result['R3b']['scores'];selected=result['R3b']['selected']
            validate_selected(eligible,scores,selected)
            target.mkdir(parents=True,exist_ok=False)
            save_rows(target/'eligible.jsonl',eligible);save_rows(target/'R3b_selected.jsonl',selected)
            np.save(target/'R3b_scores.npy',scores,allow_pickle=False)
            summary={'scene':scene,'geometry_count':len(geometry),'eligible_members':len(eligible),
                     'selected_members':len(selected),'new_AP':False,'new_training':False,'Qwen':False}
            verify(inputs);verify(files);verify(code)
            write_json(target/'receipt.json',{'identity_sha256':digest(identity),'inputs':inputs,
                'outputs':{p.name:sha256_file(p) for p in target.iterdir()},'summary':summary})
            summaries.append(summary)
            print('full frozen R3b scored',scene,summary,flush=True)
    write_json(out/'summary.json',{'complete':True,'records':summaries,'new_AP':False,'identity_sha256':digest(identity)})


def native_only_geometry(geometry):
    """Remove nonnative aliases as well as pointsets; source features must be native-only."""
    output = []
    for g in geometry:
        members = [m for m in g['legacy_members'] if m['candidate_source'] == 'native']
        if members:
            output.append({**g, 'legacy_members': members, 'dv3_plans': []})
    return output


def full_controls(args):
    """Freeze predictions only, without GT/AP; preserve the old common30 evaluator."""
    from threadpoolctl import threadpool_limits
    from tools.run_fi1_frontend_comparison import fixed_selection
    from tools.prepare_fi1_r3b_val312 import BASE, no_gt_io, load_models, score_scene, verify, require
    from tools.run_fi1_r3b_val312 import jsonl, save_rows, validate_selected, code_files
    from tools.fi1_r0_r1_protocol import PROTOCOL as full_protocol
    no_gt_io()
    selection = fixed_selection(args.selection_root)
    prep, execution, out = args.full_preparation.resolve(), args.full_execution.resolve(), args.output.resolve()
    require(out != (ROOT/'.local/fi1_frontend_comparison_20260912/analysis_v1').resolve(), 'full controls require an explicit new output directory')
    models, model_files = load_models(BASE)
    files = {**model_files, **code_files(), str(Path(__file__).resolve()): sha256_file(__file__),
             str(ROOT/'tools/run_fi1_frontend_comparison.py'): sha256_file(ROOT/'tools/run_fi1_frontend_comparison.py')}
    for path in [prep/'geometry_binding/identity.json', execution/'full/alpha_bound/identity.json',
                 execution/'full/alpha_bound/summary.json', execution/'full/predictions/identity.json',
                 execution/'full/predictions/summary.json']:
        files[str(path)] = sha256_file(path)
    require(read(execution/'full/alpha_bound/summary.json')['complete'] and read(execution/'full/predictions/summary.json')['complete'], 'historical source incomplete')
    identity = {'selection': selection, 'full_protocol': full_protocol, 'source_files': files,
                'preparation': str(prep), 'execution': str(execution), 'arms': ['H','N'],
                'scoring_head': 'frozen_R3b_full200', 'new_training': False, 'Qwen': False,
                'GT_read': False, 'new_AP': False, 'N': 'native pointsets and native aliases only; recompute context on N pool'}
    freeze_json(out/'identity.json', identity)
    records = []
    with threadpool_limits(limits=4):
        for scene in selection['scenes']:
            dest = out/scene
            if (dest/'receipt.json').exists():
                prior = read(dest/'receipt.json')
                require(prior['identity_sha256'] == digest(identity), 'control resume identity changed')
                verify(prior['inputs']);verify({str(dest/n):h for n,h in prior['outputs'].items()})
                records.append(prior['summary']);continue
            binding = prep/'geometry_binding'/scene
            evidence = execution/'full/alpha_bound'/scene
            historical = execution/'full/predictions'/scene
            br, er, hr = [read(p/'receipt.json') for p in [binding,evidence,historical]]
            for receipt, path in [(br,prep/'geometry_binding/identity.json'),
                                  (er,execution/'full/alpha_bound/identity.json'),
                                  (hr,execution/'full/predictions/identity.json')]:
                require(receipt['identity_sha256'] == digest(read(path)), 'historical per-scene identity differs')
            old_summary = read(execution/'full/predictions/summary.json')
            require(sha256_file(historical/'receipt.json') == old_summary['receipts'][str(historical/'receipt.json')], 'historical prediction receipt changed')
            inputs = {str(p/'receipt.json'):sha256_file(p/'receipt.json') for p in [binding,evidence,historical]}
            inputs.update({str(binding/n):h for n,h in br['output_files'].items()})
            inputs.update({str(evidence/n):h for n,h in er['outputs'].items()})
            inputs.update({str(historical/n):h for n,h in hr['outputs'].items()})
            verify(inputs)
            geometry = read(binding/'geometry.json')
            own = {r['geometry_key']:r for r in jsonl(evidence/'evidence.jsonl')}
            require(set(own) == {g['geometry_key'] for g in geometry}, 'own evidence coverage differs')
            with np.load(binding/'points.npz',allow_pickle=False) as packed:
                points = {g['geometry_key']:packed[g['array_key']] for g in geometry}
            native = native_only_geometry(geometry)
            require(bool(native), 'native geometry absent')
            if dest.exists():
                dest.rename(out/(scene+'.incomplete_'+str(time.time_ns())))
            counts = {};dest.mkdir(parents=True)
            for arm, gs in [('H',geometry),('N',native)]:
                result = score_scene(gs,[points[g['geometry_key']] for g in gs],
                                     {g['geometry_key']:own[g['geometry_key']] for g in gs},
                                     br['summary']['point_count'],models)
                selected, scores = result['R3b']['selected'],result['R3b']['scores']
                validate_selected(result['R1']['eligible'],scores,selected)
                if arm == 'H':
                    require(np.array_equal(scores,np.load(historical/'R3b_scores.npy',allow_pickle=False)), 'H frozen scores differ')
                    require(selected == jsonl(historical/'R3b_selected.jsonl'), 'H frozen selected members differ')
                else:
                    require(all(not m['expanded_class'] and all(a['source']=='native' for a in m['original_aliases']) for m in result['R1']['eligible']), 'N contains nonnative/expanded member')
                write_json(dest/(arm+'_geometry.json'),gs)
                np.save(dest/(arm+'_scores.npy'),scores)
                for label, rows in [('eligible',result['R1']['eligible']),('selected',selected)]:
                    save_rows(dest/(arm+'_'+label+'.jsonl'),rows)
                counts[arm] = {'geometries':len(gs),'eligible_members':len(scores),'selected_members':len(selected)}
            record = {'scene':scene,'counts':counts,'H_exact_frozen_replay':True,'N_context_recomputed':True}
            verify(inputs)
            write_json(dest/'receipt.json',{'identity_sha256':digest(identity),'inputs':inputs,
                       'outputs':{p.name:sha256_file(p) for p in dest.iterdir() if p.is_file()},'summary':record})
            records.append(record)
            print('full controls H/N replayed',scene,counts,flush=True)
    verify(files)
    write_json(out/'summary.json',{'complete':True,'scene_count':len(records),'records':records,
               'identity_sha256':digest(identity),'new_AP':False,'new_training':False,
               'full_ABCD_ready':False,'H_exact_replay':True,
               'receipts':{str(out/r['scene']/'receipt.json'):sha256_file(out/r['scene']/'receipt.json') for r in records}})


def point_hash(points):
    return hashlib.sha256(np.asarray(points, dtype='<i8').tobytes()).hexdigest()


def checked_points(path, size):
    with np.load(path, allow_pickle=False) as data:
        points = data['point_indices']
    if (points.ndim != 1 or points.dtype.kind not in 'iu' or not len(points)
            or np.any(points < 0) or np.any(points >= size)
            or len(np.unique(points)) != len(points)):
        raise ValueError(f'invalid point set: {path}')
    return np.sort(points.astype(np.int64))


def native_arrays(records, scene):
    prefix = records / scene / 'native_cache' / f'{scene}_pred_'
    data = {k: np.load(str(prefix) + k + '.npy', mmap_mode='r') for k in ('masks', 'classes', 'scores')}
    if (data['masks'].ndim != 2 or data['masks'].shape[1] != len(data['classes'])
            or data['classes'].shape != data['scores'].shape or not np.isfinite(data['scores']).all()
            or not np.isin(data['classes'], np.arange(199)).all()):
        raise ValueError(f'invalid native predictions: {scene}')
    return data


def geometry_pool(run, records, scene):
    native = native_arrays(records, scene)
    size = native['masks'].shape[0]
    points, keys, lookup, native_indexes, arm_indexes, track_rows = [], [], {}, [], {}, {}

    def add(p):
        key = point_hash(p)
        if key not in lookup:
            lookup[key] = len(points)
            points.append(p)
            keys.append(key)
        return lookup[key]

    for column in range(native['masks'].shape[1]):
        p = np.flatnonzero(native['masks'][:, column])
        if not len(p):
            raise ValueError(f'empty native mask: {scene}/{column}')
        i = add(p)
        if i not in native_indexes:
            native_indexes.append(i)
    for arm in ARMS:
        rows = read(run / 'arms' / arm / 'merged' / scene / 'automatic_tracks.json')['tracks']
        if len({r['track_id'] for r in rows}) != len(rows):
            raise ValueError('duplicate track IDs')
        indexes = []
        for row in rows:
            p = checked_points(row['points_path'], size)
            if len(p) != row['point_count']:
                raise ValueError('track point count differs')
            indexes.append(add(p))
        arm_indexes[arm], track_rows[arm] = indexes, rows
    return native, points, keys, native_indexes, arm_indexes, track_rows


def verify_files(root, files):
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f'changed file: {root / name}')


def audit(args):
    started = time.perf_counter()
    completion = read(args.run / 'observations_complete.json')
    selection, contract = read(args.run / 'selection.json'), read(args.run / 'contract.json')
    scenes = (args.run / 'scenes.txt').read_text().split()
    if (completion != {'scene_count': 20, 'arm_count': 4, 'tracking_complete': True, 'AP_evaluated': False}
            or scenes != selection['scenes'] or digest(selection) != contract['selection_sha256']):
        raise ValueError('incomplete or inconsistent source run')
    verify_files(Path('/'), contract['assets'])
    freeze_json(args.output / 'protocol.json', PROTOCOL)
    sources = [Path(__file__), ROOT/'tools/fi1_frontend_protocol.py', ROOT/'utils/__init__.py',
               ROOT/'tools/annotate_automatic_mask_tracks_yoloworld.py',
               ROOT/'pretrained/config_scannet200.yaml']
    sources += sorted((ROOT/'evaluate/scannet200').glob('*.py'))
    inputs, rows = {str(p): sha256_file(p) for p in sources}, []
    for scene in scenes:
        manifest_path = args.run / 'manifests' / f'{scene}.json'
        manifest = read(manifest_path)
        inputs[str(manifest_path)] = sha256_file(manifest_path)
        for item in manifest['files'].values():
            if sha256_file(item['path']) != item['sha256']:
                raise ValueError('geometry input changed')
        size = np.load(manifest['files']['processed']['path'], mmap_mode='r').shape[0]
        bbox = args.records / scene / 'yoloworld_bboxes_2d' / f'{scene}.pt'
        inputs[str(bbox)] = sha256_file(bbox)
        for key in ('masks', 'scores', 'classes'):
            p = args.records / scene / 'native_cache' / f'{scene}_pred_{key}.npy'
            inputs[str(p)] = sha256_file(p)
        native = native_arrays(args.records, scene)
        if native['masks'].shape[0] != size:
            raise ValueError('native/mesh point domain mismatch')
        receipts_by_frontend = {}
        for frontend in ('automatic_sam', 'grounded_sam'):
            base = {'contract': digest(contract), 'manifest': digest(manifest), 'frontend': frontend}
            frame_receipts = {}
            for frame in manifest['frames']:
                folder = args.run / 'frame_cache' / frontend / scene / f"frame{int(frame['frame_id']):06d}"
                receipt = read(folder / 'receipt.json')
                if receipt['identity'] != digest({**base, 'frame': frame}):
                    raise ValueError('frame identity mismatch')
                verify_files(folder, receipt['files'])
                frame_receipts[frame['frame_id']] = receipt
            receipts_by_frontend[frontend] = frame_receipts
        for frame in manifest['frames']:
            a, b = [receipts_by_frontend[f][frame['frame_id']] for f in receipts_by_frontend]
            if a['files']['visibility.npy'] != b['files']['visibility.npy']:
                raise ValueError('frontend visibility mismatch')
        for arm, spec in ARMS.items():
            base = args.run / 'arms' / arm
            identity = digest({'contract': digest(contract), 'manifest': digest(manifest),
                               'frontend': spec['frontend'], 'arm': arm})
            obs = read(base / 'observations' / scene / 'summary.json')
            if obs['identity'] != identity:
                raise ValueError('arm observation identity mismatch')
            verify_files(base / 'observations' / scene, obs['output_sha256'])
            seconds = 0.
            for stage in ('hierarchy', 'tracks', 'consensus', 'merged'):
                receipt_path = base / 'stage_receipts' / f'{scene}_{stage}.json'
                receipt = read(receipt_path)
                if receipt['identity'] != identity:
                    raise ValueError('geometry stage identity mismatch')
                verify_files(base / stage / scene, receipt['files'])
                inputs[str(receipt_path)] = sha256_file(receipt_path)
                seconds += receipt['seconds']
            cache = receipts_by_frontend[spec['frontend']]
            frames = [cache[str(f)] for f in manifest['arms'][arm]]
            rows.append({'scene': scene, 'arm': arm, 'sampled_frames': len(frames),
                         'valid_frames': sum(not f['inference_skipped_invalid_pose'] for f in frames),
                         'observations': sum(f['kept_observation_count'] for f in frames),
                         'inference_seconds': sum(f['inference_seconds'] for f in frames),
                         'projection_seconds': sum(f['projection_seconds'] for f in frames),
                         'geometry_seconds': seconds,
                         'track_count': read(base / 'merged' / scene / 'automatic_tracks.json')['track_count']})
        geometry_pool(args.run, args.records, scene)
        print(f'audit {scene}: all frame/stage hashes and point domains passed', flush=True)
    freeze_json(args.output / 'input_identity.json', {'files': inputs, 'run': str(args.run), 'scenes': scenes,
                                                     'protocol_sha256': digest(PROTOCOL)})
    write_json(args.output / 'audit.json', {'valid': True, 'ground_truth_usage': 'none', 'rows': rows,
                                           'scene_count': len(scenes), 'seconds': time.perf_counter()-started})


def normalize_votes(votes):
    votes = np.asarray(votes, dtype=np.float64)
    if votes.ndim != 2 or votes.shape[1] != 198 or np.any(votes < 0) or not np.isfinite(votes).all():
        raise ValueError('invalid class votes')
    maximum = votes.max(axis=1)
    result = np.zeros((len(votes), 199), dtype=np.float32)
    result[:, :198] = votes / np.maximum(maximum[:, None], 1)
    result[maximum == 0, 198] = 1.
    return result


def top_predictions(distributions, indexes, budget=600):
    indexes = np.asarray(list(dict.fromkeys(indexes)), dtype=np.int64)
    local = distributions[indexes].reshape(-1)
    selected = np.argsort(-local, kind='stable')[:budget]
    return indexes[selected // 199], selected % 199, local[selected]


def semantics(args):
    import torch
    import imageio.v2 as imageio
    from utils import WORLD_2_CAM, OpenYolo3D
    from tools.annotate_automatic_mask_tracks_yoloworld import _load_predictions
    from types import SimpleNamespace
    torch.set_num_threads(4)
    identity = read(args.output / 'input_identity.json')
    if not read(args.output / 'audit.json')['valid'] or read(args.output / 'protocol.json') != PROTOCOL:
        raise ValueError('audit/protocol missing')
    verify_files(Path('/'), identity['files'])
    summaries = []
    for scene in identity['scenes']:
        target = args.output / 'semantics' / scene
        if target.exists():
            receipt = read(target / 'receipt.json')
            if receipt['protocol_sha256'] != digest(PROTOCOL) or receipt['input_identity_sha256'] != digest(identity):
                raise ValueError('semantic resume identity mismatch')
            verify_files(target, receipt['files'])
            summaries.append(receipt['summary'])
            continue
        started = time.perf_counter()
        manifest = read(args.run / 'manifests' / f'{scene}.json')
        native, points, keys, ni, ai, _ = geometry_pool(args.run, args.records, scene)
        bbox_path = args.records / scene / 'yoloworld_bboxes_2d' / f'{scene}.pt'
        predictions, cache_contract = _load_predictions(bbox_path, scene)
        metadata = torch.load(bbox_path, map_location='cpu').get('metadata', {})
        import yaml
        prompts = yaml.safe_load((ROOT/'pretrained/config_scannet200.yaml').read_text())['network2d']['text_prompts']
        if metadata.get('text_prompts') != prompts:
            raise ValueError('YOLO class vocabulary differs')
        votes = np.zeros((len(points), 198), dtype=np.int64)
        visible_counts = np.zeros(len(points), dtype=np.int64)
        selected_frames = set(map(str, manifest['arms']['A']))
        world = WORLD_2_CAM.__new__(WORLD_2_CAM)
        world.mesh = manifest['files']['mesh']['path']
        mesh = WORLD_2_CAM.load_ply(world.mesh)
        world.load_ply = lambda _: mesh
        processed = np.load(manifest['files']['processed']['path'], mmap_mode='r')
        if not np.array_equal(mesh[0][:, :3], processed[:, :3]):
            raise ValueError('mesh/processed XYZ mismatch')
        world.intrinsics = [manifest['files']['intrinsics']['path']]
        world.depth_scale, world.vis_depth_threshold = 1000., .05
        world.device = torch.device(args.device)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        valid = 0
        for frame in manifest['frames']:
            if frame['frame_id'] not in selected_frames:
                continue
            for item in (frame['color'], frame['depth'], frame['pose']):
                if sha256_file(item['path']) != item['sha256']:
                    raise ValueError('semantic frame input changed')
            if not frame['pose_valid']:
                continue
            if frame['frame_id'] not in predictions:
                raise ValueError('missing common YOLO frame')
            world.poses = [frame['pose']['path']]
            world.color_paths = [frame['color']['path']]
            world.depth_maps_paths = [frame['depth']['path']]
            world.image_resolution = imageio.imread(world.color_paths[0]).shape[:2]
            world.depth_resolution = imageio.imread(world.depth_maps_paths[0]).shape
            world.height, world.width = world.depth_resolution
            projection, visibility = world.get_mesh_projections()
            coords, visible = projection.numpy()[0], visibility.numpy()[0]
            cached = np.load(args.run/'frame_cache'/'automatic_sam'/scene/f"frame{int(frame['frame_id']):06d}"/'visibility.npy').reshape(-1)
            if not np.array_equal(visible, cached):
                raise ValueError(f'{scene}/{frame["frame_id"]}: recomputed visibility differs in {np.sum(visible != cached)} points')
            scaling = tuple(d/c for d,c in zip(world.depth_resolution, world.image_resolution))
            model = SimpleNamespace(world2cam=world, scaling_params=scaling)
            prediction = predictions[frame['frame_id']]
            if not np.isin(prediction['labels'].numpy(), np.arange(198)).all():
                raise ValueError('invalid YOLO class index')
            label_map = OpenYolo3D.construct_label_maps(model, {frame['frame_id']: prediction}).numpy()[0]
            point_labels = np.full(len(visible), -1, dtype=np.int16)
            point_labels[visible] = label_map[coords[visible, 1], coords[visible, 0]]
            for i, p in enumerate(points):
                labels = point_labels[p]
                votes[i] += np.bincount(labels[labels >= 0], minlength=198)
                visible_counts[i] += np.count_nonzero(visible[p])
            valid += 1
        distributions = normalize_votes(votes)
        target.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(target/'evidence.npz', votes=votes, distributions=distributions, visible_counts=visible_counts)
        write_json(target/'geometry.json', {'keys': keys, 'native_indexes': ni, 'arm_indexes': ai})
        summary = {'scene': scene, 'unique_geometry_count': len(points), 'native_geometry_count': len(ni),
                   'valid_semantic_frames': valid, 'geometry_with_foreground_votes': int(np.count_nonzero(votes.sum(axis=1))),
                   'seconds': time.perf_counter()-started, 'cache_contract': cache_contract}
        write_json(target/'receipt.json', {'summary': summary, 'ground_truth_usage': 'none',
                   'protocol_sha256': digest(PROTOCOL), 'input_identity_sha256': digest(identity),
                   'files': {p.name: sha256_file(p) for p in (target/'evidence.npz', target/'geometry.json')}})
        summaries.append(summary)
        print(f'semantics {scene}: {len(points)} geometries, {valid} common frames, {summary["seconds"]:.1f}s', flush=True)
    write_json(args.output/'semantics_complete.json', {'ground_truth_usage': 'none', 'scene_count': len(summaries), 'rows': summaries})


def overlap_matrix(points, gt_ids, gt_instances):
    sizes = np.array([np.count_nonzero(gt_ids == g) for g in gt_instances])
    counts = np.zeros((len(points), len(gt_instances)), dtype=np.int64)
    lookup = {int(g): j for j, g in enumerate(gt_instances)}
    for i, p in enumerate(points):
        ids, n = np.unique(gt_ids[p], return_counts=True)
        for g, count in zip(ids, n):
            if int(g) in lookup:
                counts[i, lookup[int(g)]] = count
    ps = np.array([len(p) for p in points], dtype=np.int64)
    iou = counts / np.maximum(ps[:, None]+sizes[None, :]-counts, 1)
    return iou, counts/np.maximum(ps[:, None], 1), counts/np.maximum(sizes[None, :], 1)


def coverage_metrics(native_iou, track_iou):
    nb = native_iou.max(axis=0, initial=0)
    tb = track_iou.max(axis=0, initial=0)
    out = {'gt_count': len(nb), 'native_mean_best_iou': float(nb.mean()) if len(nb) else 0.,
           'combined_mean_best_iou': float(np.maximum(nb, tb).mean()) if len(nb) else 0.}
    for threshold in PROTOCOL['geometry_thresholds']:
        key = str(threshold)
        out[key] = {'native_covered': int(np.sum(nb > threshold)), 'track_covered': int(np.sum(tb > threshold)),
                    'combined_covered': int(np.sum(np.maximum(nb, tb) > threshold)),
                    'new_covered': int(np.sum((nb <= threshold) & (tb > threshold))),
                    'track_candidates_matching_gt': int(np.sum(track_iou.max(axis=1, initial=0) > threshold)),
                    'excess_gt_supports': int(np.maximum((track_iou > threshold).sum(axis=0)-1, 0).sum())}
    return out


def evaluate(args):
    import contextlib
    from scipy import sparse
    from evaluate.scannet200 import eval_semantic_instance as evaluator
    from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST
    identity = read(args.output/'input_identity.json')
    if read(args.output/'protocol.json') != PROTOCOL or read(args.output/'semantics_complete.json')['scene_count'] != 20:
        raise ValueError('complete frozen semantics required before GT')
    verify_files(Path('/'), identity['files'])
    scenes = identity['scenes']
    target = args.output/'evaluation'
    target.mkdir(parents=True, exist_ok=False)
    geometry_rows, per_gt, gt_identity = [], [], {}
    for scene in scenes:
        native, points, keys, ni, ai, _ = geometry_pool(args.run, args.records, scene)
        semantic_root = args.output/'semantics'/scene
        receipt = read(semantic_root/'receipt.json')
        if receipt['protocol_sha256'] != digest(PROTOCOL) or receipt['input_identity_sha256'] != digest(identity):
            raise ValueError('semantic receipt input mismatch')
        verify_files(semantic_root, receipt['files'])
        if keys != read(semantic_root/'geometry.json')['keys']:
            raise ValueError('geometry order changed after semantic export')
        gt_path = args.gt/f'{scene}.txt'
        gt_identity[str(gt_path)] = sha256_file(gt_path)
        gt = np.loadtxt(gt_path, dtype=np.int64)
        if gt.shape != (native['masks'].shape[0],):
            raise ValueError('GT point domain mismatch')
        ids, sizes = np.unique(gt, return_counts=True)
        eligible = ids[(sizes >= 100) & np.isin(ids//1000, VALID_CLASS_IDS_200_INST) & (ids >= 1000)]
        iou, precision, recall = overlap_matrix(points, gt, eligible)
        iou[np.array([len(p) < 100 for p in points])] = 0
        nb = iou[ni].max(axis=0, initial=0)
        for arm, indexes in ai.items():
            ti = iou[indexes]
            metrics = coverage_metrics(iou[ni], ti)
            # Sparse int64 products avoid uint8/boolean overlap arithmetic errors.
            ps = [points[i] for i in indexes]
            offsets = np.cumsum([0]+[len(p) for p in ps])
            matrix = sparse.csr_matrix((np.ones(int(offsets[-1]), dtype=np.int64),
                         np.concatenate(ps) if ps else np.array([], dtype=np.int64), offsets), shape=(len(ps), len(gt)))
            intersections = (matrix @ matrix.T).toarray()
            sizes_p = np.array([len(p) for p in ps])
            pairs = intersections / np.maximum(sizes_p[:, None]+sizes_p[None, :]-intersections, 1)
            np.fill_diagonal(pairs, 0)
            metrics.update({'scene': scene, 'arm': arm, 'track_count': len(indexes),
                            'unique_track_count': len(set(indexes)),
                            'exact_native_duplicate_tracks': sum(i in set(ni) for i in indexes),
                            'track_duplicate_iou50_count': int(np.sum((pairs > .5).any(axis=1))),
                            'fragmented_gt_count': int(np.sum(((precision[indexes] >= .8) & (recall[indexes] >= .1) & (recall[indexes] < .5)).sum(axis=0) >= 2))})
            geometry_rows.append(metrics)
        for j, instance in enumerate(eligible):
            per_gt.append({'scene': scene, 'gt_instance_id': int(instance), 'native_best_iou': float(nb[j]),
                           **{a: float(iou[idx, j].max(initial=0)) for a, idx in ai.items()}})
        print(f'geometry evaluation {scene}: {len(eligible)} GT instances', flush=True)
    write_json(target/'geometry.json', {'rows': geometry_rows, 'per_gt': per_gt, 'gt_files': gt_identity,
                'GT_usage': 'evaluation only; matching and fragment/redundancy statistics do not modify predictions'})

    class Predictions(Mapping):
        def __init__(self, system):
            self.system = system
        def __len__(self):
            return len(scenes)
        def __iter__(self):
            return iter(scenes)
        def __getitem__(self, scene):
            if scene not in scenes:
                raise KeyError(scene)
            native, points, keys, ni, ai, _ = geometry_pool(args.run, args.records, scene)
            if self.system == 'native_cached':
                return {f'pred_{k}': np.asarray(v) for k, v in native.items()}
            with np.load(args.output/'semantics'/scene/'evidence.npz') as data:
                distributions = data['distributions']
            indexes = ni if self.system == 'native_common30' else ni + ai[self.system]
            gi, classes, scores = top_predictions(distributions, indexes)
            masks = np.zeros((native['masks'].shape[0], len(gi)), dtype=bool)
            for column, i in enumerate(gi):
                masks[points[i], column] = True
            return {'pred_masks': masks, 'pred_classes': classes, 'pred_scores': scores}

    results = {}
    for system in ('native_cached', 'native_common30', 'A', 'B', 'C', 'D'):
        started = time.perf_counter()
        with (target/f'{system}.log').open('w') as log, contextlib.redirect_stdout(log):
            averages, _, _, _ = evaluator.evaluate(Predictions(system), str(args.gt), str(target/f'{system}.csv'), dataset='scannet200')
        results[system] = {k: float(averages[v]) for k,v in {'ap':'all_ap','ap50':'all_ap_50%','ap25':'all_ap_25%'}.items()}
        results[system]['seconds'] = time.perf_counter()-started
        write_json(target/'ap_progress.json', results)
        print(f'AP {system}: {results[system]}', flush=True)
    write_json(target/'summary.json', {'protocol': PROTOCOL, 'scene_count': 20, 'AP': results,
                 'geometry_rows': geometry_rows, 'completed': True, 'full_FI1_pipeline_evaluated': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'semantics', 'evaluate', 'full-controls', 'full-geometry', 'full-bind', 'full-score', 'full-workflow'))
    parser.add_argument('--batches', nargs='+', type=int, default=list(range(1,7)))
    parser.add_argument('--backend-root', type=Path, default=ROOT/'.local/fi1_val48_frontend_full_pipeline_20260914/backend_smoke_AC')
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=['A','C'])
    parser.add_argument('--dataset-root', type=Path, default=Path('/home/jia/Wm/Dataset/scannet200'))
    parser.add_argument('--legacy-root', type=Path, default=Path('/media/jia/软件1/scannet_validation_stream/fi1_d_v3_dm_sms1_val312_local_20260902/class_agnostic_repro_20260902'))
    parser.add_argument('--backend-config', type=Path, default=ROOT/'pretrained/config_scannet200.yaml')
    parser.add_argument('--yolo-cache-root', type=Path, default=ROOT/'.local/fi1_r3b_val312_execution_20260914/yolo')
    parser.add_argument('--dv3-models', type=Path, default=Path('/home/jia/Wm/wm_open-yolo/worktrees/OpenYOLO3D-geometry-semantic-56b2c25/pretrained/fi1_d_v3_official100_full_models_20260822'))
    parser.add_argument('--champion-models', type=Path, default=ROOT/'output/train_candidate_champion_pair_union_combined_full_official100_v2_prior_corrected')
    parser.add_argument('--selection-root', type=Path, default=ROOT/'.local/fi1_val48_frontend_full_pipeline_20260914')
    parser.add_argument('--full-preparation', type=Path, default=ROOT/'.local/fi1_r3b_val312_preparation_20260914')
    parser.add_argument('--full-execution', type=Path, default=ROOT/'.local/fi1_r3b_val312_execution_20260914/continued_after_alpha_key_fix')
    parser.add_argument('--run', type=Path, default=ROOT/'.local/fi1_frontend_comparison_20260912/run_v2')
    parser.add_argument('--output', type=Path, default=ROOT/'.local/fi1_frontend_comparison_20260912/analysis_v1')
    parser.add_argument('--records', type=Path, default=Path('/media/jia/软件1/scannet_train_stream/records'))
    parser.add_argument('--gt', type=Path, default=Path('/media/jia/软件1/scannet_train_stream/prepared/ground_truth'))
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--allow-gt-evaluation', action='store_true')
    args = parser.parse_args()
    if args.command.startswith('full-') and args.output.resolve() == (ROOT/'.local/fi1_frontend_comparison_20260912/analysis_v1').resolve():
        parser.error('full pipeline commands require an explicit new --output directory')
    if args.command == 'full-score' and (args.binding is None or args.evidence is None):
        parser.error('full-score requires --binding and --evidence')
    if args.command == 'evaluate' and not args.allow_gt_evaluation:
        raise SystemExit('evaluation requires --allow-gt-evaluation')
    globals()[args.command.replace('-','_')](args)


if __name__ == '__main__':
    main()
