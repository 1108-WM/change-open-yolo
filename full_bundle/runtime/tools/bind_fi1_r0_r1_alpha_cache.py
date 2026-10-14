#!/usr/bin/env python3
"""Bind full pre-SMS Alpha vectors to identical geometry and view requests.

Historical RGB byte hashes were not recorded by the legacy exporter. This
limitation is explicit; a cache binding is not a replay of visual inference.
No SMS keep flag or top-1 class is used to select or filter output members.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_fi1_r0_r1_inputs import digest, read, sha, write
from tools.dm_sms_core import visible_ratio_multiscale_feature


def array_hash(values):
    values = np.ascontiguousarray(values)
    h = hashlib.sha256(str(values.dtype).encode())
    h.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    h.update(values.tobytes())
    return h.hexdigest()


def same_views(requests, cached):
    if len(requests) != len(cached):
        return False
    for new, old in zip(requests, cached):
        for name in ('frame_id', 'frame_index', 'visible_point_count'):
            if new[name] != old[name]:
                return False
        if str(Path(new['image_path']).resolve()) != str(Path(old['rgb_path']).resolve()):
            return False
        if (not np.isclose(new['visible_ratio'], old['visible_ratio'], rtol=0, atol=1e-12)
                or not np.allclose(new['sam_box_prompt_xyxy'], old['sam_box_prompt_xyxy'], rtol=0, atol=1e-9)):
            return False
        if len(old['scales']) != 3 or len(new['crop_scales']) != 3:
            return False
        for current, prior in zip(new['crop_scales'], old['scales']):
            if (current['scale_index'] != prior['scale_index']
                    or current['expansion'] != prior['bbox_expansion_fraction_per_side']
                    or current['crop_xyxy'] != prior['crop_xyxy_integer_exclusive']):
                return False
    return True


def recover_alpha(row, scale_features, aggregate_features, similarities, text_features):
    per_view, ratios = [], []
    for view in row['views']:
        if not view['sam_mask_valid']:
            continue
        scales = sorted(view['scales'], key=lambda r: r['scale_index'])
        if [s['scale_index'] for s in scales] != [0,1,2] or not all(s['feature_valid'] for s in scales):
            raise ValueError('incomplete cached three-scale feature view')
        features = []
        for scale in scales:
            feature = scale_features[scale['feature_index']]
            if array_hash(feature) != scale['feature_sha256']:
                raise ValueError('cached per-scale feature hash differs')
            features.append(feature)
        per_view.append(np.stack(features))
        ratios.append(view['visible_ratio'])
    if not per_view:
        if row['alpha_feature_valid']:
            raise ValueError('cached availability differs from actual views')
        return {'acquisition_complete': True, 'available': False, 'missing_reason': 'no_valid_sam_view',
                'valid_view_count': 0}
    features = np.stack(per_view)
    if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=-1), 1, atol=2e-4):
        raise ValueError('cached visual features are not finite unit vectors')
    aggregate = visible_ratio_multiscale_feature(features, np.asarray(ratios, dtype=np.float32))
    cosine = (aggregate @ text_features.T).astype(np.float32)
    i = row['geometry_index']
    if (not row['alpha_feature_valid'] or not np.allclose(aggregate, aggregate_features[i], atol=2e-6, rtol=0)
            or not np.allclose(cosine, similarities[i], atol=2e-6, rtol=0)):
        raise ValueError('cached aggregation/cosine replay differs')
    return {'acquisition_complete': True, 'available': True, 'cosine': cosine.tolist(),
            'view_scale_std': (features @ text_features.T).reshape(-1,198).std(axis=0).tolist(),
            'missing_reason': None, 'valid_view_count': len(per_view)}


def jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source]


def cached_geometry_match(geometry, points, cached, by_hash, scene):
    """Bind historical geometry/visual_geometry namespaces by the actual point set."""
    if (points.ndim != 1 or points.dtype.kind not in 'iu'
            or not np.array_equal(points, np.unique(points)) or len(points) != geometry['point_count']):
        raise ValueError('invalid bound geometry point set')
    raw = np.asarray(points, dtype='<i8').tobytes()
    if hashlib.sha256(raw).hexdigest() != geometry['points_sha256']:
        raise ValueError('bound geometry SHA256 differs from actual points')
    point_sha1 = hashlib.sha1(raw).hexdigest()
    matches = {k:cached[k] for k in geometry['legacy_geometry_keys'] if k in cached}
    for old in by_hash.get((scene,point_sha1),[]):
        matches[old['geometry_key']] = old
    if len(matches) > 1:
        raise ValueError('multiple historical Alpha records for the same actual point set')
    if not matches:
        return None
    old = next(iter(matches.values()))
    if (old['scene_name'] != scene or old['point_count'] != len(points)
            or old['geometry_hash'] != point_sha1):
        raise ValueError('historical Alpha geometry digest/scene/count differs')
    return old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--views', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--allow-new-scenes', action='store_true', help='Acquire fresh Alpha for scenes absent from the historical cache; never fabricate cached vectors')
    args = parser.parse_args()
    args.views, args.cache, args.output = (p.resolve() for p in (args.views,args.cache,args.output))
    view_identity = read(args.views/'identity.json')
    binding = Path(view_identity['binding'])
    scenes = sorted(args.scenes or view_identity['scenes'])
    if len(scenes) != len(set(scenes)) or not set(scenes) <= set(view_identity['scenes']):
        raise ValueError('requested scenes not in view identity')
    summary = read(args.cache/'summary.json')
    model = read(args.cache/'model_provenance.json')
    prompts = read(args.cache/'text_prompts.json')
    files = {str(args.cache/n):sha(args.cache/n) for n in ('summary.json','model_provenance.json','text_prompts.json','text_features.npy')}
    if (files[str(args.cache/'model_provenance.json')] != summary['input_provenance']['model_provenance_sha256']
            or files[str(args.cache/'text_prompts.json')] != model['text_prompts_sha256']
            or files[str(args.cache/'text_features.npy')] != model['text_features_sha256']
            or prompts['class_names'] != view_identity['class_names'] or model['sam_model_type'] != 'vit_b'):
        raise ValueError('cached model/vocabulary identity differs')
    for name in ('alpha_clip_base','alpha_clip_checkpoint','sam_checkpoint'):
        files[model[name]] = sha(model[name])
        if files[model[name]] != model[name+'_sha256']:
            raise ValueError('cached model weight identity differs')
    text_features = np.load(args.cache/'text_features.npy', allow_pickle=False)
    if text_features.shape != (198,768) or not np.isfinite(text_features).all() or not np.allclose(np.linalg.norm(text_features,axis=1),1,atol=2e-4):
        raise ValueError('invalid full vocabulary text features')
    identity = {'version':'fi1_r0_r1_alpha_cache_binding_v1', 'views':str(args.views),
        'views_identity_sha256':digest(view_identity), 'cache':str(args.cache), 'cache_inputs':files,
        'code':{str(Path(__file__).resolve()):sha(__file__), str(ROOT/'tools/dm_sms_core.py'):sha(ROOT/'tools/dm_sms_core.py'),
                str(ROOT/'tools/prepare_fi1_r0_r1_inputs.py'):sha(ROOT/'tools/prepare_fi1_r0_r1_inputs.py')},
        'scenes':scenes, 'historical_rgb_content_identity_verified':False,
        'interpretation':'frozen historical evidence; exact geometry/view/crop match and feature aggregation replay; no visual-model replay',
        'sms_filter_applied':False, 'alpha_top1_applied':False}
    if args.allow_new_scenes:
        identity['allow_new_scenes'] = True
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists():
        if read(args.output/'identity.json') != identity:
            raise ValueError('resume identity differs')
    else:
        write(args.output/'identity.json',identity)
    totals = []
    for scene in scenes:
        target = args.output/scene
        if (target/'receipt.json').exists():
            receipt = read(target/'receipt.json')
            if receipt['identity_sha256'] != digest(identity):
                raise ValueError('resume receipt differs')
            for p,h in {**receipt['inputs'], **{str(target/p):h for p,h in receipt['outputs'].items()}}.items():
                if sha(p) != h:
                    raise ValueError('resume file changed')
            totals.append(receipt['summary'])
            continue
        receipt = read(args.views/scene/'receipt.json')
        if receipt['identity_sha256'] != digest(view_identity):
            raise ValueError('view scene identity differs')
        inputs = {str(args.views/scene/'receipt.json'):sha(args.views/scene/'receipt.json')}
        for p,h in {**receipt['input_files'], **{str(args.views/scene/p):h for p,h in receipt['output_files'].items()}}.items():
            if sha(p) != h:
                raise ValueError('view inputs changed')
            inputs[p] = h
        geometry = {r['geometry_key']:r for r in read(binding/scene/'geometry.json')}
        cache_scene = args.cache/'scenes'/scene
        known_scenes = {s['scene_name'] for s in summary['scene_summaries']}
        if args.allow_new_scenes and scene not in known_scenes:
            if cache_scene.exists():
                raise ValueError('unlisted partial historical scene cache exists')
            requests = jsonl(args.views/scene/'yolo_and_alpha_requests.jsonl')
            if (len(requests) != len(geometry)
                    or {r['geometry_key'] for r in requests} != set(geometry)):
                raise ValueError('new scene evidence geometry coverage differs')
            target.mkdir(exist_ok=False)
            with (target/'evidence.jsonl').open('x') as output:
                for row in requests:
                    if row['alpha']['acquisition_complete']:
                        raise ValueError('new-scene requests cannot claim existing Alpha')
                    row['alpha']['missing_reason'] = 'new_scene_without_historical_alpha'
                    output.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            result = {'scene_name':scene, 'geometry_count':len(requests), 'alpha_bound_count':0,
                      'alpha_pending_count':len(requests), 'complete':not requests,
                      'pending':[{'geometry_key':r['geometry_key'],'reason':'new_scene_without_historical_alpha'} for r in requests],
                      'ground_truth_read':False,'ap_computed':False,'historical_rgb_content_identity_verified':False}
            write(target/'receipt.json',{'identity_sha256':digest(identity),'inputs':inputs,
                  'outputs':{'evidence.jsonl':sha(target/'evidence.jsonl')},'summary':result})
            totals.append(result)
            print('Fresh Alpha required for new scene',scene,len(requests),flush=True)
            continue
        cs = read(cache_scene/'summary.json')
        inputs[str(cache_scene/'summary.json')] = sha(cache_scene/'summary.json')
        if cs != next(s for s in summary['scene_summaries'] if s['scene_name'] == scene):
            raise ValueError('cached scene summary differs from global summary')
        for name,field in [('records.jsonl','records_sha256'),('scale_features.npy','scale_features_sha256'),
                           ('aggregate_features.npy','aggregate_features_sha256'),('similarities.npy','similarities_sha256')]:
            inputs[str(cache_scene/name)] = sha(cache_scene/name)
            if inputs[str(cache_scene/name)] != cs[field]:
                raise ValueError('cached scene file hash differs')
        cached_rows = jsonl(cache_scene/'records.jsonl')
        cached = {r['geometry_key']:r for r in cached_rows}
        if len(cached) != len(cached_rows):
            raise ValueError('duplicate cached geometry key')
        by_hash = defaultdict(list)
        for record in cached_rows:
            by_hash[record['scene_name'],record['geometry_hash']].append(record)
        point_file = binding/scene/'points.npz'
        if str(point_file) not in inputs or sha(point_file) != inputs[str(point_file)]:
            raise ValueError('Alpha fallback requires the exact view-bound point arrays')
        with np.load(point_file,allow_pickle=False) as packed:
            point_sets = {k:packed[g['array_key']] for k,g in geometry.items()}
        scales = np.load(cache_scene/'scale_features.npy',mmap_mode='r',allow_pickle=False)
        aggregates = np.load(cache_scene/'aggregate_features.npy',mmap_mode='r',allow_pickle=False)
        similarities = np.load(cache_scene/'similarities.npy',mmap_mode='r',allow_pickle=False)
        target.mkdir(exist_ok=False)
        ready, pending = 0, []
        with (target/'evidence.jsonl').open('x') as output:
            for row in jsonl(args.views/scene/'yolo_and_alpha_requests.jsonl'):
                geom = geometry[row['geometry_key']]
                old = cached_geometry_match(geom,point_sets[row['geometry_key']],cached,by_hash,scene)
                reason = 'new_geometry_without_historical_alpha'
                if old is not None:
                    if old['point_count'] != geom['point_count'] or old['scene_name'] != scene:
                        raise ValueError('cached geometry binding differs')
                    if same_views(row['alpha']['view_requests'], old['views']):
                        alpha = recover_alpha(old,scales,aggregates,similarities,text_features)
                        alpha.update(cache_geometry_key=old['geometry_key'], cached_record_sha256=digest(old),
                            historical_rgb_content_identity_verified=False, sms_filter_applied=False, alpha_top1_applied=False)
                        row['alpha'] = alpha
                        ready += 1
                    else:
                        reason = 'historical_views_or_crop_recipe_differ'
                if not row['alpha']['acquisition_complete']:
                    row['alpha']['missing_reason'] = reason
                    pending.append({'geometry_key':row['geometry_key'],'reason':reason})
                output.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        result = {'scene_name':scene,'geometry_count':len(geometry),'alpha_bound_count':ready,
            'alpha_pending_count':len(pending),'pending':pending,'complete':not pending,
            'ground_truth_read':False,'ap_computed':False,'historical_rgb_content_identity_verified':False}
        write(target/'receipt.json',{'identity_sha256':digest(identity),'inputs':inputs,
            'outputs':{'evidence.jsonl':sha(target/'evidence.jsonl')},'summary':result})
        totals.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='pending'}),flush=True)
    write(args.output/'summary.json',{'scene_count':len(totals),'alpha_bound_count':sum(r['alpha_bound_count'] for r in totals),
        'alpha_pending_count':sum(r['alpha_pending_count'] for r in totals),'rows':totals,
        'complete':all(r['complete'] for r in totals),'ground_truth_read':False,'ap_computed':False,
        'identity_sha256':digest(identity)})


if __name__ == '__main__':
    main()
