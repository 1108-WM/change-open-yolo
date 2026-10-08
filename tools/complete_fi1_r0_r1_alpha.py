#!/usr/bin/env python3
"""Prepare a reviewable frozen job, then acquire missing own-mask Alpha on GPU.

Only new output directories are written. Cached evidence remains read-only.
The legacy batch encoder computes auxiliary SMS fields; they are not applied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_fi1_r0_r1_inputs import digest, read, sha, write
from tools.bind_fi1_r0_r1_alpha_cache import jsonl, recover_alpha


def check(files):
    for path, h in files.items():
        if sha(path) != h:
            raise ValueError(f'frozen input changed: {path}')


def code_files(snapshot, model):
    paths = [Path(__file__).resolve(), ROOT/'tools/bind_fi1_r0_r1_alpha_cache.py',
             ROOT/'tools/prepare_fi1_r0_r1_inputs.py', ROOT/'tools/dm_sms_core.py',
             snapshot/'tools/build_dm_sms1_alpha_embedding_ledger.py']
    for key, package in [('alpha_clip_source','alpha_clip'),('sam_source','segment_anything')]:
        paths.extend(sorted((Path(model[key])/package).rglob('*.py')))
    return {str(p):sha(p) for p in paths}


def prepare(args):
    bound, snapshot = args.bound.resolve(), args.snapshot.resolve()
    identity = read(bound/'identity.json')
    cache = Path(identity['cache'])
    model = read(cache/'model_provenance.json')
    prompts = read(cache/'text_prompts.json')
    check(identity['cache_inputs'])
    files = {str(bound/'identity.json'):sha(bound/'identity.json'),
             str(bound/'summary.json'):sha(bound/'summary.json'), **identity['cache_inputs']}
    rows = []
    for scene in identity['scenes']:
        receipt = read(bound/scene/'receipt.json')
        if receipt['identity_sha256'] != digest(identity):
            raise ValueError('bound scene identity differs')
        scene_files = {**receipt['inputs'], **{str(bound/scene/k):v for k,v in receipt['outputs'].items()},
                       str(bound/scene/'receipt.json'):sha(bound/scene/'receipt.json')}
        check(scene_files)
        records = jsonl(bound/scene/'evidence.jsonl')
        pending = [r for r in records if not r['alpha']['acquisition_complete']]
        rows.append({'scene_name':scene, 'pending_geometry_count':len(pending),
                     'pending_views':sum(len(r['alpha']['view_requests']) for r in pending),
                     'pending_keys':[r['geometry_key'] for r in pending], 'input_files':scene_files})
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    plan = {'version':'fi1_r0_r1_fresh_alpha_plan_v1','bound':str(bound),'snapshot':str(snapshot),
        'output':str(out),'bound_identity_sha256':digest(identity),'model':model,
        'text_features':str(cache/'text_features.npy'),'prompts':prompts['prompts'],
        'files':files,'code':code_files(snapshot,model),'scenes':rows,
        'sam_batch_size':8,'alpha_batch_size':32,
        'pending_geometry_count':sum(r['pending_geometry_count'] for r in rows),
        'pending_view_count':sum(r['pending_views'] for r in rows),
        'new_training':False,'ground_truth_read':False,'ap_computed':False,
        'legacy_auxiliary_sms_computed_but_not_applied':True}
    write(out/'plan.json',plan)
    print(json.dumps({'plan':str(out/'plan.json'),'pending_geometry_count':plan['pending_geometry_count'],
                      'pending_view_count':plan['pending_view_count']}))


def manifest_row(row):
    views = []
    for rank,v in enumerate(row['alpha']['view_requests']):
        views.append({'view_rank':rank,'frame_index':v['frame_index'],'frame_id':v['frame_id'],
            'rgb_path':v['image_path'],'visible_point_count':v['visible_point_count'],
            'visible_ratio':v['visible_ratio'],'sam_box_prompt_xyxy':v['sam_box_prompt_xyxy'],
            'crop_scales':[{'scale_index':s['scale_index'],'bbox_expansion_fraction_per_side':s['expansion'],
                            'crop_xyxy_integer_exclusive':s['crop_xyxy']} for s in v['crop_scales']]})
    # Compatibility metadata is unused by visual inference and never emitted as a new class.
    return {'scene_name':row['scene_name'],'geometry_key':row['geometry_key'],
        'geometry_hash':row['points_sha256'],'point_count':row['point_count'],
        'canonical_candidate_source':'own_geometry_request','canonical_candidate_id':0,
        'canonical_frozen_class_index':-1,'canonical_frozen_class_valid':False,
        'canonical_frozen_score':0.,'selected_view_count':len(views),'views':views,'members':[]}


def run(args):
    plan = read(args.plan)
    out = Path(plan['output'])
    if args.plan.resolve() != out/'plan.json':
        raise ValueError('plan path differs from frozen output')
    check(plan['files'])
    check(plan['code'])
    import tools
    tools.__path__.append(str(Path(plan['snapshot'])/'tools'))
    import torch
    from tools.build_dm_sms1_alpha_embedding_ledger import _load_alpha_clip, _load_sam, _process_scene
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: run this frozen job where the GPU is accessible')
    torch.set_num_threads(1)
    model_args = SimpleNamespace(**{k:Path(v) if k.endswith(('_source','_base','_checkpoint')) else v
                                   for k,v in plan['model'].items()})
    model_args.output_root = out/'fresh_cache'
    model_args.sam_batch_size = plan['sam_batch_size']
    model_args.alpha_batch_size = plan['alpha_batch_size']
    model_args.sms_threshold = 0.
    state = _load_alpha_clip(model_args,plan['prompts'])
    predictor = _load_sam(model_args)
    text_features = np.load(plan['text_features'],allow_pickle=False)
    # Use the same frozen text vectors for cached and fresh image features.
    write(out/'runtime.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'plan_sha256':sha(args.plan),
        'fresh_vs_frozen_text_max_abs_difference':float(np.max(np.abs(state['text_features']-text_features))),
        'scoring_text_features':'frozen cached 198 text vectors', 'new_training':False})
    bound = Path(plan['bound'])
    old_identity = read(bound/'identity.json')
    identity = {**old_identity,'version':'fi1_r0_r1_alpha_completed_v1','fresh_plan_sha256':sha(args.plan),
                'fresh_runtime_sha256':sha(out/'runtime.json'),'parent_bound':str(bound)}
    if (out/'identity.json').exists() and read(out/'identity.json') != identity:
        raise ValueError('resume completed evidence identity differs')
    write(out/'identity.json',identity)
    summaries = []
    for job in plan['scenes']:
        scene = job['scene_name']
        check(job['input_files'])
        target = out/scene
        if (target/'receipt.json').exists():
            receipt = read(target/'receipt.json')
            if receipt['identity_sha256'] != digest(identity):
                raise ValueError('resume receipt identity differs')
            check(receipt['inputs'])
            check({str(target/p):h for p,h in receipt['outputs'].items()})
            summaries.append(receipt['summary'])
            continue
        rows = jsonl(bound/scene/'evidence.jsonl')
        pending = [r for r in rows if not r['alpha']['acquisition_complete']]
        if [r['geometry_key'] for r in pending] != job['pending_keys']:
            raise ValueError('pending geometry list changed')
        files = dict(job['input_files'])
        acquired = {}
        if pending:
            requests = [manifest_row(r) for r in pending]
            print(f'GPU Alpha {scene}: {len(requests)} own geometries, {job["pending_views"]} views',flush=True)
            cs = _process_scene(scene,requests,model_args,state,predictor,text_features,digest(requests))
            cache = model_args.output_root/'scenes'/scene
            for name, field in [('records.jsonl','records_sha256'),('scale_features.npy','scale_features_sha256'),
                ('aggregate_features.npy','aggregate_features_sha256'),('similarities.npy','similarities_sha256')]:
                files[str(cache/name)] = sha(cache/name)
                if files[str(cache/name)] != cs[field]:
                    raise ValueError('fresh scene feature hash differs')
            scales = np.load(cache/'scale_features.npy',mmap_mode='r',allow_pickle=False)
            aggregates = np.load(cache/'aggregate_features.npy',mmap_mode='r',allow_pickle=False)
            similarities = np.load(cache/'similarities.npy',mmap_mode='r',allow_pickle=False)
            for row in jsonl(cache/'records.jsonl'):
                alpha = recover_alpha(row,scales,aggregates,similarities,text_features)
                alpha.update(fresh_visual_inference=True, historical_rgb_content_identity_verified=True,
                    sms_filter_applied=False,alpha_top1_applied=False,fresh_record_sha256=digest(row))
                acquired[row['geometry_key']] = alpha
        target.mkdir(exist_ok=False)
        with (target/'evidence.jsonl').open('x') as handle:
            for row in rows:
                if row['geometry_key'] in acquired:
                    row['alpha'] = acquired[row['geometry_key']]
                if not row['alpha']['acquisition_complete']:
                    raise ValueError('Alpha acquisition still incomplete')
                handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        check(job['input_files'])
        result = {'scene_name':scene,'geometry_count':len(rows),'fresh_alpha_count':len(acquired),
                  'cached_alpha_count':len(rows)-len(acquired),'complete':True,'ground_truth_read':False,'ap_computed':False}
        write(target/'receipt.json',{'identity_sha256':digest(identity),'inputs':files,
            'outputs':{'evidence.jsonl':sha(target/'evidence.jsonl')},'summary':result})
        summaries.append(result)
        print(json.dumps(result),flush=True)
    check(plan['code'])
    write(out/'summary.json',{'complete':True,'identity_sha256':digest(identity),'scene_count':len(summaries),
        'rows':summaries,'new_training':False,'ground_truth_read':False,'ap_computed':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--bound',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p = sub.add_parser('run')
    p.add_argument('--plan',type=Path,required=True)
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else run(args)
