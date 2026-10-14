#!/usr/bin/env python3
"""Export own-geometry full YOLO evidence and Alpha view requests, without GT.

Reuse the original projection, label painting and top-40 selection routines.
Alpha requests are explicitly pending evidence, never zero-score substitutes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.prepare_fi1_r0_r1_inputs import digest, read, sha, write
from tools.fi1_r0_r1_protocol import PROTOCOL


def dependencies(snapshot):
    files = {Path(__file__).resolve(), ROOT/'tools/fi1_r0_r1_protocol.py',
             ROOT/'tools/prepare_fi1_r0_r1_inputs.py'}
    for name, module in list(sys.modules.items()):
        if name == 'utils' or name.startswith('tools.'):
            file = getattr(module, '__file__', None)
            if file and Path(file).suffix == '.py':
                path = Path(file).resolve()
                if path.is_relative_to(ROOT) or path.is_relative_to(snapshot):
                    files.add(path)
    return {str(p): sha(p) for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+')
    args = parser.parse_args()
    for name in ('binding','sources','snapshot','output'):
        setattr(args, name, getattr(args, name).resolve())
    import tools
    tools.__path__.append(str(args.snapshot/'tools'))
    import torch
    import yaml
    from tools.export_fi1_refined_semantic_evidence import load_scene_context
    from tools.fi1_unified_evidence_core import baseline_maps, yolo_evidence
    from tools.fi1_full_alpha_adapter import make_views
    torch.set_num_threads(1)
    # These helpers lazily import these modules when called; bind them up front.
    import tools.build_z1_yoloworld_multiview_distribution_ledger
    import utils

    source_config = read(args.sources)
    binding_summary = read(args.binding/'summary.json')
    binding_identity = read(args.binding/'identity.json')
    if binding_summary['complete'] is not True or binding_summary['identity'] != binding_identity:
        raise ValueError('geometry binding incomplete/inconsistent')
    if binding_identity['sources_sha256'] != sha(args.sources):
        raise ValueError('source configuration differs from geometry binding')
    spec = source_config['sources'][binding_identity['source']]
    config_path = Path(source_config['config_path'])
    config = yaml.safe_load(config_path.read_text())
    labels = config['network2d']['text_prompts']
    if (len(labels) != 198 or len(set(labels)) != 198 or config['openyolo3d']['frequency'] != 10
            or config['openyolo3d']['topk'] != 40):
        raise ValueError('vocabulary/frequency/top-k differs from frozen protocol')
    scenes = sorted(args.scenes or binding_identity['scenes'])
    if len(scenes) != len(set(scenes)) or not set(scenes) <= set(binding_identity['scenes']):
        raise ValueError('scene selection differs from bound geometry')
    code = dependencies(args.snapshot)
    identity = {'version': 'fi1_r0_r1_yolo_views_v1', 'binding': str(args.binding),
                'binding_identity_sha256': digest(binding_identity), 'source': binding_identity['source'],
                'config_path': str(config_path), 'config_sha256': sha(config_path),
                'protocol': PROTOCOL, 'protocol_sha256': digest(PROTOCOL), 'code': code,
                'class_names': labels, 'scenes': scenes, 'subset': scenes != binding_identity['scenes'],
                'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available()}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists():
        if read(args.output/'identity.json') != identity:
            raise ValueError('resume identity changed; use a new output directory')
    else:
        write(args.output/'identity.json', identity)
    summaries = []
    for scene in scenes:
        target = args.output/scene
        receipt_path = target/'receipt.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt['identity_sha256'] != digest(identity):
                raise ValueError('scene receipt differs')
            for p,h in {**receipt['input_files'], **{str(target/p):h for p,h in receipt['output_files'].items()}}.items():
                if sha(p) != h:
                    raise ValueError(f'resume input/output changed: {p}')
            summaries.append(receipt['summary'])
            continue
        target.mkdir(exist_ok=False)
        started = time.monotonic()
        bound = read(args.binding/scene/'receipt.json')
        if bound['identity_sha256'] != digest(binding_identity):
            raise ValueError('geometry scene receipt not bound')
        input_files = {str(args.binding/scene/'receipt.json'): sha(args.binding/scene/'receipt.json')}
        for p,h in {**bound['input_files'], **{str(args.binding/scene/p):h for p,h in bound['output_files'].items()}}.items():
            if sha(p) != h:
                raise ValueError(f'geometry input/output changed: {p}')
            input_files[p] = h
        geometry = read(args.binding/scene/'geometry.json')
        points_file = args.binding/scene/'points.npz'
        context_args = SimpleNamespace(prepared_root=Path(spec['prepared_root']),
            yolo_cache_template=str(Path(spec['records_root'])/'{scene}'/'yoloworld_bboxes_2d'/'{scene}.pt'))
        context = load_scene_context(scene, context_args, config, labels)
        if context['visibility'].shape[1] != bound['summary']['point_count']:
            raise ValueError('projection point domain differs')
        mesh = str(Path(context['world'].mesh).resolve())
        if mesh not in bound['input_files'] or context['projection_inputs'][mesh] != bound['input_files'][mesh]:
            raise ValueError('projection mesh is not the audited geometry point domain')
        input_files.update(context['projection_inputs'])
        cache = str(Path(context_args.yolo_cache_template.format(scene=scene)))
        input_files[cache] = context['cache_hash']
        maps = baseline_maps(context, 198)
        print(f'projected {scene}: {len(context["world"].color_paths)} frames, {len(geometry)} geometries', flush=True)
        available, views_count = 0, 0
        with np.load(points_file, allow_pickle=False) as arrays, (target/'yolo_and_alpha_requests.jsonl').open('x') as output:
            for row in geometry:
                points = arrays[row['array_key']]
                yolo = yolo_evidence(points, context, maps, 198, 40)
                views = make_views(points, context)
                for view in views:
                    input_files[view['image_path']] = view['image_sha256']
                label = yolo['label_map']
                record = {'geometry_key': row['geometry_key'], 'points_sha256': row['points_sha256'],
                    'point_count': row['point_count'], 'scene_name': scene,
                    'yolo': {'acquisition_complete': True, 'available': label['available'],
                             'votes': label['counts'], 'missing_reason': label['missing_reason']},
                    'yolo_raw': yolo, 'alpha': {'acquisition_complete': False, 'available': False,
                        'missing_reason': 'not_acquired', 'view_requests': views},
                    'ground_truth_read': False, 'ap_computed': False}
                output.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
                available += label['available']
                views_count += len(views)
        if dependencies(args.snapshot) != code:
            raise ValueError('evidence implementation changed during export')
        # Also detect file changes during extraction, before marking completion.
        for p,h in input_files.items():
            if sha(p) != h:
                raise ValueError(f'evidence input changed during export: {p}')
        summary = {'scene_name': scene, 'geometry_count': len(geometry),
            'yolo_available_count': available, 'alpha_requested_views': views_count,
            'alpha_acquisition_complete': False, 'r0_r1_ap_ready': False,
            'seconds': time.monotonic()-started, 'ground_truth_read': False, 'ap_computed': False}
        write(receipt_path, {'identity_sha256': digest(identity), 'input_files': input_files,
            'output_files': {'yolo_and_alpha_requests.jsonl': sha(target/'yolo_and_alpha_requests.jsonl')},
            'summary': summary})
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        del context, maps
    write(args.output/'summary.json', {'yolo_complete': True, 'alpha_complete': False,
        'r0_r1_ap_ready': False, 'scene_count': len(summaries), 'rows': summaries,
        'identity_sha256': digest(identity), 'ground_truth_read': False, 'ap_computed': False})


if __name__ == '__main__':
    main()
