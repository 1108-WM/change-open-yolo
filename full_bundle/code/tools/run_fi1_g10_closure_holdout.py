#!/usr/bin/env python3
"""Produce one held-out GD10 frontend using the unchanged completed597 kernels."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.fi1_frontend_protocol import digest, freeze_json, sha256_file as sha, write_json
from tools.run_fi1_frontend_comparison import (assets, build_scene_manifest, verify_scene_inputs,
    check_point_order, Frontends, export_frame, assemble_arm, track_arm)
from tools.run_fi1_g10_closure_assets import LOCAL, OUT, TRAIN_FRONT, read, require


def prepare_and_export(scene):
    inventory = read(LOCAL / 'inventory.json')
    job = inventory['jobs'][scene]
    require(job['group'] in ('dev', 'val') and 'reuse' not in job, 'not a new held-out frontend')
    front = Path(job['front'])
    selection = read(Path(job['selection']) / 'selection.json')
    freeze_json(front / 'selection.json', selection)
    asset_hashes = assets()
    require(asset_hashes == read(TRAIN_FRONT / 'contract.json')['assets'], 'train/holdout frontend model/code differs')
    contract = {'selection_sha256': digest(selection), 'assets': asset_hashes,
                'stride': 10, 'variant': 'grounded_sam_stride10_matched_holdout',
                'runner_sha256': sha(__file__), 'inventory_sha256': sha(LOCAL / 'inventory.json'),
                'ground_truth_usage': 'none', 'new_visual_inference': True, 'AP_evaluated': False}
    freeze_json(front / 'contract.json', contract)
    mp = front / 'manifests' / f'{scene}.json'
    if mp.exists():
        manifest = read(mp)
    else:
        manifest = build_scene_manifest(scene, Path(job['data']), Path(job['data']))
        manifest['arms']['D'] = list(range(0, manifest['original_frame_count'], 10))
        manifest['derived_variant'] = {'name': 'grounded_sam_stride10_matched_holdout', 'stride': 10}
        freeze_json(mp, manifest)
    require(manifest['scene_name'] == scene and manifest['arms']['D'] == list(range(0, manifest['original_frame_count'], 10)),
            'held-out manifest differs')
    verify_scene_inputs(manifest)
    count = check_point_order(manifest)
    base = {'contract': digest(contract), 'manifest': digest(manifest), 'frontend': 'grounded_sam'}
    arm_identity = digest({**base, 'arm': 'D'})
    receipts = [front / 'arms/D/stage_receipts' / f'{scene}_{s}.json' for s in ('hierarchy', 'tracks', 'consensus', 'merged')]
    if not all(p.exists() for p in receipts):
        model = Frontends('grounded_sam', front / 'grounded_sam_model_receipt.json')
        frame_root = front / 'frame_cache/grounded_sam' / scene
        chosen = set(manifest['arms']['D'])
        for frame in manifest['frames']:
            if int(frame['frame_id']) not in chosen:
                continue
            r = export_frame(model, manifest, frame, frame_root / f"frame{int(frame['frame_id']):06d}",
                             digest({**base, 'frame': frame}), count)
            write_json(front / 'progress.json', {'scene': scene, 'frame_id': frame['frame_id'], 'receipt': r})
            print('FRAME', scene, frame['frame_id'], r['kept_observation_count'], flush=True)
        assemble_arm(front, 'D', manifest, frame_root, arm_identity)
        track_arm(front, 'D', manifest, frame_root, count, arm_identity)
        del model
    # Reuse the complete597 read-only auditor with explicit path injection.
    source = LOCAL / 'audit_frontend.py'
    require(sha(source) == read(LOCAL / 'frontend_code_audit.json')['sources'][str(source)], 'frontend auditor changed')
    spec = importlib.util.spec_from_file_location('closure_frontend_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FRONT = front
    module.HERE = OUT / job['group']
    report = module.scene(scene)
    require(report['complete'], 'held-out frontend audit failed')
    print('FRONTEND COMPLETE', scene, report['frames'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    args = parser.parse_args()
    prepare_and_export(args.scene)
