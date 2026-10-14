#!/usr/bin/env python3
"""Isolated registered Full replay with the author's official Replica masks."""
import argparse
import ast
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OLD = ROOT / '.local/replica_registered_full_20261011_v1'
OUT = ROOT / '.local/replica_official_masks_full_20261011_v1'
ASSETS = ROOT / '.local/official_openyolo3d_assets_20261011'
MASKS = ASSETS / 'extracted/replica_masks'

# Route existing frozen adapters to a separate output root before importing them.
from tools import run_replica_registered_full as base
base.OUT = OUT
from tools import fi1_replica_full_adapter as adapter
from tools import fi1_replica_member_scoring as scoring
adapter.OUT = scoring.OUT = OUT
from tools.fi1_frontend_protocol import freeze_json as freeze, write_json as write, sha256_file as sha
SCENES = base.SCENES


def link(target, source):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        assert target.resolve() == source.resolve(), target
    elif target.exists():
        raise FileExistsError(target)
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


def prepare():
    base.reserve()
    assets = base.read(ASSETS / 'verified.json')
    assert assets['complete'] and base.read(ASSETS / 'mesh_identity_verified.json')['all_original_meshes_exactly_equal']
    old = base.read(OLD / 'contract.json')
    for p, digest in old['models_and_inputs'].items():
        assert sha(p) == digest, p
    binding = base.read(OLD / 'backend_model_code_binding.json')
    for p, digest in binding['inputs'].items():
        assert sha(p) == digest, p
    inputs = {**old['models_and_inputs'], **binding['inputs']}
    for row in assets['scenes']:
        assert sha(row['path']) == row['sha256']
        inputs[row['path']] = row['sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['baseline.yaml', 'full.yaml']:
        target = OUT / name
        if not target.exists():
            shutil.copyfile(OLD / name, target)
        assert sha(target) == sha(OLD / name)
    reused = {}
    # These artifacts have no dependence on raw native proposal point sets.
    for name in ['prepared', 'frontend', 'common']:
        source = OLD / name
        link(OUT / name, source)
        for p in source.rglob('*'):
            if p.is_file() and 'ground_truth' not in str(p):
                reused[str(p)] = sha(p)
    for scene in SCENES:
        for name, source in [('yoloworld_bboxes_2d', OLD / 'records' / scene / 'yoloworld_bboxes_2d')]:
            link(OUT / 'records' / scene / name, source)
            for p in source.rglob('*'):
                if p.is_file():
                    reused[str(p)] = sha(p)
        reused[str(OLD / 'yolo_baseline' / (scene + '.pt'))] = sha(OLD / 'yolo_baseline' / (scene + '.pt'))
    snapshot = OLD / 'paper_protocol_audit/upstream_snapshot/utils/__init__.py'
    inputs[str(snapshot)] = sha(snapshot)
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    freeze(OUT / 'contract.json', {
        'method': old['method'], 'scenes': SCENES, 'vocabulary': old['vocabulary'],
        'models_and_inputs': inputs, 'configs': {n: sha(OUT / (n + '.yaml')) for n in ['baseline', 'full']},
        'native_proposals': 'author official Replica masks, exact archive meshes equal local meshes; no Mask3D inference or NMS',
        'baseline': 'official upstream labeling functions, Replica YAML, uniform scores; additional native score control',
        'Full': 'registered GD10 + official100 D-v3 seven models + matched597 B26 / NTUUprime / K800; no S/C/SL',
        'reuse': 'prepared superpoints, native-independent GD10 SAM/GD and YOLO-SAM observations, frozen YOLO detections only',
        'recomputed': 'native labels, native-dependent relations/filters/U/Uprime, own semantics, complete 26D features, B26/K800',
        'new_training': False, 'GT_policy': 'seal both methods for all eight scenes before evaluation reads raw GT',
        'previous_results_unchanged': True, 'author_asset_receipt_sha256': sha(ASSETS / 'verified.json'),
        'author_mesh_identity_receipt_sha256': sha(ASSETS / 'mesh_identity_verified.json')})
    freeze(OUT / 'reused_inputs.json', {'GT_read': False, 'files': reused})
    freeze(OUT / 'backend_model_code_binding.json', {'GT_read': False, 'inputs': inputs, 'production_models_unchanged': True})
    write(OUT / 'task_status.json', {'running': True, 'complete': False, 'stage': 'prepared', 'no_new_training': True})
    print('Prepared official-mask replay', len(inputs), len(reused), flush=True)


def upstream_label_class():
    """Execute unmodified author labeling methods; retain local memory-safe projection."""
    import utils
    source = (OLD / 'paper_protocol_audit/upstream_snapshot/utils/__init__.py').read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ['get_visibility_mat', 'compute_iou', 'OpenYolo3D']]
    namespace = dict(utils.__dict__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<official upstream Replica labeling>', 'exec'), namespace)
    return namespace['OpenYolo3D']


def native():
    import torch
    import yaml
    from utils import OpenYolo3D, WORLD_2_CAM
    from tools.prepare_fi1_r3b_val312 import no_gt_io
    no_gt_io()
    torch.set_num_threads(4)
    official = upstream_label_class()
    for method in ['baseline', 'full']:
        config = yaml.safe_load((OUT / (method + '.yaml')).read_text())
        for scene in SCENES:
            base.reserve()
            dest = OUT / 'baseline' / scene if method == 'baseline' else OUT / 'records' / scene / 'native_cache'
            if (dest / 'sealed.json').exists():
                continue
            # Point-cloud path has no effect on proposal features: masks are loaded, never inferred.
            obj = (official if method == 'baseline' else OpenYolo3D).__new__(official if method == 'baseline' else OpenYolo3D)
            obj.openyolo3d_config = config
            obj.num_classes = 49
            obj.datatype = 'point cloud'
            obj.world2cam = WORLD_2_CAM(str(base.DATA / scene), 6553.5, config)
            obj.mesh_projections = obj.world2cam.get_mesh_projections()
            obj.scaling_params = [obj.world2cam.depth_resolution[j] / obj.world2cam.image_resolution[j] for j in range(2)]
            obj.preds_3d = torch.load(MASKS / (scene + '.pt'), map_location='cpu')
            cache = OLD / 'yolo_baseline' / (scene + '.pt') if method == 'baseline' else OLD / 'records' / scene / 'yoloworld_bboxes_2d' / (scene + '.pt')
            payload = torch.load(cache, map_location='cpu')
            assert payload['metadata']['text_prompts'] == config['network2d']['text_prompts']
            assert payload['metadata']['openyolo3d']['frequency'] == config['openyolo3d']['frequency']
            for k in ['topk', 'th', 'nms', 'use_amp']:
                assert payload['metadata']['network2d'][k] == config['network2d'][k]
            obj.preds_2d = payload['predictions']
            result = obj.label_3d_masks_from_2d_bboxes(scene, False)[scene]
            dest.mkdir(parents=True, exist_ok=True)
            for name, value in zip(['masks', 'classes', 'scores'], result):
                np.save(dest / (scene + '_pred_' + name + '.npy'), value.detach().cpu().numpy())
            write(dest / 'sealed.json', {'GT_read': False, 'official_masks_sha256': sha(MASKS / (scene + '.pt')),
                  'YOLO_cache_sha256': sha(cache), 'labeling': 'upstream original' if method == 'baseline' else 'registered Full native top600',
                  'files': {str(p): sha(p) for p in dest.glob('*.npy')}})
            print('Native sealed', method, scene, result[0].shape, flush=True)
            del obj, payload, result
            gc.collect()
            torch.cuda.empty_cache()


def verify():
    # Existing independent scientific check, routed to the new output root.
    from tools import verify_replica_registered_full as v
    v.OUT = OUT
    v.main()
    from tools import verify_replica_agnostic_supplement as ag
    ag.OUT = OUT
    ag.main()
    for p, digest in base.read(OUT / 'reused_inputs.json')['files'].items():
        assert sha(p) == digest, p
    print('Reused upstream files unchanged', flush=True)


def chain():
    legacy = base.read(ROOT / '.local/fi1_r0_r4_preparation_20260913/dv3_generation_v2/jobs.json')['environment']
    env_gpu = {**os.environ, 'PYTHONPATH': str(ROOT) + ':' + str(ROOT / 'models/Mask3D') + ':' + str(ROOT / 'models/YOLO-World'),
               'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    env_legacy = {**env_gpu, **legacy['environment']}
    env_legacy['PYTHONPATH'] = str(ROOT) + ':' + ':'.join(env_legacy['PYTHONPATH'].split(':')[1:])
    env_legacy['PYTHONPATH'] += ':' + str(ROOT / 'models/Mask3D') + ':' + str(ROOT / 'models/YOLO-World')
    py_gpu = str(ROOT / '.venvs/yoloe_gpu/bin/python')
    commands = [('native', '/home/jia/anaconda3/envs/openyolo3d/bin/python', ['native'], env_gpu)]
    commands += [('backend_' + s, legacy['python'], ['backend', '--scenes', s], env_legacy) for s in SCENES]
    commands += [('evidence_' + s, py_gpu, ['evidence', '--scenes', s], env_gpu) for s in SCENES]
    commands += [('score', legacy['python'], ['score'], env_legacy),
                 ('evaluation', py_gpu, ['evaluate'], {**env_gpu, 'CUDA_VISIBLE_DEVICES': ''}),
                 ('verification', legacy['python'], ['verify'], env_legacy)]
    for stage, python, arguments, environment in commands:
        write(OUT / 'execution.json', {'running': True, 'stage': stage})
        with (OUT / (stage + '.log')).open('a') as log:
            result = subprocess.run([python, '-u', str(Path(__file__).resolve()), *arguments], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(OUT / 'execution.json', {'running': False, 'failed': True, 'stage': stage, 'exit_code': result.returncode})
            raise SystemExit(result.returncode)
    write(OUT / 'execution.json', {'running': False, 'complete': True, 'results': str(OUT / 'evaluation/results.json')})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'native', 'backend', 'evidence', 'score', 'evaluate', 'verify', 'chain'])
    p.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    args = p.parse_args()
    base.bootstrap()
    actions = {'prepare': prepare, 'native': native, 'backend': lambda: adapter.backend(args.scenes),
               'evidence': lambda: scoring.evidence(args.scenes), 'score': lambda: scoring.score(args.scenes),
               'evaluate': scoring.evaluate, 'verify': verify, 'chain': chain}
    actions[args.command]()


if __name__ == '__main__':
    main()
