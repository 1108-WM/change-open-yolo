#!/usr/bin/env python3
"""Matched GD10 assets for train597/dev55/val312, reusing frozen FI1 kernels.

No fitting or evaluation occurs here. Each scene has actual-geometry evidence,
hash receipts, a native/YOLO adapter and an isolated resumable output directory.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.fi1_frontend_protocol import digest, freeze_json, sha256_file as sha, write_json
from tools.fi1_g10_closure_selection import fixed_selection
from tools.run_fi1_g10_matched_pilot import link, allocated

LOCAL = ROOT / '.local/fi1_g10_full597_closure_20260921_v1'
BASE = Path('/media/jia/软件1/scannet_additional_train_stream')
OUT = BASE / 'fi1_G10_closure_20260921'
TRAIN_FRONT = BASE / 'fi1_G10_full597_20260919/frontend'
MASTER = ROOT / '资料/训练597场景清单_20260917.json'
VAL = ROOT / 'models/Mask3D/third_party/ScanNet/Tasks/Benchmark/scannetv2_val.txt'
HISTORY = ROOT / '.local/fi1_r0_r4_preparation_20260913'
V312 = ROOT / '.local/fi1_r3b_val312_preparation_20260914'
PY = ROOT / '.venvs/yoloe_gpu/bin/python'
CONFIG = ROOT / 'pretrained/config_scannet200.yaml'
SNAPSHOT = ROOT / 'full_bundle/code_sources/evidence_snapshot'
RESERVE = 50 * 1024**3
ROLES = {'train': 'train597_g10_backend', 'dev': 'dev55_g10_backend', 'val': 'val312_g10_backend'}
read = lambda p: json.loads(Path(p).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical_groups():
    m = read(MASTER)
    groups = {'train': sorted(m['roles']['train']),
              'dev': sorted(m['roles']['development_validation']), 'val': sorted(VAL.read_text().split())}
    require([len(groups[g]) for g in ROLES] == [597, 55, 312], 'split counts differ')
    mains = {g: {s.split('_')[0] for s in ss} for g, ss in groups.items()}
    for g, ss in groups.items():
        require(len(ss) == len(set(ss)), 'duplicate scenes')
        for h in groups:
            if g != h:
                require(not mains[g] & mains[h], 'main-scene split leakage')
    return groups


def build_inventory():
    groups = canonical_groups()
    source = read(HISTORY / 'input_sources_v1.json')['sources']
    train_inv = read(ROOT / '.local/fi1_g10_full597_frontend_20260919_v1/selection/source_inventory.json')
    vs = read(V312 / 'inventory/input_sources.json')['sources']['val312']
    master = read(MASTER)
    old48 = read(ROOT / '.local/fi1_val48_grounded_stride10_full597_q_20260919_v2/identity.json')
    reuse48 = {j['scene']: j for j in old48['jobs']}
    inventory = {}
    for group, scenes in groups.items():
        role = ROLES[group]
        selection = {'dataset_role': role, 'scenes': scenes, 'active_arms': ['D'],
                     'frontend': 'grounded_sam', 'stride': 10,
                     'selection_uses': 'canonical split names, no GT/AP-based selection'}
        sel = LOCAL / 'selection' / group
        freeze_json(sel / 'selection.json', selection)
        p = sel / 'scenes_execution_order.txt'
        value = '\n'.join(scenes) + '\n'
        if p.exists():
            require(p.read_text() == value, 'scene order changed')
        else:
            p.write_text(value)
        freeze_json(sel / 'selection_receipt.json', {'complete': True,
            'inputs': {str(p): sha(p) for p in (MASTER, VAL)},
            'outputs': {p.name: sha(p) for p in (sel / 'selection.json', p)}})
        require(fixed_selection(sel) == selection, 'selection adapter differs')
        for scene in scenes:
            j = {'scene': scene, 'group': group, 'role': role, 'selection': str(sel),
                 'folder': str(OUT / group / 'scenes' / scene)}
            if group == 'train' and scene in master['new397']:
                part = 'first199' if scene in master['first199'] else 'remaining198'
                a = BASE / f'fi1_A_{part}_20260917'
                j.update(data=str(BASE / 'prepared'), records=str(a / 'preparation/records_adapter' / scene),
                    native_receipt=str(a / 'preparation/native_bindings' / scene / 'receipt.json'),
                    common=str(a / 'common/03_d1_hierarchy_safe' / scene),
                    yolosam=str(a / 'common/11_yoloworld_sam_uniform30' / scene),
                    yolo=str(a / 'yolo' / scene / f'{scene}.pt'),
                    yolo_receipt=str(a / 'yolo' / scene / 'receipt.json'), alpha=vs['alpha_cache'], origin=part)
            elif group != 'val':
                r = ('ncs_validation60' if group == 'dev' else
                     {'old_official100': 'official100', 'old_ncs_train100': 'ncs_train100'}[train_inv[scene]['source']])
                spec = source[r]
                common = Path(spec['automatic_root']).parent
                ar = read(HISTORY / 'alpha_binding_v1' / r / 'identity.json')
                j.update(data=spec['prepared_root'], records=str(Path(spec['records_root']) / scene),
                    native_receipt=str(HISTORY / 'geometry_binding_v1' / r / scene / 'receipt.json'),
                    common=str(common / 'd1_hierarchy_safe' / scene),
                    yolosam=str(common / 'yoloworld_sam_uniform30' / scene),
                    yolo=str(Path(spec['records_root']) / scene / 'yoloworld_bboxes_2d' / f'{scene}.pt'),
                    yolo_receipt=str(Path(ar['views']) / scene / 'receipt.json'),
                    alpha=ar['cache'], origin=r)
            else:
                common = Path(vs['native_cache_root']).parent
                yolo = ROOT / '.local/fi1_r3b_val312_execution_20260914/yolo' / scene
                j.update(data=vs['prepared_root'], records=str(Path(vs['records_root']) / scene),
                    native_receipt=str(V312 / 'geometry_binding' / scene / 'receipt.json'),
                    common=str(common / '03_d1_hierarchy_safe' / scene),
                    yolosam=str(common / '11_yoloworld_sam_uniform30' / scene),
                    yolo=str(yolo / f'{scene}.pt'), yolo_receipt=str(yolo / 'receipt.json'),
                    alpha=vs['alpha_cache'], origin='val312')
                if scene in reuse48:
                    job = reuse48[scene]
                    j['reuse'] = {'own': str(Path(job['geometry']).parent.parent.parent),
                                 'backend': str(Path(job['dv3']).parents[2]),
                                 'front': str(ROOT / '.local/fi1_val48_grounded_stride10_frontend_20260919_v1'),
                                 'inputs': job['inputs']}
            j['front'] = str(TRAIN_FRONT if group == 'train' else OUT / group / 'frontend')
            for k in ('data', 'records', 'native_receipt', 'common', 'yolosam', 'yolo', 'yolo_receipt', 'alpha'):
                require(Path(j[k]).exists(), f'missing {scene} {k}: {j[k]}')
            inventory[scene] = j
    freeze_json(LOCAL / 'inventory.json', {'groups': groups, 'jobs': inventory,
        'inputs': {str(p): sha(p) for p in (MASTER, VAL, HISTORY / 'input_sources_v1.json', V312 / 'inventory/input_sources.json')},
        'budget': 600, 'heads': {'B': 26, 'Q': 29}, 'full312_authorized': True,
        'geometry_changes': False, 'Dv3_retraining': False})
    return read(LOCAL / 'inventory.json')


def adapters(job):
    import torch
    import yaml
    from tools.prepare_fi1_r3b_val312 import inspect_yolo
    scene, role = job['scene'], job['role']
    root = OUT / job['group'] / 'adapters' / job['origin']
    prep, common, yolo = root / 'preparation', root / 'common', root / 'yolo'
    freeze_json(prep / 'inventory/input_sources.json', {'config_path': str(CONFIG),
        'sources': {role: {'prepared_root': job['data'], 'records_root': str(prep / 'records_adapter'),
                           'alpha_cache': job['alpha']}}})
    link(prep / 'records_adapter' / scene, Path(job['records']))
    original = Path(job['native_receipt'])
    nr = read(original)
    bound = {str(original): sha(original)}
    for kind in ('masks', 'classes', 'scores'):
        name = f'{scene}_pred_{kind}.npy'
        p = Path(job['records']) / 'native_cache' / name
        h = sha(p)
        require(h in [v for k, v in nr['input_files'].items() if Path(k).name == name],
                'native differs from historical bound input: ' + scene)
        link(common / '01_native_cache' / name, p)
        bound[str(prep / 'records_adapter' / scene / 'native_cache' / name)] = h
    freeze_json(prep / 'native_bindings' / scene / 'receipt.json',
                {'scene': scene, 'dataset_role': role, 'input_files': bound})
    link(common / '03_d1_hierarchy_safe' / scene, Path(job['common']))
    link(common / '11_yoloworld_sam_uniform30' / scene, Path(job['yolosam']))
    cache = Path(job['yolo'])
    receipt = read(job['yolo_receipt'])
    h = sha(cache)
    oldhashes = {**receipt.get('input_files', {}), **receipt.get('outputs', {})}
    require(any(Path(k).resolve() == cache.resolve() and v == h for k, v in oldhashes.items()),
            'YOLO differs from historical verified evidence: ' + scene)
    payload = torch.load(cache, map_location='cpu')
    config = yaml.safe_load(CONFIG.read_text())
    check = inspect_yolo(payload, scene, config['network2d']['text_prompts'])
    require(check['historical_vocabulary_identity_verified'], 'unverified YOLO vocabulary')
    meta = payload['metadata']
    require(Path(meta['scene_path']).resolve() == (Path(job['data']) / scene).resolve(), 'YOLO data root mismatch')
    require(meta['openyolo3d']['frequency'] == 10, 'YOLO frequency differs')
    require(check['frame_ids'] == list(range(0, len(os.listdir(Path(job['data']) / scene / 'poses')), 10)),
            'YOLO cache raw-frame coverage differs')
    for k in ('topk', 'th', 'nms', 'use_amp', 'config_path', 'pretrained_path'):
        require(meta['network2d'][k] == config['network2d'][k], 'YOLO detector parameter differs: ' + k)
    p = yolo / scene / f'{scene}.pt'
    link(p, cache)
    freeze_json(p.parent / 'receipt.json', {'summary': {'complete': True, 'scene_name': scene},
        'outputs': {str(p.resolve()): h}, 'provenance': 'verified reuse, not new YOLO inference',
        'historical_receipt': {job['yolo_receipt']: sha(job['yolo_receipt'])}, 'validation': check})
    return prep, common, yolo


def audit_scene(job):
    from tools.prepare_fi1_r3b_val312 import verify
    folder = Path(job['folder'])
    own = Path(job['reuse']['own']) if 'reuse' in job else folder / 'own/D'
    backend = Path(job['reuse']['backend']) if 'reuse' in job else folder / 'backend'
    scene = job['scene']
    b = backend / 'D' / scene
    receipt = read(b / 'receipt.json')
    verify(receipt['inputs'])
    verify({str(b / p): h for p, h in receipt['outputs'].items()})
    manifests = [Path(p) for p in receipt['inputs'] if '/manifests/' in p and Path(p).name == scene + '.json']
    require(len(manifests) == 1, 'missing/ambiguous frontend manifest')
    manifest = read(manifests[0])
    require(list(map(int, manifest['arms']['D'])) == list(range(0, manifest['original_frame_count'], 10)),
            'backend does not use strict G10 observations')
    binding = own / 'geometry_binding'
    evidence = own / 'alpha_completed'
    require(read(evidence / 'summary.json')['complete'], 'Alpha incomplete')
    bi, ei, vi = (read(p / 'identity.json') for p in (binding, evidence, own / 'views'))
    require(vi['binding_identity_sha256'] == digest(bi), 'views geometry identity differs')
    require(ei['views_identity_sha256'] == digest(vi), 'Alpha views identity differs')
    br, er = read(binding / scene / 'receipt.json'), read(evidence / scene / 'receipt.json')
    require(br['identity_sha256'] == digest(bi) and er['identity_sha256'] == digest(ei), 'scene identity differs')
    verify(br['input_files'])
    verify({str(binding / scene / p): h for p, h in br['output_files'].items()})
    verify({str(evidence / scene / p): h for p, h in er['outputs'].items()})
    geometry = read(binding / scene / 'geometry.json')
    rows = [json.loads(s) for s in (evidence / scene / 'evidence.jsonl').read_text().splitlines()]
    by_key = {r['geometry_key']: r for r in geometry}
    require(len(by_key) == len(geometry) == len(rows) == len({r['geometry_key'] for r in rows}), 'duplicate geometry/evidence')
    require(set(by_key) == {r['geometry_key'] for r in rows}, 'geometry/evidence keys differ')
    for r in rows:
        require(r['points_sha256'] == by_key[r['geometry_key']]['points_sha256'], 'geometry point digest differs')
        require(r['yolo']['acquisition_complete'] and r['alpha']['acquisition_complete'], 'evidence unfinished')
    result = {'complete': True, 'scene': scene, 'strict_stride': 10, 'geometries': len(geometry),
        'own': str(own), 'backend': str(backend), 'allocated_bytes': 0 if 'reuse' in job else allocated(folder),
        'GT_read': False, 'audited_at': now(), 'inputs': {str(p): sha(p) for p in
            (b / 'receipt.json', binding / scene / 'receipt.json', evidence / scene / 'receipt.json')}}
    path = folder / 'audit.json'
    if path.exists():
        prior = read(path)
        require({k: v for k, v in prior.items() if k not in ('audited_at', 'allocated_bytes')} ==
                {k: v for k, v in result.items() if k not in ('audited_at', 'allocated_bytes')}, 'completed asset audit changed')
        return prior
    write_json(path, result)
    return result


def run(groups, pilot_only=False):
    inventory = read(LOCAL / 'inventory.json')
    audit = read(LOCAL / 'assets_code_audit.json')
    require(audit['inventory_sha256'] == sha(LOCAL / 'inventory.json'), 'inventory changed since audit')
    for p, h in audit['code'].items():
        require(sha(p) == h, 'unaudited source: ' + p)
    require(read(LOCAL / 'frontend_audit_summary.json')['complete'], 'full597 frontend audit not complete')
    old = read(HISTORY / 'dv3_generation_v2/jobs.json')['environment']
    gpu = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    cpu = dict(gpu, CUDA_VISIBLE_DEVICES='')
    legacy = {**gpu, **old['environment']}
    legacy['PYTHONPATH'] = str(ROOT) + ':' + ':'.join(legacy['PYTHONPATH'].split(':')[1:])
    state = {'status': 'running', 'groups': groups, 'pid': os.getpid(), 'started_at': now(), 'completed': [], 'stages': []}

    def command(stage, argv, env=cpu):
        for p, h in audit['code'].items():
            require(sha(p) == h, 'code changed during run: ' + p)
        require(shutil.disk_usage(BASE).free >= RESERVE, '50GiB reserve reached')
        state.update(stage=stage, updated_at=now())
        write_json(LOCAL / 'assets_execution.json', state)
        logfile = LOCAL / 'logs' / f'{stage}.log'
        logfile.parent.mkdir(parents=True, exist_ok=True)
        t = time.monotonic()
        print('START', stage, flush=True)
        with logfile.open('a') as log:
            proc = subprocess.Popen(list(map(str, argv)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    state.update(child_pid=proc.pid, updated_at=now(), free_bytes=shutil.disk_usage(BASE).free)
                    write_json(LOCAL / 'assets_execution.json', state)
                    require(state['free_bytes'] >= RESERVE, '50GiB reserve reached')
                    time.sleep(5)
                require(proc.returncode == 0, stage + ' failed; see ' + str(logfile))
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
        state['stages'].append({'stage': stage, 'seconds': time.monotonic() - t})
        print('DONE', stage, state['stages'][-1]['seconds'], flush=True)

    def tool(python, name, *args):
        return [python, ROOT / 'tools' / name, *args]

    with (LOCAL / 'assets.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for group in groups:
                scenes = inventory['groups'][group]
                if pilot_only:
                    scenes = scenes[:3]
                for index, scene in enumerate(scenes, 1):
                    job = inventory['jobs'][scene]
                    state.update(group=group, scene=scene, index=index, total=len(scenes))
                    folder = Path(job['folder']); own = folder / 'own/D'
                    if 'reuse' not in job and not (own / 'alpha_completed/summary.json').exists():
                        prep, common, yolo = adapters(job)
                        if group != 'train':
                            command(scene + '_G10_frontend', tool(PY, 'run_fi1_g10_closure_holdout.py',
                                '--scene', scene), gpu)
                        args = ['--selection-root', job['selection'], '--dataset-root', job['data'],
                                '--full-preparation', prep, '--yolo-cache-root', yolo, '--scenes', scene, '--arms', 'D']
                        command(scene + '_D_v3', tool(old['python'], 'evaluate_fi1_frontend_comparison.py',
                            'full-geometry', *args, '--run', job['front'], '--output', folder / 'backend',
                            '--legacy-root', common, '--backend-config', CONFIG), legacy)
                        command(scene + '_bind', tool(PY, 'evaluate_fi1_frontend_comparison.py', 'full-bind',
                            *args, '--backend-root', folder / 'backend', '--output', folder / 'own'))
                        command(scene + '_yolo_views', tool(PY, 'export_fi1_r0_r1_yolo_views.py',
                            '--binding', own / 'geometry_binding', '--sources', own / 'sources.json',
                            '--snapshot', SNAPSHOT, '--output', own / 'views'))
                        command(scene + '_alpha_bind', tool(PY, 'bind_fi1_r0_r1_alpha_cache.py',
                            '--views', own / 'views', '--cache', job['alpha'], '--output', own / 'alpha_bound', '--allow-new-scenes'))
                        if not (own / 'alpha_completed/plan.json').exists():
                            command(scene + '_alpha_plan', tool(PY, 'complete_fi1_r0_r1_alpha.py', 'prepare',
                                '--bound', own / 'alpha_bound', '--snapshot', SNAPSHOT, '--output', own / 'alpha_completed'))
                        command(scene + '_alpha_gpu', tool(PY, 'complete_fi1_r0_r1_alpha.py', 'run',
                            '--plan', own / 'alpha_completed/plan.json'), gpu)
                    result = audit_scene(job)
                    state['completed'].append(result)
                    write_json(LOCAL / 'assets_execution.json', state)
                    print('SCENE COMPLETE', group, index, len(scenes), scene, flush=True)
                    if index == 3:
                        size = max(r['allocated_bytes'] for r in state['completed'][-3:])
                        projected = int(size * (len(inventory['groups'][group]) - 3) * 1.2)
                        require(shutil.disk_usage(BASE).free >= RESERVE + projected, 'pilot storage projection exceeds reserve')
                        write_json(LOCAL / f'{group}_pilot_gate.json', {'complete': True, 'scenes': scenes[:3],
                            'projected_remaining_bytes': projected, 'audits': state['completed'][-3:], 'finished_at': now()})
            state.update(status='pilot_complete' if pilot_only else 'complete', finished_at=now())
        except BaseException as e:
            state.update(status='failed', error=repr(e), updated_at=now())
            write_json(LOCAL / 'assets_execution.json', state)
            raise
        write_json(LOCAL / 'assets_execution.json', state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['inventory', 'run'])
    p.add_argument('--groups', nargs='+', choices=list(ROLES), default=['train'])
    p.add_argument('--pilot-only', action='store_true')
    a = p.parse_args()
    if a.command == 'inventory':
        build_inventory()
    else:
        run(a.groups, a.pilot_only)
