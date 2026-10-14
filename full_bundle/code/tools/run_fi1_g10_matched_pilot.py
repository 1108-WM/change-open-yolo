#!/usr/bin/env python3
"""Build strict G10 assets for a frozen 30-scene matched-training pilot.

The selection uses scan names only, excludes every main scene touched by the
2026-09-18 pairwise development experiment, and freezes five six-scene holdout
folds.  One real scene is completed and audited before the remaining 29 scenes
are allowed to run.  This command does not train a model or compute AP.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
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
from tools.run_fi1_frontend_comparison import assets, build_scene_manifest, fixed_selection

BASE = Path('/media/jia/软件1/scannet_additional_train_stream')
DATA = BASE / 'prepared'
OUT = BASE / 'fi1_G10_matched_pilot30_20260919'
LOCAL = ROOT / '.local/fi1_g10_matched_pilot30_20260919_v1'
SELECTION = LOCAL / 'selection'
FRONTEND = OUT / 'frontend'
PREPARATION = OUT / 'preparation'
COMMON = OUT / 'common'
YOLO = OUT / 'yolo'
MASTER = ROOT / '资料/scannet_new397_20260914_master397.txt'
PAIRWISE_SPLIT = ROOT / '.local/fi1_pairwise_small_20260918_v5/split.json'
SEED = 'fi1-g10-matched-pilot30-20260919-v1|'
ROLE = 'additional_train_g10_pilot30'
PY = ROOT / '.venvs/yoloe_gpu/bin/python'
OPENPY = Path('/home/jia/anaconda3/envs/openyolo3d/bin/python')
SNAPSHOT = ROOT / 'full_bundle/code_sources/evidence_snapshot'
CONFIG = ROOT / 'pretrained/config_scannet200.yaml'
RESERVE = 50 * 1024**3


def read(path: Path):
    return json.loads(Path(path).read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def selected_scenes(master: list[str], excluded_main: set[str], count: int = 30) -> list[str]:
    """Select one scan per untouched main scene using the frozen hash rule."""
    require(len(master) == 397 and len(set(master)) == 397, 'new397 master differs')
    representatives: dict[str, str] = {}
    for scan in sorted(master):
        require(scan.startswith('scene') and '_' in scan, 'invalid ScanNet scan name')
        representatives.setdefault(scan.split('_')[0], scan)
    ranked = sorted(
        ((main, scan) for main, scan in representatives.items() if main not in excluded_main),
        key=lambda item: (hashlib.sha256((SEED + item[0]).encode()).hexdigest(), item[0]),
    )
    require(len(ranked) >= count, 'not enough untouched main scenes')
    return [scan for _, scan in ranked[:count]]


def scene_folds(scenes: list[str]) -> dict:
    """Make five deterministic, main-scene-isolated folds of six scans each."""
    require(len(scenes) == 30 and len({s.split('_')[0] for s in scenes}) == 30,
            'pilot requires 30 distinct main scenes')
    ranked = sorted(
        scenes,
        key=lambda scene: (hashlib.sha256((SEED + 'fold|' + scene.split('_')[0]).encode()).hexdigest(), scene),
    )
    folds = []
    for index in range(5):
        holdout = ranked[index * 6:(index + 1) * 6]
        folds.append({'fold': index, 'fit_scans': [s for s in scenes if s not in holdout],
                      'holdout_scans': holdout,
                      'fit_main_scenes': [s.split('_')[0] for s in scenes if s not in holdout],
                      'holdout_main_scenes': [s.split('_')[0] for s in holdout]})
    return {'version': 1, 'seed': SEED, 'assignment': 'SHA-256 rank then contiguous six-scan blocks',
            'folds': folds}


def source_partition(scene: str) -> Path:
    master = MASTER.read_text().split()
    require(scene in master, 'scene outside new397')
    name = 'first199' if master.index(scene) < 199 else 'remaining198'
    return BASE / f'fi1_A_{name}_20260917'


def link(path: Path, target: Path) -> None:
    target = target.resolve()
    require(target.exists(), 'missing source asset: ' + str(target))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.symlink_to(target, target_is_directory=target.is_dir())
    require(path.resolve() == target, 'adapter target changed: ' + str(path))


def freeze_selection() -> tuple[list[str], dict]:
    master = MASTER.read_text().split()
    split = read(PAIRWISE_SPLIT)
    excluded = set(split['holdout_main_scenes'])
    scenes = selected_scenes(master, excluded)
    folds = scene_folds(scenes)
    control = {
        'dataset_role': ROLE,
        'seed': SEED,
        'pilot_scenes': scenes[:1],
        'active_arms': ['D'],
        'frontend': 'grounded_sam',
        'stride': 10,
        'selection_uses': 'scan/main-scene names only; no labels, assets, losses, or AP',
        'excluded_pairwise_main_scenes': sorted(excluded),
        'auto_expand_after_single_scene_gate': True,
        'new_training': False,
        'AP': False,
    }
    freeze_json(SELECTION / 'selection.json', control)
    scene_text = '\n'.join(scenes) + '\n'
    scene_path = SELECTION / 'scenes_execution_order.txt'
    if scene_path.exists():
        require(scene_path.read_text() == scene_text, 'frozen scene order changed')
    else:
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        scene_path.write_text(scene_text)
    freeze_json(SELECTION / 'folds.json', folds)
    receipt = {
        'complete': True,
        'inputs': {str(MASTER): sha(MASTER), str(PAIRWISE_SPLIT): sha(PAIRWISE_SPLIT)},
        'outputs': {str(path.relative_to(SELECTION)): sha(path) for path in (
            SELECTION / 'selection.json', scene_path, SELECTION / 'folds.json')},
    }
    freeze_json(SELECTION / 'selection_receipt.json', receipt)
    expanded = fixed_selection(SELECTION)
    require(expanded['scenes'] == scenes and expanded['stride'] == 10, 'expanded selection differs')
    return scenes, expanded


def build_adapters(scenes: list[str]) -> str:
    historical = read(ROOT / '.local/fi1_r3b_val312_preparation_20260914/inventory/input_sources.json')
    alpha = historical['sources']['val312']['alpha_cache']
    freeze_json(PREPARATION / 'inventory/input_sources.json', {
        'config_path': str(CONFIG),
        'sources': {ROLE: {'prepared_root': str(DATA),
                           'records_root': str(PREPARATION / 'records_adapter'),
                           'alpha_cache': alpha}},
    })
    for scene in scenes:
        source = source_partition(scene)
        source_binding = source / 'preparation/native_bindings' / scene / 'receipt.json'
        require(read(source_binding)['scene'] == scene, 'source native binding scene differs')
        link(PREPARATION / 'records_adapter' / scene,
             source / 'preparation/records_adapter' / scene)
        inputs = {str(source_binding): sha(source_binding)}
        for name in ('masks', 'classes', 'scores'):
            filename = f'{scene}_pred_{name}.npy'
            target = source / 'common/01_native_cache' / filename
            bound = PREPARATION / 'records_adapter' / scene / 'native_cache' / filename
            link(COMMON / '01_native_cache' / filename, target)
            require(bound.resolve() == target.resolve(), 'records/native target differs')
            inputs[str(bound)] = sha(target)
        freeze_json(PREPARATION / 'native_bindings' / scene / 'receipt.json', {
            'dataset_role': ROLE, 'scene': scene, 'source_partition': str(source),
            'input_files': inputs,
        })
        for folder in ('03_d1_hierarchy_safe', '11_yoloworld_sam_uniform30'):
            link(COMMON / folder / scene, source / 'common' / folder / scene)
        link(YOLO / scene, source / 'yolo' / scene)
    return alpha


def prepare_frontend(selection: dict) -> None:
    freeze_json(FRONTEND / 'selection.json', selection)
    scenes_text = '\n'.join(selection['scenes']) + '\n'
    scenes_path = FRONTEND / 'scenes.txt'
    if scenes_path.exists():
        require(scenes_path.read_text() == scenes_text, 'frontend scene order changed')
    else:
        scenes_path.parent.mkdir(parents=True, exist_ok=True)
        scenes_path.write_text(scenes_text)
    inventory = []
    for scene in selection['scenes']:
        manifest = build_scene_manifest(scene, DATA, DATA)
        g10 = list(range(0, manifest['original_frame_count'], 10))
        require(g10 and all(frame % 10 == 0 for frame in g10), 'invalid G10 frames')
        manifest['arms']['D'] = g10
        manifest['derived_variant'] = {
            'name': 'grounded_sam_stride10_matched_pilot30',
            'compatible_arm': 'D', 'stride': 10,
            'new_visual_inference': True,
        }
        freeze_json(FRONTEND / 'manifests' / f'{scene}.json', manifest)
        inventory.append({'scene': scene, 'ready': True, 'G10_frames': len(g10),
                          'original_frame_count': manifest['original_frame_count']})
    asset_hashes = assets()
    freeze_json(FRONTEND / 'contract.json', {
        'selection_sha256': digest(selection), 'assets': asset_hashes,
        'variant': 'grounded_sam_stride10_matched_pilot30', 'stride': 10,
        'runner': str(Path(__file__).resolve()), 'runner_sha256': sha(__file__),
        'ground_truth_usage': 'none', 'new_visual_inference': True,
    })
    write_json(FRONTEND / 'preflight.json', {
        'state': 'ready_for_export', 'ready_scene_count': len(inventory),
        'required_scene_count': len(inventory), 'inventory': inventory,
        'dataset_root': str(DATA), 'ground_truth_usage': 'none',
        'real_inference_complete': False, 'AP_evaluated': False,
    })


def allocated(path: Path) -> int:
    total = 0
    for base, _, files in os.walk(path, followlinks=False):
        for name in files:
            item = Path(base) / name
            if not item.is_symlink():
                total += item.stat().st_blocks * 512
    return total


def audit_scene(folder: Path, scene: str) -> dict:
    from tools.prepare_fi1_r3b_val312 import verify
    manifest = read(FRONTEND / 'manifests' / f'{scene}.json')
    require(manifest['arms']['D'] == list(range(0, manifest['original_frame_count'], 10)),
            'frontend is not strict G10')
    for stage in ('hierarchy', 'tracks', 'consensus', 'merged'):
        receipt = FRONTEND / 'arms/D/stage_receipts' / f'{scene}_{stage}.json'
        require(receipt.is_file(), 'missing G10 frontend stage: ' + stage)
    backend = folder / 'backend'
    backend_receipt = read(backend / 'D' / scene / 'receipt.json')
    verify(backend_receipt['inputs'])
    verify({str(backend / 'D' / scene / name): value
            for name, value in backend_receipt['outputs'].items()})
    own = folder / 'own/D'
    binding = own / 'geometry_binding'
    evidence = own / 'alpha_completed'
    require(read(evidence / 'summary.json')['complete'], 'Alpha evidence incomplete')
    geometry = read(binding / scene / 'geometry.json')
    evidence_rows = [json.loads(line) for line in (evidence / scene / 'evidence.jsonl').read_text().splitlines()]
    require(len(geometry) == len(evidence_rows), 'geometry/evidence count differs')
    by_key = {row['geometry_key']: row for row in geometry}
    require(set(by_key) == {row['geometry_key'] for row in evidence_rows},
            'geometry/evidence keys differ')
    for row in evidence_rows:
        require(row['points_sha256'] == by_key[row['geometry_key']]['points_sha256'],
                'own evidence point set differs')
        require(row['yolo']['acquisition_complete'] and row['alpha']['acquisition_complete'],
                'own evidence acquisition incomplete')
    result = {
        'complete': True, 'scene': scene, 'strict_stride': 10,
        'G10_frames': len(manifest['arms']['D']), 'geometry_count': len(geometry),
        'alpha_available': sum(row['alpha']['available'] for row in evidence_rows),
        'yolo_available': sum(row['yolo']['available'] for row in evidence_rows),
        'allocated_bytes': allocated(folder), 'ground_truth_read': False,
        'training_run': False, 'AP_run': False,
        'audited_at': datetime.datetime.now().astimezone().isoformat(),
    }
    write_json(folder / 'audit.json', result)
    return result


def run(check_only: bool = False) -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    scenes, selection = freeze_selection()
    alpha = build_adapters(scenes)
    prepare_frontend(selection)
    if check_only:
        print(json.dumps({'complete': True, 'scenes': scenes, 'folds': scene_folds(scenes),
                          'output': str(OUT)}, ensure_ascii=False, indent=2))
        return

    code_paths = [Path(__file__), ROOT / 'tools/run_fi1_frontend_comparison.py',
                  ROOT / 'tools/evaluate_fi1_frontend_comparison.py',
                  ROOT / 'tools/export_fi1_r0_r1_yolo_views.py',
                  ROOT / 'tools/bind_fi1_r0_r1_alpha_cache.py',
                  ROOT / 'tools/complete_fi1_r0_r1_alpha.py']
    code = {str(path.resolve()): sha(path) for path in code_paths}
    old = read(ROOT / '.local/fi1_r0_r4_preparation_20260913/dv3_generation_v2/jobs.json')['environment']
    gpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    cpu_env = dict(gpu_env, CUDA_VISIBLE_DEVICES='')
    legacy_env = {**gpu_env, **old['environment']}
    legacy_env['PYTHONPATH'] = str(ROOT) + ':' + ':'.join(legacy_env['PYTHONPATH'].split(':')[1:])
    state = {'status': 'running', 'pid': os.getpid(), 'started_at': datetime.datetime.now().astimezone().isoformat(),
             'completed': [], 'total': len(scenes), 'pilot': scenes[:1], 'code': code,
             'selection': str(SELECTION), 'folds': str(SELECTION / 'folds.json'), 'stage_history': []}

    def command(stage: str, argv: list, env=cpu_env) -> None:
        for path, expected in code.items():
            require(sha(path) == expected, 'source changed during run: ' + path)
        require(shutil.disk_usage(BASE).free >= RESERVE, 'less than 50 GiB free')
        state.update(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat())
        write_json(LOCAL / 'execution.json', state)
        logfile = LOCAL / 'logs' / f'{stage}.log'
        logfile.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        print('START', stage, flush=True)
        with logfile.open('a') as log:
            proc = subprocess.Popen(list(map(str, argv)), cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    state.update(child_pid=proc.pid,
                                 updated_at=datetime.datetime.now().astimezone().isoformat(),
                                 free_bytes=shutil.disk_usage(BASE).free)
                    write_json(LOCAL / 'execution.json', state)
                    require(state['free_bytes'] >= RESERVE, 'disk reserve reached')
                    time.sleep(5)
                require(proc.returncode == 0, stage + ' failed; see ' + str(logfile))
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        state['stage_history'].append({'stage': stage, 'seconds': time.monotonic() - started})
        print('DONE', stage, state['stage_history'][-1]['seconds'], flush=True)

    def tool(python: Path, name: str, *argv) -> list:
        return [python, ROOT / 'tools' / name, *argv]

    with (LOCAL / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for index, scene in enumerate(scenes, 1):
                state.update(current=scene, index=index)
                folder = OUT / 'scenes' / scene
                folder.mkdir(parents=True, exist_ok=True)
                scene_list = folder / 'scene.txt'
                scene_list.write_text(scene + '\n')
                own = folder / 'own/D'
                if not (own / 'alpha_completed/summary.json').exists():
                    command(scene + '_G10_frontend', tool(
                        PY, 'run_fi1_frontend_comparison.py', 'export', '--output', FRONTEND,
                        '--dataset-root', DATA, '--scenes', scene, '--arms', 'D'), gpu_env)
                    common = ['--selection-root', SELECTION, '--dataset-root', DATA,
                              '--full-preparation', PREPARATION, '--yolo-cache-root', YOLO,
                              '--scenes', scene, '--arms', 'D']
                    command(scene + '_D_v3', tool(
                        Path(old['python']), 'evaluate_fi1_frontend_comparison.py', 'full-geometry',
                        *common, '--run', FRONTEND, '--output', folder / 'backend',
                        '--legacy-root', COMMON, '--backend-config', CONFIG), legacy_env)
                    command(scene + '_bind', tool(
                        PY, 'evaluate_fi1_frontend_comparison.py', 'full-bind', *common,
                        '--backend-root', folder / 'backend', '--output', folder / 'own'))
                    command(scene + '_yolo_views', tool(
                        PY, 'export_fi1_r0_r1_yolo_views.py', '--binding', own / 'geometry_binding',
                        '--sources', own / 'sources.json', '--snapshot', SNAPSHOT,
                        '--output', own / 'views'))
                    command(scene + '_alpha_bind', tool(
                        PY, 'bind_fi1_r0_r1_alpha_cache.py', '--views', own / 'views',
                        '--cache', alpha, '--output', own / 'alpha_bound', '--allow-new-scenes'))
                    if not (own / 'alpha_completed/plan.json').exists():
                        command(scene + '_alpha_plan', tool(
                            PY, 'complete_fi1_r0_r1_alpha.py', 'prepare', '--bound', own / 'alpha_bound',
                            '--snapshot', SNAPSHOT, '--output', own / 'alpha_completed'))
                    command(scene + '_alpha_gpu', tool(
                        PY, 'complete_fi1_r0_r1_alpha.py', 'run',
                        '--plan', own / 'alpha_completed/plan.json'), gpu_env)
                audit = audit_scene(folder, scene)
                state['completed'].append(audit)
                write_json(LOCAL / 'execution.json', state)
                if index == 1:
                    used = allocated(OUT)
                    projected = audit['allocated_bytes'] * (len(scenes) - 1)
                    require(shutil.disk_usage(BASE).free >= RESERVE + int(projected * 1.2),
                            'single-scene gate passed but projected storage exceeds reserve')
                    write_json(LOCAL / 'pilot_gate.json', {
                        'complete': True, 'scene': scene, 'audit': audit,
                        'allocated_bytes_total': used, 'projected_remaining_bytes': projected,
                        'auto_expand_scene_count': len(scenes),
                        'passed_at': datetime.datetime.now().astimezone().isoformat(),
                    })
                print('SCENE COMPLETE', index, len(scenes), scene, flush=True)
            state.update(status='complete', current=None,
                         finished_at=datetime.datetime.now().astimezone().isoformat())
        except BaseException as exc:
            state.update(status='failed', error=repr(exc),
                         updated_at=datetime.datetime.now().astimezone().isoformat())
            write_json(LOCAL / 'execution.json', state)
            raise
        write_json(LOCAL / 'execution.json', state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Freeze/audit contracts without GPU inference')
    args = parser.parse_args()
    run(check_only=args.check)


if __name__ == '__main__':
    main()
