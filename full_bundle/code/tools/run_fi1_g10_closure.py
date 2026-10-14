#!/usr/bin/env python3
"""Serial, resumable GD10 train597 -> B/Q -> dev55 -> full312 closure."""
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.run_fi1_g10_closure_assets import LOCAL, OUT, BASE, PY, RESERVE, read, require, now
from tools.fi1_frontend_protocol import freeze_json, sha256_file as sha, write_json as write
from tools.run_fi1_train597 import verify


def stages():
    result = [('train_assets', 'run_fi1_g10_closure_assets.py', ['run', '--groups', 'train']),
              ('train_features', 'run_fi1_g10_closure_learning.py', ['prepare', '--group', 'train']),
              ('fit_B_Q', 'run_fi1_g10_closure_learning.py', ['fit'])]
    for group in ('dev', 'val'):
        result.append((group + '_assets', 'run_fi1_g10_closure_assets.py', ['run', '--groups', group]))
        for name, command in (('features', 'prepare'), ('predictions', 'predict'), ('AP', 'evaluate')):
            result.append((group + '_' + name, 'run_fi1_g10_closure_learning.py', [command, '--group', group]))
    return result


def stage_outputs(name):
    inventory = read(LOCAL / 'inventory.json')
    if name.endswith('_assets'):
        group = name.split('_')[0]
        state = read(LOCAL / 'assets_execution.json')
        require(state['status'] == 'complete' and state['groups'] == [group], 'assets stage not complete')
        return [Path(inventory['jobs'][s]['folder']) / 'audit.json' for s in inventory['groups'][group]]
    if name == 'fit_B_Q':
        return [OUT / 'learning/models/summary.json', OUT / 'learning/models/B.json', OUT / 'learning/models/Q.json']
    group, tail = name.split('_', 1)
    suffix = {'features': 'dataset', 'predictions': 'predictions', 'AP': 'summary'}[tail]
    return [OUT / 'learning' / (group + '_' + suffix + '.json')]


def main():
    audit = read(LOCAL / 'pipeline_code_audit.json')
    verify(audit['code'])
    inventory = read(LOCAL / 'inventory.json')
    require(audit['inventory_sha256'] == sha(LOCAL / 'inventory.json'), 'pipeline inventory differs')
    contract = {'code': audit['code'], 'inventory_sha256': audit['inventory_sha256'],
        'stages': [{'name': n, 'script': script, 'args': args} for n, script, args in stages()],
        'train': 597, 'dev': 55, 'val': 312, 'full312_authorized': True, 'Q_primary_B_control': True,
        'GT_policy': 'train labels only in training; heldout GT only after frozen predictions',
        'models_and_features_frozen_before_evaluation': True, 'parameter_search': False,
        'single_GPU_child': True, 'reserve_bytes': RESERVE}
    freeze_json(LOCAL / 'pipeline_contract.json', contract)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    state = {'status': 'running', 'started_at': now(), 'pid': os.getpid(), 'completed_stages': [],
             'stage_count': len(stages()), 'contract_sha256': sha(LOCAL / 'pipeline_contract.json')}
    with (LOCAL / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # These are correctness/storage gates, never validation-AP selection gates.
            pilot = read(OUT / 'learning/pilot_models/summary.json')
            require(pilot['complete'] and pilot['pilot'] and pilot['fit_scenes'] == inventory['groups']['train'][:3], 'training pilot missing')
            verify(pilot['receipts'])
            for group in ('train', 'dev'):
                gate = read(LOCAL / (group + '_pilot_gate.json'))
                require(gate['complete'] and gate['scenes'] == inventory['groups'][group][:3], 'asset pilot gate missing')
            for name, script, args in stages():
                verify(audit['code'])
                receipt = LOCAL / 'pipeline_stages' / (name + '.json')
                if receipt.exists():
                    r = read(receipt)
                    require(r['contract_sha256'] == state['contract_sha256'], 'stage belongs to another pipeline')
                    verify(r['outputs'])
                    state['completed_stages'].append(name)
                    continue
                state.update(stage=name, updated_at=now())
                write(LOCAL / 'execution.json', state)
                argv = [str(PY), str(ROOT / 'tools' / script), *args]
                log_path = LOCAL / 'logs' / (name + '.log')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                print('PIPELINE START', name, now(), flush=True)
                with log_path.open('a') as log:
                    process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    try:
                        while process.poll() is None:
                            free = shutil.disk_usage(BASE).free
                            require(free >= RESERVE, 'pipeline disk reserve reached')
                            state.update(updated_at=now(), child_pid=process.pid, free_bytes=free)
                            write(LOCAL / 'execution.json', state)
                            time.sleep(15)
                        require(process.returncode == 0, name + ' failed; see ' + str(log_path))
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try: process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL); process.wait()
                outputs = stage_outputs(name)
                for p in outputs:
                    require(read(p).get('complete') is True, 'incomplete stage output: ' + str(p))
                write(receipt, {'complete': True, 'stage': name, 'contract_sha256': state['contract_sha256'],
                    'argv': argv, 'seconds': time.monotonic() - started, 'finished_at': now(),
                    'outputs': {str(p): sha(p) for p in outputs}})
                state['completed_stages'].append(name)
                print('PIPELINE COMPLETE STAGE', name, flush=True)
            state.update(status='complete', finished_at=now())
            write(LOCAL / 'summary.json', {'complete': True, 'contract_sha256': state['contract_sha256'],
                'AP': {g: read(OUT / 'learning' / f'{g}_summary.json')['AP'] for g in ('dev', 'val')},
                'AP_units': 'fraction', 'matched_G10_training': True, 'train_scans': 597,
                'new_Dv3_training': False, 'new_B_Q_training': True, 'finished_at': now()})
        except BaseException as e:
            state.update(status='failed', error=repr(e), updated_at=now())
            write(LOCAL / 'execution.json', state)
            raise
        write(LOCAL / 'execution.json', state)


if __name__ == '__main__':
    main()
