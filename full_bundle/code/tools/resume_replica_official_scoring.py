#!/usr/bin/env python3
"""Use the registered B26 sklearn1.3.2 environment after D-v3 sklearn1.3.0."""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '.local/replica_official_masks_full_20261011_v1'
PYTHON = ROOT / '.venvs/yoloe_gpu/bin/python'


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    failure = OUT / 'score_failed_sklearn130.log'
    if not failure.exists():
        shutil.copyfile(OUT / 'score.log', failure)
        shutil.copyfile(OUT / 'execution.json', OUT / 'execution_failed_sklearn130.json')
    environment = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '4',
                   'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    write('B26_environment_resume.json', {'reason': 'D-v3 sklearn1.3.0 cannot deserialize B26 tree node dtype for prediction',
          'B26_python': str(PYTHON), 'sklearn': '1.3.2', 'same_as_previous_B26_inference': True,
          'completed_geometry_and_own_evidence_reused': True, 'models_unchanged': True, 'new_training': False})
    for stage in ['score', 'evaluate', 'verify']:
        write('execution.json', {'running': True, 'stage': stage, 'B26_environment': 'sklearn1.3.2'})
        with (OUT / (stage + '_resume.log')).open('a') as log:
            result = subprocess.run([str(PYTHON), '-u', 'tools/run_replica_official_masks_full.py', stage],
                     cwd=ROOT, env={**environment, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write('execution.json', {'running': False, 'failed': True, 'stage': stage, 'exit_code': result.returncode})
            raise SystemExit(result.returncode)
    write('execution.json', {'running': False, 'complete': True, 'results': str(OUT / 'evaluation/results.json')})


if __name__ == '__main__':
    main()
