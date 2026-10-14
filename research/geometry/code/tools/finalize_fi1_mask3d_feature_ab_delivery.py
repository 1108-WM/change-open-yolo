#!/usr/bin/env python3
"""Verify sealed A/B artifacts and snapshot current handoff; no learning or GT reads."""
from pathlib import Path
import hashlib
import json
import shutil
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / '.local/fi1_mask3d_feature_ab_internal_20261013_v1'
DOCS = [
    'CURRENT_EXPERIMENT_STATUS.md', '新开对话阅读内容.md',
    '资料/结果.md', '资料/当前基线修改方向.md',
    '资料/代码文件索引与当前主线地图.md',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(mapping):
    for path, digest in mapping.items():
        assert sha(path) == digest, path


def save(path, value):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def finalize():
    assert not (OUT / 'delivery_verified.json').exists()
    for path in (ROOT, OUT.resolve()):
        assert shutil.disk_usage(path).free >= 50 * 2**30
    c = read(OUT / 'contract.json')
    complete = read(OUT / 'completion_verified.json')
    supplemental = read(OUT / 'supplemental_verified.json')
    independent = read(OUT / 'independent_verified.json')
    assert complete['complete'] and complete['experiment_finished']
    assert independent['complete'] and supplemental['complete']
    assert sha(OUT / 'completion_verified.json') == supplemental['primary_completion_sha256']
    assert sha(OUT / 'independent_verified.json') == complete['independent_sha256']
    check(c['code'])
    check(c['frozen_SL_models'])
    check(supplemental['source_outputs_unchanged'])
    check(supplemental['outputs'])
    assert sha(ROOT / 'tools/report_fi1_mask3d_feature_ab.py') == supplemental['reporting_code_sha256']
    artifacts = {}
    for stage in ('instance_features', 'models', 'probabilities', 'predictions', 'evaluation'):
        p = OUT / (stage + '_frozen.json')
        receipt = read(p)
        assert receipt['complete']
        check(receipt['outputs'])
        artifacts[str(p)] = sha(p)
    interpretation = read(OUT / 'interpretation.json')
    assert interpretation['complete']
    assert not interpretation['new_training']
    assert not interpretation['new_rules']
    assert not interpretation['new_predictions']
    snapshot = OUT / 'handoff_final_20261013'
    assert not snapshot.exists()
    documents = {}
    for name in DOCS:
        p = ROOT / name
        assert 'fi1_mask3d_feature_ab_internal_20261013_v1' in p.read_text()
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        assert sha(dest) == sha(p)
        documents[str(p)] = {'sha256': sha(p), 'snapshot': str(dest)}
    sources = {}
    for name in ('tools/report_fi1_mask3d_feature_ab.py',
                 'tools/finalize_fi1_mask3d_feature_ab_delivery.py'):
        p = ROOT / name
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        assert sha(dest) == sha(p)
        sources[str(p)] = {'sha256': sha(p), 'snapshot': str(dest)}
    handoff = dict(complete=True, updated=time.time(), documents=documents,
                   supplemental_sha256=sha(OUT / 'supplemental_verified.json'),
                   primary_completion_sha256=sha(OUT / 'completion_verified.json'),
                   old_receipts_preserved=True, next_experiment_started=False)
    save(OUT / 'handoff_update_verified_20261013.json', handoff)
    save(OUT / 'delivery_verified.json', dict(
        complete=True, experiment_finished=True, stage_artifacts_checked=artifacts,
        handoff_sha256=sha(OUT / 'handoff_update_verified_20261013.json'),
        reporting_sources=sources, independent_counts=independent['counts'],
        unit_tests=dict(passed=8, previously_run=True,
                        test_source_sha256=sha(ROOT / 'tests/test_fi1_mask3d_feature_ab.py'),
                        command='OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 '
                        '.venvs/yoloe_gpu/bin/python -m pytest -q '
                        'tests/test_fi1_mask3d_feature_ab.py'),
        new_training=False, new_rules=False, new_predictions=False,
        next_experiment_started=False, GT_read=False,
        decision=complete['decision'], disk_reserve_GiB=50))
    print('A/B artifacts and current handoff verified; experiment complete.', flush=True)


if __name__ == '__main__':
    finalize()
