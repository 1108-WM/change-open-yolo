#!/usr/bin/env python3
"""Seal delivery only after independent verification; never refit or overwrite."""
from pathlib import Path
import shutil
import time

import run_fi1_relation_train200_dev55 as r


def main():
    r.guard(gt=True)
    c = r.contract()
    a = r.a
    a.reserve()
    assert not (r.OUT / 'delivery_verified.json').exists()
    verified = a.read(r.OUT / 'verification.json')
    assert verified['complete'] and verified['additional_fits'] == 0
    assert verified['checks']['scene_geometries_and_measurements'] == 255
    for name in ('data_frozen.json', 'training_labels_frozen.json',
                 'predictions_frozen.json', 'evaluation_frozen.json'):
        r.verify(a.read(r.OUT / name)['outputs'])
    additions = ('verification_runtime_addendum.json',
                 'cache_projection_verifier_before_training.json',
                 'data_parallel_plan.json', 'data_parallel_tail_plan.json',
                 'training_label_allowlist_addendum.json')
    code = dict(c['code'])
    for name in additions:
        extra = a.read(r.OUT / name)
        r.verify(extra['code'])
        code.update(extra['code'])
    code[str(Path(__file__).resolve())] = a.sha(__file__)
    old = a.read(r.SPLIT)
    assert c['train_scenes'] == old['train_scenes']
    assert c['scenes'] == old['scenes']
    data = a.read(r.OUT / 'data_frozen.json')
    assert {q['scene'] for q in data['scenes']} == set(c['train_scenes'] + c['scenes'])
    assert all(q['complete'] for q in data['scenes'])
    assert data['reference82_feature_parity'] == c['small_scenes']
    parity = a.read(r.OUT / 'reference82_training_parity_verified.json')
    assert parity['complete'] and parity['original_training_rows'] == 3525
    assert all(parity[k] for k in ('clear_labels_exact', 'features_exact',
                                  'training_membership_exact'))
    corrected = []
    for s in c['train_scenes'] + c['scenes']:
        correction = a.read(r.OUT / 'measurements/predictions' / s / 'cache_corrections.json')
        for q in correction['corrections']:
            corrected.append(dict(scene=s, frame_id=q['frame_id'],
                                  removed_points=len(q['removed_point_ids']),
                                  affected_observations=len(q['affected_observations'])))
    projection = a.read(r.OUT / 'cache_projection_verified.json')
    assert projection['complete'] and corrected == projection['frames']
    receipt = a.read(r.OUT / 'models/receipt.json')
    assert receipt['fits'] == 1 and receipt['baseline_refits'] == 0
    r.verify(receipt['outputs'])
    assert a.sha(r.OUT / 'models/reference82.joblib') == a.sha(r.SMALL / 'models/fold0/double.joblib')
    end = time.time()
    a.seal(r.OUT / 'runtime.json', dict(
        protocol_start_unix=c['time_unix'], completion_unix=end,
        elapsed_wall_seconds=end - c['time_unix'],
        scene_preparation_seconds_sum=sum(q['seconds'] for q in data['scenes']),
        scene_preparation_timing='Sum of per-scene durations; includes concurrent CPU preparation, not wall time.',
        fit_seconds=receipt['seconds'], new_fits=1, baseline_refits=0,
        verification_refits=0, free_disk_bytes=shutil.disk_usage(r.OUT).free))
    # Hash actual files once; directory symlinks are represented separately.
    files = {}
    symlinks = {}
    for p in sorted(r.OUT.rglob('*')):
        if p.relative_to(r.OUT).parts[0] == 'tmp':
            continue
        if p.is_symlink():
            symlinks[str(p)] = str(p.resolve())
        elif p.is_file():
            files[str(p)] = a.sha(p)
    files.update(code)
    a.seal(r.OUT / 'delivery_verified.json', dict(
        complete=True, files=files, symlink_targets=symlinks,
        hash_scope='All new experiment files except temporary files and this seal, plus all task code.',
        recorded_source_files_unchanged=True, source_files=c['source_files'],
        original_split_unchanged=True, train_scans=200, dev_scans=55,
        dev_physical_sites=29, excluded_scenes=[], reference82_exact_parity=True,
        new_fits=1, baseline_refits=0, verification_refits=0,
        AP=False, three_class=False, production_writes=False, old_seal_writes=False,
        verified_checks=verified['checks']))
    delivery = a.read(r.OUT / 'delivery_verified.json')
    r.verify(delivery['files'])
    r.verify(delivery['source_files'])
    assert all(str(Path(p).resolve()) == dest for p, dest in symlinks.items())
    print('DELIVERY VERIFIED', len(files), 'files;', len(symlinks), 'directory links;',
          'train200/dev55; one fit; no AP', flush=True)


if __name__ == '__main__':
    main()
