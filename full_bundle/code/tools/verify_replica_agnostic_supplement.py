#!/usr/bin/env python3
"""Read-only replay of sealed Replica output deduplication and cached AP."""
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluate.replica import eval_semantic_instance as evaluator

OUT = ROOT / '.local/replica_registered_full_20261011_v1'


def read(path):
    return json.loads(path.read_text())


def main():
    ag = types.ModuleType('replica_agnostic_verification')
    exec(compile(Path(evaluator.__file__).read_text(), evaluator.__file__, 'exec'), ag.__dict__)
    ag.CLASS_LABELS = ['object']
    ag.VALID_CLASS_IDS = np.array([1])
    ag.ID_TO_LABEL = {1: 'object'}
    ag.LABEL_TO_ID = {'object': 1}
    ag.PRED_ID_TO_ID = {0: 1}
    receipt = read(OUT / 'evaluation/receipt.json')
    # Verify outputs only: never open raw or transformed GT files.
    checked = 0
    for filename, digest in receipt['outputs'].items():
        path = Path(filename)
        if path.parent.name == 'agnostic_gt':
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, filename
        checked += 1
    scenes = read(OUT / 'contract.json')['scenes']
    stats = {}
    for method in ['baseline_official', 'baseline_native', 'Full']:
        directory = OUT / 'evaluation' / method
        with (directory / 'actual_class_agnostic_matches.pkl').open('rb') as handle:
            matches = pickle.load(handle)
        np.testing.assert_array_equal(ag.evaluate_matches(matches), np.load(directory / 'actual_class_agnostic_ap.npy'))
        stats[method] = {}
        for scene in scenes:
            groups = {}
            if method == 'Full':
                geometry = {g['geometry_key']: g['array_key'] for g in read(OUT / 'own' / scene / 'geometry.json')}
                with np.load(OUT / 'own' / scene / 'points.npz') as arrays:
                    rows = [json.loads(line) for line in (OUT / 'full' / scene / 'selected.jsonl').read_text().splitlines()]
                    for row in rows:
                        if 0 <= row['class_index'] < 48:
                            points = np.asarray(arrays[geometry[row['geometry_key']]], dtype='<i8')
                            key = points.tobytes()  # Exact indices, independently of production mask packing.
                            groups[key] = (len(points), max(row['score'], groups.get(key, (0, -np.inf))[1]))
            else:
                directory_baseline = OUT / 'baseline' / scene
                masks = np.load(directory_baseline / (scene + '_pred_masks.npy'), mmap_mode='r')
                classes = np.load(directory_baseline / (scene + '_pred_classes.npy'))
                scores = np.load(directory_baseline / (scene + '_pred_scores.npy'))
                for i, category in enumerate(classes):
                    if 0 <= category < 48:
                        points = np.flatnonzero(masks[:, i]).astype('<i8')
                        key = points.tobytes()
                        score = 1.0 if method == 'baseline_official' else float(scores[i])
                        groups[key] = (len(points), max(score, groups.get(key, (0, -np.inf))[1]))
                del masks
            expected = [value for value in groups.values() if value[0] >= ag.opt['min_region_sizes'][0]]
            actual = [(p['vert_count'], p['confidence']) for p in matches[scene]['pred']['object']]
            assert expected == actual, (method, scene)
            np.testing.assert_array_equal(ag.evaluate_matches({scene: matches[scene]}), np.load(directory / ('actual_class_agnostic_' + scene + '_ap.npy')))
            stats[method][scene] = {'selected_unique_geometry': len(groups), 'evaluator_predictions': len(actual)}
    result = {'complete': True, 'GT_read': False, 'new_training_or_prediction': False,
              'new_evaluation_experiment': False, 'cached_agnostic_AP_arrays_exact_replay': 27,
              'evaluation_output_hashes_checked': checked,
              'independent_exact_points_selected_max_score': True, 'scenes': stats}
    (OUT / 'agnostic_supplement_verified.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
