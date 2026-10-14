#!/usr/bin/env python3
"""Fit two fixed-capacity diagnostic heads on fit-only action targets.

No heldout target, model selection, score policy or B2 output is consumed or
produced here. The sparse positive coverage remains an explicit limitation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fi1_action_training_data import load_complete_dataset, effective_training_rows, coverage_report, FAMILIES
from tools.fi1_evaluator_delta_core import finite
from tools.fi1_gain_attribution_core import Bindings, sha, write_json, write_rows
from tools.fi1_scene_reference import main_scene

PARAMS = {'loss': 'squared_error', 'learning_rate': .05, 'max_iter': 80,
          'max_leaf_nodes': 4, 'max_depth': 2, 'min_samples_leaf': 5,
          'l2_regularization': 1., 'early_stopping': False, 'random_state': 0}


def feature_matrix(features, names):
    if len(names) != len(set(names)) or any(k in names for k in ('class_index', 'prediction_id', 'scene_name', 'delta', 'target', 'label')):
        raise ValueError('invalid feature schema')
    for row in features:
        if set(row) != set(names):
            raise ValueError('feature schema differs')
    return np.asarray([[np.nan if row[name] is None else finite(row[name], name) for name in names]
                       for row in features], dtype=float).reshape(len(features), len(names))


def balanced_weights(rows):
    """Equal main-scene total, then geometry, candidate and distinct score action."""
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, row in enumerate(rows):
        i = row['identity']
        tree[main_scene(i['scene_name'])][i['geometry_key']][i['prediction_id']].append(index)
    weights = np.empty(len(rows), dtype=float)
    for geometries in tree.values():
        for candidates in geometries.values():
            for indices in candidates.values():
                weights[indices] = len(rows) / (len(tree) * len(geometries) * len(candidates) * len(indices))
    return weights


def fit_family(rows, names, family):
    selected = effective_training_rows(rows, family)
    if len({r['labels']['target_class_ap_delta'] for r in selected}) < 2:
        raise ValueError('nonconstant defined targets required for diagnostic fitting')
    x = feature_matrix([r['features'] for r in selected], names)
    # Scale units to class AP percentage points, without changing any labels/signs.
    target = np.asarray([100 * r['labels']['target_class_ap_delta'] for r in selected])
    weights = balanced_weights(selected)
    model = HistGradientBoostingRegressor(**PARAMS).fit(x, target, sample_weight=weights)
    predicted = model.predict(x)
    if not np.isfinite(predicted).all():
        raise ValueError('nonfinite fitted contribution')
    receipt = {'family': family, 'fit_action_count': len(selected),
        'fit_candidate_count': len({r['identity']['prediction_id'] for r in selected}),
        'fit_main_scene_count': len({main_scene(r['identity']['scene_name']) for r in selected}),
        'target_units': 'class AP percentage points', 'prediction_is_ap_score': False,
        'training_weight_sum': float(weights.sum()),
        'weighted_fit_mae': float(np.average(abs(predicted - target), weights=weights)),
        'weighted_zero_predictor_mae': float(np.average(abs(target), weights=weights)),
        'weighted_fit_rmse': float(np.sqrt(np.average((predicted - target) ** 2, weights=weights))),
        'metrics_scope': 'in_sample_fit_only; not generalization or joint AP',
        'fit_prediction_ids': [r['identity']['prediction_id'] for r in selected]}
    output = [{'identity': r['identity'], 'target_class_ap_delta': r['labels']['target_class_ap_delta'],
               'predicted_class_ap_delta': float(v / 100), 'sample_weight': float(w)}
              for r, v, w in zip(selected, predicted, weights)]
    return model, output, receipt


def train(args):
    bind = Bindings()
    # Explicit fit manifest gate happens before any target-bearing file is opened.
    rows, ref, names, bind = load_complete_dataset(args.fit_dataset_manifest, 'fit', bind)
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'run_state.json', {'status': 'running'})
    protocol = {'version': 'fi1_action_contribution_diagnostic_v1', 'params': PARAMS,
        'families': list(FAMILIES), 'feature_names': names,
        'target': '100 * original class AP delta; retain raw labels including tiny values',
        'weighting': 'equal main scene, geometry, candidate, distinct effective action; mean weight1',
        'missing_values': 'native HGB NaN handling, with existing missing flags; no target imputation',
        'controls': 'skip keep/identical-original-score actions; deduplicate same candidate/proposed score',
        'hyperparameter_search': False, 'rowwise_validation_split': False, 'early_stopping': False,
        'heldout_targets_read': False, 'heldout_features_read': False,
        'scope': 'training plumbing and fit diagnostics only; no policy or AP benefit claim',
        'historical_supervised_upstream_reused': True, 'whole_system_oof': False,
        'fit_reference_id': ref['reference_id'], 'fit_dataset_manifest_sha256': sha(args.fit_dataset_manifest)}
    write_json(out / 'protocol.json', protocol)
    try:
        coverage = coverage_report(rows, ref, names); write_json(out / 'fit_coverage.json', coverage)
        receipts = []
        for family in FAMILIES:
            model, predictions, receipt = fit_family(rows, names, family)
            artifact = {'model': model, 'feature_names': names, 'family': family, 'target_scale': 100.,
                'fit_reference_id': ref['reference_id'], 'fit_main_scenes': sorted({main_scene(r['identity']['scene_name']) for r in rows}),
                'protocol_sha256': sha(out / 'protocol.json'), 'whole_system_oof': False, 'production_ready': False}
            path = out / (family + '.joblib'); joblib.dump(artifact, path)
            reloaded = joblib.load(path)
            effective = effective_training_rows(rows, family)
            replay = reloaded['model'].predict(feature_matrix([r['features'] for r in effective], reloaded['feature_names'])) / reloaded['target_scale']
            if not np.array_equal(replay, [r['predicted_class_ap_delta'] for r in predictions]):
                raise ValueError('saved model cannot exactly replay predictions')
            write_rows(out / (family + '_fit_predictions.jsonl'), predictions)
            receipt.update(model_sha256=sha(path), serialized_predictions_exact=True,
                           gain_coverage=coverage['families'][family])
            receipts.append(receipt)
            print(json.dumps({k: receipt[k] for k in ('family', 'fit_action_count', 'weighted_fit_mae', 'weighted_zero_predictor_mae')}), flush=True)
        summary = {'version': protocol['version'], 'reference_id': ref['reference_id'], 'outer_fold_index': ref['outer_fold_index'],
            'partition_role': 'fit', 'model_count': len(receipts), 'models': receipts,
            'input_sha256': bind.files, 'heldout_targets_read': False, 'heldout_features_read': False,
            'code_sha256': {n: sha(ROOT / n) for n in ('tools/train_fi1_action_contribution.py', 'tools/fi1_action_training_data.py')},
            'model_fitted': True, 'training_ready': False, 'production_ready': False, 'whole_system_oof': False,
            'joint_output_generated': False, 'method_ap_evaluated': False,
            'limitations': ['fit-only diagnostic; neither fitted loss nor numerical replay proves AP benefit',
                'material positive candidates scarce and concentrated; action rows are not independent instances',
                'no action policy or score calibration accepted', 'raw real-scene GT association still relies on audited caches',
                'inner model selection requires separate fit/calibration label references']}
        summary['hashes'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run_state.json'}
        write_json(out / 'summary.json', summary)
        write_json(out / 'run_state.json', {'status': 'complete', 'summary_sha256': sha(out / 'summary.json')})
    except Exception as exc:
        write_json(out / 'run_state.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-dataset-manifest', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    train(p.parse_args())


if __name__ == '__main__':
    main()
