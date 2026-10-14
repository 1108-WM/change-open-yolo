"""Frozen-reference counterfactuals evaluated by the original ScanNet evaluator.

These are development diagnostics, not scene-isolated training targets. No
custom matching or AP formula is used. Each result is conditional on one
explicit action and the complete reference scene set.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np

from tools.fi1_gain_attribution_core import (
    Bindings, configure_evaluator, points_from_population, read_rows, sha,
    sparse_assign, unique, write_json, write_rows,
)

VERSION = 'fi1_evaluator_counterfactual_v2'
ACTIONS = ('restore_existing', 'append_new', 'rescore_existing')
IDENTITY = ('prediction_id', 'scene_name', 'geometry_key', 'class_index')


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} outside valid range')
    return value


def finite(value, name='score'):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be numeric')
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


def validate_prediction(row, foreground=False):
    pid = row.get('prediction_id')
    if isinstance(pid, bool) or not isinstance(pid, (int, str)) or pid == '' or (isinstance(pid, int) and pid < 0):
        raise ValueError('missing or invalid prediction_id')
    for name in ('scene_name', 'geometry_key'):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f'missing {name}')
    integer(row['class_index'], 'class_index', maximum=197 if foreground else 198)
    finite(row['score'])


def identity_equal(left, right):
    return all(type(left.get(k)) is type(right.get(k)) and left.get(k) == right.get(k) for k in IDENTITY)


def indexed_b1(rows):
    """Bind stable global IDs to frozen file order; retain scene column order."""
    output = []
    columns = defaultdict(list)
    for pid, row in enumerate(rows):
        if 'prediction_id' in row and row['prediction_id'] != pid:
            raise ValueError('B1 prediction_id differs from frozen file order')
        item = {**row, 'prediction_id': pid}
        validate_prediction(item)
        col = integer(item['column_index'], 'column_index')
        columns[item['scene_name']].append(col)
        output.append(item)
    if not output:
        raise ValueError('empty B1')
    for scene, cols in columns.items():
        if cols != list(range(len(cols))):
            raise ValueError(f'B1 column order differs: {scene}')
    return output


def intervention(baseline, request, action):
    """Return exact before/after ordered member lists without mutating B1."""
    if action not in ACTIONS:
        raise ValueError('unknown action')
    unique(baseline, 'prediction_id')
    for row in baseline:
        validate_prediction(row)
    validate_prediction(request, foreground=True)
    found = [i for i, r in enumerate(baseline) if r['prediction_id'] == request['prediction_id']]
    if action == 'append_new':
        if found:
            raise ValueError('append prediction_id already in B1')
        if any(r['geometry_key'] == request['geometry_key'] for r in baseline):
            raise ValueError('append_new requires geometry outside B1')
        return list(baseline), [*baseline, dict(request)]
    if len(found) != 1:
        raise ValueError('existing prediction_id absent from B1')
    index = found[0]
    original = baseline[index]
    if not identity_equal(original, request) or finite(request['score']) != finite(original['score']):
        raise ValueError('existing member identity/score differs from B1')
    if action == 'restore_existing':
        return baseline[:index] + baseline[index + 1:], list(baseline)
    after = list(baseline)
    after[index] = {**original, 'score': finite(request['new_score'], 'new_score')}
    return list(baseline), after


def metric_values(array, cid, evaluator):
    # Use the evaluator's aggregation too. Changing reduction order can change
    # the last floating-point bit even when the per-class AP arrays are exact.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        averages = evaluator.compute_averages(array)
    cls = averages['classes'][evaluator.CLASS_LABELS[cid]]
    values = {'class_ap': cls['ap'], 'class_ap50': cls['ap50%'],
              'overall_ap': averages['all_ap'], 'overall_ap50': averages['all_ap_50%']}
    return {name: float(value) if np.isfinite(value) else None for name, value in values.items()}


class CounterfactualEvaluator:
    def __init__(self, baseline, caches, evaluator):
        unique(baseline, 'prediction_id')
        self.baseline = baseline
        self.caches = caches
        self.evaluator = evaluator
        self.by_scene = {s: [] for s in caches}
        for row in baseline:
            validate_prediction(row)
            if row['scene_name'] not in caches:
                raise ValueError('prediction outside reference scene set')
            self.by_scene[row['scene_name']].append(row)
        # Even scenes without any predictions remain in the GT reference.
        self.matches = {s: self.assign(s, rows) for s, rows in self.by_scene.items()}
        self.baseline_array = evaluator.evaluate_matches(self.matches)[0]

    def assign(self, scene, rows):
        cache = self.caches[scene]
        return sparse_assign(rows, cache['geometry_stats'], cache['gt_inventory'], self.evaluator, scene)

    def evaluate(self, request, action):
        before, after = intervention(self.baseline, request, action)
        scene = request['scene_name']
        if scene not in self.caches:
            raise ValueError('candidate outside reference scene set')
        if request['geometry_key'] not in self.caches[scene]['geometry_stats']:
            raise ValueError('candidate geometry absent from scene cache')

        def array(rows, is_baseline):
            if is_baseline:
                return self.baseline_array.copy()
            matches = dict(self.matches)
            matches[scene] = self.assign(scene, [r for r in rows if r['scene_name'] == scene])
            return self.evaluator.evaluate_matches(matches)[0]

        before_array = array(before, action != 'restore_existing')
        after_array = array(after, action == 'restore_existing')
        b = metric_values(before_array, request['class_index'], self.evaluator)
        a = metric_values(after_array, request['class_index'], self.evaluator)
        result = {k: request[k] for k in IDENTITY}
        result.update(version=VERSION, action=action, evaluation_scope='full_reference_scenes',
                      reference_scene_count=len(self.caches), before_count=len(before), after_count=len(after),
                      score_before=None if action == 'append_new' else request['score'],
                      score_after=request['new_score'] if action == 'rescore_existing' else request['score'],
                      before=b, after=a,
                      delta={k: None if b[k] is None or a[k] is None else a[k] - b[k] for k in b},
                      training_ready=False)
        return result, before_array, after_array


def load_reference(b1_path, attribution_root, audit_path, bindings):
    """Require the historical audit binding, all 100 caches, and frozen B1."""
    root = Path(attribution_root)
    audit = json.loads(bindings.check(audit_path).read_text())
    if audit.get('audit_valid') is not True or audit.get('all_seven_ap_arrays_exactly_replayed') is not True:
        raise ValueError('historical attribution audit not accepted')
    summary_path = bindings.check(root / 'summary.json', audit['summary_sha256'])
    summary = json.loads(summary_path.read_text())
    if summary['ground_truth_scope'] != 'official100 only' or summary['scene_count'] != 100:
        raise ValueError('only frozen official100 diagnostic reference is supported')
    hashes = summary['hashes']
    b1 = indexed_b1(read_rows(bindings.check(b1_path, hashes['b1_frozen_predictions.jsonl'])))
    names = sorted(k for k in hashes if k.startswith('scene_cache/') and k.endswith('.json'))
    if len(names) != 100:
        raise ValueError('reference must include all 100 scene caches')
    caches = {}
    for name in names:
        scene = Path(name).stem
        if name != f'scene_cache/{scene}.json':
            raise ValueError('invalid cache path')
        cache = json.loads(bindings.check(root / name, hashes[name]).read_text())
        if cache['scene_name'] != scene or scene in caches:
            raise ValueError('cache scene identity differs')
        caches[scene] = cache
    if {r['scene_name'] for r in b1} - caches.keys():
        raise ValueError('B1 outside official100 reference')
    return b1, caches, summary


def load_population(root, bindings):
    root = Path(root)
    summary = json.loads(bindings.check(root / 'summary.json').read_text())
    pop = unique(read_rows(bindings.check(root / 'population.jsonl', summary['hashes']['population.jsonl'])), 'geometry_key')
    from evaluate.scannet200 import eval_semantic_instance as ev
    configure_evaluator(ev)
    if summary['class_names'] != list(ev.CLASS_LABELS):
        raise ValueError('population vocabulary differs from evaluator')
    for key, row in pop.items():
        integer(row['fold_index'], 'fold_index', maximum=4)
        if key != f"{row['scene_name']}:geometry:{row['points_sha256']}":
            raise ValueError('population geometry key/hash differs')
    return pop, summary


def validate_population_requests(requests, baseline, population, caches, root, bindings, action):
    unique(requests, 'prediction_id')
    b1keys = {r['geometry_key'] for r in baseline}
    seen_pairs = set()
    checked_points = set()
    for request in requests:
        validate_prediction(request, foreground=True)
        key = request['geometry_key']
        pop = population[key]
        if request['scene_name'] != pop['scene_name']:
            raise ValueError('candidate/population scene differs')
        if request.get('fold_index') != pop['fold_index']:
            raise ValueError('candidate/population fold differs')
        cache = caches[pop['scene_name']]
        if cache['points_sha256'][key] != pop['points_sha256']:
            raise ValueError('population/cache point hash differs')
        if cache['geometry_stats'][key]['point_count'] != pop['point_count']:
            raise ValueError('population/cache point count differs')
        if action == 'append_new':
            pair = (key, request['class_index'])
            if pair in seen_pairs:
                raise ValueError('duplicate new geometry-class hypothesis')
            seen_pairs.add(pair)
            if key in b1keys or pop['is_new_geometry'] is not True or pop['b1_members']:
                raise ValueError('candidate is not new geometry')
            if request.get('is_new_geometry') is not True or request.get('source') != pop['sources']:
                raise ValueError('new candidate source differs from population')
        elif key not in b1keys:
            raise ValueError('existing candidate not in B1')
        if key not in checked_points:
            points_from_population(Path(root), pop, bindings)
            checked_points.add(key)
        intervention(baseline, request, action)


def code_bindings():
    root = Path(__file__).resolve().parents[3]
    names = ['tools/fi1_evaluator_delta_core.py', 'tools/fi1_gain_attribution_core.py',
             'tools/audit_fi1_candidate_evaluator_delta.py', 'tools/audit_fi1_new_geometry_evaluator_delta.py',
             'evaluate/scannet200/eval_semantic_instance.py', 'evaluate/scannet200/scannet_constants.py',
             'evaluate/scannet200/util_3d.py']
    return {name: sha(root / name) for name in names}


def run_audit(args, action, requests):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'run_state.json', {'status': 'running', 'version': VERSION})
    try:
        bind = Bindings()
        b1, caches, summary = load_reference(args.b1_predictions, args.attribution_root, args.attribution_audit, bind)
        pop, _ = load_population(args.population_root, bind)
        if {r['scene_name'] for r in pop.values()} != set(caches):
            raise ValueError('population/reference scene set differs')
        selected = requests(b1, pop, bind)
        if not selected:
            raise ValueError('no explicit diagnostic requests')
        validate_population_requests(selected, b1, pop, caches, args.population_root, bind, action)
        from evaluate.scannet200 import eval_semantic_instance as ev
        configure_evaluator(ev)
        engine = CounterfactualEvaluator(b1, caches, ev)
        historical = np.load(bind.check(Path(args.attribution_root) / 'ap_arrays/O0_b1.npy',
                                       summary['hashes']['ap_arrays/O0_b1.npy']), allow_pickle=False)
        if not np.array_equal(engine.baseline_array, historical, equal_nan=True):
            raise ValueError('B1 AP array differs from historical control')
        reference_id = hashlib.sha256(json.dumps({'b1': bind.files[str(Path(args.b1_predictions).resolve())],
            'caches': {s: summary['hashes'][f'scene_cache/{s}.json'] for s in caches}}, sort_keys=True).encode()).hexdigest()
        write_rows(out / 'reference_predictions.jsonl', b1)
        write_rows(out / 'actions.jsonl', selected)
        np.save(out / 'baseline_ap.npy', engine.baseline_array, allow_pickle=False)
        rows = []
        arrays = out / 'ap_arrays'
        arrays.mkdir()
        for i, request in enumerate(selected):
            result, before, after = engine.evaluate(request, action)
            result.update(reference_id=reference_id, array_file=f'ap_arrays/{i:04d}.npz')
            np.savez(out / result['array_file'], before=before, after=after)
            rows.append(result)
            print(f'{action}: {i + 1}/{len(selected)} evaluated over {len(caches)} scenes', flush=True)
        write_rows(out / 'evaluator_delta.jsonl', rows)
        outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run_state.json'}
        write_json(out / 'summary.json', {'version': VERSION, 'action': action, 'reference_id': reference_id,
            'evaluation_scope': 'full_reference_scenes', 'reference_scenes': list(caches),
            'thresholds': list(map(float, ev.opt['overlaps'])), 'candidate_count': len(rows),
            'b1_mean_ap': metric_values(engine.baseline_array, 0, ev)['overall_ap'],
            'baseline_array_exactly_replayed': True, 'input_sha256': bind.files,
            'code_sha256': code_bindings(), 'hashes': outputs, 'numpy_version': np.__version__,
            'training_ready': False, 'model_fitted': False, 'b2_written': False,
            'limitations': ['single-action development diagnostics, not a new method result',
                'full official100 GT reference must not be used as held-out training targets',
                'historical supervised upstream is frozen, not rebuilt OOF',
                'candidate scores are supplied controls, not learned calibration',
                'single-action deltas are not additive']})
        write_json(out / 'run_state.json', {'status': 'complete', 'summary_sha256': sha(out / 'summary.json')})
    except Exception as exc:
        write_json(out / 'run_state.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
