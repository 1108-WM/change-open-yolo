"""Physical-scene grouping and heldout noninterference for downstream targets."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from evaluate.scannet200 import eval_semantic_instance as ev
from tools.fi1_evaluator_delta_core import CounterfactualEvaluator, configure_evaluator
from tools.fi1_gain_attribution_core import Bindings, geometry_statistics, point_hash, sha, write_json, write_rows
from tools.fi1_scene_reference import (
    LABEL_VERSION, REFERENCE_VERSION, digest, load_partition, main_scene,
    make_split, partition_scenes, reference_payload, validate_partition_requests, validate_reference,
)


def test_groups_are_isolated_and_assignment_ignores_input_order():
    scenes = [f'scene{i:04d}_00' for i in range(20)] + ['scene0001_01', 'scene0002_02']
    split = make_split(scenes)
    assert split == make_split(list(reversed(scenes)))
    seen = []
    for fold in range(5):
        fit = partition_scenes(split, fold, 'fit'); heldout = partition_scenes(split, fold, 'heldout')
        assert set(fit).isdisjoint(heldout)
        assert set(fit + heldout) == set(scenes)
        assert {main_scene(s) for s in fit}.isdisjoint(main_scene(s) for s in heldout)
        seen += heldout
    assert sorted(seen) == sorted(scenes)


@pytest.mark.parametrize('scenes', [[], ['scene0001_00'] * 5, ['bad'] * 5, ['scene0001_00', 'scene0001_01']])
def test_invalid_scene_universe_is_rejected(scenes):
    with pytest.raises(ValueError):
        make_split(scenes)


@pytest.fixture
def partition_world(tmp_path):
    configure_evaluator(ev)
    scenes = [f'scene{i:04d}_00' for i in range(10)] + ['scene0000_01']
    split = make_split(scenes)
    new_fold = {s: f for f, members in enumerate(split['heldout_folds']) for s in members}
    lid, other = ev.PRED_ID_TO_ID[1], ev.PRED_ID_TO_ID[2]
    gt = np.array([lid * 1000 + 1] * 100 + [other * 1000 + 1] * 100)
    pts = {'tp': np.arange(100), 'fp': np.arange(100, 200)}
    caches = tmp_path / 'attr' / 'scene_cache'; caches.mkdir(parents=True)
    hashes = {}; pred = []; geos = []; candidate_by_scene = {}
    for pid, scene in enumerate(sorted(scenes)):
        keys = {name: f'{scene}:geometry:{point_hash(points)}' for name, points in pts.items()}
        inv, stats = geometry_statistics({keys[n]: p for n, p in pts.items()}, gt, ev)
        cache = {'scene_name': scene, 'geometry_stats': stats, 'gt_inventory': inv,
                 'points_sha256': {keys[n]: point_hash(p) for n, p in pts.items()}}
        path = caches / f'{scene}.json'; write_json(path, cache); hashes[scene] = sha(path)
        pred.append({'prediction_id': pid, 'scene_name': scene, 'geometry_key': keys['tp'],
                     'class_index': 1, 'score': .8, 'source': 'native', 'column_index': 0})
        for name in pts:
            geos.append({'geometry_key': keys[name], 'scene_name': scene, 'legacy_fold_index': 3,
                         'split_fold_index': new_fold[scene], 'points_sha256': point_hash(pts[name]),
                         'point_count': 100, 'is_new_geometry': name == 'fp', 'has_b1_members': name == 'tp',
                         'sources': ['temporary_refined'] if name == 'fp' else ['native']})
        candidate_by_scene[scene] = {'prediction_id': f'new:{scene}', 'scene_name': scene, 'geometry_key': keys['fp'],
            'class_index': 1, 'score': .99, 'source': ['temporary_refined'], 'fold_index': 3,
            'is_new_geometry': True, 'candidate_score_source': 'fixed_test', 'action': 'append_new'}
    folders = {}; refs = {}
    for role in ['fit', 'heldout']:
        allowed = partition_scenes(split, 0, role)
        folder = tmp_path / role; folder.mkdir(); folders[role] = folder
        write_rows(folder / 'predictions.jsonl', [r for r in pred if r['scene_name'] in allowed])
        write_rows(folder / 'geometry_index.jsonl', [r for r in geos if r['scene_name'] in allowed])
        ref = {'version': REFERENCE_VERSION, 'split': split, 'split_id': digest(split), 'outer_fold_index': 0,
               'role': role, 'reference_scenes': allowed, 'class_names': list(ev.CLASS_LABELS),
               'predictions_sha256': sha(folder / 'predictions.jsonl'),
               'geometry_index_sha256': sha(folder / 'geometry_index.jsonl'),
               'cache_sha256': {s: hashes[s] for s in allowed}, 'attribution_root': str(tmp_path / 'attr'),
               'population_root': str(tmp_path / 'pop')}
        ref['reference_id'] = digest(reference_payload(ref)); refs[role] = ref
        write_json(folder / 'reference.json', ref)
    return folders, refs, candidate_by_scene


def run_fit(world):
    folders, refs, candidates = world
    bind = Bindings()
    ref, rows, geos, caches = load_partition(folders['fit'], bind, expected_role='fit')
    candidate = candidates[ref['reference_scenes'][0]]
    validate_partition_requests([candidate], rows, geos, ref)
    result, before, after = CounterfactualEvaluator(rows, caches, ev).evaluate(candidate, candidate['action'])
    return result, before, after, bind, ref


def test_deleting_all_heldout_gt_and_predictions_cannot_change_fit_targets(partition_world):
    folders, refs, candidates = partition_world
    first = run_fit(partition_world)
    for scene in refs['heldout']['reference_scenes']:
        (Path(refs['heldout']['attribution_root']) / 'scene_cache' / f'{scene}.json').unlink()
    (folders['heldout'] / 'predictions.jsonl').write_text('not even valid JSON')
    second = run_fit(partition_world)
    assert first[0] == second[0]
    np.testing.assert_equal(first[1], second[1]); np.testing.assert_equal(first[2], second[2])
    assert first[4]['reference_id'] == second[4]['reference_id']
    read_cache_names = {Path(p).stem for p in second[3].files if Path(p).parent.name == 'scene_cache'}
    assert read_cache_names == set(refs['fit']['reference_scenes'])
    assert second[0]['delta']['class_ap50'] < 0


def test_changing_fit_gt_changes_target_so_noninterference_is_not_constant_output(partition_world):
    folders, refs, candidates = partition_world
    original = run_fit(partition_world)
    ref = deepcopy(refs['fit']); scene = ref['reference_scenes'][0]
    path = Path(ref['attribution_root']) / 'scene_cache' / f'{scene}.json'
    cache = json.loads(path.read_text())
    cache['gt_inventory'][ev.CLASS_LABELS[1]][0]['vert_count'] = 200
    write_json(path, cache)
    ref['cache_sha256'][scene] = sha(path); ref['reference_id'] = digest(reference_payload(ref))
    write_json(folders['fit'] / 'reference.json', ref)
    changed = run_fit(partition_world)
    assert original[0]['before']['class_ap50'] != changed[0]['before']['class_ap50']
    assert original[0]['delta']['class_ap50'] != changed[0]['delta']['class_ap50']


def test_wrong_role_is_rejected_before_any_cache_open(partition_world):
    folders, refs, _ = partition_world
    class NoCache(Bindings):
        def check(self, path, expected=None):
            assert Path(path).parent.name != 'scene_cache', 'opened heldout GT before rejecting role'
            return super().check(path, expected)
    with pytest.raises(ValueError, match='role differs'):
        load_partition(folders['heldout'], NoCache(), expected_role='fit')


def test_adding_heldout_scene_to_fit_reference_is_rejected_even_with_updated_hash(partition_world):
    folders, refs, _ = partition_world
    ref = deepcopy(refs['fit']); wrong = refs['heldout']['reference_scenes'][0]
    ref['reference_scenes'].append(wrong)
    ref['cache_sha256'][wrong] = refs['heldout']['cache_sha256'][wrong]
    ref['reference_id'] = digest(reference_payload(ref))
    with pytest.raises(ValueError, match='scope differs'):
        validate_reference(ref)


def test_heldout_request_rejected_by_fit_partition(partition_world):
    folders, refs, candidates = partition_world
    ref, rows, geos, _ = load_partition(folders['fit'], Bindings(), expected_role='fit')
    req = candidates[refs['heldout']['reference_scenes'][0]]
    with pytest.raises(ValueError, match='outside partition'):
        validate_partition_requests([req], rows, geos, ref)


def test_scope_change_cannot_reindex_existing_members(partition_world):
    folders, refs, _ = partition_world
    ref, rows, geos, _ = load_partition(folders['fit'], Bindings(), expected_role='fit')
    candidate = {**rows[-1], 'fold_index': 3, 'action': 'restore_existing'}
    validate_partition_requests([candidate], rows, geos, ref)
    # IDs remain those of the full B1 source, not range(len(partition)).
    original_ids = [r['prediction_id'] for r in rows]
    from tools.fi1_evaluator_delta_core import intervention
    before, after = intervention(rows, candidate, 'restore_existing')
    assert [r['prediction_id'] for r in after] == original_ids
    assert candidate['prediction_id'] not in [r['prediction_id'] for r in before]


def scoped_example(world, role='fit', cid=1):
    folders, refs, candidates = world
    ref, b1, _, caches = load_partition(folders[role], Bindings(), expected_role=role)
    c = {**candidates[ref['reference_scenes'][0]], 'class_index': cid}
    raw, _, _ = CounterfactualEvaluator(b1, caches, ev).evaluate(c, c['action'])
    label = {**raw, 'version': LABEL_VERSION, 'evaluation_scope': 'partition_reference_scenes',
             'reference_id': ref['reference_id'], 'split_id': ref['split_id'],
             'outer_fold_index': 0, 'partition_role': role,
             'label_usage': 'fit_only' if role == 'fit' else 'evaluation_only'}
    summary = {'version': LABEL_VERSION, 'evaluation_scope': 'partition_reference_scenes',
               'reference_id': ref['reference_id'], 'split_id': ref['split_id'], 'outer_fold_index': 0,
               'partition_role': role, 'reference_scenes': ref['reference_scenes'],
               'reference_prediction_count': len(b1), 'thresholds': list(map(float, ev.opt['overlaps'])),
               'label_usage': label['label_usage']}
    return c, label, summary, ref


def test_fit_and_heldout_bundles_keep_distinct_roles_and_no_gt_features(partition_world):
    from tools.build_fi1_scene_margin_dataset import make_bundle, validate_scoped_labels
    from tools.build_fi1_margin_dataset import FEATURES
    for role in ['fit', 'heldout']:
        c, label, summary, ref = scoped_example(partition_world, role)
        bundle = make_bundle([c], [label], summary, ref, role,
            {c['prediction_id']: {k: .1 for k in FEATURES}}, {c['prediction_id']: {}}, {c['prediction_id']: []})
        assert bundle['partition_role'] == role
        assert bundle['training_ready'] is False
        assert bundle['whole_system_oof'] is False
        assert 'delta' not in bundle['rows'][0]['features']
        assert bundle['rows'][0]['identity']['legacy_fold_index'] == 3
        if role == 'heldout':
            assert bundle['label_usage'] == 'evaluation_only'
            with pytest.raises(ValueError, match='evaluation-only'):
                validate_scoped_labels([c], [label], summary, ref, 'fit')


@pytest.mark.parametrize('field,value', [
    ('version', 'fi1_evaluator_counterfactual_v2'),
    ('reference_id', 'full100-reference'), ('partition_role', 'heldout'),
    ('outer_fold_index', 4), ('label_usage', 'evaluation_only'), ('score_after', .1),
    ('before_count', 999),
])
def test_mixed_or_unscoped_labels_cannot_enter_fit_bundle(partition_world, field, value):
    from tools.build_fi1_scene_margin_dataset import validate_scoped_labels
    c, label, summary, ref = scoped_example(partition_world)
    label[field] = value
    with pytest.raises(ValueError):
        validate_scoped_labels([c], [label], summary, ref, 'fit')


def test_no_gt_in_entire_fit_partition_stays_null_not_full100_fallback(partition_world):
    from tools.build_fi1_scene_margin_dataset import make_bundle
    from tools.build_fi1_margin_dataset import FEATURES
    c, label, summary, ref = scoped_example(partition_world, cid=5)
    assert label['before']['class_ap'] is None and label['after']['class_ap'] is None
    bundle = make_bundle([c], [label], summary, ref, 'fit',
        {c['prediction_id']: {k: .1 for k in FEATURES}}, {c['prediction_id']: {}}, {c['prediction_id']: []})
    assert bundle['row_count'] == 1 and bundle['defined_target_count'] == 0
    assert bundle['rows'][0]['labels']['target_status'] == 'class_absent_from_partition'
    assert bundle['rows'][0]['labels']['delta']['class_ap'] is None
