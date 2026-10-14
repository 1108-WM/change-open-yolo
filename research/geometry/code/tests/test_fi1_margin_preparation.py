"""Sampling/evidence failures that would silently alter downstream supervision."""
from copy import deepcopy
from collections import defaultdict
import hashlib

import numpy as np
import pytest

from tools.fi1_margin_preparation import (
    VISUAL_FEATURES, VisualIndex, representative_geometries, candidates_from_selection,
    verify_quality, expanded_features, fit_score_anchors, action_requests, action_features, parent_index,
)
from tools.fi1_scene_reference import make_split


def visual_rows(pop):
    result = []
    for key, p in pop.items():
        for cid in range(198):
            f = {name: 0.0 for name in VISUAL_FEATURES}
            for name in f:
                if name.startswith('source_') or name.endswith('_available') or name == 'b1_geometry_present':
                    f[name] = name.endswith('_available') or name == 'source_track'
            f['independent_geometry_quality'] = None
            f['refinement_parent_retention_min'] = None
            f['refinement_parent_retention_max'] = None
            for channel, top in [('alpha', 1), ('label_map', 5), ('bbox', 9)]:
                f[channel + '_class_support'] = .9 if cid == top else (.5 if cid == top + 1 else 0.0)
            result.append({'geometry_key': key, 'scene_name': p['scene_name'], 'fold_index': p['fold_index'],
                'geometry_sha256': p['points_sha256'], 'class_index': cid, 'class_name': f'c{cid}',
                'evidence_sha256': 'ev:' + key, 'features': f, 'label': {'poison_gt': 100}})
    return result


def poprow(key, scene='scene0001_00', new=False, sources=None):
    return {'geometry_key': key, 'scene_name': scene, 'fold_index': 3, 'points_sha256': key,
            'point_count': 100, 'is_new_geometry': new, 'b1_members': [] if new else [{}],
            'sources': sources or (['temporary_refined'] if new else ['track']), 'refinement_aliases': []}


def visual(pop, rows=None):
    return VisualIndex(visual_rows(pop) if rows is None else rows, pop, [f'c{i}' for i in range(198)])


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'class_name', 'scene', 'fold', 'hash', 'nan', 'missing_field', 'availability', 'quality'])
def test_bad_visual_identity_and_missing_contract_rejected(mutation):
    pop = {'g': poprow('g')}; rows = visual_rows(pop)
    if mutation == 'duplicate': rows.append(deepcopy(rows[0]))
    if mutation == 'missing': rows.pop()
    if mutation == 'class_name': rows[0]['class_name'] = 'wrong'
    if mutation == 'scene': rows[0]['scene_name'] = 'scene0002_00'
    if mutation == 'fold': rows[0]['fold_index'] = 1
    if mutation == 'hash': rows[0]['geometry_sha256'] = 'wrong'
    if mutation == 'nan': rows[0]['features']['alpha_class_support'] = float('nan')
    if mutation == 'missing_field': del rows[0]['features']['log1p_point_count']
    if mutation == 'availability': rows[0]['features']['alpha_available'] = False
    if mutation == 'quality': rows[0]['features']['independent_geometry_quality'] = .7
    with pytest.raises(ValueError):
        visual(pop, rows)


def test_sampling_is_order_and_gt_independent_and_balances_main_scenes():
    scenes = [f'scene{i:04d}_00' for i in range(15)] + ['scene0001_01']
    split = make_split(scenes); pop = {}; b1 = []
    for scene in scenes:
        for j in range(3):
            key = f'{scene}:{j}'; pop[key] = poprow(key, scene)
            b1.append({'geometry_key': key, 'source': 'track', 'class_index': 1})
    v = visual(pop)
    first = representative_geometries(pop, b1, v, split)
    poisoned = deepcopy(pop)
    for p in poisoned.values(): p['gt_iou'] = -999; p['label'] = 'do not read'
    second = representative_geometries(dict(reversed(list(poisoned.items()))), list(reversed(b1)), v, split)
    assert first == second
    selected, inventory = first
    assert len(selected) == 10
    for fold in range(5):
        rows = [p for p in selected if p['split_fold_index'] == fold]
        assert len({pop[r['geometry_key']]['scene_name'].split('_')[0] for r in rows}) == 2
    assert all(r['selected_count'] == 0 for r in inventory if r['source'] != 'track')


def test_multiclass_union_uses_channel_ranks_parent_members_and_hash_control():
    pop = {'child': poprow('child', new=True), 'parent': poprow('parent')}
    v = visual(pop); parents = defaultdict(list, child=[{'parent_geometry_key': 'parent'}])
    b1 = [{'geometry_key': 'parent', 'class_index': 42}]
    selected = [{'geometry_key': 'child', 'source_stratum': 'temporary_refined'}]
    candidates = candidates_from_selection(selected, pop, b1, v, parents)
    by_class = {c['class_index']: c for c in candidates}
    assert {1, 2, 5, 6, 9, 10, 42} <= set(by_class)
    assert len(by_class) == 8
    assert sum(c['hypothesis_reasons'] == ['hash_control'] for c in candidates) == 1
    assert all('score' not in c for c in candidates)
    # A strictly monotone channel transformation cannot switch its class ordering.
    rows = visual_rows(pop)
    for r in rows: r['features']['alpha_class_support'] *= .01
    assert candidates_from_selection(selected, pop, b1, visual(pop, rows), parents) == candidates


def test_flat_or_unavailable_channels_do_not_invent_hypotheses():
    pop = {'child': poprow('child', new=True)}; rows = visual_rows(pop)
    for r in rows:
        r['features']['alpha_class_support'] = .1
        r['features']['bbox_available'] = False
        r['features']['bbox_class_support'] = None
    c = candidates_from_selection([{'geometry_key': 'child', 'source_stratum': 'temporary_refined'}],
                                 pop, [], visual(pop, rows), defaultdict(list))
    assert len(c) == 3
    assert {5, 6} < {r['class_index'] for r in c}


def test_existing_columns_are_not_collapsed_by_class_or_geometry():
    pop = {'g': poprow('g')}
    b1 = [{'prediction_id': i, 'geometry_key': 'g', 'class_index': cid, 'source': 'track'}
          for i, cid in enumerate([1, 2, 2])]
    candidates = candidates_from_selection([{'geometry_key': 'g', 'source_stratum': 'track'}],
        pop, b1, visual(pop), defaultdict(list))
    assert [c['prediction_id'] for c in candidates] == [0, 1, 2]


def quality_fixture():
    points = np.arange(100, dtype='<i8'); pop = {'g': poprow('g')}
    meta = {'geometry_key': 'g', 'scene_name': 'scene0001_00', 'fold_index': 3,
        'legacy_stage_a_quality': .7, 'candidate_source': 'track', 'source_row_id': 'source',
        'nested_exclusion_verified': False, 'use_as_training_feature': False}
    source = {'row_id': 'source', 'scene_name': 'scene0001_00', 'fold_index': 3,
        'oof_unified_quality': .7, 'candidate_source': 'track', 'geometry_hash': hashlib.sha1(points.tobytes()).hexdigest()}
    return points, pop, meta, source


@pytest.mark.parametrize('mutation', ['points', 'value', 'fold', 'source', 'duplicate'])
def test_quality_join_rejects_wrong_geometry_or_provenance(mutation):
    points, pop, meta, source = quality_fixture(); metadata = [meta]
    if mutation == 'points': points = points + 1
    if mutation == 'value': source['oof_unified_quality'] = .8
    if mutation == 'fold': meta['fold_index'] = 0
    if mutation == 'source': source['candidate_source'] = 'native'
    if mutation == 'duplicate': metadata.append(deepcopy(meta))
    with pytest.raises(ValueError):
        verify_quality({'g'}, metadata, [source], pop, lambda k: points)


def test_frozen_quality_keeps_original_nested_flags():
    points, pop, meta, source = quality_fixture()
    assert verify_quality({'g'}, [meta], [source], pop, lambda k: points)['g'] == meta


def region_fixture():
    pop = {'child': poprow('child', new=True), 'parent': poprow('parent')}
    pop['parent']['point_count'] = 150
    pop['child']['refinement_aliases'] = ['a']
    alias = {'alias_key': 'a', 'geometry_key': 'child', 'parent_geometry_key': 'parent',
        'scene_name': 'scene0001_00', 'fold_index': 3, 'points_sha256': 'child', 'point_count': 100,
        'removed_point_count': 50, 'same_as_parent': False, 'accepted_by_historical_guard': False, 'final_status': 'rejected'}
    points = {'child': np.arange(100), 'parent': np.arange(150)}
    parents = parent_index([alias], pop)
    c = {'prediction_id': 'new', 'geometry_key': 'child', 'class_index': 1}
    return pop, alias, points, parents, c


def test_parent_quality_and_region_support_never_replace_missing_child_quality():
    pop, alias, points, parents, c = region_fixture()
    q = {'child': None, 'parent': {'legacy_stage_a_quality': .8}}
    out = expanded_features([c], pop, [], visual(pop), parents, q, points.__getitem__)['new']
    f = out['features']
    assert f['independent_geometry_quality'] is None and f['independent_geometry_quality_missing'] == 1
    assert f['parent_quality_max'] == .8
    assert f['parent_retained_fraction_min'] == 2/3
    assert f['parent_removed_fraction_max'] == 1/3
    assert f['parent_added_fraction_max'] == 0
    assert 'accepted_by_historical_guard' not in f
    assert 'label' not in f


def test_region_relation_is_recomputed_from_points():
    pop, alias, points, parents, c = region_fixture()
    points['child'] = np.arange(50, 150)
    parents['child'][0]['removed_point_count'] = 40
    with pytest.raises(ValueError, match='actual point sets'):
        expanded_features([c], pop, [], visual(pop), parents, {'child': None, 'parent': None}, points.__getitem__)


def test_parent_cross_scene_identity_rejected():
    pop, alias, _, _, _ = region_fixture(); pop['parent']['scene_name'] = 'scene0002_00'
    with pytest.raises(ValueError, match='identity'):
        parent_index([alias], pop)


def test_score_anchors_fit_only_fallback_and_explicit_score_actions():
    b1 = [{'scene_name': 'fit', 'class_index': 1, 'score': s} for s in [.1, .5, .9]]
    anchors = fit_score_anchors(b1, {'fit'})
    np.testing.assert_allclose(anchors['1']['scores'], [.3, .5, .7])
    assert anchors['1']['fallback_all_fit_foreground'] is False
    assert anchors['7']['fallback_all_fit_foreground'] is True
    with pytest.raises(ValueError, match='only fit'):
        fit_score_anchors(b1 + [{'scene_name': 'heldout', 'class_index': 1, 'score': 1}], {'fit'})
    c = {'prediction_id': 1, 'class_index': 1, 'score': .98}
    assert action_requests([c], 'existing_keep', anchors)[0]['new_score'] == .98
    assert action_requests([c], 'existing_up', anchors)[0]['new_score'] == 1
    new = {'prediction_id': 'n', 'class_index': 1, 'is_new_geometry': True}
    req = action_requests([new], 'new_q50', anchors)[0]
    assert req['score'] == .5 and req['action'] == 'append_new'


def test_competition_tracks_proposed_score_and_excludes_self():
    pop = {'g': poprow('g'), 'peer': poprow('peer')}
    old = {'prediction_id': 0, 'scene_name': 'scene0001_00', 'geometry_key': 'g', 'class_index': 1, 'score': .4}
    peer = {**old, 'prediction_id': 1, 'geometry_key': 'peer', 'score': .45}
    req = {**old, 'action': 'rescore_existing', 'new_score': .5}
    edges = [{'left': 'g', 'right': 'peer', 'scene_name': old['scene_name'], 'intersection': 50, 'iou': 1/3}]
    f = action_features([req], [old, peer], edges, pop, {0: {'features': {}}})[0]['features']
    assert f['same_class_overlap_higher_score_count'] == 1
    assert f['same_class_overlap_higher_proposed_count'] == 0
    assert f['same_class_reference_member_count'] == 1


def test_retired_sampler_fails_before_any_io():
    from tools.build_fi1_new_geometry_candidates import main
    with pytest.raises(RuntimeError, match='retired'):
        main()
