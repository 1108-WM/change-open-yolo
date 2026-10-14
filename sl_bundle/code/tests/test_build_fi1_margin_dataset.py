from copy import deepcopy
import numpy as np
import pytest

from tools.build_fi1_margin_dataset import (
    FEATURES, build_bundle, extract_visual_features, overlapping_context, validate_labels,
)
from tools.fi1_evaluator_delta_core import VERSION


def fixtures():
    names = [f'c{i}' for i in range(198)]
    pop = {'g': {'scene_name': 's', 'fold_index': 1, 'points_sha256': 'h', 'point_count': 100},
           'peer': {'scene_name': 's', 'fold_index': 1, 'points_sha256': 'p', 'point_count': 100}}
    c = {'prediction_id': 'new', 'scene_name': 's', 'geometry_key': 'g', 'class_index': 4,
         'score': .8, 'fold_index': 1, 'source': ['temporary_refined']}
    visual = [{'geometry_key': 'g', 'class_index': i, 'class_name': names[i], 'scene_name': 's',
               'fold_index': 1, 'geometry_sha256': 'h', 'features': {k: .1 for k in FEATURES},
               # Poison top-level/GT fields must not become feature values.
               'alpha_class_support': -999., 'label': 12345} for i in range(198)]
    return c, visual, pop, names


def test_nested_features_are_real_values_and_gt_never_copied():
    c, visual, pop, names = fixtures()
    values = extract_visual_features([c], iter(visual), pop, names)['new']
    assert values == {k: .1 for k in FEATURES}
    assert 'label' not in values


@pytest.mark.parametrize('mutation,error', [
    ('duplicate', 'duplicate visual'), ('missing_class', 'unique198'),
    ('missing_feature', 'numeric'), ('nan', 'finite'), ('wrong_scene', 'identity'),
    ('wrong_fold', 'identity'), ('wrong_hash', 'identity'), ('wrong_class', 'vocabulary'),
])
def test_corrupt_evidence_is_rejected(mutation, error):
    c, visual, pop, names = fixtures()
    if mutation == 'duplicate': visual.append(deepcopy(visual[0]))
    if mutation == 'missing_class': visual.pop()
    if mutation == 'missing_feature': del visual[4]['features']['alpha_class_support']
    if mutation == 'nan': visual[4]['features']['alpha_class_support'] = float('nan')
    if mutation == 'wrong_scene': visual[4]['scene_name'] = 'elsewhere'
    if mutation == 'wrong_fold': visual[4]['fold_index'] = 3
    if mutation == 'wrong_hash': visual[4]['geometry_sha256'] = 'wrong'
    if mutation == 'wrong_class': visual[4]['class_name'] = 'wrong'
    with pytest.raises(ValueError, match=error):
        extract_visual_features([c], visual, pop, names)


def label_fixture():
    c, _, _, _ = fixtures()
    summary = {'version': VERSION, 'evaluation_scope': 'full_reference_scenes',
               'reference_scenes': ['s'] + [f's{i}' for i in range(99)], 'action': 'append_new',
               'reference_id': 'ref', 'baseline_array_exactly_replayed': True,
               'thresholds': list(np.append(np.arange(.5, .95, .05), .25))}
    d = {**{k: c[k] for k in ('prediction_id', 'scene_name', 'geometry_key', 'class_index')},
         'version': VERSION, 'action': 'append_new', 'evaluation_scope': 'full_reference_scenes',
         'reference_scene_count': 100, 'reference_id': 'ref', 'score_before': None, 'score_after': .8,
         'before_count': 10, 'after_count': 11,
         'before': {k: .5 for k in ('class_ap', 'class_ap50', 'overall_ap', 'overall_ap50')},
         'after': {k: .25 for k in ('class_ap', 'class_ap50', 'overall_ap', 'overall_ap50')},
         'delta': {k: -.25 for k in ('class_ap', 'class_ap50', 'overall_ap', 'overall_ap50')}}
    return c, d, summary


def test_dataset_separates_gt_labels_and_cannot_claim_training_ready():
    c, d, summary = label_fixture()
    bundle = build_bundle([c], [d], summary, {'new': {k: .1 for k in FEATURES}}, {'new': {}}, {'new': []})
    assert bundle['training_ready'] is False
    assert 'delta' in bundle['rows'][0]['labels']
    assert 'delta' not in bundle['rows'][0]['features']
    assert bundle['rows'][0]['labels']['delta']['class_ap50'] == -.25
    assert bundle['required_feature_nonnull_counts'] == {k: 1 for k in FEATURES}


@pytest.mark.parametrize('mutation,error', [
    ('legacy', 'v2'), ('duplicate', 'duplicate'), ('score', 'score'), ('identity', 'identity'),
    ('scope', 'reference'), ('delta', 'invalid AP'), ('missing', 'sets differ'),
    ('wrong_before_score', 'original score'), ('wrong_count', 'member counts'),
])
def test_unbound_or_legacy_labels_rejected(mutation, error):
    c, d, summary = label_fixture()
    rows = [d]
    if mutation == 'legacy': summary['version'] = 'old_local'
    if mutation == 'duplicate': rows.append(deepcopy(d))
    if mutation == 'score': d['score_after'] = .1
    if mutation == 'identity': d['geometry_key'] = 'wrong'
    if mutation == 'scope': d['reference_scene_count'] = 1
    if mutation == 'delta': d['delta']['class_ap'] = .99
    if mutation == 'missing': rows.clear()
    if mutation == 'wrong_before_score': d['score_before'] = .8
    if mutation == 'wrong_count': d['after_count'] = d['before_count']
    with pytest.raises(ValueError, match=error):
        validate_labels([c], rows, summary)


def test_spatial_competitors_exclude_nonoverlapping_high_score_and_self():
    c, _, pop, _ = fixtures()
    b1 = [{**c, 'prediction_id': 0, 'geometry_key': 'peer', 'score': .2},
          {**c, 'prediction_id': 1, 'geometry_key': 'far', 'score': .99}]
    edges = [{'left': 'g', 'right': 'peer', 'scene_name': 's', 'intersection': 50, 'iou': 1 / 3}]
    values, identities = overlapping_context([c], b1, edges, pop)
    assert values['new']['same_class_overlap_max_score'] == .2
    assert values['new']['same_class_overlap_higher_score_count'] == 0
    assert identities['new'] == [0]
