import copy
import unittest

import numpy as np

from tools.fi1_r0_r1_protocol import build_pair, direct_score


def fixture(source='track', digest='a'):
    key = 'scene0001_00:geometry:' + digest*64
    geometry = {'geometry_key': key, 'points_sha256': digest*64, 'point_count': 100,
                'legacy_members': [{'candidate_source': source, 'candidate_id': 0, 'frozen_class_index': 1}],
                'dv3_plans': []}
    votes = [0]*198
    votes[1], votes[2] = 1, 9
    evidence = {'geometry_key': key, 'points_sha256': digest*64,
                'yolo': {'acquisition_complete': True, 'available': True, 'votes': votes},
                'alpha': {'acquisition_complete': True, 'available': False, 'missing_reason': 'no_visible_views'}}
    return geometry, evidence


class DirectProtocolTest(unittest.TestCase):
    def test_restoration_changes_members_only(self):
        g, e = fixture()
        result = build_pair([g], {g['geometry_key']: e})
        self.assertEqual([r['class_index'] for r in result['R0']['selected']], [1])
        self.assertEqual([r['class_index'] for r in result['R1']['selected']], [2, 1])
        self.assertEqual(result['R0']['selected'][0]['score'], result['R1']['selected'][1]['score'])

    def test_native_only_geometry_is_not_expanded(self):
        g,e = fixture('native')
        result = build_pair([g], {g['geometry_key']: e})
        self.assertEqual(result['R0'], result['R1'])

    def test_identical_geometry_class_deduplicates_without_losing_aliases(self):
        g,e = fixture('native')
        g['legacy_members'].append({'candidate_source': 'track', 'candidate_id': 0, 'frozen_class_index': 1})
        result = build_pair([g], {g['geometry_key']: e})
        self.assertEqual(len(result['R0']['selected']), 1)
        self.assertEqual(len(result['R0']['selected'][0]['original_aliases']), 2)
        self.assertEqual(result['R1']['audit']['selected_expanded_classes'], 1)

    def test_budget_native_eviction_is_counted(self):
        native,en = fixture('native', 'a')
        track,et = fixture('track', 'b')
        result = build_pair([native,track], {native['geometry_key']: en,track['geometry_key']: et}, budget=1)
        self.assertEqual(result['R0']['audit']['budget_evicted_native_alias_count'], 0)
        self.assertEqual(result['R1']['audit']['budget_evicted_native_alias_count'], 1)
        self.assertEqual(result['R1']['audit']['budget_evicted_native_geometry_count'], 1)

    def test_tie_rule_is_input_order_independent(self):
        first,e1 = fixture('track', 'a')
        second,e2 = fixture('track', 'b')
        evidence = {first['geometry_key']:e1, second['geometry_key']:e2}
        self.assertEqual(build_pair([first,second], evidence), build_pair([second,first], evidence))

    def test_incomplete_modality_cannot_silently_fallback(self):
        _,e = fixture()
        e['alpha']['acquisition_complete'] = False
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            direct_score(e)

    def test_full_alpha_distribution_is_used(self):
        _,e = fixture()
        e['alpha'] = {'acquisition_complete': True, 'available': True, 'cosine': [0.]*198}
        scores = direct_score(e)
        self.assertAlmostEqual(scores[1], .5*(.1 + 1/198))
        self.assertEqual(np.count_nonzero(scores), 198)
        self.assertAlmostEqual(scores.sum(), 1)

    def test_missing_both_retains_existing_zero_class_only(self):
        g,e = fixture()
        e['yolo'] = {'acquisition_complete': True, 'available': False, 'missing_reason': 'no_label_votes'}
        result = build_pair([g], {g['geometry_key']:e})
        self.assertEqual(result['R0'], result['R1'])
        self.assertEqual(result['R1']['selected'][0]['score'], 0.)

    def test_invalid_and_background_never_become_foreground(self):
        g,e = fixture()
        g['legacy_members'][0]['frozen_class_index'] = 198
        result = build_pair([g], {g['geometry_key']:e})
        self.assertEqual(result['R0']['selected'], [])
        self.assertEqual(result['R1']['audit']['legacy_background_sentinel_alias_count'], 1)
        g['legacy_members'][0]['frozen_class_index'] = 199
        with self.assertRaisesRegex(ValueError, 'invalid original class'):
            build_pair([g], {g['geometry_key']:e})

    def test_parent_evidence_cannot_be_reused_for_different_geometry(self):
        g,e = fixture()
        e['points_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'identity'):
            build_pair([g], {g['geometry_key']:e})

    def test_cross_scene_budget_is_rejected(self):
        g,e = fixture()
        other = copy.deepcopy(g)
        other['geometry_key'] = other['geometry_key'].replace('0001','0002')
        with self.assertRaisesRegex(ValueError, 'per scene'):
            build_pair([g,other], {g['geometry_key']:e,other['geometry_key']:e})


if __name__ == '__main__':
    unittest.main()
