import unittest

import numpy as np

import fi1_cross_track_dual_core as core


def frame(a, b, sizes, owners, va=30, vb=30):
    a, b, sizes = [np.asarray(x, dtype=np.int64) for x in (a, b, sizes)]
    return core.frame_hypotheses(a, b, np.zeros(len(a), int), va, vb, 0,
        sizes, np.arange(len(a)), np.asarray(owners), {1}, {2})


class DualTests(unittest.TestCase):
    def test_zero_overlap_does_not_reject(self):
        f = frame([30], [30], [60], [1])
        self.assertEqual(f['state'], 'merge_wins')
        other = frame([30], [30], [60], [2])
        r = core.decide([f, other], (30, 30), [])
        self.assertEqual(r['state'], 'eligible')

    def test_repeated_cross_mask_alone_is_unknown(self):
        f = frame([30], [30], [60], [1])
        self.assertEqual(core.decide([f]*10, (30, 30), [])['reason'],
                         'joint_support_concentrated_in_one_lineage')

    def test_selective_alternatives_compete_with_joint_mask(self):
        f = frame([30, 30, 0], [30, 0, 30], [60, 30, 30], [1, 1, 2])
        self.assertEqual(f['state'], 'competing_or_close')
        self.assertTrue(f['separate_plausible'])
        self.assertEqual(core.decide([f, f], (30, 30), [])['state'], 'unknown')

    def test_explicit_separation_and_absence_differ(self):
        f = frame([30, 0], [0, 30], [30, 30], [1, 2])
        self.assertEqual(core.decide([f, f], (30, 30), [])['state'], 'reject')
        missing = dict(state='not_jointly_reliably_visible')
        self.assertEqual(core.decide([missing]*10, (30, 30), [])['reason'],
                         'no_jointly_reliable_views')
        absent = frame([], [], [], [])
        self.assertEqual(absent['state'], 'no_fitting_explanation')
        self.assertEqual(core.decide([absent]*2, (30, 30), [])['state'], 'unknown')

    def test_third_instance_and_multiple_partners_abstain(self):
        a, b = frame([30], [30], [60], [1]), frame([30], [30], [60], [2])
        r = core.decide([a, b], (30, 30), [dict(node=9, supported_conflict=True)])
        self.assertEqual(r['reason'], 'supported_third_instance_explanation')
        rows = [dict(left=0, right=1, state='eligible'), dict(left=1, right=2, state='eligible')]
        self.assertTrue(all(r['state'] == 'unknown' for r in core.resolve_partner_competition(rows)))

    def test_same_frame_duplicate_masks_do_not_make_two_frames(self):
        f = frame([30, 30], [30, 30], [60, 60], [1, 2])
        self.assertEqual(core.decide([f], (30, 30), [])['state'], 'unknown')

    def test_two_track_ids_in_one_dependence_family_are_not_independent(self):
        args = (np.array([30, 30]), np.array([30, 30]), np.zeros(2, int),
                30, 30, 0, np.array([60, 60]), np.array([4, 5]), np.array([1, 2]), {1}, {2})
        f = core.frame_hypotheses(*args, source_families=np.array([1, 1]))
        self.assertEqual(core.decide([f, f], (30, 30), [])['reason'],
                         'joint_support_concentrated_in_one_lineage')


if __name__ == '__main__':
    unittest.main()
