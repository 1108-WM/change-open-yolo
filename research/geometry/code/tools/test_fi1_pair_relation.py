import unittest
import numpy as np
import fi1_pair_relation_core as c


class PairRelationTests(unittest.TestCase):
    def test_zero_proposal_scene_has_no_actions(self):
        self.assertEqual(c.decide([], np.array([]), []), [])

    def test_full_endpoint_mixed_not_negative(self):
        left = c.endpoint_label([1001, 2001], [980, 20])
        right = c.endpoint_label([1001], [100])
        self.assertEqual(left['kind'], 'mixed')
        self.assertEqual(c.labels(left, right), 'mixed')
        self.assertEqual(c.endpoint_label([0, 1001], [11, 89])['kind'], 'unknown')

    def test_symmetry_and_exclusive_contact(self):
        raw = np.zeros((6, 9))
        raw[:, 0] = np.arange(6)*.01
        raw[:, 3] = np.arange(6)*30
        raw[:, 8] = 1
        left, right = np.array([0, 1, 2, 3]), np.array([2, 3, 4, 5])
        g, v, aa, bb, cp = c.surface(raw, left, right)
        gg, vv, *_ = c.surface(raw, right, left)
        np.testing.assert_allclose(g, gg, equal_nan=True)
        np.testing.assert_allclose(v, vv, equal_nan=True)
        self.assertEqual(set(aa), {0, 1})
        self.assertEqual(set(bb), {4, 5})
        self.assertFalse(set(cp.ravel()) & {2, 3})

    def test_sensor_depth_not_camera_delta(self):
        depth = np.ones((20, 20))
        a, b = np.array([[1, 10]]), np.array([[18, 10]])
        before = c.depth_path_evidence(depth, a, b)
        depth[:, 10:] = 2
        after = c.depth_path_evidence(depth, a, b)
        self.assertEqual(before[2], 0.)
        self.assertEqual(after[2], 1.)
        missing = c.depth_path_evidence(np.zeros_like(depth), a, b)
        self.assertTrue(np.isnan(missing[:3]).all())
        self.assertEqual(missing[3], 1.)

    def test_no_mask_template_requirement_and_disjoint_matching(self):
        rows = [dict(pair_id='p0', left=0, right=1), dict(pair_id='p1', left=1, right=2),
                dict(pair_id='p2', left=3, right=4), dict(pair_id='p3', left=5, right=6)]
        p = c.decide(rows, [.95, .99, .02, .99], [True, True, True, False])
        self.assertEqual([r['state'] for r in p], ['unknown', 'accept', 'reject', 'unknown'])
        # Extra legacy mask counts are never read by the relation selector.
        for row in rows:
            row.update(merge_winning_frames=0, separate_winning_frames=100)
        pp = c.decide(rows, [.95, .99, .02, .99], [True, True, True, False])
        self.assertEqual([r['state'] for r in p], [r['state'] for r in pp])

    def test_projection_and_missing_visibility(self):
        px, z, valid = c.project(np.array([[1., 2., 1.], [0., 0., -1.]]), np.eye(4), np.eye(4))
        np.testing.assert_equal(px[0], [1, 2])
        self.assertEqual(valid.tolist(), [True, False])
        self.assertFalse(c.distinct_pose(np.eye(4), np.eye(4)))
        other = np.eye(4);other[0, 3] = .3
        self.assertTrue(c.distinct_pose(np.eye(4), other))


if __name__ == '__main__':
    unittest.main()
