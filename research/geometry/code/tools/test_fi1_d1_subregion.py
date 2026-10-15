"""Small synthetic checks for the fixed readout's voting and partition contracts."""
import unittest

import numpy as np

import fi1_d1_subregion_core as core


class ReadoutTests(unittest.TestCase):
    def test_same_frame_masks_vote_once(self):
        obs = {1: dict(frame_index=7), 2: dict(frame_index=7)}
        pp = {1: np.array([0, 1]), 2: np.array([1, 2])}
        frames = core.frame_unions([1, 2, 1], obs, pp)
        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0][2], [0, 1, 2])
        r = core.readout(np.zeros(3, int), np.ones((1, 3), bool), np.ones((1, 3), bool))
        self.assertFalse(r['kept'].any())

    def test_original_tracks_cannot_pool_single_frame_votes(self):
        one = np.ones((1, 3), bool)
        labels = np.zeros(3, int)
        self.assertFalse(core.readout(labels, one, one)['kept'].any())
        self.assertTrue(core.readout(labels, np.vstack([one, one]),
                                     np.vstack([one, one]))['kept'].all())

    def test_exact_signature_artifact_and_zero_region(self):
        support = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]], bool)
        vis = np.ones_like(support)
        _, _, whole, sub = core.track_readout(np.arange(6), np.zeros(6, int), support, vis)
        self.assertFalse(whole['kept'].any())  # supported mean .5 fails .55
        self.assertEqual(int(sub['kept'].sum()), 1)
        np.testing.assert_array_equal(np.flatnonzero(sub['kept'][sub['labels']]), [0, 1, 2])
        self.assertTrue(np.isin(sub['inside'] / sub['visible'], [0, 1]).all())
        self.assertFalse(sub['initial'][sub['inside'].sum(axis=0) == 0].any())

    def test_tiny_atom_has_no_three_point_exception(self):
        support = np.array([[1, 1, 0], [1, 1, 0]], bool)
        _, _, _, sub = core.track_readout(np.arange(3), np.zeros(3, int),
                                         support, np.ones_like(support))
        self.assertFalse(sub['kept'].any())


if __name__ == '__main__':
    unittest.main()
