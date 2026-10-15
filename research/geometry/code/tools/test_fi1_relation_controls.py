import unittest
import numpy as np
import fi1_relation_controls_core as c


class LearningControlsTests(unittest.TestCase):
    def test_scene_mass_and_present_cell_balance(self):
        scenes = ['a']*5+['b']*3
        overlap = np.array([0, 0, 0, 0, 1, 0, 0, 1], bool)
        y = np.array([1, 1, 1, 0, 1, 1, 0, 1])
        wa = c.weights(scenes, overlap, y)
        wb = c.weights(scenes, overlap, y, balanced=True)
        for s in ('a', 'b'):
            ii = np.array(scenes) == s
            self.assertAlmostEqual(wa[ii].sum(), 4.)
            self.assertAlmostEqual(wb[ii].sum(), 4.)
            keys = [c.cell(overlap[i], y[i]) for i in np.flatnonzero(ii)]
            mass = [wb[ii][np.array(keys) == k].sum() for k in set(keys)]
            np.testing.assert_allclose(mass, [4/3]*3)
        self.assertTrue(np.all(wb > 0))
        self.assertAlmostEqual(wb.sum(), len(y))
        self.assertFalse(np.array_equal(wa, wb))

    def test_scene_with_one_present_cell(self):
        w = c.weights(['a', 'a', 'b'], [False, False, False], [1, 1, 0], True)
        np.testing.assert_allclose(w, [.75, .75, 1.5])

    def test_positive_and_neighbor_diagnostics(self):
        m = c.metrics([1, 1, 0, 0], [.95, .7, .96, .05], ['s1', 's1', 's2', 's2'])
        self.assertEqual(m['positive']['high90'], 1)
        self.assertEqual(m['negative']['high90'], 1)
        self.assertEqual(m['negative']['low10'], 1)
        self.assertTrue(np.isfinite(m['logloss']))
        self.assertEqual(c.metrics([], [], [])['rows'], 0)

    def test_more_positive_scores_alone_do_not_pass(self):
        a = c.metrics([1, 1, 0, 0], [.8, .7, .2, .1], ['s']*4)
        b = c.metrics([1, 1, 0, 0], [.95, .96, .99, .3], ['s']*4)
        flags = c.improvement(a, b)
        self.assertTrue(flags['positive_high90_not_lower'])
        self.assertFalse(flags['negative_high90_not_higher'])
        self.assertFalse(all(flags.values()))


if __name__ == '__main__':
    unittest.main()
