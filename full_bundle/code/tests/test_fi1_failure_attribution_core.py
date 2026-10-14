import unittest
import numpy as np
from tools.fi1_failure_attribution_core import geometry_overlaps, coverage_funnel


class FailureAttributionTest(unittest.TestCase):
    def test_duplicates_are_coverage(self):
        r = coverage_funnel(np.ones((2, 1)), [{3}, {3}], [{3}, {3}], [3], .5)[0]
        self.assertEqual(r['state'], 'selected_covered')
        self.assertEqual(r['duplicate_correct_members'], 1)

    def test_four_states_and_strict_threshold(self):
        iou = np.diag([.5, .8, .8, 1.])
        rows = coverage_funnel(iou, [{1}, {9}, {3}, {4}], [set(), {9}, set(), {4}], [1, 2, 3, 4], .5)
        self.assertEqual([r['state'] for r in rows], ['geometry_missing', 'correct_class_missing', 'budget_excluded', 'selected_covered'])

    def test_minimum_size_and_contamination(self):
        gt = np.repeat([2001, 4001], 100)
        iou, inter, sizes = geometry_overlaps([np.arange(99), np.arange(100), np.arange(200)], gt, [2001, 4001], [100, 100])
        np.testing.assert_array_equal(iou, [[0, 0], [1, 0], [.5, .5]])
        np.testing.assert_array_equal(inter, [[99, 0], [100, 0], [100, 100]])

    def test_invalid_point_domain(self):
        for points in ([1, 1], [2, 1], [-1], [4]):
            with self.assertRaises(ValueError):
                geometry_overlaps([np.array(points)], np.zeros(4, dtype=int), [2001], [100])

    def test_empty_geometry_pool(self):
        iou, _, _ = geometry_overlaps([], np.zeros(100, dtype=int), [2001], [100])
        self.assertEqual(coverage_funnel(iou, [], [], [1], .5)[0]['state'], 'geometry_missing')


if __name__ == '__main__':
    unittest.main()
