import unittest
import numpy as np

from tools.fi1_same_class_mask_nms import select


def member(key, score=0.5, cls=0, native=False):
    return dict(geometry_key=key, class_index=cls, score=score,
                has_original_native_alias=native, original_aliases=[])


class SameClassNmsTests(unittest.TestCase):
    def run_selection(self, members, points, budget=800):
        return select(members, {k: np.asarray(v, dtype=np.int64) for k, v in points.items()}, 200, budget)

    def test_identical_points_same_class(self):
        out, review = self.run_selection([member('a', .9), member('b', .8)], {'a': range(40), 'b': range(40)})
        self.assertEqual([m['geometry_key'] for m in out], ['a'])
        self.assertEqual(review['suppressed'][0]['iou'], 1)

    def test_cross_class_preserved(self):
        out, _ = self.run_selection([member('a', .9), member('b', .8, 1)], {'a': range(40), 'b': range(40)})
        self.assertEqual(len(out), 2)

    def test_exact_boundary_preserved(self):
        out, review = self.run_selection([member('a', .9), member('b', .8)], {'a': range(19), 'b': range(20)})
        self.assertEqual(len(out), 2)
        self.assertEqual(review['near_geometry_pairs'], [])

    def test_above_boundary_suppressed(self):
        out, review = self.run_selection([member('a', .9), member('b', .8)], {'a': range(20), 'b': range(21)})
        self.assertEqual(len(out), 1)
        self.assertEqual(review['suppressed'][0]['intersection'], 20)

    def test_tie_native_then_geometry_class(self):
        out, _ = self.run_selection([member('a'), member('b', native=True)], {'a': range(40), 'b': range(40)})
        self.assertEqual(out[0]['geometry_key'], 'b')
        out, _ = self.run_selection([member('b'), member('a')], {'a': range(40), 'b': range(40)})
        self.assertEqual(out[0]['geometry_key'], 'a')

    def test_suppressed_member_cannot_suppress(self):
        out, review = self.run_selection([member('a', .9), member('b', .8), member('c', .7)],
                                        {'a': range(40), 'b': range(1, 41), 'c': range(2, 42)})
        self.assertEqual([m['geometry_key'] for m in out], ['a', 'c'])
        self.assertEqual(len(review['suppressed']), 1)

    def test_refill_outside_original_budget(self):
        out, review = self.run_selection([member('a', .9), member('b', .8), member('c', .7)],
                                        {'a': range(40), 'b': range(40), 'c': range(100, 140)}, 2)
        self.assertEqual([m['geometry_key'] for m in out], ['a', 'c'])
        self.assertEqual(review['examined'], 3)

    def test_insufficient_remaining_no_restoration(self):
        out, review = self.run_selection([member('a', .9), member('b', .8)], {'a': range(40), 'b': range(40)}, 5)
        self.assertEqual(len(out), 1)
        self.assertEqual(review['unexamined'], 0)

    def test_partial_class_suppression(self):
        out, _ = self.run_selection([member('a', .9), member('b', .8), member('b', .7, 1)],
                                   {'a': range(40), 'b': range(40)})
        self.assertEqual([(m['geometry_key'], m['class_index']) for m in out], [('a', 0), ('b', 1)])

    def test_multiple_witnesses_first_retained_wins(self):
        out, review = self.run_selection([member('a', .9), member('c', .8), member('b', .7)],
                                        {'a': range(40), 'b': range(1, 41), 'c': range(2, 42)})
        self.assertEqual(len(out), 2)
        self.assertEqual(review['suppressed'][0]['witness']['geometry_key'], 'a')

    def test_shared_aliases_do_not_change_tie(self):
        a = {**member('a', .9), 'original_aliases': [{'source': 'native'}, {'source': 'track'}]}
        out, _ = self.run_selection([a, member('b', .8)], {'a': range(40), 'b': range(40)})
        self.assertEqual(out, [a])

    def test_empty_members(self):
        out, review = self.run_selection([], {})
        self.assertEqual(out, [])
        self.assertEqual(review['examined'], 0)

    def test_duplicate_member_rejected(self):
        with self.assertRaises(ValueError):
            self.run_selection([member('a'), member('a')], {'a': range(40)})

    def test_invalid_index_rejected(self):
        with self.assertRaises(ValueError):
            self.run_selection([member('a')], {'a': [1, 1]})

    def test_invalid_score_rejected(self):
        with self.assertRaises(ValueError):
            self.run_selection([member('a', float('nan'))], {'a': range(40)})

    def test_budget_and_class_validation(self):
        with self.assertRaises(ValueError):
            self.run_selection([member('a')], {'a': range(40)}, 0)
        with self.assertRaises(ValueError):
            self.run_selection([member('a', cls=198)], {'a': range(40)})

    def test_exact_intersection_not_superpoint_or_box(self):
        out, review = self.run_selection([member('a', .9), member('b', .8)],
                                        {'a': range(0, 100, 2), 'b': range(1, 100, 2)})
        self.assertEqual(len(out), 2)
        self.assertEqual(review['near_geometry_pairs'], [])


if __name__ == '__main__':
    unittest.main()
