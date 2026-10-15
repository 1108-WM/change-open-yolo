import unittest
import numpy as np
import fi1_physical_relation_balance_core as c


class PhysicalRelationBalanceTests(unittest.TestCase):
    def test_formula_preserves_scene_and_each_class_mass(self):
        keys = [('a', 1, 10, 10)]*3+[('a', 1, 20, 20)]+[('a', 0, 10, 20)]*2+[
            ('a', 0, 10, 30), ('b', 1, 10, 10), ('b', 1, 20, 20)]
        reference = np.array([2.]*7+[3., 3.])
        w = c.balance(keys, reference)
        np.testing.assert_allclose(w, [4/3]*3+[4.]+[1.5, 1.5, 3., 3., 3.])
        for scene in ('a', 'b'):
            for category in (-1, 0, 1):
                mask = np.array([k[0] == scene and (category < 0 or k[1] == category) for k in keys])
                self.assertAlmostEqual(w[mask].sum(), reference[mask].sum())
        self.assertTrue(np.all(w > 0))

    def test_unordered_relation_and_scene_namespace(self):
        labels = {s: dict(nodes={'1': dict(kind='clear', target=1001),
                                  '2': dict(kind='clear', target=2001)}) for s in ('a', 'b')}
        rows = [dict(scene='a', left=1, right=2), dict(scene='a', left=2, right=1),
                dict(scene='b', left=1, right=2), dict(scene='a', left=1, right=1)]
        keys, y = c.group_keys(rows, labels)
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])
        self.assertEqual(y.tolist(), [0, 0, 0, 1])

    def test_mixed_or_unavailable_scene_cannot_group(self):
        labels = dict(a=dict(nodes={'1': dict(kind='mixed', target=1001),
                                     '2': dict(kind='clear', target=1001)}))
        with self.assertRaises(ValueError):
            c.group_keys([dict(scene='a', left=1, right=2)], labels)
        with self.assertRaises(KeyError):
            c.group_keys([dict(scene='heldout', left=1, right=2)], labels)

    def test_single_group_keeps_original_uniform_weights(self):
        np.testing.assert_allclose(c.balance([('a', 1, 1, 1)]*4, [.5]*4), [.5]*4)
        with self.assertRaises(ValueError):
            c.balance([('a', 1, 1, 1)], [0])


if __name__ == '__main__':
    unittest.main()
