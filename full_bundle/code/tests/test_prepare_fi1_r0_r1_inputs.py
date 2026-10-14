import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.prepare_fi1_r0_r1_inputs import (
    digest, load_partition, main_scene, make_partition, sha, validate_partition,
)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.a = [f'scene{i:04d}_00' for i in range(8)]
        self.b = ['scene0000_01', 'scene0010_00']
        self.v = ['scene0001_01', 'scene0011_00', 'scene0011_01']

    def test_related_scans_are_diagnostic_and_all_scans_are_accounted_for(self):
        p = make_partition(self.a, self.b, self.v)
        self.assertEqual(p['roles']['diagnostic'], ['scene0001_01'])
        self.assertEqual(p['roles']['development_validation'], ['scene0011_00', 'scene0011_01'])
        self.assertEqual(set(sum(p['roles'].values(), [])), set(self.a + self.b + self.v))

    def test_source_order_does_not_change_partition(self):
        self.assertEqual(make_partition(self.a, self.b, self.v),
                         make_partition(self.a[::-1], self.b[::-1], self.v[::-1]))

    def test_training_main_scene_never_crosses_folds(self):
        p = make_partition(self.a, self.b, self.v)
        seen = {}
        for i, fold in enumerate(p['training_heldout_folds']):
            for scene in fold:
                self.assertEqual(seen.setdefault(main_scene(scene), i), i)
        self.assertEqual(sorted(sum(p['training_heldout_folds'], [])), p['roles']['train'])

    def test_duplicate_or_invalid_scans_are_rejected(self):
        for rows in (self.a + [self.a[0]], self.a + ['scene0011'], self.a + ['../scene0011_00']):
            with self.assertRaises(ValueError):
                make_partition(rows, self.b, self.v)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            make_partition(self.a, [self.a[0]], self.v)

    def test_tampered_role_or_fold_is_rejected(self):
        p = make_partition(self.a, self.b, self.v)
        changed = copy.deepcopy(p)
        changed['roles']['development_validation'].append(changed['roles']['diagnostic'].pop())
        with self.assertRaises(ValueError):
            validate_partition(changed)
        changed = copy.deepcopy(p)
        changed['whole_system_oof'] = True
        with self.assertRaises(ValueError):
            validate_partition(changed)

    def test_frozen_file_mutation_rejected_before_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = make_partition(self.a, self.b, self.v)
            (root / 'partition.json').write_text(json.dumps(p))
            (root / 'train.txt').write_text('\n'.join(p['roles']['train']))
            (root / 'manifest.json').write_text(json.dumps({
                'partition_id': digest(p),
                'files': {name: sha(root / name) for name in ('partition.json', 'train.txt')},
            }))
            self.assertEqual(load_partition(root), p)
            (root / 'train.txt').write_text('scene0001_01\n')
            with self.assertRaisesRegex(ValueError, 'frozen split changed'):
                load_partition(root)


if __name__ == '__main__':
    unittest.main()
