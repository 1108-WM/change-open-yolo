import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from tools.audit_fi1_r0_r1_geometry import audit_scene, checked_points, point_digest


class GeometryBindingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.scene = 'scene0001_00'
        self.spec = {'prepared_root': str(root/'prepared'), 'records_root': str(root/'records')}
        prepared = root/'prepared'/self.scene
        native = root/'records'/self.scene/'native_cache'
        prepared.mkdir(parents=True)
        native.mkdir(parents=True)
        self.xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.processed = prepared/'0001_00.npy'
        np.save(self.processed, self.xyz)
        vertices = np.array([tuple(p) for p in self.xyz], dtype=[('x','f4'),('y','f4'),('z','f4')])
        PlyData([PlyElement.describe(vertices, 'vertex')]).write(prepared/f'{self.scene}_vh_clean_2.ply')
        masks_path = native/f'{self.scene}_pred_masks.npy'
        classes_path = native/f'{self.scene}_pred_classes.npy'
        np.save(masks_path, np.array([[1,1],[1,1],[0,0],[0,0]], dtype=bool))
        np.save(classes_path, [1,2])
        np.save(native/f'{self.scene}_pred_scores.npy', [.7,.4])
        points = np.array([0,1], dtype='<i8')
        h = point_digest(points, 'sha1')
        key = f'{self.scene}:geometry:{h}'
        members = []
        for i, (cls, score) in enumerate([(1,.7),(2,.4)]):
            members.append({'candidate_id': i, 'candidate_source': 'native',
                'frozen_class_index': cls, 'frozen_score': score, 'scene_name': self.scene,
                'geometry_hash': h, 'point_count': 2,
                'geometry_locator': {'kind': 'native_mask_column', 'masks_path': str(masks_path), 'column_index': i},
                'semantic_provenance': {'classes_path': str(classes_path), 'class_array_index': i}})
        self.legacy = [{'scene_name': self.scene, 'geometry_key': key, 'geometry_hash': h,
            'point_count': 2, 'canonical_geometry_locator': members[0]['geometry_locator'],
            'member_count': 2, 'members': members}]
        self.plans = [{'scene_name': self.scene, 'geometry_key': key, 'plan_key': key,
            'geometry_digest': h, 'geometry_digest_algorithm': 'sha1_point_indices',
            'point_count': 2, 'geometry_locator_read_only': members[0]['geometry_locator'], 'append_only': False}]

    def run_audit(self):
        return audit_scene(self.scene, self.spec, self.legacy, self.plans)

    def test_exact_geometry_dedup_preserves_all_native_classes(self):
        rows, points, _, summary = self.run_audit()
        self.assertEqual(len(rows), 1)
        self.assertEqual([m['frozen_class_index'] for m in rows[0]['legacy_members']], [1,2])
        self.assertEqual(summary['native_member_count'], 2)
        np.testing.assert_array_equal(points['g00000'], [0,1])

    def test_point_order_mismatch_is_rejected(self):
        np.save(self.processed, self.xyz[::-1])
        with self.assertRaisesRegex(ValueError, 'XYZ'):
            self.run_audit()

    def test_corrupt_geometry_claim_is_rejected(self):
        self.plans[0]['geometry_digest'] = '0'*40
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.run_audit()

    def test_lost_native_member_is_rejected(self):
        self.legacy[0]['members'].pop()
        self.legacy[0]['member_count'] = 1
        with self.assertRaisesRegex(ValueError, 'drops/duplicates'):
            self.run_audit()

    def test_changed_native_class_is_rejected(self):
        self.legacy[0]['members'][1]['frozen_class_index'] = 3
        with self.assertRaisesRegex(ValueError, 'mask/class'):
            self.run_audit()

    def test_refined_alias_is_not_counted_as_novel_geometry(self):
        refined = copy.deepcopy(self.plans[0])
        refined.update(plan_key='refined', append_only=True, candidate_source='refined_union',
                       original_union_geometry_key=refined['geometry_key'])
        self.plans.append(refined)
        rows, _, _, summary = self.run_audit()
        self.assertEqual(summary['refined_plan_count'], 1)
        self.assertEqual(summary['novel_refined_geometry_count'], 0)
        self.assertEqual(len(rows[0]['dv3_plans']), 2)

    def test_missing_parent_is_rejected(self):
        self.plans[0].update(append_only=True, candidate_source='refined_union', original_union_geometry_key='missing')
        with self.assertRaisesRegex(ValueError, 'parent'):
            self.run_audit()

    def test_bounds_and_integer_validation(self):
        for raw in (np.array([-1, 2]), np.array([4]), np.array([1.5])):
            with self.assertRaises(ValueError):
                checked_points(raw, 4)
        np.testing.assert_array_equal(checked_points(np.array([2,0,2]), 4), [0,2])


if __name__ == '__main__':
    unittest.main()
