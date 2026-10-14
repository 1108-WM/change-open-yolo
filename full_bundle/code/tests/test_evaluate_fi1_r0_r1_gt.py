import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData,PlyElement

from tools.evaluate_fi1_r0_r1_gt import reconstruct_gt, prediction_geometry_paths


class GroundTruthBindingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scene = 'scene0001_00'
        raw = self.root/self.scene
        raw.mkdir()
        self.xyz = np.arange(12,dtype=np.float32).reshape(4,3)
        fields = [('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1'),('label','u2')]
        vertices = np.array([(*p,0,0,0,0) for p in self.xyz],dtype=fields)
        for suffix in ['_vh_clean_2.ply','_vh_clean_2.labels.ply']:
            PlyData([PlyElement.describe(vertices,'vertex')]).write(raw/(self.scene+suffix))
        (raw/(self.scene+'_vh_clean_2.0.010000.segs.json')).write_text(json.dumps({'segIndices':[1,1,2,3]}))
        (raw/(self.scene+'.aggregation.json')).write_text(json.dumps({'segGroups':[
            {'id':0,'label':'chair','segments':[1]},{'id':1,'label':'table','segments':[2]}]}))
        self.processed = self.root/'processed.npy'
        np.save(self.processed,self.xyz)
        self.labels = self.root/'map.tsv'
        self.labels.write_text('raw_category\tid\nchair\t2\ntable\t4\n')
        self.gt = self.root/'gt.txt'
        np.savetxt(self.gt,[2001,2001,4002,0],fmt='%d')

    def run_check(self):
        return reconstruct_gt(self.scene,self.root,self.processed,self.labels,self.gt)

    def test_cached_ids_match_reconstructed_class_and_instance_ids(self):
        result = self.run_check()
        self.assertTrue(result['raw_gt_reconstruction_equal'])
        self.assertEqual(result['point_count'],4)

    def test_same_length_shuffled_gt_is_rejected(self):
        np.savetxt(self.gt,[2001,4002,2001,0],fmt='%d')
        with self.assertRaisesRegex(ValueError,'reconstruction'):
            self.run_check()

    def test_same_shape_wrong_point_order_is_rejected(self):
        np.save(self.processed,self.xyz[::-1])
        with self.assertRaisesRegex(ValueError,'XYZ'):
            self.run_check()

    def test_wrong_class_mapping_is_rejected(self):
        self.labels.write_text('raw_category\tid\nchair\t3\ntable\t4\n')
        with self.assertRaisesRegex(ValueError,'reconstruction'):
            self.run_check()

    def test_verified_point_file_cannot_differ_from_consumed_binding(self):
        binding = self.root/'binding'
        receipt = {'geometry_file':str(binding/self.scene/'geometry.json'),
                   'points_file':str(self.root/'different_but_valid_points.npz')}
        with self.assertRaisesRegex(ValueError,'verified and consumed'):
            prediction_geometry_paths(receipt,binding,self.scene)

    def test_verified_canonical_point_file_is_returned_for_consumption(self):
        binding = self.root/'binding'
        receipt = {'geometry_file':str(binding/self.scene/'geometry.json'),
                   'points_file':str(binding/self.scene/'points.npz')}
        paths = prediction_geometry_paths(receipt,binding,self.scene)
        self.assertEqual(paths['points_file'],binding/self.scene/'points.npz')


if __name__ == '__main__':
    unittest.main()
