import copy
import hashlib
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from tests.test_fi1_r0_r1_protocol import fixture
from tools.prepare_fi1_r3b_val312 import overlaps,inspect_yolo,score_scene,alpha_plan_hash


class FixedModel:
    def __init__(self,n):self.n_features_in_=n;self.seen=None
    def predict(self,x):self.seen=x.copy();return x[:,0]


class PreparationTest(unittest.TestCase):
    def test_refined_sha256_is_verified_then_converted_to_alpha_sha1(self):
        with tempfile.TemporaryDirectory() as td:
            pts=np.arange(100,dtype='<i8');p=Path(td)/'points.npz';np.savez(p,point_indices=pts)
            plan={'geometry_digest_algorithm':'sha256_point_indices','geometry_digest':hashlib.sha256(pts.tobytes()).hexdigest(),
                  'point_count':100,'geometry_locator_read_only':{'kind':'point_indices_npz','points_path':str(p)}}
            self.assertEqual(alpha_plan_hash(plan),hashlib.sha1(pts.tobytes()).hexdigest())
            np.savez(p,point_indices=pts+1)
            with self.assertRaises(ValueError):alpha_plan_hash(plan)

    def test_main_scene_overlap_detected_without_exact_scan_overlap(self):
        self.assertEqual(overlaps(['scene0001_00'],['scene0001_01']),{'scans':[],'main_scenes':['scene0001']})
        with self.assertRaises(ValueError):overlaps(['scene0001_00']*2,[])

    def test_bare_cache_not_misrepresented_as_provenance_verified(self):
        p={'0':{'bbox':torch.tensor([[1.,2.,3.,4.]]),'labels':torch.tensor([0]),'scores':torch.tensor([.5])}}
        r=inspect_yolo(p,'scene0001_00',['chair']);self.assertFalse(r['historical_vocabulary_identity_verified']);self.assertEqual(r['bbox_count'],1)
        wrapped={'metadata':{'scene_name':'scene0001_00','text_prompts':['chair']},'predictions':p}
        self.assertTrue(inspect_yolo(wrapped,'scene0001_00',['chair'])['current_exporter_metadata_present'])
        with self.assertRaises(ValueError):inspect_yolo(wrapped,'scene0002_00',['chair'])
        with self.assertRaises(ValueError):inspect_yolo(wrapped,'scene0001_00',['table'])

    def test_bad_cache_label_bounds_duplicate_frames_and_shapes_rejected(self):
        p={'0':{'bbox':torch.zeros((1,4)),'labels':torch.tensor([1]),'scores':torch.tensor([.5])}}
        with self.assertRaises(ValueError):inspect_yolo(p,'scene0001_00',['chair'])
        p['0']['labels']=torch.tensor([0]);p['00']=copy.deepcopy(p['0'])
        with self.assertRaises(ValueError):inspect_yolo(p,'scene0001_00',['chair'])
        p.pop('00');p['0']['scores']=torch.tensor([float('nan')])
        with self.assertRaises(ValueError):inspect_yolo(p,'scene0001_00',['chair'])

    def data(self):
        g,e=fixture();pts=np.arange(100,dtype='<i8');h=hashlib.sha256(pts.tobytes()).hexdigest();key='scene0001_00:geometry:'+h
        g.update(geometry_key=key,points_sha256=h);e.update(geometry_key=key,points_sha256=h)
        return g,e,pts,{'R2':FixedModel(20),'R3b':FixedModel(26)}

    def test_deployment_preserves_member_sets_and_feature_dimensions(self):
        g,e,pts,models=self.data();r=score_scene([g],[pts],{g['geometry_key']:e},100,models)
        self.assertEqual(models['R2'].seen.shape,(2,20));self.assertEqual(models['R3b'].seen.shape,(2,26))
        np.testing.assert_array_equal(models['R3b'].seen[:,:20],models['R2'].seen)
        np.testing.assert_array_equal(models['R3b'].seen[:,20:],0)
        self.assertEqual(r['R2']['selected'],r['R1']['selected']);self.assertEqual(r['R3b']['selected'],r['R1']['selected'])
        self.assertEqual(len(r['R0']['selected']),1)

    def test_missing_alpha_or_wrong_point_identity_cannot_score(self):
        g,e,pts,models=self.data();e['alpha']['acquisition_complete']=False
        with self.assertRaises(ValueError):score_scene([g],[pts],{g['geometry_key']:e},100,models)
        e['alpha']['acquisition_complete']=True
        with self.assertRaises(ValueError):score_scene([g],[pts+1],{g['geometry_key']:e},101,models)
        models['R3b'].n_features_in_=20
        with self.assertRaises(ValueError):score_scene([g],[pts],{g['geometry_key']:e},100,models)


if __name__=='__main__':unittest.main()
