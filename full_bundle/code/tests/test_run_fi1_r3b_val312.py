import tempfile
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools.run_fi1_r3b_val312 import validate_selected, bind_gt_arrays, evaluate
from tools.prepare_fi1_r0_r1_inputs import write
from tools.bind_fi1_r0_r1_alpha_cache import cached_geometry_match


class Val312ExecutionTest(unittest.TestCase):
    def test_alpha_namespace_fallback_requires_actual_point_digest(self):
        pts = np.arange(100,dtype='<i8'); raw = pts.tobytes(); h = hashlib.sha1(raw).hexdigest()
        g = {'point_count':100,'points_sha256':hashlib.sha256(raw).hexdigest(),
             'legacy_geometry_keys':['scene0011_00:geometry:'+h]}
        old = {'geometry_key':'scene0011_00:visual_geometry:'+h,'geometry_hash':h,'point_count':100,'scene_name':'scene0011_00'}
        cached = {old['geometry_key']:old}; index = {('scene0011_00',h):[old]}
        self.assertEqual(cached_geometry_match(g,pts,cached,index,'scene0011_00'),old)
        with self.assertRaises(ValueError): cached_geometry_match(g,pts+1,cached,index,'scene0011_00')
        self.assertIsNone(cached_geometry_match(g,pts,cached,index,'scene0015_00'))
        duplicate = {**old,'geometry_key':'different-alias'}
        with self.assertRaises(ValueError): cached_geometry_match(g,pts,cached,{('scene0011_00',h):[old,duplicate]},'scene0011_00')

    def test_selected_scores_require_exact_member_alignment(self):
        members = [{'geometry_key':'a','class_index':0,'has_original_native_alias':False},
                   {'geometry_key':'b','class_index':1,'has_original_native_alias':True}]
        scores = np.array([.3,.8]); selected = [{**members[1],'score':.8},{**members[0],'score':.3}]
        validate_selected(members,scores,selected)
        for bad in [scores[::-1],np.array([.8]),np.array([float('nan'),.8]),np.array([.3,1.1])]:
            with self.assertRaises(ValueError): validate_selected(members,bad,selected)

    def test_tied_scores_keep_original_native_priority(self):
        members = [{'geometry_key':'a','class_index':0,'has_original_native_alias':False},
                   {'geometry_key':'b','class_index':1,'has_original_native_alias':True}]
        expected = [{**members[1],'score':.5},{**members[0],'score':.5}]
        validate_selected(members,np.array([.5,.5]),expected)
        with self.assertRaises(ValueError): validate_selected(members,np.array([.5,.5]),expected[::-1])

    def test_gt_binding_rejects_wrong_point_order_and_labels(self):
        a = np.zeros((2,12)); a[:,0] = [1.,2.]; a[:,10] = [3,4]; a[:,11] = [0,1]
        xyz = a[:,:3].copy(); gt = np.array([3001,4002])
        self.assertEqual(bind_gt_arrays(a,gt,xyz),2)
        for bad in [gt[::-1],gt+1,gt[:1]]:
            with self.assertRaises(ValueError): bind_gt_arrays(a,bad,xyz)
        with self.assertRaises(ValueError): bind_gt_arrays(a,gt,xyz[::-1])
        a[0,10] = float('nan')
        with self.assertRaises(ValueError): bind_gt_arrays(a,gt,xyz)

    def test_evaluation_requires_explicit_stage_and_complete_predictions(self):
        with self.assertRaises(ValueError): evaluate(SimpleNamespace(allow_gt_evaluation=False))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); write(p/'identity.json',{}); write(p/'summary.json',{'complete':False})
            with self.assertRaises(ValueError):
                evaluate(SimpleNamespace(allow_gt_evaluation=True,predictions=p,output=p/'eval',preparation=p/'prep'))


if __name__ == '__main__':
    unittest.main()
