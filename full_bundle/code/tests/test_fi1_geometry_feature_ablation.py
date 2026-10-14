import hashlib
import unittest
from collections import defaultdict
import numpy as np
from tools.fi1_geometry_feature_ablation import extra_features,matrix,replay_scores,member_identity

class FeatureAblationTest(unittest.TestCase):
    def test_replay_accepts_roundoff_but_rejects_material_or_invalid_scores(self):
        self.assertLess(replay_scores(np.array([.1]),np.array([.1+1e-16]))['max_abs_error'],1e-12)
        for x in [np.array([.1+1e-10]),np.array([np.nan]),np.array([.1,.2])]:
            with self.assertRaises(ValueError):replay_scores(np.array([.1]),x)
        a=[{'geometry_key':'a','score':.1},{'geometry_key':'b','score':.1}]
        self.assertNotEqual(member_identity(a),member_identity(a[::-1]))
    def fixture(self):
        scene='scene0001_00';points=[np.arange(200,dtype='<i8'),np.arange(100,dtype='<i8')]
        hashes=[hashlib.sha256(p.tobytes()).hexdigest() for p in points]
        legacy=scene+':geometry:'+hashlib.sha1(points[0].tobytes()).hexdigest()
        gs=[{'geometry_key':scene+':geometry:'+hashes[0],'points_sha256':hashes[0],'point_count':200,'legacy_geometry_keys':[legacy],'dv3_plans':[]},
            {'geometry_key':scene+':geometry:'+hashes[1],'points_sha256':hashes[1],'point_count':100,'legacy_geometry_keys':[],'dv3_plans':[{'append_only':True,'original_union_geometry_key':legacy}]}]
        aa={legacy:{'scene_name':scene,'geometry_hash':hashlib.sha1(points[0].tobytes()).hexdigest(),'deployment_unified_quality':.4}}
        cc={(scene,legacy,hashes[1]):{'append_eligible_refined_union':True,'temporary_refined_point_count':100,'original_union_sha256':hashes[0],'corrected_oof_temporary_refined_quality':.8}}
        return scene,gs,points,(aa,cc,defaultdict(list))
    def test_refined_uses_own_quality_and_pointwise_parent_change(self):
        scene,gs,points,ix=self.fixture();x,s=extra_features(scene,gs,points,ix)
        np.testing.assert_allclose(x[0],[.4,0,0,0,0,0,0])
        np.testing.assert_allclose(x[1],[.8,0,0,np.log(2),.5,0,.5])
        self.assertEqual(s['parent_geometries'],1)
    def test_union_has_two_distinct_parents_and_duplicate_edges_do_not_reweight(self):
        scene='scene0001_00';points=[np.arange(200,dtype='<i8'),np.arange(100,300,dtype='<i8'),np.arange(300,dtype='<i8')]
        gs=[];aa={};keys=[]
        for i,p in enumerate(points):
            h1=hashlib.sha1(p.tobytes()).hexdigest();h2=hashlib.sha256(p.tobytes()).hexdigest();key=scene+':geometry:'+h1;keys.append(key)
            gs.append({'geometry_key':h2,'points_sha256':h2,'point_count':len(p),'legacy_geometry_keys':[key],'dv3_plans':[]})
            aa[key]={'scene_name':scene,'geometry_hash':h1,'deployment_unified_quality':[.3,.6,.8][i]}
        links=[{'geometry_key_a':keys[2],'geometry_key_b':keys[0],'source_a':'pair_union','source_b':'native'},
               {'geometry_key_a':keys[1],'geometry_key_b':keys[2],'source_a':'track','source_b':'pair_union'}]
        x,stats=extra_features(scene,gs,points,(aa,{}, {scene:links+[links[0]]}))
        np.testing.assert_allclose(x[2],[.8,0,0,np.log(3),2/3,1/3,0])
        self.assertEqual(stats['parent_geometries'],1)
    def test_missing_or_wrong_quality_identity_is_rejected(self):
        scene,gs,points,ix=self.fixture();ix[0].clear()
        with self.assertRaises(ValueError):extra_features(scene,gs,points,ix)
        scene,gs,points,ix=self.fixture();points[0]=points[0]+1
        with self.assertRaises(ValueError):extra_features(scene,gs,points,ix)
    def test_features_are_separate_factors_and_base_is_unchanged(self):
        d={'X':np.ones((5,20)),'context':np.ones((5,6))*2,'extra':np.tile(np.arange(7),(5,1))}
        for arm,n in [('B',26),('Q',29),('P',30),('QP',33)]:
            x=matrix(d,arm);self.assertEqual(x.shape,(5,n));np.testing.assert_array_equal(x[:,:26],matrix(d,'B'))
        np.testing.assert_array_equal(matrix(d,'Q')[:,26:],d['extra'][:,:3])
        np.testing.assert_array_equal(matrix(d,'P')[:,26:],d['extra'][:,3:])

if __name__=='__main__':unittest.main()
