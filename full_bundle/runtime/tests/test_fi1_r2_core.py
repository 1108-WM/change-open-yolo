import copy
import unittest

import numpy as np

from tools.fi1_r2_core import geometry_features, geometry_targets, valid_instances, scene_weights, select


class R2CoreTest(unittest.TestCase):
    def setUp(self):
        self.g={'geometry_key':'s:geometry:a','points_sha256':'a','point_count':100,
            'legacy_members':[{'candidate_source':'track','candidate_id':'x','frozen_class_index':0}], 'dv3_plans':[]}
        self.e={'geometry_key':'s:geometry:a','points_sha256':'a',
            'yolo':{'acquisition_complete':True,'available':True,'votes':[2,2]+[0]*196},
            'alpha':{'acquisition_complete':True,'available':True,'cosine':[0.]*198,'view_scale_std':[0.1]*198}}

    def test_twenty_features_tie_rank_and_gap(self):
        x=geometry_features(self.g,self.e)
        self.assertEqual(x.shape,(198,20))
        self.assertAlmostEqual(x[0,0],(0.5+1/198)/2)
        self.assertEqual(x[0,5],0)
        self.assertEqual(x[1,5],1/197)
        self.assertAlmostEqual(x[0,9],np.log(2))
        self.assertAlmostEqual(x[0,10],np.log(198))
        self.assertEqual(x[2,7],-0.5)
        self.assertEqual(x[0,12],1)
        self.assertEqual(x[0,15],1)

    def test_missing_modality_zeroes_own_features_and_cross_terms(self):
        self.e['alpha']={'acquisition_complete':True,'available':False,'missing_reason':'no_view'}
        x=geometry_features(self.g,self.e)
        np.testing.assert_array_equal(x[:,[2,3,4,6,8,10,11,12]],0)
        np.testing.assert_array_equal(x[:,19],1)
        self.assertEqual(x[0,0],0.5)

    def test_incomplete_acquisition_is_rejected(self):
        self.e['alpha']['acquisition_complete']=False
        with self.assertRaisesRegex(ValueError,'incomplete'):geometry_features(self.g,self.e)

    def test_geometry_quality_does_not_affect_features(self):
        before=geometry_features(self.g,self.e)
        self.g['dv3_score']=1000
        np.testing.assert_array_equal(before,geometry_features(self.g,self.e))

    def test_target_uses_max_same_class_iou_and_full_union(self):
        gt=np.array([2001]*100+[2002]*100+[3001]*100+[0]*100)
        instances=valid_instances(gt,[2,3])
        pts=np.r_[np.arange(50),np.arange(100,125),np.arange(300,325)]
        target=geometry_targets(pts,gt,instances,[2,3])
        np.testing.assert_allclose(target,[50/150,0])

    def test_small_gt_and_small_prediction_have_zero_target(self):
        gt=np.array([2001]*99+[3001]*101)
        inst=valid_instances(gt,[2,3])
        self.assertNotIn(2001,inst)
        np.testing.assert_array_equal(geometry_targets(np.arange(99),gt,inst,[2,3]),0)
        np.testing.assert_allclose(geometry_targets(np.arange(100),gt,inst,[2,3]),[0,1/200])

    def test_per_geometry_and_per_scene_equal_weights(self):
        members=[{'geometry_key':k} for k in ['a','b','b','b']]
        w=scene_weights(members)
        self.assertAlmostEqual(w.sum(),1)
        self.assertAlmostEqual(w[0],w[1:].sum())

    def test_select_keeps_native_tie_priority_and_changes_score_only(self):
        members=[{'geometry_key':'s:geometry:a','class_index':0,'score':0.9,'has_original_native_alias':False},
            {'geometry_key':'s:geometry:b','class_index':1,'score':0.,'has_original_native_alias':True}]
        before=copy.deepcopy(members)
        selected=select(members,[0.5,0.5])
        self.assertEqual(selected,[{**members[1],'score':0.5},{**members[0],'score':0.5}])
        self.assertEqual(members,before)

    def test_select_scores_all_prebudget_members(self):
        members=[{'geometry_key':f's:geometry:{i:04d}','class_index':0,'score':0.,'has_original_native_alias':False} for i in range(601)]
        scores=np.zeros(601);scores[600]=1
        selected=select(members,scores)
        self.assertEqual(len(selected),600)
        self.assertEqual(selected[0]['geometry_key'],members[600]['geometry_key'])

    def test_duplicate_and_nonfinite_predictions_rejected(self):
        m={'geometry_key':'s:geometry:a','class_index':0,'has_original_native_alias':False}
        with self.assertRaisesRegex(ValueError,'duplicate'):select([m,m],[0,0])
        with self.assertRaisesRegex(ValueError,'invalid'):select([m],[np.nan])

    def test_independent_features_across_all_missingness_patterns(self):
        from tools.audit_fi1_r2_dataset import features
        rng=np.random.default_rng(20260913)
        for ya,aa in [(True,True),(True,False),(False,True),(False,False)]:
            with self.subTest(yolo=ya,alpha=aa):
                e=copy.deepcopy(self.e)
                e['yolo'].update(available=ya,missing_reason=None if ya else 'no_votes',votes=rng.integers(0,100,198).tolist())
                e['alpha'].update(available=aa,missing_reason=None if aa else 'no_view',cosine=rng.uniform(-1,1,198).tolist())
                np.testing.assert_allclose(geometry_features(self.g,e),features(self.g,e),atol=1e-12,rtol=0)

    def test_randomized_targets_match_independent_boolean_iou(self):
        rng=np.random.default_rng(20260913)
        gt=np.repeat([2001,2002,3001,3002,0],[100,150,99,201,250])
        inst=valid_instances(gt,[2,3])
        for count in [0,99,100,150,400]:
            pts=np.sort(rng.choice(len(gt),size=count,replace=False))
            predicted=np.zeros(len(gt),dtype=bool);predicted[pts]=True
            expected=[]
            for cls in [2,3]:
                values=[0.]
                if count>=100:
                    for i in np.unique(gt):
                        truth=gt==i
                        if i//1000==cls and truth.sum()>=100:
                            values.append((truth&predicted).sum()/(truth|predicted).sum())
                expected.append(max(values))
            np.testing.assert_allclose(geometry_targets(pts,gt,inst,[2,3]),expected,atol=1e-15,rtol=0)


if __name__=='__main__':unittest.main()
