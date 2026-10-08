import unittest
import numpy as np
from tools.fi1_r3_core import pair_overlap,context_features,gt_iou,assign_competition,independent_context,independent_assignment


def member(key,cls=0,score=.5,native=False):
    return {'geometry_key':key,'class_index':cls,'score':score,'has_original_native_alias':native}


class R3CoreTest(unittest.TestCase):
    def test_overlap_matches_boolean_and_excludes_cross_class_and_self(self):
        pts=[np.arange(100),np.arange(20,120),np.arange(200,300)]
        iou,cov=pair_overlap(pts,300)
        self.assertEqual(iou[0,1],80/120);self.assertEqual(cov[0,1],.8)
        m=[member('a',score=.3),member('b',score=.7),member('c',1)]
        x=context_features(m,['a','b','c'],iou,cov)
        np.testing.assert_allclose(x[0],[2/3,np.log(2),2/3*.7,-.4,1,.8])
        np.testing.assert_array_equal(x[2],0)
        np.testing.assert_allclose(x,independent_context(m,['a','b','c'],iou,cov),atol=1e-15)

    def test_half_threshold_strict(self):
        ov=np.array([[1.,.5],[.5,1.]])
        x=context_features([member('a'),member('b')],['a','b'],ov,ov)
        np.testing.assert_array_equal(x[:,1],0)
        np.testing.assert_array_equal(x[:,3:5],0)

    def test_capacity_and_next_available_gt(self):
        m=[member('a'),member('b')];ov=np.array([[.9,.8],[.85,.7]])
        y,a=assign_competition(m,['a','b'],ov,np.array([2001,2002]),[2])
        np.testing.assert_array_equal(y,[.9,.7]);np.testing.assert_array_equal(a,[2001,2002])
        # Deliberately greedy: global optimum .8+.85 is not this protocol.
        self.assertLess(y.sum(),.8+.85)

    def test_duplicate_gt_loser_zero(self):
        y,a=assign_competition([member('a'),member('b')],['a','b'],np.array([[.8],[.9]]),np.array([2001]),[2])
        np.testing.assert_array_equal(y,[0,.9]);np.testing.assert_array_equal(a,[-1,2001])

    def test_tie_direct_then_native_then_key(self):
        m=[member('a',score=.6),member('b',score=.7),member('c',score=.7,native=True)]
        y,_=assign_competition(m,['a','b','c'],np.ones((3,1))*.8,np.array([2001]),[2])
        np.testing.assert_array_equal(y,[0,0,.8])
        m[2]['has_original_native_alias']=False
        y,_=assign_competition(m,['a','b','c'],np.ones((3,1))*.8,np.array([2001]),[2])
        np.testing.assert_array_equal(y,[0,.8,0])

    def test_same_geometry_separate_class_capacity(self):
        m=[member('a',0),member('a',1)]
        y,a=assign_competition(m,['a'],np.array([[.6,.5]]),np.array([2001,3001]),[2,3])
        np.testing.assert_array_equal(y,[.6,.5]);np.testing.assert_array_equal(a,[2001,3001])

    def test_gt_minimum_and_prediction_minimum(self):
        gt=np.repeat([2001,2002,3001],[100,99,101])
        ov,ids=gt_iou([np.arange(99),np.arange(100),np.arange(199,300)],gt,[2,3])
        np.testing.assert_array_equal(ids,[2001,3001]);np.testing.assert_array_equal(ov[0],0)
        np.testing.assert_array_equal(ov[1],[1,0]);np.testing.assert_array_equal(ov[2],[0,1])

    def test_random_assignment_independent_replay(self):
        rng=np.random.default_rng(5);keys=[str(i) for i in range(8)]
        m=[member(k,c,float(rng.uniform()),bool(i%2)) for i,k in enumerate(keys) for c in [0,1]]
        ov=rng.integers(0,5,(8,7))/5;ids=np.array([2001,2002,2003,3001,3002,3003,3004])
        actual=assign_competition(m,keys,ov,ids,[2,3]);expected=independent_assignment(m,keys,ov,ids,[2,3])
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)

    def test_member_permutation_invariant(self):
        m=[member('a'),member('b')];ov=np.array([[.8],[.8]]);ids=np.array([2001])
        y,_=assign_competition(m,['a','b'],ov,ids,[2])
        yy,_=assign_competition(m[::-1],['a','b'],ov,ids,[2])
        np.testing.assert_array_equal(y,yy[::-1])

    def test_invalid_points_rejected(self):
        for p in [np.array([1,1]),np.array([-1]),np.array([100])]:
            with self.assertRaisesRegex(ValueError,'point'):pair_overlap([p],100)


if __name__=='__main__':unittest.main()
