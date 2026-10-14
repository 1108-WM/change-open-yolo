"""Regression for independent replay eligibility and set IoU agreement."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from verify_fi1_local_edit_evidence_counts import counts,iou
from verify_fi1_nt_region_identity import vector

class ReplayTest(unittest.TestCase):
    def test_small_candidate_has_no_pool_value(self):
        gt=np.repeat([1001,2001],150);ids=np.array([1001,2001]);sizes=np.array([150,150])
        points=np.arange(99)
        np.testing.assert_array_equal(iou(counts(points,gt),len(points),ids,sizes),[0,0])
        self.assertGreater(iou(counts(np.arange(100),gt),100,ids,sizes)[0],0)

    def test_count_replay_agrees_with_independent_set_edits(self):
        gt=np.repeat([0,1001,2001],150);ids=np.array([1001,2001]);sizes=np.array([150,150])
        parent=np.arange(75,250);hist=counts(parent,gt)
        for sign,block in [(1,np.arange(250,420)),(-1,np.arange(200,250)),(-1,np.arange(75,200))]:
            actual=set(parent)|set(block) if sign==1 else set(parent)-set(block)
            replay=iou(hist+sign*counts(block,gt),len(parent)+sign*len(block),ids,sizes)
            np.testing.assert_allclose(replay,vector(actual,gt,ids,sizes),rtol=0,atol=1e-12)

if __name__=='__main__':unittest.main()
