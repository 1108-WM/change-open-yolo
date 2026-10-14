from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import fi1_local_edit_evidence_core as core

class EditEvidenceTest(unittest.TestCase):
    def test_proposals_cover_both_missing_and_members_and_keep_visibility(self):
        labels=np.array([0,0,1,1]);member=np.array([True,True,False,False]);visible=np.array([2,1,2,2]);p=np.array([[.9,.1],[.8,.2],[.1,.9],[.9,.1]]);nn=np.tile(np.arange(4),(4,1));xyz=np.zeros((4,3))
        pieces=list(core.pieces(labels,member,visible,p,0,nn,xyz));whole=[r for r in pieces if r[1]=='region'];self.assertEqual(len(whole),2);np.testing.assert_array_equal(whole[0][3],[0]);np.testing.assert_array_equal(whole[1][3],[2,3]);parts=[r for r in pieces if r[1]=='query_boundary'];self.assertEqual(len(parts),2)
    def test_competitor_excludes_parent_query(self):
        r=core.query_evidence(np.array([[.9,.2,.3],[.8,.1,.4]]),0);self.assertEqual(r['competitor_id'],2);self.assertGreater(r['query_margin_mean'],0)
    def test_delete_reference_excludes_edit_and_missing_is_not_support(self):
        masks=np.array([[1,0],[0,1],[0,1]],bool);r=core.frame_evidence(np.array([0]),np.array([1,2]),np.ones(3,bool),masks);self.assertTrue(r['conflict']);self.assertFalse(r['common']);missing=core.frame_evidence(np.array([0]),np.array([],int),np.ones(3,bool),masks);self.assertTrue(missing['missing']);self.assertFalse(missing['conflict'])
    def test_same_mask_support_and_competition_are_descriptive(self):
        masks=np.ones((3,1),bool);r=core.frame_evidence(np.array([0]),np.array([1,2]),np.ones(3,bool),masks);self.assertTrue(r['common']);self.assertFalse(r['conflict'])
        row=dict(action='add',query_mean=.9,query_margin_mean=.2,competitor_mean=.7,common_frames=2,conflict_frames=0);self.assertTrue(core.screens(row)['query_contrast_multiview'])

if __name__=='__main__':unittest.main()
