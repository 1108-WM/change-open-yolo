import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import fi1_rejectable_local_block_core as c
class TestLocalAttribution(unittest.TestCase):
    def test_missing_weak_tie_conflict_are_distinct(self):
        edit=np.array([0,1]);ref=np.array([2,3]);vis=np.ones(4,bool)
        self.assertEqual(c.frame_witness(edit,ref,vis,np.zeros((4,0),bool))['state'],'missing')
        self.assertEqual(c.frame_witness(edit,ref,vis,np.zeros((4,2),bool))['state'],'observed_no_support')
        self.assertEqual(c.frame_witness(edit,ref,vis,np.ones((4,2),bool))['state'],'ambiguous_tie')
        mask=np.array([[1,0],[1,0],[0,1],[0,1]],bool);row=c.frame_witness(edit,ref,vis,mask)
        self.assertEqual(row['state'],'conflict');self.assertTrue(c.attribution([row,{'state':'common'},{'state':'common'}],.9).startswith('reject'))
    def test_alternate_query_can_have_zero_parent_support(self):
        p=np.array([[0,.9],[0,.8],[1,0],[1,0]],np.float32);query,mean=c.supporter(p,np.array([0,1]),[0,1]);self.assertEqual(query,1);self.assertEqual(p[2:,query].sum(),0)
        state=c.attribution([{'state':'common'},{'state':'common'}],mean);self.assertTrue(state.startswith('accept'))
    def test_clipping_does_not_import_query_points_from_other_regions(self):
        p=np.array([[.9],[.1],[.99],[.99]],np.float32);whole,clip=c.readout(np.array([0,1]),p,0,True);np.testing.assert_array_equal(whole,[0,1]);np.testing.assert_array_equal(clip,[0]);self.assertNotIn(2,clip)
        w,z=c.readout(np.array([0,1]),p,0,False);self.assertEqual(len(w)+len(z),0)
if __name__=='__main__':unittest.main()
