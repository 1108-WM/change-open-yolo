from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import fi1_t_add_union_core as core

class TAddTest(unittest.TestCase):
    def test_whole_blocks_and_competitor_mean_not_pointwise_max(self):
        halo=np.arange(6);parent=halo[:2];labels=np.array([0,0,1,1,2,2]);visible=np.full(6,2)
        p=np.array([[0,0,0],[0,0,0],[.5,.9,0],[.5,0,.9],[.7,.4,.1],[.7,.4,.1]])
        b=list(core.blocks(parent,halo,labels,visible,p,0))
        self.assertEqual([x['region'] for x in b],[1,2]);self.assertTrue(b[0]['A']);self.assertFalse(b[0]['B'])
        self.assertAlmostEqual(b[0]['competitor_mean'],.45);self.assertTrue(b[1]['B'])

    def test_union_keeps_all_parent_and_all_selected_blocks(self):
        parent=np.array([0,2,4]);p,a=core.union(parent,[np.array([1,3]),np.array([5])])
        np.testing.assert_array_equal(p,np.arange(6));np.testing.assert_array_equal(a,[1,3,5])
        np.testing.assert_array_equal(core.union(parent,[])[0],parent)

    def test_visibility_is_existing_domain_not_new_rule(self):
        halo=np.arange(4);parent=halo[:1];labels=np.array([0,1,1,1]);visible=np.array([2,2,1,3]);p=np.full((4,2),.8)
        b=list(core.blocks(parent,halo,labels,visible,p,0));np.testing.assert_array_equal(b[0]['indices'],[1,3]);self.assertTrue(b[0]['A']);self.assertFalse(b[0]['B'])

if __name__=='__main__':unittest.main()
