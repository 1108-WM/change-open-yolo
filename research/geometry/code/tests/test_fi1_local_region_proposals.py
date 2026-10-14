import unittest
import numpy as np
from tools import fi1_local_region_proposals as core
from tools import run_fi1_local_region_supply as run


def evidence(n=100):
    xyz=np.zeros((n,3),np.float32);xyz[:,0]=np.arange(n)*.002
    nn=np.array([[max(0,i-1),min(n-1,i+1)]+[i]*6 for i in range(n)],np.int32)
    return xyz,np.full((n,3),100,np.float32),np.tile([0,0,1.],(n,1)),np.arange(n),np.arange(n),np.full(n,2),nn,np.zeros(n,bool)


class TestRegionProposals(unittest.TestCase):
    def test_no_disagreement_still_proposes_and_preserves_visibility(self):
        args=list(evidence());args[5][:5]=1
        blocks,meta,budget,trace=core.propose(*args)
        self.assertGreater(len(budget['96']),0);self.assertEqual(set(core.union(blocks)),set(range(5,100)))
        for ids in budget.values():self.assertEqual(len(ids),len(set(ids)));self.assertLessEqual(len(ids),192)
    def test_color_boundary_splits_same_tile_without_gt(self):
        args=list(evidence(10));args[1][5:]=255
        blocks,meta,budget,trace=core.propose(*args)
        self.assertTrue(all(not (np.any(b<5) and np.any(b>=5)) for b in blocks))
    def test_reverse_normals_do_not_create_fake_boundary(self):
        args=list(evidence(10));before=core.propose(*args)[0];args[2][5:]*=-1;after=core.propose(*args)[0]
        self.assertEqual([b.tolist() for b in before],[b.tolist() for b in after])
    def test_asymmetric_knn_edge_is_retained(self):
        nn=np.tile(np.arange(3)[:,None],(1,8));nn[2,0]=0
        result=core.components(np.arange(3),nn);self.assertEqual([b.tolist() for b in result],[[0,2],[1]])
    def test_probability_boundary_is_explicit_float32(self):
        args=evidence(3);prob=np.array([[0,.8,.2],[0,.801,.199],[0,.799,.201]],np.float16)
        blocks=core.probability_blocks(args[3],args[4],prob,args[5],args[6]);self.assertEqual(core.union(blocks).tolist(),[1])
    def test_budget_bounds_and_multi_size_supply(self):
        sizes=[2,20,100,300];xyz=np.concatenate([np.full((n,3),k,np.float32) for k,n in enumerate(sizes)])
        blocks=np.split(np.arange(sum(sizes)),np.cumsum(sizes)[:-1]);order=core.inspection_order(blocks,xyz,np.zeros(len(xyz),bool),4)
        self.assertEqual(set(order),set(range(4)))
    def test_guard_excludes_holdout_gt_and_learned_proposal_generation(self):
        c=dict(scenes=['scene0000_00'],folds={'scene0000_00':0},GT_inputs={})
        for mode,path in [('generate',run.PREVIOUS/'generators/outer0/full/model.pt'),('generate',run.SOURCE/'labels/scene0000_00/meta.json'),('diagnose','/tmp/ground_truth/scene0000_00.txt')]:
            with self.assertRaises(PermissionError):run.Guard(c,mode,0).hook('open',(str(path),'r',0))


if __name__=='__main__':unittest.main()
