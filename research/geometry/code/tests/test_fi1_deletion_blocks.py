import unittest
from pathlib import Path
import numpy as np
import torch
from tools import fi1_deletion_block_core as core
from tools import run_fi1_deletion_blocks as run


class TestDeletionBlocks(unittest.TestCase):
    def test_components_do_not_use_gt_and_only_visible_parent(self):
        halo=np.arange(7);parent=np.arange(5);prob=np.tile([.05,.9,.05],(7,1));seen=np.array([2,2,2,2,1,2,2])
        nn=np.array([[1],[0],[3],[2],[3],[6],[5]])
        blocks=core.deletion_blocks(parent,halo,prob,seen,nn)
        self.assertEqual([b.tolist() for b in blocks],[[0,1],[2,3]])
    def test_partial_blocks_can_help_when_whole_action_harms(self):
        parent=np.arange(120);halo=parent.copy();gt=np.full(120,2001);gt[:100]=1001
        blocks=[np.arange(100,120),np.arange(20)]
        labels=core.consequences(gt,parent,halo,blocks,dict(stable=True,target=1001))
        self.assertEqual(labels[0][0][0],1);self.assertEqual(labels[1][0][0],0)
        self.assertTrue(np.array_equal(core.delete(parent,halo,blocks,[True,False]),np.arange(100)))
        self.assertTrue(np.array_equal(core.delete(parent,halo,blocks),parent))
    def test_ambiguous_identity_has_no_forced_target(self):
        gt=np.array([1001]*55+[2001]*45);p=np.arange(100);y,meta=core.stable_supervision(gt,p,p,dict(target=1001,good=False),[1,2])
        self.assertFalse(meta['stable']);self.assertIsNone(meta['target']);self.assertTrue((y==2).all())
        actions=core.consequences(gt,p,p,[np.arange(10)],meta);self.assertEqual(actions[0][0],[0.,0.,0.,1.,0.])
    def test_cost_bounded_and_full_domain_unknown_loss(self):
        logits=torch.zeros((100,3),requires_grad=True);labels=torch.zeros(100,dtype=torch.long);labels[0]=1
        parent=torch.ones(100,dtype=torch.bool);loss=core.corrected_loss(logits,labels,parent,False);loss.backward()
        ratio=abs(float(logits.grad[0,1]))/abs(float(logits.grad[1,0]));self.assertLess(ratio,4)
        unknown=core.corrected_loss(torch.zeros((100,3)),torch.full((100,),2),parent,False)
        self.assertAlmostEqual(float(unknown),float(np.log(3)),places=6)
    def test_guard_rejects_both_levels_heldout_and_posthoc(self):
        c=dict(GT_inputs={'/tmp/ground_truth/a.txt':'h'},jobs={'a':{'GT':'/tmp/ground_truth/a.txt'}})
        for mode,p in [('train',run.OUT/'labels'/'b'/'data.npz'),('predict',run.OUT/'labels'/'a'/'data.npz'),('train','/tmp/ground_truth/a.txt'),('predict','/tmp/independent_analysis/error.csv')]:
            with self.assertRaises(PermissionError):run.Guard(c,mode,{'a'}).hook('open',(str(p),'r',0))
    def test_relation_models_absent_neighbors_are_finite(self):
        x=torch.zeros((2,7,140));x[:,0,-1]=1
        for kind in ('summary','relation'):
            model=core.BlockVerifier(kind);logits=model(x);self.assertEqual(tuple(logits.shape),(2,5));self.assertTrue(torch.isfinite(logits).all())
            loss=core.verifier_loss(logits,torch.zeros(2,5),torch.ones(2));loss.backward()


if __name__=='__main__':unittest.main()
