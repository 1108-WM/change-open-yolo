import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
P=Path(__file__).resolve().parents[1]/'tools/run_fi1_query_same_parent_pilot.py'
spec=importlib.util.spec_from_file_location('pilot',P);pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)

class PilotTest(unittest.TestCase):
    def example(self):
        rng=np.random.default_rng(1);base=rng.normal(size=(120,64)).astype(np.float32);base[:,9]=1
        return dict(base=base,head=rng.normal(size=(120,128)).astype(np.float32),query=np.linspace(0,1,120,dtype=np.float32),scene='scene0001_00',candidate_id='c000',y=np.arange(120)%3)
    def test_only_query_changes_and_labels_never_change(self):
        e=self.example();y=e['y'].copy();a,b=pilot.inputs(e,'A'),pilot.inputs(e,'B')
        np.testing.assert_array_equal(a[:,:192],b[:,:192]);self.assertEqual(a.shape,(120,193));self.assertTrue((a[:,-1]==0).all());np.testing.assert_array_equal(b[:,-1],e['query'])
        for mode in pilot.MODES:pilot.inputs(e,mode)
        np.testing.assert_array_equal(y,e['y']);np.testing.assert_array_equal(pilot.inputs(e,'B_zero'),a)
    def test_shuffle_preserves_distribution_and_is_reproducible(self):
        e=self.example()
        for i in range(3):
            v=pilot.inputs(e,f'B_shuffle_{i}');np.testing.assert_array_equal(v,pilot.inputs(e,f'B_shuffle_{i}'));np.testing.assert_array_equal(np.sort(v[:,-1]),e['query']);self.assertFalse(np.array_equal(v[:,-1],e['query']));np.testing.assert_array_equal(v[:,:192],pilot.inputs(e,'B')[:,:192])
    def test_channel_reaches_forward(self):
        # Initial zero decoder deliberately ignores all learned channels; test a nonzero trained head.
        torch.manual_seed(1);net=pilot.CandidateLocalSegmenter(193,64).eval();torch.nn.init.normal_(net.decoder[-1].weight,std=.1);e=self.example();nn=torch.arange(120)[:,None].expand(-1,8);w=torch.ones(120,8)/8
        with torch.no_grad():a=net(torch.tensor(pilot.inputs(e,'A')),nn,w,[(0,120)]);b=net(torch.tensor(pilot.inputs(e,'B')),nn,w,[(0,120)])
        self.assertGreater(float((a-b).abs().max()),1e-5)
    def test_invisible_keep_and_min100(self):
        parent=np.arange(100);halo=np.arange(120);p=np.tile([.01,.98,.01],(120,1));groups=np.ones(120)
        out,_=pilot.repair(parent,halo,p,groups);np.testing.assert_array_equal(out,parent)
        out,meta=pilot.repair(parent,halo,p,np.ones(120)*2);np.testing.assert_array_equal(out,parent);self.assertEqual(meta['reason'],'minimum_mask_abstain')

if __name__=='__main__':unittest.main()
