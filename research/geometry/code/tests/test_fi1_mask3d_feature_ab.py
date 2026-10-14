import numpy as np
import pytest
import torch
from tools import fi1_mask3d_feature_ab_core as core
from tools.run_fi1_mask3d_feature_ab import Guard

def test_annotation_columns_do_not_enter_adapter():
    raw=np.arange(120,dtype=float).reshape(10,12);a=core.safe_columns(raw);raw[:,9:]=np.nan
    assert np.array_equal(a,core.safe_columns(raw))

def test_A_is_zero_after_normalization_and_B_aligned():
    base=np.zeros((7,64));feat=np.ones((7,128));mean=np.full(128,3);std=np.full(128,2)
    assert not core.inputs(base,feat,mean,std,'A')[:,64:].any()
    assert np.all(core.inputs(base,feat,mean,std,'B')[:,64:]==-1)
    with pytest.raises(ValueError):core.inputs(base,feat[:6],mean,std,'B')

def test_same_voxel_points_can_be_edited_independently():
    # Same instance-feature vector never ties decisions: original indices and point probabilities remain distinct.
    parent=np.arange(101);halo=np.arange(103);p=np.tile([.1,.1,.8],(103,1));p[0]=[.01,.98,.01];p[1]=[.98,.01,.01];p[102]=[.98,.01,.01]
    q=core.repair(parent,halo,p,np.full(103,2),.8)
    assert 0 not in q and 1 in q and 102 in q
    assert np.array_equal(core.repair(parent,halo,p,np.zeros(103),.8),parent)

def test_graph_network_has_no_atomic_voxel_binding():
    torch.manual_seed(1);net=core.model().eval()
    with torch.no_grad():net.decoder[-1].weight.normal_(std=.05)
    x=torch.zeros(101,192);x[:,9]=1;x[:,64:]=.123;x[0,:3]=torch.tensor([.001,.002,.003]);x[1,:3]=torch.tensor([.019,.018,.017]);nn=torch.arange(101)[:,None].expand(-1,8);w=torch.ones(101,8)/8
    with torch.no_grad():p=net(x,nn,w,[(0,101)])
    assert not torch.equal(p[0],p[1])

def test_physical_selection_does_not_split_scans():
    scenes=[f'scene{i:04d}_00' for i in range(30)]+['scene0000_01']
    split=core.selection_split(scenes)
    assert not {s.split('_')[0] for s in split['fit']}&{s.split('_')[0] for s in split['selection']}
    assert split==core.selection_split(list(reversed(scenes)))

def test_unstable_identity_and_unknown_are_not_safe():
    m=dict(actions=5,action_scenes=3,added=10,added_target=8,deleted=10,deleted_other=8,unknown_edits=4,changed_parent_target=100,deleted_target=0,good_parents=100,good_degraded=0)
    checks=core.protection(m);assert not checks['add_precision'] and not checks['delete_precision'] and not checks['unknown']

def test_no_action_does_not_pass_support():assert not core.protection({})['support']

def test_training_and_prediction_label_guards():
    c=dict(scenes=['scene0000_00','scene0001_00'],splits=dict(calibration=['scene0001_00']),GT_inputs={})
    guard=Guard(c,'train',['scene0000_00'])
    guard.hook('open',('/tmp/labels/scene0000_00/data.npz',))
    with pytest.raises(PermissionError):guard.hook('open',('/tmp/labels/scene0001_00/data.npz',))
    guard.mode='predict'
    with pytest.raises(PermissionError):guard.hook('open',('/tmp/labels/scene0000_00/data.npz',))
    with pytest.raises(PermissionError):guard.hook('open',('/tmp/ground_truth/scene0000_00.txt',))
