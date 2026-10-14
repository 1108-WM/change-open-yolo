import numpy as np
from tools.fi1_complementary_fragment_evidence import bbox_gap,boundary,view_evidence,signatures

def frames(groups,masks):
    return [dict(frame=i,group=g,visible=np.ones(6,bool),pixels=np.array([[i,0] for i in range(6)]),depth=np.ones(6),masks=masks) for i,g in enumerate(groups)]

def test_disjoint_spatial_fragments_can_be_retrieved():
    raw=np.zeros((6,9));raw[:,0]=np.arange(6)*.005;raw[:,6]=1
    bd=boundary(np.arange(3),np.arange(3,6),raw)
    assert bd['contacts10']==3 and bd['contacts05']==3

def test_broad_background_mask_is_not_compact_identity():
    f=frames([1,2],[np.arange(6)]);a=np.array([0]);b=np.array([5])
    # More points to reach physical minimum; mask includes unrelated region.
    a=np.arange(3);b=np.arange(3,6);f[0]['masks']=[np.arange(10)];f[1]['masks']=[np.arange(10)]
    v=view_evidence(a,b,f,[]);assert len(v['joint_groups'])==2 and not v['compact_groups']

def test_same_pose_repeats_do_not_supply_two_groups():
    v=view_evidence(np.arange(3),np.arange(3,6),frames([1,1],[np.arange(6)]),[])
    assert len(v['joint_groups'])==1

def test_separation_vetoes_joint_and_missing_depth_abstains():
    f=frames([1,2],[np.arange(6),np.arange(3),np.arange(3,6)]);v=view_evidence(np.arange(3),np.arange(3,6),f,[])
    bd=dict(contacts05=3,normal_difference=0.,color_difference=0.,normal_valid_fraction=1.)
    z=signatures(bd,v);assert z['joint_only'] and not z['compact_no_separation'] and not z['compact_boundary_depth']
    v=view_evidence(np.arange(3),np.arange(3,6),frames([1,2],[np.arange(6)]),[]);assert signatures(bd,v)['compact_no_separation'] and not signatures(bd,v)['compact_boundary_depth']

def test_bbox_gap_is_symmetric_and_requires_no_intersection():
    a=(np.array([0,0]),np.array([2,2]));b=(np.array([4,0]),np.array([6,2]));assert bbox_gap(a,b)==bbox_gap(b,a)==2
