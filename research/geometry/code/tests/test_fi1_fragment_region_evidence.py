import numpy as np
from tools.fi1_fragment_region_visual_evidence import depth_path_evidence,region_depth_evidence,chosen_views,pool_tokens

def test_real_depth_path_detects_occluder_despite_equal_endpoint_depth():
    depth=np.ones((20,20),np.float32);depth[:,9:12]=2
    a=np.array([[1,10],[2,10],[3,10]]);b=np.array([[17,10],[18,10],[19,10]])
    result=depth_path_evidence(depth,a,b)
    assert result[2]>0.5 # max measured jump, not delta-z of the endpoints.
    assert np.allclose(result,depth_path_evidence(depth,b,a))

def test_missing_depth_is_explicit_and_not_depth_continuity():
    depth=np.zeros((5,5));a=np.array([[0,2]]);b=np.array([[4,2]])
    v=depth_path_evidence(depth,a,b);assert v[3]==1 and v[4]==1

def test_region_occlusion_uses_sensor_not_neighbor_geometry():
    d=np.ones((5,5));pixels=np.array([[2,2],[3,2]]);camera=np.array([2.,2.])
    v=region_depth_evidence(d,pixels,camera,np.arange(2));assert v[1]==1 and v[2]==0

def test_view_selection_preserves_distinct_groups_and_abstains_no_joint_visibility():
    a=np.arange(3);b=np.arange(3,6)
    frames=[dict(frame=i,group=g,visible=np.ones(6,bool)) for i,g in enumerate([0,0,1,2])]
    assert chosen_views(a,b,frames)==[0,2,3]
    for f in frames:f['visible'][b]=False
    assert chosen_views(a,b,frames)==[]

def test_token_pool_uses_only_projected_region_cells():
    t=np.zeros((24,32,384));t[0,0,0]=1;t[0,1,1]=1
    v,n=pool_tokens(t,np.array([[0,0],[1,1]]),640,480)
    assert n==1 and np.isclose(v[0],1) and v[1]==0
