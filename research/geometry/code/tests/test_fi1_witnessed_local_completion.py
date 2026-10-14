import numpy as np
from tools.fi1_witnessed_local_completion import witnessed_addition,PARAMETERS


def frames(groups,separate=False):
    m=[np.arange(6)] if not separate else [np.arange(3),np.arange(3,6)]
    return [dict(frame=i,group=g,visible=np.ones(6,bool),masks=m) for i,g in enumerate(groups)]


def test_one_pose_repeat_cannot_make_independent_support():
    xyz=np.zeros((6,3));p,r=witnessed_addition(np.arange(3),np.arange(6),frames([1,1]),xyz)
    assert not r['accepted'] and np.array_equal(p,np.arange(3))


def test_separation_vetoes_broad_joint():
    xyz=np.zeros((6,3));ff=frames([1,2]);ff+=frames([3],True)
    p,r=witnessed_addition(np.arange(3),np.arange(6),ff,xyz)
    assert r['reason']=='separation_counterevidence'


def test_completion_adds_only_supported_points_and_keeps_parent():
    xyz=np.zeros((6,3));p,r=witnessed_addition(np.arange(3),np.arange(6),frames([1,2]),xyz)
    assert r['accepted'] and np.array_equal(p,np.arange(6)) and r['added_points']==3
