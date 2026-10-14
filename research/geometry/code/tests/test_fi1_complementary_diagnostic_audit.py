import numpy as np
import pytest
from tools import fi1_complementary_fragment_evidence as core
from tools import run_fi1_complementary_fragment_diagnostic as run


@pytest.mark.parametrize('shared',[np.array([],int),np.array([2])])
def test_precomputed_mask_evidence_equals_direct_visible_intersections(shared):
    a=np.array([0,1,3]);b=np.array([4,5,6]);pa=np.union1d(a,shared);pb=np.union1d(b,shared)
    masks=[np.arange(7),pa,pb];frames=[]
    for group in [0,1]:
        f=dict(frame=group,group=group,visible=np.ones(7,bool),pixels=np.column_stack([np.arange(7),np.zeros(7)]),depth=np.ones(7),masks=masks)
        f.update(candidate_visible=np.array([len(pa),len(pb)]),candidate_boxes=[(np.array([0,0]),np.array([3,0])),(np.array([2 if len(shared) else 4,0]),np.array([6,0]))],mask_sizes=np.array([len(m) for m in masks]),candidate_intersections=np.array([[len(np.intersect1d(p,m)) for m in masks] for p in [pa,pb]]))
        frames.append(f)
    cp=[[0,4],[1,5],[3,6]]
    assert core.view_evidence(a,b,frames,cp)==core.view_evidence(a,b,frames,cp,endpoints=(0,1),shared=shared)


def test_raw_gt_guard_requires_seal_and_exact_whitelist(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path)
    path=tmp_path/'ground_truth'/'scene0002_00.txt';path.parent.mkdir();path.write_text('1001')
    c=dict(scenes=['scene0002_00'],GT_inputs={str(path):'unused'})
    for labels in (False,True):
        with pytest.raises(PermissionError):run.Guard(c,labels).hook('open',(str(path),))
    (tmp_path/'features_frozen.json').write_text('{}')
    run.Guard(c,True).hook('open',(str(path),))
    with pytest.raises(PermissionError):run.Guard(c,False).hook('open',(str(path),))


def test_projection_checks_positive_camera_depth():
    xyz=np.array([[1.,2.,1.],[0.,0.,-1.],[0.,0.,0.]])
    pixels,depth,valid=core.project_points(xyz,np.eye(4),np.eye(4))
    assert valid.tolist()==[True,False,False]
    assert pixels[0].tolist()==[1,2] and depth.tolist()==[1,-1,0]
