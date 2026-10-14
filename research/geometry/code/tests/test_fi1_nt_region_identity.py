"""Boundary tests for partition, mappings and diagnostic action semantics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import fi1_nt_region_identity_core as c


def test_partition_empty_intersection_containment_competitors_and_zero_members():
    xyz=np.array([[0,0,0],[.01,0,0],[.02,0,0],[.03,0,0],[.3,0,0]])
    membership=np.array([[1,1,0],[1,1,0],[1,0,1],[0,0,1],[0,0,0]],bool)
    neighbors=np.array([[1,2],[0,2],[1,3],[2,4],[3,4]])
    labels,order,offset,edges,distance=c.partition(xyz,neighbors,membership)
    assert labels.tolist()==[0,0,1,2,3]
    assert sorted(order.tolist())==list(range(5)) and offset[-1]==5
    assert edges.tolist()==[[0,1],[1,2]]
    assert np.all(distance<=.05)


def test_probability_readout_visibility_and_minimum_fallback():
    halo=np.arange(130);parent=np.arange(100)
    p=np.ones(130);p[:10]=0
    visible=np.full(130,2);visible[:5]=0
    out=c.probe(parent,halo,p,visible)
    assert np.array_equal(out,np.r_[np.arange(5),np.arange(10,130)])
    assert np.array_equal(c.probe(parent,halo,np.zeros(130),visible),parent)


def test_unknown_not_deleted_and_mixed_region_not_pure():
    halo=np.arange(130);parent=np.arange(100);gt=np.full(130,1001);gt[:5]=0;gt[5:10]=2001
    visible=np.full(130,2)
    point=c.feasible(parent,halo,gt,1001,visible)
    assert np.array_equal(point,np.r_[np.arange(5),np.arange(10,130)])
    region=c.feasible(parent,halo,gt,1001,visible,[np.arange(130)])
    assert np.array_equal(region,parent)


def test_coordinate_row_reordering_and_safe_columns():
    xyz=np.array([[0,1,2,3],[0,2,3,4],[0,3,4,5]])
    assert c.coordinate_rows(xyz,xyz[[2,0,1]]).tolist()==[2,0,1]
    raw=np.zeros((5,12));raw[:,9:]=np.nan
    assert np.array_equal(c.safe_input(raw),np.zeros((5,9)))


def test_full_volume_association_and_ties_are_recorded():
    p=np.array([[.9,.8,.8],[.9,.8,.8]])
    link=c.association(p,np.array([100.,1.6,1.6]),np.arange(2))
    assert link['top']==[1,2] and link['ambiguous'] and link['gap']==0
    assert c.association(np.zeros((2,1)),np.array([0]),np.arange(2))['missing']


def test_region_point_probe_difference_is_preserved():
    p=np.r_[np.full(100,.49),np.full(30,.9)][:,None];labels=np.zeros(130,np.int32)
    means,std=c.region_stats(p,labels)
    assert means[0,0]>.5 and std[0,0]>0
    assert np.array_equal(c.probe(np.arange(100),np.arange(130),p[:,0],np.full(130,2)),np.arange(100))
    assert len(c.probe(np.arange(100),np.arange(130),means[labels,0],np.full(130,2)))==130
