import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from fi1_nt_instance_ownership_core import duplicate_groups, credible_groups, ownership


def test_large_bridge_does_not_transitively_merge_disjoint_instances():
    # A broad N overlaps two disjoint T objects by exactly half its volume.
    sizes=np.array([100,200,100])
    intersection=np.array([[100,100,0],[100,200,100],[0,100,100]])
    groups=duplicate_groups(intersection,sizes,['a','bridge','c'])
    assert all(not (0 in group and 2 in group) for group in groups)


def test_unique_reference_accepts_without_query_or_parent_block_overlap():
    groups=[[0,1],[2,3]];credible=[0,1]
    result=ownership([100,90,0,0],100,[0,25,0,0],25,groups,credible)
    assert result['state']=='accept'
    # Parent's own entry is zero on the missing block; another N supplies its link.


def test_competition_abstains_and_different_instance_rejects():
    groups=[[0,1],[2,3]];credible=[0,1]
    assert ownership([100,90,0,0],100,[0,25,25,25],25,groups,credible)['state']=='unknown'
    assert ownership([100,90,0,0],100,[0,0,25,25],25,groups,credible)['state']=='reject'
    assert ownership([100,90,0,0],100,[0,0,0,0],25,groups,credible)['state']=='unknown'


def test_aliases_are_not_independent_cross_source_agreement():
    assert credible_groups([[0]],[100],[['native','track']])==[]
    assert credible_groups([[0,1]],[100,110],[['native'],['track']])==[0]
    assert credible_groups([[0,1]],[99,110],[['native'],['track']])==[]
