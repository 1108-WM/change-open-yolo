"""Semantic safeguards for posthoc diagnosis, not tests of fitted thresholds."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_fi1_local_block_attribution_selection import census, union
from fi1_rejectable_local_block_core import readout


def test_unknown_is_neither_target_nor_neighbor():
    gt=np.array([1001,2001,0,1])
    c=census(np.arange(4),gt,1001)
    assert (c['target'],c['other'],c['unknown'])==(1,1,2)
    unresolved=census(np.arange(4),gt,None)
    assert unresolved['target'] is None and unresolved['other'] is None
    assert unresolved['annotated']==2 and unresolved['unknown']==2


def test_clipping_addition_preserves_parent_and_excludes_other_blocks():
    parent=np.array([0,1]);block=np.array([2,3]);p=np.array([[0.],[0.],[.9],[.1],[1.]])
    whole,clip=readout(block,p,0,True)
    assert np.array_equal(np.union1d(parent,clip),[0,1,2])
    assert np.array_equal(whole,[2,3])
    assert 4 not in clip


def test_union_does_not_pick_gt_good_block():
    good=np.array([2,3]);bad=np.array([4,5]);gt=np.array([1001,1001,1001,1001,2001,0])
    added=union([good,bad]);c=census(added,gt,1001)
    assert (c['target'],c['other'],c['unknown'])==(2,1,1)
    assert c['ownership']=='mixed_known'
