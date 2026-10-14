import numpy as np
from tools.diagnose_fi1_sl_remaining_development200 import missing_kind, prefix, frontier, Guard
import pytest


def test_mutually_exclusive_strict_threshold():
    t=.7500000000000002
    assert missing_kind(t,t,t,t)=='no_NT_geometry'
    assert missing_kind(.9,t,0,t)=='no_correct_class_eligibility'
    assert missing_kind(.9,.9,t,t)=='outside_S800'
    assert missing_kind(.9,.9,.9,t)=='covered'


def test_ties_not_given_arbitrary_rank():
    assert prefix([.1,.2,.2,.4],.2)=={'above':1,'tied':2}
    assert prefix([],.3)=={'above':0,'tied':0}


def test_correct_class_frontier_not_geometry_presence():
    io=np.array([[.8,.1],[.3,.9]])
    rows=[{'geometry_key':'a','class_index':1},{'geometry_key':'b','class_index':1}]
    assert frontier(rows,{'a':0,'b':1},io,np.array([0,1])).tolist()==[0,.9]


def test_read_boundary():
    g=Guard(['scene0002_00'])
    for q in ('/x/ground_truth/scene0002_00.txt','/x/scene0003_00/a.json','/x/val312/summary.json'):
        with pytest.raises(PermissionError):g.hook('open',(q,))
    g.hook('open',('/x/scene0002_00/overlap.npz',))
