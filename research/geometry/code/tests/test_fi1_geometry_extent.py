from tools.diagnose_fi1_nt_geometry_gaps import extent_kind,metrics
import numpy as np


def test_extent_decomposes_missed_and_extra_points():
    gt=np.array([1001]*10+[2001]*10)
    q=metrics(np.array([0,1,2,3,4]),gt,1001)
    assert q['coverage']==.5 and q['purity']==1 and q['iou']==.5
    assert extent_kind(q['purity'],q['coverage'],q['intersection'])=='incomplete_clean'
    q=metrics(np.arange(20),gt,1001)
    assert extent_kind(q['purity'],q['coverage'],q['intersection'])=='complete_contaminated'


def test_zero_intersection_is_not_fragment():
    assert extent_kind(0,0,0)=='no_overlap'
    assert extent_kind(.5,.5,5)=='incomplete_contaminated'
    assert extent_kind(.8,.8,8)=='boundary_both'
