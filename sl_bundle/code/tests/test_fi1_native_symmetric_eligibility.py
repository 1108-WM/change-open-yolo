import copy

import numpy as np
import pytest

from tools.run_fi1_native_symmetric_eligibility_development200 import symmetric_members
from tools.fi1_r0_r1_protocol import build_pair


def fixture(source='native'):
    key='scene0002_00:geometry:'+'a'*64
    g=dict(geometry_key=key,points_sha256='a'*64,point_count=100,
        legacy_members=[dict(candidate_source=source,candidate_id=1,frozen_class_index=1)],dv3_plans=[])
    votes=[0]*198; votes[2]=9; votes[1]=1
    e=dict(geometry_key=key,points_sha256='a'*64,
        yolo=dict(acquisition_complete=True,available=True,votes=votes),
        alpha=dict(acquisition_complete=True,available=False,missing_reason='no_visible_views'))
    return g,{key:e}


def test_new_native_class_preserves_real_source_and_alias_identity():
    g,e=fixture(); before=copy.deepcopy((g,e))
    old=build_pair([g],e)['R1']['eligible']; new=symmetric_members([g],e)
    assert [m['class_index'] for m in new]==[2,1]
    assert new[1]==old[0]
    assert new[0]['expanded_class'] and not new[0]['has_original_native_alias']
    assert new[0]['original_aliases']==new[0]['native_alias_keys']==[]
    assert (g,e)==before


@pytest.mark.parametrize('source',['track','pair_union'])
def test_nonnative_r1_members_are_identical(source):
    g,e=fixture(source)
    assert symmetric_members([g],e)==build_pair([g],e)['R1']['eligible']


def test_shared_exact_pointset_does_not_invent_geometry_or_members():
    g,e=fixture();g['legacy_members'].append(dict(candidate_source='track',candidate_id=8,frozen_class_index=1))
    assert symmetric_members([g],e)==build_pair([g],e)['R1']['eligible']


def test_missing_both_retains_original_zero_score_class():
    g,e=fixture();e[g['geometry_key']]['yolo']=dict(acquisition_complete=True,available=False,missing_reason='no_votes')
    assert symmetric_members([g],e)==build_pair([g],e)['R1']['eligible']


def test_no_topk_filter_and_native_background_does_not_block_expansion():
    g,e=fixture();g['legacy_members'][0]['frozen_class_index']=198
    e[g['geometry_key']]['alpha']=dict(acquisition_complete=True,available=True,cosine=[0.]*198)
    rows=symmetric_members([g],e)
    assert len(rows)==198 and all(m['expanded_class'] and not m['has_original_native_alias'] for m in rows)
    assert {m['class_index'] for m in rows}==set(range(198))


def test_empty_pointset_never_outputs_members():
    g,e=fixture();g['point_count']=0
    assert symmetric_members([g],e)==[]


def test_parent_evidence_rejected():
    g,e=fixture();e[g['geometry_key']]['points_sha256']='b'*64
    with pytest.raises(ValueError,match='identity'): symmetric_members([g],e)


def test_small_pointset_is_eligible_and_not_filtered_early():
    g,e=fixture();g['point_count']=20
    assert len(symmetric_members([g],e))==2


def test_incomplete_evidence_cannot_be_zero_fallback():
    g,e=fixture();e[g['geometry_key']]['alpha']['acquisition_complete']=False
    with pytest.raises(ValueError,match='incomplete'): symmetric_members([g],e)
