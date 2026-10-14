import copy
import pytest
from tools.run_fi1_union_refinement_ablation_development200 import candidate_pool, refinement_links, effects
from tools.fi1_r0_r1_protocol import original_aliases


def geometry(key, sources, parent=None):
    return dict(geometry_key=key, legacy_geometry_keys=['legacy_' + key], point_count=120,
        legacy_members=[dict(candidate_source=s, candidate_id=i, frozen_class_index=0)
                        for i, s in enumerate(sources) if s != 'refined_union'],
        dv3_plans=[dict(append_only=True, candidate_source='refined_union', plan_key='ref_' + key,
                       frozen_class_index=0, original_union_geometry_key=parent)] if 'refined_union' in sources else [])


def source_map(pool):
    return {g['geometry_key']: {a['source'] for a in original_aliases(g)} for g in pool}


def test_rejecting_parent_does_not_delete_refinement_or_shared_native():
    gs = [geometry('u', ['pair_union']), geometry('up', ['refined_union'], 'legacy_u'),
          geometry('shared', ['native', 'pair_union', 'refined_union'], 'legacy_u')]
    saved = copy.deepcopy(gs)
    pool = source_map(candidate_pool(gs, 'screened_Full', set()))
    assert pool == {'up': {'refined_union'}, 'shared': {'native', 'refined_union'}}
    assert source_map(candidate_pool(gs, 'screened_NTU', set())) == {'shared': {'native'}}
    assert gs == saved
    links = refinement_links(gs, set())
    assert len(links) == 2 and all(not r['parent_U_admitted'] for r in links)
    assert all(r['parent_U'] == 'u' for r in links)
    assert [r['independent_geometry'] for r in links] == [True, False]


def test_refinement_only_output_preserves_internal_parent_identity():
    gs = [geometry('u', ['pair_union']), geometry('up', ['refined_union'], 'legacy_u')]
    assert source_map(candidate_pool(gs, 'NTUp', {'u'})) == {'up': {'refined_union'}}
    assert refinement_links(gs, {'u'})[0]['parent_U_admitted']
    assert source_map(candidate_pool(gs, 'NTU', set())) == {'u': {'pair_union'}}


def test_shared_mask_not_removed_with_union_alias():
    gs = [geometry('g', ['native', 'track', 'pair_union'])]
    assert source_map(candidate_pool(gs, 'NT', set())) == {'g': {'native', 'track'}}
    assert source_map(candidate_pool(gs, 'screened_Full', set())) == {'g': {'native', 'track'}}
    assert source_map(candidate_pool(gs, 'Full', set()))['g'] == {'native', 'track', 'pair_union'}


def test_invalid_gate_and_ambiguous_parent_fail_closed():
    with pytest.raises(ValueError): candidate_pool([geometry('u', ['pair_union'])], 'screened_Full', {'missing'})
    a, b = geometry('a', ['pair_union']), geometry('b', ['pair_union'])
    b['legacy_geometry_keys'] = a['legacy_geometry_keys']
    with pytest.raises(ValueError): refinement_links([a, b], set())
    with pytest.raises(KeyError): refinement_links([geometry('up', ['refined_union'], 'missing')], set())


def test_effects_are_source_comparisons_in_percentage_points():
    ap = {a: dict(ap=i/100, ap50=i/100, ap25=i/100)
          for i, a in enumerate(('NT', 'NTU', 'NTUp', 'Full', 'screened_NTU', 'screened_Full'))}
    e = effects(ap)
    assert e['Full_minus_NT']['ap'] == 3
    assert abs(e['screened_Full_minus_screened_NTU']['ap'] - 1) < 1e-12
