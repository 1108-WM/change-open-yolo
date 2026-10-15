"""Fixed pairwise instance hypotheses. Counts are evidence, never identity truth."""
from collections import Counter

import numpy as np


PARAMETERS = dict(spatial_distance=.15, proposal_cover=.30, containment_cover=.50,
                  min_exclusive_points=20, min_visible_points=20, min_visible_ratio=.10,
                  coverage=.60, selective_cross_coverage=.10, fit=.60, margin=.15,
                  min_frames=2, competitor_side_cover=.80, competitor_other_cover=.10,
                  competitor_outside_fraction=.20, source_duplicate_iou=.80,
                  source_duplicate_frames=2)


def reliable(count, size):
    return count >= PARAMETERS['min_visible_points'] and count / max(1, size) >= PARAMETERS['min_visible_ratio']


def frame_hypotheses(a, b, inter, va, vb, vi, mask_sizes, global_ids, raw_owners,
                     lineage_a, lineage_b, source_families=None):
    """Score one-mask union versus two distinct selective-mask explanations."""
    ea, eb = a - inter, b - inter
    vea, veb = va - vi, vb - vi
    ca, cb = ea / max(1, vea), eb / max(1, veb)
    union_inside, union_visible = a + b - inter, va + vb - vi
    merged_fit = union_inside / np.maximum(1, mask_sizes + union_visible - union_inside)
    fit_a = a / np.maximum(1, mask_sizes + va - a)
    fit_b = b / np.maximum(1, mask_sizes + vb - b)
    joint = (ca >= PARAMETERS['coverage']) & (cb >= PARAMETERS['coverage'])
    sa = (ca >= PARAMETERS['coverage']) & (cb <= PARAMETERS['selective_cross_coverage'])
    sb = (cb >= PARAMETERS['coverage']) & (ca <= PARAMETERS['selective_cross_coverage'])

    def best(values, eligible):
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            return 0., None
        # Input rows are globally observation-ID sorted; ties are evidence locators only.
        k = int(candidates[np.argmax(values[candidates])])
        return float(values[k]), int(global_ids[k])

    merge, mid = best(merged_fit, joint)
    score_a, aid = best(fit_a, sa)
    score_b, bid = best(fit_b, sb)
    separate = (score_a + score_b)/2 if aid is not None and bid is not None else 0.
    assert aid is None or bid is None or aid != bid
    merge_plausible = merge >= PARAMETERS['fit']
    separate_plausible = separate >= PARAMETERS['fit'] and min(score_a, score_b) >= PARAMETERS['fit']
    if merge_plausible and merge - separate >= PARAMETERS['margin']:
        state = 'merge_wins'
    elif separate_plausible and separate - merge >= PARAMETERS['margin']:
        state = 'separate_wins'
    elif merge_plausible or separate_plausible:
        state = 'competing_or_close'
    else:
        state = 'no_fitting_explanation'
    qualified_joint = joint & (merged_fit >= PARAMETERS['fit'])
    owner_set = {int(raw_owners[k]) for k in np.flatnonzero(qualified_joint) if raw_owners[k] >= 0}
    if source_families is None:
        source_families = raw_owners
    dependency_set = {int(source_families[k]) for k in np.flatnonzero(qualified_joint) if source_families[k] >= 0}
    return dict(state=state, merge_fit=merge, separate_fit=separate,
        selective_A_fit=score_a, selective_B_fit=score_b, fit_margin=merge-separate,
        merge_observation_id=mid, separate_observation_ids=[aid, bid],
        merge_plausible=merge_plausible, separate_plausible=separate_plausible,
        joint_observation_ids=global_ids[qualified_joint].tolist(),
        joint_raw_track_ids=sorted(owner_set),
        joint_dependency_family_ids=sorted(dependency_set),
        joint_from_A=sorted(owner_set & set(lineage_a)),
        joint_from_B=sorted(owner_set & set(lineage_b)),
        selective_A_observation_ids=global_ids[sa].tolist(),
        selective_B_observation_ids=global_ids[sb].tolist(),
        masks_touching_either=int(((a > 0) | (b > 0)).sum()))


def decide(frames, exclusive_sizes, competitors):
    counts = Counter(f['state'] for f in frames)
    checked = [f for f in frames if f['state'] != 'not_jointly_reliably_visible']
    wins = [f for f in checked if f['state'] == 'merge_wins']
    source_a = sorted({x for f in wins for x in f['joint_from_A']})
    source_b = sorted({x for f in wins for x in f['joint_from_B']})
    families = sorted({x for f in wins for x in f['joint_dependency_family_ids']})
    plausible_separate = sum(f['separate_plausible'] for f in checked)
    conflicts = [c['node'] for c in competitors if c['supported_conflict']]
    diagnostics = dict(frame_states=dict(counts), jointly_reliable_frames=len(checked),
        merge_winning_frames=len(wins), separate_winning_frames=counts['separate_wins'],
        separate_plausible_frames=plausible_separate, joint_source_A=source_a,
        joint_source_B=source_b, joint_source_families=families,
        joint_only_one_source_family=len(families) <= 1,
        supported_competitor_nodes=conflicts)
    if min(exclusive_sizes) < PARAMETERS['min_exclusive_points']:
        state, reason = 'unknown', 'insufficient_distinct_geometry'
    elif counts['separate_wins'] >= PARAMETERS['min_frames'] and not wins:
        state, reason = 'reject', 'repeated_better_separate_explanation'
    elif not checked:
        state, reason = 'unknown', 'no_jointly_reliable_views'
    elif conflicts:
        state, reason = 'unknown', 'supported_third_instance_explanation'
    elif plausible_separate:
        state, reason = 'unknown', 'separate_explanation_remains_plausible'
    elif len(wins) < PARAMETERS['min_frames']:
        state, reason = 'unknown', 'insufficient_merge_discrimination'
    elif not source_a or not source_b or len(families) < 2:
        state, reason = 'unknown', 'joint_support_concentrated_in_one_lineage'
    else:
        state, reason = 'eligible', 'merge_explanation_wins_with_cross_lineage_checks'
    return dict(state=state, reason=reason, **diagnostics)


def resolve_partner_competition(rows):
    """Only mutually unique eligible partners combine; no recursive matching/tree."""
    degree = Counter(n for r in rows if r['state'] == 'eligible' for n in (r['left'], r['right']))
    for r in rows:
        r['eligible_partner_counts'] = [degree[r['left']], degree[r['right']]]
        if r['state'] == 'eligible':
            if degree[r['left']] == degree[r['right']] == 1:
                r.update(state='accept', reason='unique_pair_after_hypothesis_competition')
            else:
                r.update(state='unknown', reason='multiple_competing_pair_explanations')
    return rows
