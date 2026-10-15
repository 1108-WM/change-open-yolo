"""Postseal strata and named evidence schema; no new scores or decisions."""
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import run_fi1_pair_relation as r


def main():
    r.guard(False)
    r.verify(r.a.read(r.OUT/'predictions_frozen.json')['outputs'])
    result = r.a.read(r.OUT/'results.json')['results']
    rows = r.a.read(r.OUT/'feature_rows.json')
    source_cases = r.a.read(r.SOURCE/'postseal_cases.json')
    geometry = np.load(r.OUT/'features.npz')['geometry']
    strata = {}
    for arm in r.ARMS:
        joined = r.a.read(r.OUT/'evaluation'/f'{arm}_pairs.json')
        assert [(x['scene'], x['pair_id']) for x in rows] == [(x['scene'], x['pair_id']) for x in joined]
        sets = dict(all=np.ones(len(rows), bool), zero_overlap=geometry[:, 4] == 0,
                    with_overlap=geometry[:, 4] > 0,
                    valuable_supply=np.array([q['safe_gain02'] for q in joined]))
        strata[arm] = {}
        for name, subset in sets.items():
            selected = [q for q, ok in zip(joined, subset) if ok]
            clear = [q for q in selected if q['GT_relation_strict'] in ('same', 'different')]
            y = [q['GT_relation_strict'] == 'same' for q in clear]
            score = [q['score'] for q in clear]
            strata[arm][name] = dict(pairs=len(selected), relations=dict(Counter(q['GT_relation_strict'] for q in selected)),
                accepted=sum(q['state'] == 'accept' for q in selected), emitted=sum(q['emit'] for q in selected),
                clear_ROC_AUC=float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
                clear_pair_average_precision=float(average_precision_score(y, score)) if any(y) else None)
    opportunities = []
    for g, v in zip(result['geometry']['valuable_supply'], result['geometry_rgbd']['valuable_supply']):
        assert (g['scene'], g['pair_id']) == (v['scene'], v['pair_id'])
        key = g['scene']+'/'+g['pair_id']
        row = next(x for x in rows if (x['scene'], x['pair_id']) == (g['scene'], g['pair_id']))
        oldframes = source_cases[key]['reliable_frames']
        old_joint = sum(bool(f['joint_observation_ids']) for f in oldframes)
        opportunities.append(dict(scene=g['scene'], pair_id=g['pair_id'],
            geometry_score=g['score'], rgbd_score=v['score'], state=v['state'],
            same_target_pool_delta=v['delta'], strict_relation=v['strict_relation'],
            contacts=row['contacts'], reliable_common_frames=row['reliable_common_frames'],
            selected_frames=row['selected_frame_ids'], old_joint_mask_frames=old_joint))
    base = ['rgb_mean_r', 'rgb_mean_g', 'rgb_mean_b', 'rgb_std_r', 'rgb_std_g', 'rgb_std_b']
    base += [f'rgb_hist_{channel}_bin{j}' for channel in 'rgb' for j in range(8)]
    base += ['abs_normal_mean_'+a for a in 'xyz']+['normal_std_'+a for a in 'xyz']
    geometry_names = ['log_small_size', 'log_large_size', 'log_small_exclusive_size', 'log_large_exclusive_size',
        'intersection_over_union', 'center_distance_m']
    geometry_names += [f'{what}_sorted{i}' for what in ('mean_extents', 'absolute_extent_difference', 'union_extents') for i in range(3)]
    geometry_names += ['nearest_distance_m', 'bidir_distance_q10', 'bidir_distance_q50', 'bidir_distance_q90',
        'bidir_fraction_within_05m', 'bidir_fraction_within_10m', 'reliable_common_frame_count', 'selected_view_count']
    path = ['path_mean_median_jump', 'path_mean_q90_jump', 'path_mean_max_jump', 'path_mean_missing',
            'path_max_missing', 'path_median_length_px', 'path_max_length_px', 'path_count_over8']
    path += ['same_frame_endpoint_mean_'+s for s in base[:6]]+['same_frame_endpoint_absdiff_'+s for s in base[:6]]
    path += ['visible_contact_count', 'contact_known_depth_fraction', 'contact_depth_jump_q10', 'contact_depth_jump_q50', 'contact_depth_jump_q90']
    visual_names = ['endpoint_mean_'+s for s in base]+['endpoint_absdiff_'+s for s in base]
    visual_names += ['has_10cm_contact', 'contact_color_q10', 'contact_color_q50', 'contact_color_q90',
        'contact_normal_valid_fraction', 'contact_normal_difference_q10', 'contact_normal_difference_q50', 'contact_normal_difference_q90', 'measured_view_count']
    visual_names += ['view_mean_'+s for s in path]+['view_std_'+s for s in path]
    assert len(geometry_names) == 23 and len(visual_names) == 131 and len(path) == 25
    r.a.seal(r.OUT/'feature_schema.json', dict(geometry=geometry_names, rgbd=visual_names,
        view_values=path, rgb_range=[0, 1], length_units='metres except explicit pixels',
        missing='NaN for unknown measurement, explicit count/presence alongside. Zero complete depth jumps are distinct from unavailable paths.',
        no_masks_or_semantic_labels=True, schema_written_postseal=True, predictions_changed=False))
    r.a.seal(r.OUT/'postseal_diagnostic.json', dict(code_sha256=r.a.sha(__file__), strata=strata,
        opportunities=opportunities, old_joint_mask_frame_census=dict(Counter(x['old_joint_mask_frames'] for x in opportunities)),
        all_opportunities_eligible=all(x['contacts'] or x['selected_frames'] for x in opportunities),
        all_opportunities_measured_views=sum(bool(x['selected_frames']) for x in opportunities),
        metrics_are_not_instance_AP=True, used_for_fitting_or_selection=False, predictions_changed=False))
    print(r.a.plain(dict(strata=strata, opportunities=opportunities)), flush=True)


if __name__ == '__main__':
    main()
