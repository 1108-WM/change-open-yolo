"""Join sealed source/hypothesis fields with postseal outcomes; no new predictions."""
from collections import Counter, defaultdict

import run_fi1_cross_track_dual as run


def main():
    run.guard(False)
    run.verify_map(run.a.read(run.OUT / 'predictions_frozen.json')['outputs'])
    run.verify_map(run.a.read(run.OUT / 'evaluation_frozen.json')['outputs'])
    sc = run.a.read(run.SOURCE / 'contract.json')
    outcomes = run.a.read(run.OUT / 'evaluation/pairs.json')
    index = {(r['scene'], r['pair_id']): r for r in outcomes}
    witnesses = {(r['scene'], r['observation_id']): r for r in run.a.read(run.OUT / 'evaluation/witness_masks.json')}
    rows, cases, dependence = [], {}, Counter()
    for scene in sc['jobs']:
        d = run.OUT / 'predictions' / scene
        pairs = run.a.read(d / 'pairs.json')
        frames = run.a.read(d / 'pair_frames.json')
        fs = defaultdict(list)
        for f in frames:
            fs[f['pair_id']].append(f)
        dep = run.a.read(d / 'source_dependence.json')
        dependence['edges'] += len(dep['dependence_edges'])
        dependence['families'] += len(set(dep['family_by_raw_track'].values()))
        for p in pairs:
            o = index[scene, p['pair_id']]
            all_joint = {oid for f in fs[p['pair_id']] for oid in f.get('joint_observation_ids', [])}
            cross = [oid for oid in all_joint if witnesses[scene, oid]['cross_known_instances']]
            row = dict(scene=scene, pair_id=p['pair_id'], left_track_ids=p['left_track_ids'],
                right_track_ids=p['right_track_ids'], proposal_reasons=p['proposal_reasons'],
                minimum_distance=p['minimum_distance'], shared_points=p['intersection'],
                union_points=p['union_points'], exclusive_A_points=p['exclusive_A_points'],
                exclusive_B_points=p['exclusive_B_points'], containing_nodes=p['containing_nodes'],
                state=p['state'], reason=p['reason'], emit=p['emit'],
                reliable_views=p['jointly_reliable_frames'],
                missing_reliable_views=p['frame_states'].get('not_jointly_reliably_visible', 0),
                neither_explanation_fits_views=p['frame_states'].get('no_fitting_explanation', 0),
                close_or_competing_views=p['frame_states'].get('competing_or_close', 0),
                merge_winning_views=p['merge_winning_frames'], separate_winning_views=p['separate_winning_frames'],
                separate_plausible_views=p['separate_plausible_frames'],
                joint_source_A=p['joint_source_A'], joint_source_B=p['joint_source_B'],
                joint_dependence_families=p['joint_source_families'],
                only_one_or_no_joint_family=p['joint_only_one_source_family'],
                supported_competitor_nodes=p['supported_competitor_nodes'],
                eligible_partner_counts=p['eligible_partner_counts'],
                GT_posthoc_relation=o['relation'], GT_posthoc_union_purity=o['union_target_purity'],
                GT_posthoc_same_target_pool_delta=o['same_target_pool_delta'],
                GT_posthoc_safe_supply=o['safe_gain02'], GT_posthoc_any_target_gain=o['any_target_gain02'],
                GT_posthoc_joint_masks_with_multiple_known_instances=len(cross),
                joint_mask_count=len(all_joint), duplicate_original_NT=p['duplicate_original_NT'],
                pair_evidence=f'predictions/{scene}/pairs.json:{p["pair_id"]}',
                frame_evidence=f'predictions/{scene}/pair_frames.json:{p["pair_id"]}',
                pointset=f'predictions/{scene}/data.npz:{p["pair_id"]}_union')
            rows.append(row)
            if p['state'] == 'accept' or o['safe_gain02'] or (scene == 'scene0369_00' and set(p['left_track_ids']+p['right_track_ids']) == {33, 35}):
                cases[scene + '/' + p['pair_id']] = dict(prediction=p, outcome=o,
                    reliable_frames=[f for f in fs[p['pair_id']] if f['state'] != 'not_jointly_reliably_visible'])
    run.csv_seal(run.OUT / 'pair_evidence.csv', rows)
    run.a.seal(run.OUT / 'postseal_cases.json', cases)
    safe = [r for r in outcomes if r['safe_gain02']]
    summary = dict(complete=True, code_sha256=run.a.sha(__file__), used_for_predictions=False,
        paired_evidence_rows=len(rows), dependence=dependence,
        safe_supply_pairs=len(safe), safe_supply_targets=len({(r['scene'], r['same_target']) for r in safe}),
        safe_supply_scenes=len({r['scene'] for r in safe}),
        safe_supply_reasons=dict(Counter(r['reason'] for r in safe)),
        zero_overlap_safe_supply_pairs=sum(r['shared_points'] == 0 and r['GT_posthoc_safe_supply'] for r in rows),
        joint_masks_cross_known_pair_associations=sum(r['GT_posthoc_joint_masks_with_multiple_known_instances'] for r in rows),
        cases=list(cases), note='GT marks explain all sealed pairs; no GT filtering or new output branch.')
    # Native dependence is reported from actual proposed pair containment IDs.
    containing_native = 0
    for scene in sc['jobs']:
        nodes = run.a.read(run.OUT / 'predictions' / scene / 'nodes.json')
        for p in run.a.read(run.OUT / 'predictions' / scene / 'pairs.json'):
            containing_native += any('native' in nodes[n]['sources'] for n in p['containing_nodes'])
    summary['no_N_containing_proposal_pairs'] = len(rows)-containing_native
    run.a.seal(run.OUT / 'supplementary_summary.json', summary)
    run.a.seal(run.OUT / 'supplementary_frozen.json', dict(complete=True, GT_posthoc_only=True,
        predictions_changed=False, outputs={str(run.OUT / name): run.a.sha(run.OUT / name)
            for name in ('pair_evidence.csv', 'postseal_cases.json', 'supplementary_summary.json')}))
    print('POSTSEAL TABLES', summary, flush=True)


if __name__ == '__main__':
    main()
