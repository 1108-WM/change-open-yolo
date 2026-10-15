#!/usr/bin/env python3
"""Postseal GT interpretation of all provenance rows; no inferred output selection."""
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fi1_t_source_chain as audit
import evaluate_fi1_nt_region_identity as met
from analyze_fi1_local_block_attribution_selection import census
import run_fi1_nt_region_identity as diag

OUT = audit.OUT
MODES = ('lineage_observed', 'd1_initial', 'merge_input', 'other_tracks', 'all_raw')
STAGES = MODES + ('d1_kept', 'merge_removed', 'post_hierarchy', 'kept_unassigned')


def csv_seal(path, rows):
    with path.open('x', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def summarize(rows):
    stable = [r for r in rows if r['stable']]
    added = sum(r['added'] for r in stable)
    return dict(parents=len(rows), stable_parents=len(stable), actions=sum(r['changed'] for r in rows),
        stable_actions=sum(r['changed'] for r in stable),
        unresolved_actions=sum(r['changed'] and not r['stable'] for r in rows),
        stable_added=added, target_added=sum(r.get('added_target', 0) for r in stable),
        other_added=sum(r.get('added_other', 0) for r in stable),
        unknown_added=sum(r['unknown_edits'] for r in stable),
        precision=sum(r.get('added_target', 0) for r in stable)/added if added else None,
        same_parent_gain02=sum(r.get('pool_delta', -1) >= .02-1e-12 for r in stable),
        safe_gain02=sum(r.get('pool_delta', -1) >= .02-1e-12 and r['added'] > 0
                        and r.get('added_target', 0)/r['added'] >= .9 for r in stable),
        same_parent_gain02_scenes=sorted({r['scene'] for r in stable if r.get('pool_delta', -1) >= .02-1e-12}),
        good_copy_loss_gt02=sum(r.get('good_loss', 0) > .02+1e-12 for r in stable),
        degraded_ge02=sum(r.get('parent_delta', 0) <= -.02+1e-12 for r in stable),
        parent_delta_sum=sum(r.get('parent_delta', 0) for r in stable),
        positive_pool_delta_sum=sum(max(0, r.get('pool_delta', 0)) for r in stable))


def main():
    frozen = audit.read(OUT / 'source_evidence_frozen.json')
    assert frozen['complete'] and frozen['GT_used'] is False
    for p, h in frozen['outputs'].items():
        assert audit.sha(p) == h, p
    c = audit.read(OUT / 'contract.json')
    old = audit.read(audit.OLD / 'contract.json')
    rows, blocks, observations, pairs, scopes = [], [], [], [], []
    gtinputs, outputs = {}, {}
    for i, (s, spec) in enumerate(c['jobs'].items()):
        print('postseal_GT', i+1, 52, s, flush=True)
        j = spec['job']
        assert audit.sha(j['GT']) == old['GT_inputs'][j['GT']]
        gtinputs[j['GT']] = old['GT_inputs'][j['GT']]
        gt = np.loadtxt(j['GT'], dtype=np.int64)
        ids, sizes = met.valid_targets(gt, old['class_ids'])
        _, nt = diag.load_nt(j)
        baseline = met.pool_base(nt, gt, ids, sizes)
        evidence = OUT / 'evidence' / s
        br = audit.read(evidence / 'blocks.json')
        obs = audit.read(evidence / 'block_observations.json')
        relations = audit.read(evidence / 'block_merge_relations.json')
        parentrefs = audit.read(evidence / 'parents.json')
        scopes_before = {r['candidate_id']: r for r in audit.read(audit.OLD / 'evaluation' / s / 'unions.json') if r['arm']=='all_C'}
        obindex = defaultdict(list)
        for r in obs:
            obindex[r['block_id']].append(r)
        relindex = defaultdict(list)
        for r in relations:
            relindex[r['block_id']].append(r)
        scene_blocks, scene_rows, scene_obs, scene_pairs = [], [], [], []
        obs_cache = {}
        with np.load(evidence / 'pointsets.npz') as z:
            for p in parentrefs:
                cid = p['candidate_id']
                parent = z[cid + '_parent']
                meta = met.identity(parent, gt, ids, sizes)
                assert meta['scope'] == scopes_before[cid]['scope']
                target = meta['target'] if meta['stable'] else None
                scopes.append(dict(scene=s, candidate_id=cid, **meta))
                bs = [b for b in br if b['candidate_id'] == cid]
                editable = audit.union(z[b['block_id'] + '_block'] for b in bs)
                full_oracle = np.union1d(parent, editable[gt[editable] == target]) if target else parent
                oracle, _ = met.outcome(parent, full_oracle, gt, ids, sizes, meta, baseline)
                opportunity = oracle.get('pool_delta', -1) >= .02-1e-12
                for mode in MODES:
                    pp = z[cid + '_' + mode + '_union']
                    add = np.setdiff1d(pp, parent)
                    outcome, _ = met.outcome(parent, pp, gt, ids, sizes, meta, baseline)
                    scene_rows.append(dict(scene=s, candidate_id=cid, mode=mode,
                        construction='fixed_raw_restoration_diagnostic', **outcome,
                        original_editable_point_oracle=oracle.get('pool_delta'),
                        original_opportunity=opportunity))
                    upper = np.union1d(parent, add[gt[add] == target]) if target else parent
                    upperrow, _ = met.outcome(parent, upper, gt, ids, sizes, meta, baseline)
                    scene_rows.append(dict(scene=s, candidate_id=cid, mode=mode + '_target_oracle',
                        construction='posthoc_target_points_upper_bound_not_method', **upperrow,
                        original_editable_point_oracle=oracle.get('pool_delta'),
                        original_opportunity=opportunity))
                for b in bs:
                    bid = b['block_id']
                    whole = z[bid + '_block']
                    bc = census(whole, gt, target)
                    stages = {name: census(z[bid + '_' + name], gt, target) for name in STAGES}
                    # Exact differences identify where specific editable points were lost.
                    specific = dict(
                        raw_not_initial=np.setdiff1d(z[bid+'_lineage_observed'], z[bid+'_d1_initial']),
                        D1_removed=np.setdiff1d(z[bid+'_d1_initial'], z[bid+'_d1_kept']),
                        D1_kept_absent_final=z[bid+'_d1_kept'],
                        hierarchy_only=np.setdiff1d(z[bid+'_all_raw'], z[bid+'_post_hierarchy']),
                        outside_lineage=np.setdiff1d(z[bid+'_all_raw'], z[bid+'_lineage_observed']))
                    loss_GT = {name:census(pp, gt, target) for name,pp in specific.items()}
                    row = dict(b, stable=meta['stable'], parent_scope=meta['scope'], target=target,
                        block_GT=bc, stage_GT=stages, exact_loss_GT=loss_GT,
                        parent_editable_oracle_gain02=opportunity,
                        opportunity_contribution=bool(opportunity and bc['target']),
                        accepted_NT_block=b['prior_ownership_state']=='accept')
                    scene_blocks.append(row)
                    for o in obindex[bid]:
                        path = o['point_indices_path']
                        if path not in obs_cache:
                            assert audit.sha(path) == o['point_file_sha256']
                            obs_cache[path] = audit.points(path)
                        pp = obs_cache[path]
                        bp = audit.hit(whole, pp)
                        assert len(bp) == o['block_hits']
                        oc = census(pp, gt, target)
                        scene_obs.append(dict(o, stable=meta['stable'], parent_scope=meta['scope'],
                            target=target, full_observation_GT=oc, block_hit_GT=census(bp,gt,target),
                            cross_known_instances=oc['known_instances']>1,
                            simultaneous_parent_and_other=bool(target and o['parent_hits'] and (gt[bp]>=1000).any()
                                                              and ((gt[bp]>=1000)&(gt[bp]!=target)).any())))
                    for r in relindex[bid]:
                        scene_pairs.append(dict(r, stable=meta['stable'], parent_scope=meta['scope'],
                                                block_GT=bc, opportunity_contribution=row['opportunity_contribution']))
        directory = OUT / 'evaluation' / s
        directory.mkdir(parents=True)
        for name, value in [('blocks', scene_blocks), ('unions', scene_rows),
                            ('observations', scene_obs), ('merge_relations', scene_pairs)]:
            audit.seal(directory / (name + '.json'), value)
            outputs[str(directory / (name + '.json'))] = audit.sha(directory / (name + '.json'))
        blocks += scene_blocks
        rows += scene_rows
        observations += scene_obs
        pairs += scene_pairs
    assert len(blocks) == 1308 and len(scopes) == 207
    assert sum(r['stable'] for r in scopes) == 143
    grouped = defaultdict(list)
    for b in blocks:
        grouped[(b['scene'], b['candidate_id'])].append(b)
    table = []
    for b in blocks:
        row = dict(scene=b['scene'], cohort=b['cohort'], candidate_id=b['candidate_id'],
            block_id=b['block_id'], track_id=b['track_id'], points=b['points'],
            parent_scope=b['parent_scope'], target=b['target'], ownership=b['block_GT']['ownership'],
            target_points=b['block_GT']['target'], other_points=b['block_GT']['other'],
            unknown_points=b['block_GT']['unknown'], prior_state=b['prior_ownership_state'],
            point_oracle_opportunity=b['opportunity_contribution'],
            raw_observation_ids=';'.join(map(str,b['source_observation_ids'])),
            lineage_observation_ids=';'.join(map(str,b['lineage_hit_observation_ids'])),
            other_track_observation_ids=';'.join(map(str,b['other_track_hit_observation_ids'])),
            raw_hits=b['stages']['all_raw'], lineage_raw_hits=b['stages']['lineage_observed'],
            D1_initial=b['stages']['d1_initial'], D1_kept=b['stages']['d1_kept'],
            exact_D1_removed=b['exact_loss_GT']['D1_removed']['points'],
            merge_input=b['stages']['merge_input'], merge_removed=b['stages']['merge_removed'], final_T=b['stages']['final_T'],
            other_tracks=b['stages']['other_tracks'], max_other_merge_iou=b['max_recorded_other_point_iou'],
            eligible_other_merge_pairs=b['recorded_eligible_other_pairs'],
            association_rejection_reason=b['association_rejection_reason'],
            removed_SP_evidence=';'.join((str(r['stage'])+':'+str(r['superpoint_id'])+':'+','.join(r['failed'])) for r in b['removed_superpoint_evidence']),
            source_evidence=f'evidence/{b["scene"]}/block_observations.json',
            merge_evidence=f'evidence/{b["scene"]}/block_merge_relations.json')
        table.append(row)
    csv_seal(OUT / 'block_source_evidence.csv', table)
    audit.seal(OUT / 'evaluation' / 'parent_scopes.json', scopes)
    audit.seal(OUT / 'evaluation' / 'all_unions.json', rows)
    audit.seal(OUT / 'evaluation' / 'opportunity_blocks.json', [b for b in blocks if b['opportunity_contribution']])
    metrics = {mode:summarize([r for r in rows if r['mode']==mode])
               for mode in MODES + tuple(m+'_target_oracle' for m in MODES)}
    strata = {}
    for name, bs in [('all',blocks), ('stable_all',[b for b in blocks if b['stable']]),
                     ('opportunity_contribution',[b for b in blocks if b['opportunity_contribution']]),
                     ('stable_negatives',[b for b in blocks if b['stable'] and b['block_GT']['other']]),
                     ('accepted_other',[b for b in blocks if b['stable'] and b['accepted_NT_block'] and b['block_GT']['other']]),
                     ('unresolved_parent',[b for b in blocks if not b['stable']])]:
        strata[name] = dict(blocks=len(bs), scenes=len({b['scene'] for b in bs}),
            ownership=dict(Counter(b['block_GT']['ownership'] for b in bs)),
            flags={k:sum(b['flags'][k] for b in bs) for k in blocks[0]['flags']},
            stages={stage:dict(points=sum(b['stage_GT'][stage]['points'] for b in bs),
                target=sum(b['stage_GT'][stage]['target'] or 0 for b in bs) if all(b['stable'] for b in bs) else None,
                other=sum(b['stage_GT'][stage]['other'] or 0 for b in bs) if all(b['stable'] for b in bs) else None,
                unknown=sum(b['stage_GT'][stage]['unknown'] for b in bs)) for stage in STAGES},
            loss={stage:dict(points=sum(b['exact_loss_GT'][stage]['points'] for b in bs),
                target=sum(b['exact_loss_GT'][stage]['target'] or 0 for b in bs) if all(b['stable'] for b in bs) else None,
                other=sum(b['exact_loss_GT'][stage]['other'] or 0 for b in bs) if all(b['stable'] for b in bs) else None)
                  for stage in blocks[0]['exact_loss_GT']})
    summary = dict(counts=dict(scenes=52, parents=len(scopes), blocks=len(blocks),
            stable_parents=sum(r['stable'] for r in scopes), block_observation_links=len(observations),
            block_merge_relations=len(pairs)),
        parent_scopes=dict(Counter(r['scope'] for r in scopes)),
        modes=metrics, strata=strata,
        opportunities=[r for r in rows if r['original_opportunity']],
        cross_object_observation_links=sum(o['cross_known_instances'] for o in observations),
        simultaneous_parent_and_other_links=sum(o['simultaneous_parent_and_other'] for o in observations),
        evidence_seal_sha256=audit.sha(OUT / 'source_evidence_frozen.json'),
        cautions=['All modes are attribution-free supply diagnostics, not safe repair methods.',
                  'GT used only after source_evidence_frozen.json; never to select source observations.',
                  'Overlapping block/parent counts are incidences, not unique scene points.',
                  'Unresolved parent identity keeps target/other null.',
                  'No inference from aggregate removed counts; loss comes from exact stage SP/point differences.'])
    audit.seal(OUT / 'evaluation_summary.json', summary)
    outputs.update({str(p):audit.sha(p) for p in [OUT/'block_source_evidence.csv', OUT/'evaluation_summary.json',
        OUT/'evaluation/parent_scopes.json', OUT/'evaluation/all_unions.json', OUT/'evaluation/opportunity_blocks.json']})
    audit.seal(OUT / 'evaluation_frozen.json', dict(complete=True, GT_used=True, GT_inputs=gtinputs,
        evidence_seal_sha256=audit.sha(OUT/'source_evidence_frozen.json'), code_sha256=audit.sha(__file__), outputs=outputs))
    print('GT_interpretation_sealed', {k:metrics[k] for k in MODES}, flush=True)


if __name__ == '__main__':
    main()
