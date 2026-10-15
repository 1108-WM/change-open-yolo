#!/usr/bin/env python3
"""GT-postseal factorial diagnosis. No labels feed back into sealed references."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fi1_local_block_attribution_selection as run
import evaluate_fi1_nt_region_identity as met
D = run.diag
OUT = run.OUT
MODES = ('W', 'C', 'oracle_pure_W', 'oracle_pure_C', 'oracle_target_points')


def census(points, gt, target):
    labels, nums = np.unique(gt[points], return_counts=True)
    known = int(nums[labels >= 1000].sum())
    target_count = int((gt[points] == target).sum()) if target is not None else None
    other = known - target_count if target is not None else None
    if target is None:
        ownership = 'unresolved_parent'
    elif not len(points):
        ownership = 'empty'
    elif known == 0:
        ownership = 'unknown_only'
    elif target_count and other:
        ownership = 'mixed_known'
    elif target_count:
        ownership = 'target_only' if known == len(points) else 'target_unknown'
    else:
        ownership = 'other_only' if known == len(points) else 'other_unknown'
    return dict(points=len(points), target=target_count, other=other,
                annotated=known, unknown=len(points)-known,
                known_instances=int((labels >= 1000).sum()), ownership=ownership,
                distribution={str(g): int(n) for g, n in zip(labels, nums)})


def union(parts):
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)


def safe_gain(row):
    """Posthoc diagnostic, never a deployed filter; unknown stays in denominator."""
    return bool(row.get('pool_delta', -1) >= .02-1e-12 and row['added'] and
                row.get('added_target', 0)/row['added'] >= .9)


def evaluate():
    contract = D.read(OUT/'contract.json')
    frozen = D.read(OUT/'predictions_frozen.json')
    assert frozen['complete'] and frozen['GT_used'] is False
    D.verify(contract['code']); D.verify(frozen['outputs']); D.verify(contract['GT_inputs'])
    all_blocks, all_rows, all_pool, all_any_pool, opportunities, outputs = [], [], [], [], [], {}
    scopes = []
    for si, (scene, job) in enumerate(contract['jobs'].items()):
        run.stage('postseal_GT_diagnosis', scene=scene, completed=si, total=52)
        gt = np.loadtxt(job['GT'], dtype=np.int64)
        ids, sizes = met.valid_targets(gt, contract['class_ids'])
        _, nt = D.load_nt(job)
        baseline = met.pool_base(nt, gt, ids, sizes)
        parents = D.read(run.SOURCE/'relations'/scene/'parents.json')
        block_refs = D.read(OUT/'predictions'/scene/'blocks.json')
        old = {r['candidate_id']: r for r in D.read(run.SOURCE/'evaluation'/scene/'posthoc_supply.json')}
        same_best = {m: baseline.copy() for m in MODES}
        any_best = {m: baseline.copy() for m in MODES}
        scene_blocks, scene_rows, scene_opps, arrays = [], [], [], {}
        with np.load(OUT/'predictions'/scene/'data.npz') as z, np.load(Path(job['asset_root'])/'structure'/scene/'data.npz') as st:
            for p in parents:
                cid = p['candidate_id']; parent = st[cid+'_parent']
                identity = met.identity(parent, gt, ids, sizes)
                target = identity['target'] if identity['stable'] else None
                scopes.append(dict(scene=scene, cohort=job['cohort'], candidate_id=cid, **identity,
                                   known_instance_count=sum(int(g)>=1000 for g in identity['distribution'])))
                refs = [b for b in block_refs if b['candidate_id'] == cid]
                pure_w, pure_c, true_parts = [], [], []
                for b in refs:
                    block = z[b['block_id']+'_block']; clip = z[b['block_id']+'_clipped_proposal']
                    bc, cc = census(block, gt, target), census(clip, gt, target)
                    dropped = census(np.setdiff1d(block, clip), gt, target)
                    r = dict(b, stable=identity['stable'], parent_scope=identity['scope'], target=target,
                             block_GT=bc, clipped_GT=cc, dropped_GT=dropped,
                             parent_identity_distribution=identity['distribution'])
                    for arm, pts in [('W', block), ('C', clip)]:
                        val, _ = met.outcome(parent, np.union1d(parent, pts), gt, ids, sizes, identity, baseline)
                        r[arm+'_single'] = val
                    if target is not None:
                        true_parts.append(block[gt[block] == target])
                        if bc['ownership'] == 'target_only':
                            pure_w.append(block); pure_c.append(clip)
                    scene_blocks.append(r)
                additions = dict(W=z[cid+'_W_added'], C=z[cid+'_C_added'],
                                 oracle_pure_W=union(pure_w), oracle_pure_C=union(pure_c),
                                 oracle_target_points=union(true_parts))
                by_mode = {}
                for mode, added in additions.items():
                    points = np.union1d(parent, added)
                    arrays[cid+'_'+mode] = points
                    row, scores = met.outcome(parent, points, gt, ids, sizes, identity, baseline)
                    row.update(scene=scene, cohort=job['cohort'], candidate_id=cid, mode=mode,
                               output_kind='GT_oracle' if mode.startswith('oracle') else 'unattributed_reference',
                               safe_gain02=safe_gain(row), added_GT=census(added, gt, target))
                    row['same_parent_gain02'] = row.get('pool_delta', -1) >= .02-1e-12
                    scene_rows.append(row); by_mode[mode] = row
                    any_best[mode] = np.maximum(any_best[mode], scores)
                    if identity['stable']:
                        ix = identity['index']; same_best[mode][ix] = max(same_best[mode][ix], scores[ix])
                # Leave-one-block-out diagnoses contribution; never chooses an output.
                for b in [r for r in scene_blocks if r['candidate_id'] == cid]:
                    block = z[b['block_id']+'_block']
                    if target is None:
                        b.update(point_oracle_contribution=None, pure_oracle_contribution=None)
                        continue
                    ix = identity['index']
                    for mode, key, removed in [
                        ('oracle_target_points', 'point_oracle_contribution', block[gt[block] == target]),
                        ('oracle_pure_W', 'pure_oracle_contribution', block if b['block_GT']['ownership']=='target_only' else block[:0])]:
                        without = np.setdiff1d(additions[mode], removed)
                        score = met.ious(np.union1d(parent, without), gt, ids, sizes)[ix]
                        b[key] = float(by_mode[mode]['repaired_iou']-score)
                    b['valuable_point_contributor'] = bool(by_mode['oracle_target_points']['same_parent_gain02'] and b['point_oracle_contribution'] > 1e-12)
                    b['valuable_pure_contributor'] = bool(by_mode['oracle_pure_W']['same_parent_gain02'] and b['pure_oracle_contribution'] > 1e-12)
                if identity['stable']:
                    prev = old[cid]
                    # Reproduce old oracle supply; this round has exactly the same editable domain.
                    assert abs(by_mode['oracle_pure_W']['repaired_iou']-prev['pure_regions_iou']) < 1e-12
                    assert abs(by_mode['oracle_target_points']['repaired_iou']-prev['point_optimum_iou']) < 1e-12
                    if by_mode['oracle_target_points']['same_parent_gain02'] or prev['best_candidate_single_gain02']:
                        scene_opps.append(dict(scene=scene, candidate_id=cid, cohort=job['cohort'], target=target,
                            parent_iou=identity['parent_iou'], original_NT_best=float(baseline[identity['index']]),
                            modes={m: dict(iou=r['repaired_iou'], gain_over_NT=r['pool_delta'],
                                added=r['added'], target=r['added_target'], other=r['added_other'],
                                unknown=r['unknown_edits'], safe_gain02=r['safe_gain02']) for m,r in by_mode.items()},
                            previous_best_candidate_query=prev['best_candidate_query'],
                            previous_best_candidate_gain=prev['best_candidate_single_gain_over_NT'],
                            contributing_blocks=[b['block_id'] for b in scene_blocks if b['candidate_id']==cid and b.get('valuable_point_contributor')],
                            posthoc_diagnostic_only=True))
        directory = OUT/'evaluation'/scene; directory.mkdir(parents=True)
        for name, data in [('blocks', scene_blocks), ('unions', scene_rows), ('opportunities', scene_opps)]:
            path = directory/(name+'.json'); D.seal(path, data); outputs[str(path)] = D.sha(path)
        path = directory/'oracle_diagnostic_points.npz'; np.savez_compressed(path, **arrays); outputs[str(path)] = D.sha(path)
        all_blocks += scene_blocks; all_rows += scene_rows; opportunities += scene_opps
        for m in MODES:
            all_pool += met.pool_rows(scene, m, baseline, same_best[m], ids)
            all_any_pool += met.pool_rows(scene, m, baseline, any_best[m], ids)
    directory = OUT/'evaluation'
    for name,data in [('pool_same_parent',all_pool),('pool_any_target',all_any_pool),('parent_scopes',scopes),('opportunities',opportunities)]:
        path=directory/(name+'.json'); D.seal(path,data); outputs[str(path)]=D.sha(path)
    summary = dict(complete=True, scenes=52, sampled_T_parents=len(scopes), blocks=len(all_blocks),
        scopes=dict(Counter(r['scope'] for r in scopes)),
        stable_parent_known_instance_mixture=dict(Counter(str(r['known_instance_count']) for r in scopes if r['stable'])),
        ownership=dict(Counter(r['block_GT']['ownership'] for r in all_blocks)),
        ownership_by_parent_scope={sc:dict(Counter(r['block_GT']['ownership'] for r in all_blocks if r['parent_scope']==sc)) for sc in sorted({r['scope'] for r in scopes})},
        modes={m:dict(unions=met.summarize_rows([r for r in all_rows if r['mode']==m]),
            same_parent_pool=met.pool_summary([r for r in all_pool if r['mode']==m]),
            any_target_pool=met.pool_summary([r for r in all_any_pool if r['mode']==m]),
            same_parent_gain02_parents=sum(r['same_parent_gain02'] for r in all_rows if r['mode']==m),
            same_parent_gain02_scenes=len({r['scene'] for r in all_rows if r['mode']==m and r['same_parent_gain02']}),
            safe_gain02_parents=sum(r['safe_gain02'] for r in all_rows if r['mode']==m),
            safe_gain02_scenes=len({r['scene'] for r in all_rows if r['mode']==m and r['safe_gain02']}),
            by_parent_scope={sc:dict(parents=len(rs),added=sum(r['added'] for r in rs),annotation_unknown=sum(r['added_GT']['unknown'] for r in rs),annotated=sum(r['added_GT']['annotated'] for r in rs)) for sc in sorted({r['scope'] for r in all_rows}) for rs in [[r for r in all_rows if r['mode']==m and r['scope']==sc]]}) for m in MODES},
        actual_accepted_output=False, AP_run=False, GT_oracles_are_not_methods=True)
    path=OUT/'evaluation_summary.json';D.seal(path,summary);outputs[str(path)]=D.sha(path)
    D.seal(OUT/'evaluation_frozen.json',dict(complete=True,outputs=outputs,
        predictions_sha256=D.sha(OUT/'predictions_frozen.json'), GT_after_predictions=True))
    run.stage('separate_GT_diagnoses_completed', parents=len(scopes), blocks=len(all_blocks))


def block_summary(rows):
    stable=[r for r in rows if r['stable']]
    counts={}
    for name, field in [('W','block_GT'),('C','clipped_GT'),('discarded','dropped_GT')]:
        c={k:sum(r[field][k] for r in stable) for k in ('points','target','other','unknown')}
        c.update(target_precision=c['target']/c['points'] if c['points'] else None,
                 annotated_target_precision=c['target']/(c['target']+c['other']) if c['target']+c['other'] else None)
        counts[name]=c
    counts['target_retention']=counts['C']['target']/counts['W']['target'] if counts['W']['target'] else None
    counts['neighbor_rejection']=counts['discarded']['other']/counts['W']['other'] if counts['W']['other'] else None
    counts['unknown_retention']=counts['C']['unknown']/counts['W']['unknown'] if counts['W']['unknown'] else None
    return dict(blocks=len(rows),scenes=len({r['scene'] for r in rows}),stable_blocks=len(stable),
        ownership=dict(Counter(r['block_GT']['ownership'] for r in rows)),point_selection=counts,
        zero_parent_support=sum(r['query_parent_hard_fraction']==0 for r in rows),
        zero_anchor_overlap=sum(r['anchor_supporter_hard_intersection']==0 for r in rows),
        has_conflict=sum(r['frame_states'].get('conflict',0)>0 for r in rows),
        has_common=sum(r['frame_states'].get('common',0)>0 for r in rows),
        no_common=sum(r['frame_states'].get('common',0)==0 for r in rows),
        has_unknown=sum(r['block_GT']['unknown']>0 for r in rows),
        low_parent_coverage_le_half=sum(r['query_parent_hard_fraction']<=.5 for r in rows),
        # .5 here is a descriptive half-coverage bin, never an acceptance condition.
        parent_hard_fraction_quantiles=np.quantile([r['query_parent_hard_fraction'] for r in rows],[0,.25,.5,.75,1]).tolist() if rows else [])


def analyze():
    D.verify(D.read(OUT/'evaluation_frozen.json')['outputs'])
    c=D.read(OUT/'contract.json'); rows=[]
    for scene in c['scenes']:rows+=D.read(OUT/'evaluation'/scene/'blocks.json')
    cells=defaultdict(list)
    for r in rows:cells[tuple(r['evidence_signature'])].append(r)
    table=[]
    for signature,rr in cells.items():
        pos=[r for r in rr if r['block_GT']['ownership']=='target_only']
        neg=[r for r in rr if r['block_GT']['ownership'] in ('other_only','other_unknown','mixed_known')]
        table.append(dict(signature=list(signature),blocks=len(rr),scenes=len({r['scene'] for r in rr}),
            ownership=dict(Counter(r['block_GT']['ownership'] for r in rr)),
            valuable_point_blocks=sum(r.get('valuable_point_contributor',False) for r in rr),
            valuable_pure_blocks=sum(r.get('valuable_pure_contributor',False) for r in rr),
            target_and_neighbor_collision=bool(pos and neg),
            target_and_neighbor_cross_scene_collision=any(p['scene']!=n['scene'] for p in pos for n in neg)))
    valuable=[r for r in rows if r.get('valuable_point_contributor')]
    pure_valuable=[r for r in rows if r.get('valuable_pure_contributor')]
    evidence=[]
    features=('supporter_mean','anchor_block_mean','query_parent_mean','query_parent_hard_fraction','minimum_distance')
    for r in valuable:
        rr=cells[tuple(r['evidence_signature'])]
        negatives=[n for n in rr if n['block_GT']['ownership'] in ('other_only','other_unknown','mixed_known') and (n['scene'],n['candidate_id'],n['region'])!=(r['scene'],r['candidate_id'],r['region'])]
        # Raw-feature range overlap is descriptive only; no fitted distance/selection.
        ranges={f:dict(value=r[f],minimum=min(n[f] for n in negatives),maximum=max(n[f] for n in negatives),inside_range=min(n[f] for n in negatives)<=r[f]<=max(n[f] for n in negatives)) for f in features} if negatives else {}
        evidence.append(dict(scene=r['scene'],candidate_id=r['candidate_id'],region=r['region'],
            block_id=r['block_id'],supporter=r['supporter'],anchor=r['anchor'],
            evidence={f:r[f] for f in features},query_parent_hard_fraction=r['query_parent_hard_fraction'],
            anchor_hard_iou=r['anchor_supporter_hard_intersection']/r['anchor_supporter_hard_union'] if r['anchor_supporter_hard_union'] else None,
            frame_states=r['frame_states'],block_GT=r['block_GT'],clipped_GT=r['clipped_GT'],
            point_oracle_contribution=r['point_oracle_contribution'],pure_oracle_contribution=r['pure_oracle_contribution'],
            signature=r['evidence_signature'],same_signature_neighbors=len(negatives),
            cross_scene_neighbors=sum(n['scene']!=r['scene'] for n in negatives),
            continuous_negative_ranges=ranges,
            negative_examples=[dict(scene=n['scene'],candidate_id=n['candidate_id'],region=n['region'],
                ownership=n['block_GT']['ownership'],block_GT=n['block_GT'],
                **{f:n[f] for f in features}) for n in negatives[:5]]))
    groups=dict(all=rows, stable=[r for r in rows if r['stable']],
        unresolved_parent=[r for r in rows if not r['stable']],
        pure_target=[r for r in rows if r['block_GT']['ownership']=='target_only'],
        neighbor_only=[r for r in rows if r['block_GT']['ownership']=='other_only'],
        target_neighbor_mixed=[r for r in rows if r['block_GT']['ownership']=='mixed_known'],
        valuable_point_contributors=valuable,valuable_pure_contributors=pure_valuable)
    summary=dict(groups={k:block_summary(rr) for k,rr in groups.items()},
        point_selection_by_ownership={k:block_summary([r for r in rows if r['block_GT']['ownership']==k]) for k in sorted({r['block_GT']['ownership'] for r in rows})},
        signature_cells=len(cells),cells_with_target_neighbor_collision=sum(t['target_and_neighbor_collision'] for t in table),
        valuable_blocks_with_neighbor_signature=sum(e['same_signature_neighbors']>0 for e in evidence),
        valuable_blocks_with_cross_scene_neighbor_signature=sum(e['cross_scene_neighbors']>0 for e in evidence),
        scope='207 sampled T parents/52 exposed physical scenes/original editable domain only',
        categorical_collisions_do_not_prove_raw_information_inseparability=True,
        no_GT_fitted_rule=True,no_AUC_threshold=True)
    d=OUT/'analysis';d.mkdir()
    for name,val in [('signature_cells',table),('valuable_block_evidence',evidence),('summary',summary)]:D.seal(d/(name+'.json'),val)
    D.seal(OUT/'analysis_frozen.json',dict(complete=True,outputs={str(p):D.sha(p) for p in d.iterdir()},code_sha256=D.sha(Path(__file__))))
    run.stage('evidence_analysis_completed',valuable_blocks=len(valuable),signature_collisions=summary['valuable_blocks_with_neighbor_signature'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('evaluate','analyze'))
    globals()[parser.parse_args().mode]()
