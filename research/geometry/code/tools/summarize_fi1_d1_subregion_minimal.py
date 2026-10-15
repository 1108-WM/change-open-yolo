"""Postseal tables and explanatory source-point oracle, never prediction selection."""
from collections import Counter
import csv

import numpy as np

import run_fi1_d1_subregion_minimal as run
import evaluate_fi1_nt_region_identity as met
import run_fi1_nt_region_identity as diag
from analyze_fi1_t_source_chain import summarize


def csv_seal(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def main():
    run.guard(False)
    frozen = run.audit.read(run.OUT / 'predictions_frozen.json')
    run.verify_map(frozen['outputs'])
    run.verify_map(run.audit.read(run.OUT / 'evaluation_frozen.json')['outputs'])
    evaluation = run.OUT / 'evaluation'
    for name in ('parents', 'targets', 'paired'):
        csv_seal(run.OUT / (name + '.csv'), run.audit.read(evaluation / (name + '.json')))
    blocks = run.audit.read(evaluation / 'blocks.json')
    flat = []
    for b in blocks:
        r = {k: v for k, v in b.items() if k not in run.ARMS}
        for arm in run.ARMS:
            r.update({arm + '_' + k: v for k, v in b[arm].items()})
        flat.append(r)
    csv_seal(run.OUT / 'blocks.csv', flat)
    old = run.audit.read(run.audit.OLD / 'contract.json')
    sc = run.audit.read(run.SOURCE / 'contract.json')
    parent_rows = run.audit.read(evaluation / 'parents.json')
    parent_index = {(r['scene'], r['candidate_id']): r for r in parent_rows if r['mode'] == 'sub_full'}
    regions, oracle = [], []
    census = Counter()
    for scene, spec in sc['jobs'].items():
        print('postseal_tables', scene, flush=True)
        gt = np.loadtxt(spec['job']['GT'], dtype=np.int64)
        ids, sizes = met.valid_targets(gt, old['class_ids'])
        _, nt = diag.load_nt(spec['job'])
        base = met.pool_base(nt, gt, ids, sizes)
        dest = run.OUT / 'predictions' / scene
        refs = run.audit.read(dest / 'parents.json')
        tracks = {r['track_id']: r for r in run.audit.read(dest / 'tracks.json')}
        # The broad oracle includes source-supported points outside D1 initial SPs,
        # but still requires repetition within one original track, not pooled tracks.
        repeated = {}
        for tid, t in tracks.items():
            counts = Counter()
            for frame in t['frames']:
                union = set()
                for o in frame['observations']:
                    union.update(run.audit.points(o['point_indices_path']).tolist())
                counts.update(union)
            repeated[tid] = np.array(sorted(p for p, n in counts.items() if n >= 2), np.int64)
        with np.load(dest / 'data.npz') as archive:
            z = {k: archive[k] for k in archive.files}
        for t in tracks.values():
            key = t['key']
            census['regions'] += len(z[key + '_sub_sizes'])
            census['kept_regions'] += int(z[key + '_sub_kept'].sum())
            census['tiny_regions'] += int((z[key + '_sub_sizes'] < 3).sum())
            census['zero_signature_regions'] += int((z[key + '_sub_inside'].sum(axis=0) == 0).sum())
        for p in refs:
            cid = p['candidate_id']
            parent, editable = z[cid + '_parent'], z[cid + '_editable']
            meta = met.identity(parent, gt, ids, sizes)
            target = meta['target'] if meta['stable'] else None
            supply = run.audit.hit(run.audit.union(repeated[tid] for tid in p['source_track_ids']), editable)
            pp = np.union1d(parent, supply[gt[supply] == target]) if target else parent
            r, _ = met.outcome(parent, pp, gt, ids, sizes, meta, base)
            oracle.append(dict(scene=scene, candidate_id=cid, mode='all_original_source_points_old_two_frames_target_oracle', **r))
            for tid in p['source_track_ids']:
                t = tracks[tid]
                key = t['key']
                domain = z[key + '_domain']
                labels = z[key + '_sub_labels']
                for rid in np.flatnonzero(z[key + '_sub_kept']):
                    where = np.flatnonzero(labels == rid)
                    added = np.setdiff1d(domain[where], parent)
                    if not len(added):
                        continue
                    inside, visible = z[key + '_sub_inside'][:, rid], z[key + '_sub_visible'][:, rid]
                    votes = (visible >= 3) & (inside / np.maximum(1, visible) >= .5)
                    negative = (visible >= 3) & ~votes
                    support_ids = [f['frame_id'] for f, vote in zip(t['frames'], votes) if vote]
                    support_obs = [o for f, vote in zip(t['frames'], votes) if vote for o in f['observation_ids']]
                    sp = int(z[key + '_sub_sp'][rid])
                    regions.append(dict(scene=scene, candidate_id=cid, stable=meta['stable'], target=target,
                        source_track_id=tid, region_id=int(rid), original_SP=sp,
                        historical_whole_SP_kept=sp in t['whole_kept_SP_ids'],
                        region_points=len(where), added=len(added), old_domain_added=len(run.audit.hit(added, editable)),
                        added_target=int((gt[added] == target).sum()) if target else None,
                        added_other=int(((gt[added] >= 1000) & (gt[added] != target)).sum()) if target else None,
                        added_unknown=int((gt[added] < 1000).sum()),
                        support_frames=int(votes.sum()), negative_frames=int(negative.sum()),
                        insufficient_visibility_frames=int((visible < 3).sum()),
                        supported_frame_ids=';'.join(map(str, support_ids)),
                        supported_frame_observation_ids=';'.join(map(str, support_obs)),
                        evidence=f'predictions/{scene}/data.npz:{key}_sub_labels=={rid}',
                        source_evidence=f'predictions/{scene}/tracks.json:track_id={tid}',
                        parent_full_pool_delta=parent_index[scene, cid].get('pool_delta')))
    csv_seal(run.OUT / 'sub_added_region_evidence.csv', regions)
    run.audit.seal(run.OUT / 'broad_two_frame_oracle.json', dict(
        GT_posthoc_only=True, used_for_selection=False,
        scope='Original editable whole-block domain; direct mask points, no extrapolation; >=2 distinct frames within each original track. Broader than common D1 initial domain.',
        summary=summarize(oracle), gains=[r for r in oracle if r.get('pool_delta', -1) >= .02-1e-12], parents=oracle))
    paired = run.audit.read(evaluation / 'paired.json')
    pairs = {}
    for view in run.VIEWS:
        rr = [r for r in paired if r['view'] == view and r['stable']]
        pairs[view] = dict(sub_only=sum(r['sub_only'] for r in rr),
            sub_only_target=sum(r['sub_only_target'] for r in rr),
            sub_only_other=sum(r['sub_only_other'] for r in rr),
            sub_only_unknown=sum(r['sub_only_unknown'] for r in rr),
            whole_only=sum(r['whole_only'] for r in rr),
            improved_gt02=sum(r['sub_minus_whole_iou'] > .02 for r in rr),
            harmed_gt02=sum(r['sub_minus_whole_iou'] < -.02 for r in rr))
    run.audit.seal(run.OUT / 'supplementary_summary.json', dict(
        code_sha256=run.audit.sha(__file__), region_census=census, paired=pairs,
        region_table_rows=len(regions), region_table_counts='Track-region associations may overlap; use deduplicated parent rows for final point totals.',
        broad_oracle=summarize(oracle),
        prediction_seal_sha256=run.audit.sha(run.OUT / 'predictions_frozen.json')))
    paths = list(run.OUT.glob('*.csv')) + [run.OUT / n for n in ('broad_two_frame_oracle.json', 'supplementary_summary.json')]
    run.audit.seal(run.OUT / 'supplementary_frozen.json', dict(complete=True,
        GT_posthoc_only=True, predictions_changed=False, outputs={str(p): run.audit.sha(p) for p in paths}))
    print('supplementary complete', dict(census), pairs, summarize(oracle), flush=True)


if __name__ == '__main__':
    main()
