"""Independent raw membership, per-frame hypotheses, decisions and pool checks."""
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

import run_fi1_cross_track_dual as run


def scalar_frame(a, b, inter, va, vb, vi, sizes, ids, owners, families, la, lb):
    joint, select_a, select_b = [], [], []
    for k in range(len(ids)):
        ca, cb = (int(a[k])-int(inter[k]))/max(1, va-vi), (int(b[k])-int(inter[k]))/max(1, vb-vi)
        hit = int(a[k]+b[k]-inter[k])
        score = hit/max(1, int(sizes[k])+va+vb-vi-hit)
        if ca >= .6 and cb >= .6:
            joint.append((score, int(ids[k]), k))
        if ca >= .6 and cb <= .1:
            select_a.append((int(a[k])/max(1, int(sizes[k])+va-int(a[k])), int(ids[k]), k))
        if cb >= .6 and ca <= .1:
            select_b.append((int(b[k])/max(1, int(sizes[k])+vb-int(b[k])), int(ids[k]), k))
    def best(rows):
        return sorted(rows, key=lambda r: (-r[0], r[1]))[0][:2] if rows else (0., None)
    merge, mid = best(joint)
    sa, aid = best(select_a)
    sb, bid = best(select_b)
    split = (sa+sb)/2 if aid is not None and bid is not None else 0.
    mp, sp = merge >= .6, split >= .6 and min(sa, sb) >= .6
    state = 'merge_wins' if mp and merge-split >= .15 else 'separate_wins' if sp and split-merge >= .15 else 'competing_or_close' if mp or sp else 'no_fitting_explanation'
    qualified = [k for score, _, k in joint if score >= .6]
    os = {int(owners[k]) for k in qualified if owners[k] >= 0}
    return dict(state=state, merge_fit=merge, separate_fit=split,
        merge_observation_id=mid, separate_observation_ids=[aid, bid], separate_plausible=sp,
        joint_observation_ids=sorted(int(ids[k]) for k in qualified), joint_raw_track_ids=sorted(os),
        joint_from_A=sorted(os & set(la)), joint_from_B=sorted(os & set(lb)),
        joint_dependency_family_ids=sorted({int(families[k]) for k in qualified if families[k] >= 0}))


def independent_identity(pp, gt, ids, sizes):
    counts = np.array([np.count_nonzero(gt[pp] == g) for g in ids])
    values = counts/np.maximum(1, len(pp)+sizes-counts)
    idx = int(values.argmax()) if len(ids) and values.max() > 0 else None
    stable = idx is not None and counts[idx] >= 20 and counts[idx]/len(pp) >= .6
    return bool(stable), int(ids[idx]) if stable else None


def main():
    run.guard(False)
    run.check_contract()
    prediction = run.a.read(run.OUT / 'predictions_frozen.json')
    run.verify_map(prediction['outputs'])
    run.verify_map(run.a.read(run.OUT / 'evaluation_frozen.json')['outputs'])
    sc = run.a.read(run.SOURCE / 'contract.json')
    old = run.a.read(run.a.OLD / 'contract.json')
    import evaluate_fi1_nt_region_identity as met
    erows = run.a.read(run.OUT / 'evaluation/pairs.json')
    eindex = {(r['scene'], r['pair_id']): r for r in erows}
    targets = run.a.read(run.OUT / 'evaluation/targets.json')
    tally, inputs = Counter(), {}
    for ix, (scene, spec) in enumerate(sc['jobs'].items(), 1):
        print('independent_verify', ix, 52, scene, flush=True)
        d = run.OUT / 'predictions' / scene
        receipt = run.a.read(d / 'receipt.json')
        run.verify_map(receipt['inputs'])
        inputs.update(receipt['inputs'])
        observations = run.a.read(d / 'observations.json')
        nodes = run.a.read(d / 'nodes.json')
        pairs = run.a.read(d / 'pairs.json')
        frames = run.a.read(d / 'pair_frames.json')
        visibility_refs = run.a.read(d / 'visibility.json')
        dependence = run.a.read(d / 'source_dependence.json')
        with np.load(d / 'data.npz') as archive:
            z = {k: archive[k] for k in archive.files}
        pp = [z['n%04d' % n['node']] for n in nodes]
        n = spec['job']['binding']['point_count']
        sizes = np.array([len(p) for p in pp])
        G = coo_matrix((np.ones(int(sizes.sum()), np.int32),
                       (np.concatenate(pp), np.repeat(np.arange(len(pp)), sizes))), shape=(n, len(pp))).tocsr()
        raw = [run.a.points(o['point_indices_path']) for o in observations]
        osizes = np.array([len(p) for p in raw])
        rawmatrix = coo_matrix((np.ones(int(osizes.sum()), np.int32),
            (np.repeat(np.arange(len(raw)), osizes), np.concatenate(raw))), shape=(len(raw), n)).tocsc()
        hits = np.array([np.asarray(G[p].sum(axis=0)).ravel() for p in raw], dtype=np.int64)
        np.testing.assert_array_equal(hits, z['mask_node_hits'])
        vis = np.stack([np.load(r['path']) for r in visibility_refs])
        visible = np.array([np.asarray(G[v].sum(axis=0)).ravel() for v in vis], dtype=np.int64)
        np.testing.assert_array_equal(visible, z['frame_node_visible'])
        for p, o in zip(raw, observations):
            j = next(i for i, r in enumerate(visibility_refs) if r['frame_index'] == o['frame_index'])
            assert vis[j, p].all()
        byframe = defaultdict(list)
        for i, o in enumerate(observations):
            byframe[o['frame_index']].append(i)
        # Rebuild dependence edges and connected sets, using explicit graph traversal.
        expected_edges = defaultdict(set)
        csr = rawmatrix.tocsr()
        for fi, rows in byframe.items():
            sub = csr[rows]
            inter = (sub @ sub.T).toarray()
            for ai, l in enumerate(rows):
                for bi in range(ai+1, len(rows)):
                    r = rows[bi]
                    lo, ro = observations[l]['raw_track_id'], observations[r]['raw_track_id']
                    if min(lo, ro) < 0 or lo == ro:
                        continue
                    hit = int(inter[ai, bi])
                    if hit/max(1, int(osizes[l]+osizes[r]-hit)) >= .8:
                        expected_edges[tuple(sorted((lo, ro)))].add(fi)
        valid_edges = {edge: fs for edge, fs in expected_edges.items() if len(fs) >= 2}
        assert set(valid_edges) == {tuple(e['raw_track_ids']) for e in dependence['dependence_edges']}
        adjacency = defaultdict(set)
        for left, right in valid_edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        family = {}
        for root in map(int, dependence['family_by_raw_track']):
            if root in family:
                continue
            stack, members = [root], set()
            while stack:
                current = stack.pop()
                if current not in members:
                    members.add(current)
                    stack.extend(adjacency[current]-members)
            for member in members:
                family[member] = min(members)
        assert {str(k): v for k, v in family.items()} == dependence['family_by_raw_track']
        owners = np.array([o['raw_track_id'] for o in observations])
        families = np.array([family.get(int(o), -1) for o in owners])
        np.testing.assert_array_equal(families, z['dependence_family_ids'])
        frameindex = {(f['pair_id'], f['frame_index']): f for f in frames}
        oid = np.array([o['observation_id'] for o in observations])
        eligible, bypair = [], {}
        for pair in pairs:
            pid, l, r = pair['pair_id'], pair['left'], pair['right']
            both = np.intersect1d(pp[l], pp[r])
            np.testing.assert_array_equal(both, z[pid + '_intersection'])
            union = np.union1d(pp[l], pp[r])
            np.testing.assert_array_equal(union, z[pid + '_union'])
            ih = np.asarray(rawmatrix[:, both].sum(axis=1)).ravel()
            iv = vis[:, both].sum(axis=1)
            np.testing.assert_array_equal(ih, z[pid + '_intersection_inside'])
            np.testing.assert_array_equal(iv, z[pid + '_intersection_visible'])
            ea, eb = np.setdiff1d(pp[l], pp[r]), np.setdiff1d(pp[r], pp[l])
            replay = []
            for j, vr in enumerate(visibility_refs):
                fi = vr['frame_index']
                stored = frameindex[pid, fi]
                va, vb = int(visible[j, l]-iv[j]), int(visible[j, r]-iv[j])
                reliable = va >= 20 and vb >= 20 and va/max(1, len(ea)) >= .1 and vb/max(1, len(eb)) >= .1
                if not reliable:
                    assert stored['state'] == 'not_jointly_reliably_visible'
                    continue
                rows = np.array(byframe.get(fi, []), int)
                expected = scalar_frame(hits[rows, l], hits[rows, r], ih[rows], int(visible[j, l]),
                    int(visible[j, r]), int(iv[j]), osizes[rows], oid[rows], owners[rows], families[rows],
                    nodes[l]['source_track_ids'], nodes[r]['source_track_ids'])
                for key, value in expected.items():
                    if isinstance(value, float):
                        assert abs(stored[key]-value) < 1e-12, (scene, pid, fi, key)
                    else:
                        assert stored[key] == value, (scene, pid, fi, key)
                replay.append(expected)
                tally['reliable_pair_frames'] += 1
            # Recompute supported third explanations from raw outside-region points.
            actual_competitors = []
            for candidate in range(len(nodes)):
                if candidate in (l, r) or not len(ea) or not len(eb):
                    continue
                ca = len(np.intersect1d(pp[candidate], ea))/len(ea)
                cb = len(np.intersect1d(pp[candidate], eb))/len(eb)
                side = 'A' if ca >= .8 and cb <= .1 else 'B' if cb >= .8 and ca <= .1 else None
                outside = np.setdiff1d(pp[candidate], union)
                if side is None or len(outside) < 20 or len(outside)/len(pp[candidate]) < .2:
                    continue
                actual_competitors.append(candidate)
                stored = next(c for c in pair['competitors'] if c['node'] == candidate)
                oh = np.asarray(rawmatrix[:, outside].sum(axis=1)).ravel()
                ov = vis[:, outside].sum(axis=1)
                sidepoints, otherpoints = (ea, eb) if side == 'A' else (eb, ea)
                sh = np.asarray(rawmatrix[:, sidepoints].sum(axis=1)).ravel()
                th = np.asarray(rawmatrix[:, otherpoints].sum(axis=1)).ravel()
                sv, tv = vis[:, sidepoints].sum(axis=1), vis[:, otherpoints].sum(axis=1)
                witness = []
                for j, vr in enumerate(visibility_refs):
                    fi = vr['frame_index']
                    if frameindex[pid, fi]['state'] == 'not_jointly_reliably_visible':
                        continue
                    if ov[j] < 20 or ov[j]/len(outside) < .1 or visible[j, candidate] < 20 or visible[j, candidate]/len(pp[candidate]) < .1:
                        continue
                    ids = []
                    for k in byframe.get(fi, []):
                        score = hits[k, candidate]/max(1, int(osizes[k]+visible[j, candidate]-hits[k, candidate]))
                        if score >= .6 and sh[k]/max(1, sv[j]) >= .6 and th[k]/max(1, tv[j]) <= .1 and oh[k]/max(1, ov[j]) >= .6:
                            ids.append(int(oid[k]))
                    if ids:
                        witness.append((fi, ids))
                assert witness == [(f['frame_index'], f['observation_ids']) for f in stored['frames']]
                assert stored['supported_conflict'] == (len(witness) >= 2)
                tally['competitor_records'] += 1
            assert sorted(actual_competitors) == sorted(c['node'] for c in pair['competitors'])
            wins = [f for f in replay if f['state'] == 'merge_wins']
            split = sum(f['state'] == 'separate_wins' for f in replay)
            source_a = {t for f in wins for t in f['joint_from_A']}
            source_b = {t for f in wins for t in f['joint_from_B']}
            groups = {t for f in wins for t in f['joint_dependency_family_ids']}
            if min(len(ea), len(eb)) < 20:
                state, reason = 'unknown', 'insufficient_distinct_geometry'
            elif split >= 2 and not wins:
                state, reason = 'reject', 'repeated_better_separate_explanation'
            elif not replay:
                state, reason = 'unknown', 'no_jointly_reliable_views'
            elif any(c['supported_conflict'] for c in pair['competitors']):
                state, reason = 'unknown', 'supported_third_instance_explanation'
            elif any(f['separate_plausible'] for f in replay):
                state, reason = 'unknown', 'separate_explanation_remains_plausible'
            elif len(wins) < 2:
                state, reason = 'unknown', 'insufficient_merge_discrimination'
            elif not source_a or not source_b or len(groups) < 2:
                state, reason = 'unknown', 'joint_support_concentrated_in_one_lineage'
            else:
                state, reason = 'eligible', None
                eligible.append(pair)
            bypair[pid] = (state, reason)
            tally['proposed_pairs'] += 1
        degrees = Counter(n for p in eligible for n in (p['left'], p['right']))
        endpoints, codes = set(), set()
        _, nt = run.diag.load_nt(spec['job'])
        original_codes = {run.a.digest(p) for p in nt.values()}
        for p in pairs:
            state, reason = bypair[p['pair_id']]
            if state == 'eligible':
                state, reason = ('accept', 'unique_pair_after_hypothesis_competition') if degrees[p['left']] == degrees[p['right']] == 1 else ('unknown', 'multiple_competing_pair_explanations')
            assert (state, reason) == (p['state'], p['reason'])
            digest = run.a.digest(z[p['pair_id'] + '_union'])
            assert digest == p['union_sha256']
            emit = state == 'accept' and digest not in original_codes and digest not in codes
            assert emit == p['emit']
            if state == 'accept':
                assert not endpoints & {p['left'], p['right']}
                endpoints.update((p['left'], p['right']))
                codes.add(digest)
        # Check proposal completeness over all unordered source-T geometries.
        tnodes = [n['node'] for n in nodes if n['final_track_ids']]
        proposed = {(p['left'], p['right']): p for p in pairs}
        intersections = z['node_intersections']
        xyz = np.load(spec['job']['paths']['processed'], mmap_mode='r')[:, :3]
        trees = {}
        mins, maxs = [xyz[p].min(axis=0) for p in pp], [xyz[p].max(axis=0) for p in pp]
        for ai, l in enumerate(tnodes):
            for r in tnodes[ai+1:]:
                reasons = []
                if intersections[l, r] > 0:
                    reasons.append('shared_points')
                else:
                    lower = np.linalg.norm(np.maximum(0, np.maximum(mins[l]-maxs[r], mins[r]-maxs[l])))
                    if lower <= .15:
                        if r not in trees:
                            trees[r] = cKDTree(xyz[pp[r]])
                        dist, _ = trees[r].query(xyz[pp[l]], distance_upper_bound=.15+1e-12)
                        if np.isfinite(dist).any():
                            reasons.append('spatial_near')
                contains = [c for c in range(len(nodes)) if c not in (l, r) and intersections[c, l] >= .5*len(pp[l]) and intersections[c, r] >= .5*len(pp[r])]
                if contains:
                    reasons.append('existing_containing_explanation')
                raw_proposal = []
                for j, vr in enumerate(visibility_refs):
                    va, vb = visible[j, l], visible[j, r]
                    if va < 20 or vb < 20 or va/len(pp[l]) < .1 or vb/len(pp[r]) < .1:
                        continue
                    for k in byframe.get(vr['frame_index'], []):
                        if hits[k, l] >= 20 and hits[k, r] >= 20 and hits[k, l]/va >= .3 and hits[k, r]/vb >= .3:
                            raw_proposal.append(int(oid[k]))
                if raw_proposal:
                    reasons.append('raw_common_mask')
                assert bool(reasons) == ((l, r) in proposed)
                if reasons:
                    assert reasons == proposed[l, r]['proposal_reasons']
                    assert raw_proposal == proposed[l, r]['proposal_observation_ids']
                    assert contains == proposed[l, r]['containing_nodes']
                tally['all_track_pairs'] += 1
        # Independent label counts and expanded pool IoU, including tiny T identities.
        assert run.a.sha(spec['job']['GT']) == old['GT_inputs'][spec['job']['GT']]
        gt = np.loadtxt(spec['job']['GT'], dtype=np.int64)
        gids, gsizes = met.valid_targets(gt, old['class_ids'])
        def ious(points):
            if len(points) < 100:
                return np.zeros(len(gids))
            values, counts = np.unique(gt[points], return_counts=True)
            table = dict(zip(values.tolist(), counts.tolist()))
            return np.array([table.get(int(g), 0)/(len(points)+sz-table.get(int(g), 0)) for g, sz in zip(gids, gsizes)])
        base = np.zeros(len(gids))
        for points in nt.values():
            base = np.maximum(base, ious(points))
        best = {arm: base.copy() for arm in ('proposed', 'accepted')}
        for p in pairs:
            row = eindex[scene, p['pair_id']]
            ml = independent_identity(pp[p['left']], gt, gids, gsizes)
            mr = independent_identity(pp[p['right']], gt, gids, gsizes)
            relation = 'same' if ml[0] and mr[0] and ml[1] == mr[1] else 'different' if ml[0] and mr[0] else 'unresolved'
            assert relation == row['relation']
            union = z[p['pair_id'] + '_union']
            assert int((gt[union] < 1000).sum()) == row['union_GT']['unknown']
            values, counts = np.unique(gt[union], return_counts=True)
            assert {str(g): int(n) for g, n in zip(values, counts)} == row['union_GT']['distribution']
            score = ious(union)
            if relation == 'same':
                idx = int(np.flatnonzero(gids == ml[1])[0])
                delta = float(score[idx]-base[idx])
                purity = float((gt[union] == ml[1]).mean())
                assert abs(delta-row['same_target_pool_delta']) < 1e-12
                assert row['safe_gain02'] == (purity >= .9 and delta >= .02-1e-12)
            best['proposed'] = np.maximum(best['proposed'], score)
            if p['emit']:
                best['accepted'] = np.maximum(best['accepted'], score)
        for r in [r for r in targets if r['scene'] == scene]:
            idx = int(np.flatnonzero(gids == r['gt_id'])[0])
            assert abs(r['baseline']-base[idx]) < 1e-12
            expected = base[idx] if r['mode'] == 'NT' else best[r['mode']][idx]
            assert abs(r['expanded']-expected) < 1e-12
            tally['pool_target_rows'] += 1
        tally.update(scenes=1, observations=len(observations), final_T_records=receipt['counts']['final_T_records'])
    assert tally['scenes'] == 52 and tally['observations'] == 51463 and tally['final_T_records'] == 2300
    run.a.seal(run.OUT / 'verification.json', dict(complete=True, counts=tally, input_files_rehashed=len(inputs),
        independent_mask_membership=True, independent_visibility=True, independent_scalar_hypotheses=True,
        independent_dependence_graph=True, independent_competitor_checks=True,
        proposal_completeness=True, independent_decisions=True, nonrecursive_matching=True,
        independent_pair_GT_and_pool=True, old_seals_and_production_writes_forbidden=True))
    print('VERIFIED', dict(tally), flush=True)


if __name__ == '__main__':
    main()
