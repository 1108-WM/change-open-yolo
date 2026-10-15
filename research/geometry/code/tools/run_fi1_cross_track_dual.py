#!/usr/bin/env python3
"""Cross-track fragmentation: competing split/merge hypotheses, then pair unions."""
import argparse
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

import audit_fi1_t_source_chain as a
import fi1_cross_track_dual_core as core
import run_fi1_nt_region_identity as diag

SOURCE = a.OUT
OUT = a.EXP / 'fi1_cross_track_dual_hypothesis_exposed52_20261015_v2'
HERE = Path(__file__).resolve().parent
SCRIPTS = ('run_fi1_cross_track_dual.py', 'fi1_cross_track_dual_core.py',
           'test_fi1_cross_track_dual.py', 'verify_fi1_cross_track_dual.py',
           'audit_fi1_t_source_chain.py', 'run_fi1_nt_region_identity.py',
           'evaluate_fi1_nt_region_identity.py')


def guard(no_gt=True):
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if no_gt and any(x in str(p) for x in ('ground_truth', '/evaluation/', '/diagnosis/',
                            'parent_scopes', 'evaluation_summary', 'supplementary_source_vote',
                            'broad_two_frame_oracle')):
            raise PermissionError('GT/old diagnosis forbidden: ' + str(p))
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if writing and not p.is_relative_to(OUT):
            raise PermissionError('Write restricted to new experiment: ' + str(p))
    sys.addaudithook(hook)


def verify_map(mapping):
    for path, sha in mapping.items():
        assert a.sha(path) == sha, path


def check_contract():
    c = a.read(OUT / 'contract.json')
    verify_map(c['code'])
    assert core.PARAMETERS == c['parameters']
    assert a.sha(SOURCE / 'source_evidence_frozen.json') == c['source_seal_sha256']
    assert a.sha(SOURCE / 'contract.json') == c['source_contract_sha256']
    return c


def prepare():
    guard()
    a.reserve()
    assert not OUT.exists()
    source = a.read(SOURCE / 'source_evidence_frozen.json')
    assert source['complete'] and source['GT_used'] is False
    verify_map(source['outputs'])
    files = [HERE / name for name in SCRIPTS]
    files += [a.ROOT / 'full_bundle/code/tools' / name for name in ('fi1_core_ablation.py', 'fi1_r0_r1_protocol.py')]
    code = {str(p): a.sha(p) for p in files}
    predecessor = a.EXP / 'fi1_cross_track_dual_hypothesis_exposed52_20261015_v1'
    assert a.read(predecessor / 'aborted_before_GT.json')['GT_used'] is False
    assert a.read(predecessor / 'contract.json')['parameters'] == core.PARAMETERS
    a.seal(OUT / 'contract.json', dict(version=OUT.name, time_unix=time.time(),
        predecessor_abort_sha256=a.sha(predecessor / 'aborted_before_GT.json'),
        source_seal_sha256=a.sha(SOURCE / 'source_evidence_frozen.json'),
        source_contract_sha256=a.sha(SOURCE / 'contract.json'), code=code,
        scope='All 52 exposed scenes, all 2300 historical final T records, including small tracks not present in NT binding; exact duplicate geometries share one node, all source IDs retained. Preserve sampled207 as diagnostic stratum, no GT positive selection.',
        parameters=core.PARAMETERS,
        proposal='All unordered distinct source-T geometry pairs. Propose if exact minimum distance<=.15m, any shared points, a third existing N/T candidate covers>=.5 of each, or a cached raw mask covers>=.3 of each in a jointly reliable view. No top-k, no GT or query selection. Save proposal/rejection census for all pairs.',
        hypotheses='Hmerge: one raw mask fits visible full A union B by IoU and covers>=.6 of each exclusive side. Hseparate: two distinct selective masks each cover>=.6 of their exclusive side and <=.1 of the other, each IoU>=.6 to its visible full endpoint; score is average IoU. Margin .15. Each source frame contributes at most one event regardless mask count.',
        reliability='Each exclusive side has>=20 cached depth-visible points and >=.1 visible fraction. Outside reliable covisibility means missing evidence, never evidence of different objects. No 3D overlap is not a rejection.',
        dependence='Raw tracks whose masks have IoU>=.8 in>=2 same source frames share an evidence-dependence family (connected components only for conservative dependence bookkeeping, never instance merging). Unassigned masks have no independent track family. Repeated masks/frames are not identity confirmation.',
        competitors='Record every one-sided N/T competitor covering>=.8 of one exclusive side and <=.1 other, with>=20 and>=.2 of its geometry outside pair. Supported conflict needs>=2 reliable frames with a mask fitting competitor>=.6, covering side and observed outside>=.6, avoiding other<=.1. N is optional competing explanation, never mandatory joint reference.',
        decision='Reject when>=2 separate-winning frames and zero merge-winning frames. Missing, close/contradictory, one-lineage/dependence-family-only, or supported-third-instance evidence => unknown. Eligible requires>=2 merge-winning frames, no plausible separate frame, both endpoint original lineages represented,>=2 dependence families, no supported competitor. Accept only mutually unique eligible partners. All other eligible edges abstain; no score tie breaker, recursive merging or tree.',
        construction='One frozen geometry: full point union of two existing source T nodes after original merge. No point edits, query, clipping, boundary fixes, D1 reread or extrapolation. Preserve all original NT; no replacements/deletions. Each accepted endpoint used at most once. Save ALL proposed unions, status, evidence and dedup flags; accepted branch emits only changed, nonduplicate NT/new unions.',
        comparisons=['NT unchanged','all proposed pair unions: unsafe supply diagnostic','accepted pair unions: fixed dual-hypothesis prototype'],
        gates=dict(safe_gain02_scenes_min=2, union_target_purity_min=.90,
                   accepted_resolved_same_identity_fraction_min=.90,
                   accepted_confirmed_cross_identity_max=0, accepted_unresolved_identity_max=0,
                   good_endpoint_union_loss_gt02_max=0, unknown_annotation_fraction_max=.10),
        evaluation='Both endpoints stable same GT defines resolved same identity; stable different GT confirmed cross identity; otherwise unresolved, with mixed/unknown distributions retained. Stable requires>=20 points and>=.6 purity for a valid GT target, including source fragments<100 points; evaluation IoU retains official candidate-size>=100. Safe gain uses resolved-same pairs, union target purity>=.9 and same-target IoU minus COMPLETE NT>=.02. Separately evaluate any-target expanded pool, decision confusions, mixed/negative cases and sampled parents. GT only after complete predictions seal.',
        stop='One fixed prototype, no post-GT threshold search. Failure stops this rule. No automatic independent scenes, replacement or downstream AP.',
        limits=dict(training=False, new_frames=False, point_edits=False, split=False, query=False,
                    recursive_merge=False, instance_tree=False, replacement=False, AP=False,
                    production_modified=False, old_seals_modified=False)))
    for p in files:
        dest = OUT / 'code_snapshot' / p.relative_to(a.ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE),
                           '-p', 'test_fi1_cross_track_dual.py', '-v'], capture_output=True, text=True)
    assert test.returncode == 0, test.stderr
    a.seal(OUT / 'tests.json', dict(passed=True, stdout=test.stdout, stderr=test.stderr))
    print('contract frozen; tests passed', flush=True)


class Scene:
    def __init__(self, scene, spec, source_contract, source_seal):
        self.scene, self.spec = scene, spec
        self.frontend = Path(source_contract['frontend'])
        self.arm = self.frontend / 'arms/D'
        self.used = {}
        ed = SOURCE / 'evidence' / scene
        path = ed / 'alignment.json'
        assert a.sha(path) == source_seal['outputs'][str(path)]
        self.used[str(path)] = a.sha(path)
        self.expected = a.read(path)['inputs']

    def input(self, path):
        path = Path(path)
        key = str(path)
        if key not in self.used:
            self.used[key] = a.sha(path)
        assert self.used[key] == self.expected[key], key
        return path

    def load(self):
        s = self.scene
        mf = a.read(self.input(self.frontend / 'manifests' / (s + '.json')))
        data = np.load(self.input(self.spec['job']['paths']['processed']), mmap_mode='r')
        self.xyz, self.sp = data[:, :3], np.asarray(data[:, 9], np.int64)
        self.n = len(data)
        assert self.n == self.spec['job']['binding']['point_count']
        self.input(self.spec['job']['binding']['geometry'])
        self.input(self.spec['job']['binding']['points'])
        geometry, nt = diag.load_nt(self.spec['job'])
        self.nt = nt
        gmap = {g['geometry_key']: g for g in geometry}
        self.nodes, self.points = [], []
        digest_index = {}
        for key in sorted(nt):
            g = gmap[key]
            sources = sorted({r['source'] for r in diag.original_aliases(g)})
            digest = a.digest(nt[key])
            assert digest not in digest_index
            idx = len(self.nodes)
            digest_index[digest] = idx
            self.nodes.append(dict(node=idx, key=key, points=len(nt[key]), sources=sources,
                in_original_NT=True, final_track_ids=[], source_track_ids=[], observation_ids=[],
                geometry_sha256=digest, locator=dict(points_path=self.spec['job']['binding']['points'], array_key=g['array_key'])))
            self.points.append(nt[key])
        self.final = a.read(self.input(self.arm / 'merged' / s / 'automatic_tracks.json'))['tracks']
        merged_receipt = a.read(self.input(self.arm / 'stage_receipts' / (s + '_merged.json')))
        for relative, digest in merged_receipt['files'].items():
            key = str(self.arm / 'merged' / s / relative)
            if key in self.expected:
                assert self.expected[key] == digest
            self.expected[key] = digest
        self.raw = a.read(self.input(self.arm / 'tracks' / s / 'automatic_tracks.json'))['tracks']
        raw_by_id = {int(t['track_id']): t for t in self.raw}
        owner = {int(oid): int(t['track_id']) for t in self.raw for oid in t['observation_ids']}
        assert len(owner) == sum(len(t['observation_ids']) for t in self.raw)
        final_owner = {int(tid): int(t['track_id']) for t in self.final for tid in t['source_track_ids']}
        final_paths = {}
        for t in sorted(self.final, key=lambda x: x['track_id']):
            pp = a.points(self.input(t['points_path']))
            assert np.array_equal(pp, np.flatnonzero(np.isin(self.sp, t['superpoint_ids'])))
            assert set(t['observation_ids']) == {oid for tid in t['source_track_ids'] for oid in raw_by_id[tid]['observation_ids']}
            digest = a.digest(pp)
            if digest not in digest_index:
                idx = len(self.nodes)
                digest_index[digest] = idx
                self.nodes.append(dict(node=idx, key=s + ':source_T:' + digest, points=len(pp),
                    sources=['track'], in_original_NT=False, final_track_ids=[], source_track_ids=[],
                    observation_ids=[], geometry_sha256=digest,
                    locator=dict(points_path=t['points_path'], array_key='point_indices')))
                self.points.append(pp)
            node = self.nodes[digest_index[digest]]
            node['final_track_ids'].append(int(t['track_id']))
            node['source_track_ids'] = sorted(set(node['source_track_ids']) | set(t['source_track_ids']))
            node['observation_ids'] = sorted(set(node['observation_ids']) | set(t['observation_ids']))
            final_paths[int(t['track_id'])] = t['points_path']
        self.track_nodes = [n['node'] for n in self.nodes if n['final_track_ids']]
        rawobs = a.jsonlines(self.input(self.arm / 'observations' / s / 'automatic_observations.jsonl'))
        assert rawobs == a.jsonlines(self.input(self.arm / 'hierarchy' / s / 'original_automatic_observations.jsonl'))
        actions = {int(r['observation_id']): r for r in a.jsonlines(self.input(self.arm / 'hierarchy' / s / 'observation_action_ledger.jsonl'))}
        frame_meta = {int(f['frame_index']): f for f in mf['frames']}
        self.fis = sorted(fi for fi, f in frame_meta.items() if int(f['frame_id']) in mf['arms']['D'])
        self.firow = {fi: j for j, fi in enumerate(self.fis)}
        self.frame_ids = [str(frame_meta[fi]['frame_id']) for fi in self.fis]
        vis, cachemeta = [], {}
        self.visibility_paths = []
        for fi in self.fis:
            fid = frame_meta[fi]['frame_id']
            cache = self.frontend / 'frame_cache/grounded_sam' / s / ('frame%06d' % int(fid))
            rc = a.read(self.input(cache / 'receipt.json'))
            rows = a.read(self.input(cache / 'observations.json'))
            assert self.used[str(cache / 'observations.json')] == rc['files']['observations.json']
            cachemeta[fi] = {r['point_indices_path']: r for r in rows}
            path = self.input(cache / 'visibility.npy')
            assert self.used[str(path)] == rc['files']['visibility.npy']
            v = np.load(path, allow_pickle=False)
            assert v.dtype == bool and v.shape == (self.n,)
            vis.append(v)
            self.visibility_paths.append(str(path))
        self.vis = np.stack(vis)
        self.observations, self.obs_points = [], []
        self.frame_rows = defaultdict(list)
        for o in sorted(rawobs, key=lambda x: x['observation_id']):
            oid, fi = int(o['observation_id']), int(o['frame_index'])
            local = cachemeta[fi][o['point_indices_path']]
            assert all(o[k] == local[k] for k in ('frame_id', 'frame_index', 'scene_name', 'mask_rle', 'predicted_iou', 'stability_score'))
            assert o['scene_name'] == s and str(o['frame_id']) == str(frame_meta[fi]['frame_id'])
            pp = a.points(self.input(o['point_indices_path']))
            assert self.vis[self.firow[fi], pp].all()
            row = len(self.observations)
            self.observations.append(dict(row=row, observation_id=oid, frame_index=fi,
                frame_id=str(o['frame_id']), scene_name=s, point_indices_path=o['point_indices_path'],
                point_file_sha256=self.used[o['point_indices_path']], cache_local_observation_id=local['observation_id'],
                raw_track_id=owner.get(oid, -1), final_track_id=final_owner.get(owner.get(oid)),
                hierarchy_action=actions[oid]['action'] if 'action' in actions[oid] else actions[oid],
                predicted_iou=o['predicted_iou'], stability_score=o['stability_score']))
            self.obs_points.append(pp)
            self.frame_rows[self.firow[fi]].append(row)
        sizes = np.array([len(p) for p in self.obs_points], np.int64)
        self.O = coo_matrix((np.ones(int(sizes.sum()), np.int32),
            (np.repeat(np.arange(len(sizes)), sizes), np.concatenate(self.obs_points))),
            shape=(len(sizes), self.n)).tocsc()
        nsizes = np.array([len(p) for p in self.points], np.int64)
        G = coo_matrix((np.ones(int(nsizes.sum()), np.int32),
            (np.concatenate(self.points), np.repeat(np.arange(len(nsizes)), nsizes))),
            shape=(self.n, len(nsizes))).tocsr()
        self.H = (self.O @ G).toarray().astype(np.int64)
        self.V = np.asarray([np.asarray(G[v].sum(axis=0)).ravel() for v in self.vis], dtype=np.int64)
        self.I = (G.T @ G).toarray().astype(np.int64)
        assert np.array_equal(np.diag(self.I), nsizes)
        self.obs_sizes = sizes
        self.ids = np.array([r['observation_id'] for r in self.observations], np.int64)
        self.owners = np.array([r['raw_track_id'] for r in self.observations], np.int64)
        self.dependence = self.dependencies()
        self.families = np.array([self.dependence['family_by_raw_track'].get(str(o), -1) for o in self.owners], np.int64)
        return self

    def dependencies(self):
        edges = defaultdict(list)
        csr = self.O.tocsr()
        for j, rows in self.frame_rows.items():
            row = csr[rows]
            intersections = (row @ row.T).toarray()
            for ii, left in enumerate(rows):
                ra = int(self.owners[left])
                if ra < 0:
                    continue
                for jj in range(ii+1, len(rows)):
                    right, rb = rows[jj], int(self.owners[rows[jj]])
                    if rb < 0 or ra == rb:
                        continue
                    hit = int(intersections[ii, jj])
                    iou = hit / max(1, int(self.obs_sizes[left]+self.obs_sizes[right]-hit))
                    if iou >= core.PARAMETERS['source_duplicate_iou']:
                        edges[tuple(sorted((ra, rb)))].append(dict(frame_index=self.fis[j], frame_id=self.frame_ids[j],
                            observation_ids=[int(self.ids[left]), int(self.ids[right])], iou=iou))
        parent = {int(t['track_id']): int(t['track_id']) for t in self.raw}
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        accepted = []
        for (left, right), records in sorted(edges.items()):
            if len({r['frame_index'] for r in records}) >= core.PARAMETERS['source_duplicate_frames']:
                l, r = root(left), root(right)
                parent[max(l, r)] = min(l, r)
                accepted.append(dict(raw_track_ids=[left, right], frames=records))
        return dict(family_by_raw_track={str(k): root(k) for k in sorted(parent)}, dependence_edges=accepted,
                    caveat='Dependence families are conservative shared-mask bookkeeping, not instances or independent identity certificates.')

    def region_counts(self, pp):
        return np.asarray(self.O[:, pp].sum(axis=1)).ravel().astype(np.int64), self.vis[:, pp].sum(axis=1).astype(np.int64)

    def competitors(self, left, right, ea, eb, union, covisible):
        result = []
        if not len(ea) or not len(eb):
            return result
        for n in self.nodes:
            c = n['node']
            if c in (left, right):
                continue
            cover_a = len(a.hit(self.points[c], ea))/len(ea)
            cover_b = len(a.hit(self.points[c], eb))/len(eb)
            side = 'A' if cover_a >= .8 and cover_b <= .1 else 'B' if cover_b >= .8 and cover_a <= .1 else None
            if side is None:
                continue
            outside = np.setdiff1d(self.points[c], union)
            if len(outside) < 20 or len(outside)/len(self.points[c]) < .2:
                continue
            oh, ov = self.region_counts(outside)
            sidepoints, otherpoints = (ea, eb) if side == 'A' else (eb, ea)
            sh, sv = self.region_counts(sidepoints)
            th, tv = self.region_counts(otherpoints)
            witnesses = []
            for j in covisible:
                if not core.reliable(int(ov[j]), len(outside)) or not core.reliable(int(self.V[j, c]), len(self.points[c])):
                    continue
                rows = np.asarray(self.frame_rows.get(j, []), int)
                fit = self.H[rows, c] / np.maximum(1, self.obs_sizes[rows]+self.V[j, c]-self.H[rows, c])
                good = (fit >= .6) & (sh[rows]/max(1, sv[j]) >= .6) & (th[rows]/max(1, tv[j]) <= .1) & (oh[rows]/max(1, ov[j]) >= .6)
                if good.any():
                    witnesses.append(dict(frame_index=self.fis[j], frame_id=self.frame_ids[j],
                        observation_ids=self.ids[rows[good]].tolist(), outside_visible=int(ov[j])))
            result.append(dict(node=c, sources=n['sources'], side=side, cover_A=cover_a, cover_B=cover_b,
                outside_points=len(outside), supported_conflict=len(witnesses) >= 2, frames=witnesses))
        return result


def propose(obj):
    points, nodes = obj.points, obj.nodes
    mins = [obj.xyz[p].min(axis=0) for p in points]
    maxs = [obj.xyz[p].max(axis=0) for p in points]
    trees, proposed, census = {}, [], Counter()
    for ai, left in enumerate(obj.track_nodes):
        for right in obj.track_nodes[ai+1:]:
            census['all_track_pairs'] += 1
            reasons = []
            inter = int(obj.I[left, right])
            if inter:
                distance = 0.
                reasons.append('shared_points')
            else:
                gap = np.maximum(0, np.maximum(mins[left]-maxs[right], mins[right]-maxs[left]))
                distance = None
                if np.linalg.norm(gap) <= .15:
                    small, large = (left, right) if len(points[left]) < len(points[right]) else (right, left)
                    if large not in trees:
                        trees[large] = cKDTree(obj.xyz[points[large]])
                    dist, _ = trees[large].query(obj.xyz[points[small]], k=1, distance_upper_bound=.15+1e-12, workers=1)
                    if np.isfinite(dist).any():
                        distance = float(dist.min())
                        reasons.append('spatial_near')
            contains = [n['node'] for n in nodes if n['node'] not in (left, right)
                and obj.I[n['node'], left] >= .5*len(points[left])
                and obj.I[n['node'], right] >= .5*len(points[right])]
            if contains:
                reasons.append('existing_containing_explanation')
            raw = []
            for j in range(len(obj.fis)):
                if not core.reliable(int(obj.V[j, left]), len(points[left])) or not core.reliable(int(obj.V[j, right]), len(points[right])):
                    continue
                rows = np.asarray(obj.frame_rows.get(j, []), int)
                good = (obj.H[rows, left] >= 20) & (obj.H[rows, right] >= 20) & (obj.H[rows, left]/obj.V[j, left] >= .3) & (obj.H[rows, right]/obj.V[j, right] >= .3)
                raw.extend(obj.ids[rows[good]].tolist())
            if raw:
                reasons.append('raw_common_mask')
            if reasons:
                pid = 'p%05d' % len(proposed)
                proposed.append(dict(pair_id=pid, left=left, right=right, proposal_reasons=reasons,
                    minimum_distance=distance, intersection=inter, containing_nodes=contains,
                    proposal_observation_ids=raw))
            else:
                census['not_proposed'] += 1
    return proposed, census


def generate():
    guard()
    check_contract()
    source_contract = a.read(SOURCE / 'contract.json')
    source_seal = a.read(SOURCE / 'source_evidence_frozen.json')
    totals, outputs = Counter(), {}
    for i, (scene, spec) in enumerate(source_contract['jobs'].items(), 1):
        print('GT_free', i, 52, scene, flush=True)
        a.reserve()
        obj = Scene(scene, spec, source_contract, source_seal).load()
        pairs, census = propose(obj)
        arrays = dict(mask_node_hits=obj.H, frame_node_visible=obj.V, node_intersections=obj.I,
            observation_ids=obj.ids, raw_track_owners=obj.owners, dependence_family_ids=obj.families,
            mask_sizes=obj.obs_sizes, frame_indices=np.array(obj.fis), frame_ids=np.array(obj.frame_ids))
        for n, pp in enumerate(obj.points):
            arrays['n%04d' % n] = pp
        frames_out = []
        for pair in pairs:
            pid, left, right = pair['pair_id'], pair['left'], pair['right']
            pa, pb = obj.points[left], obj.points[right]
            inter = a.hit(pa, pb)
            ea, eb = np.setdiff1d(pa, pb), np.setdiff1d(pb, pa)
            union = np.union1d(pa, pb)
            ih, iv = obj.region_counts(inter)
            arrays[pid + '_union'], arrays[pid + '_intersection'] = union, inter
            arrays[pid + '_intersection_inside'], arrays[pid + '_intersection_visible'] = ih, iv
            fr, covisible = [], []
            for j, fi in enumerate(obj.fis):
                va, vb = int(obj.V[j, left]-iv[j]), int(obj.V[j, right]-iv[j])
                row = dict(pair_id=pid, frame_index=fi, frame_id=obj.frame_ids[j],
                    exclusive_A_visible=va, exclusive_B_visible=vb,
                    exclusive_A_visible_fraction=va/max(1, len(ea)), exclusive_B_visible_fraction=vb/max(1, len(eb)))
                if not core.reliable(va, len(ea)) or not core.reliable(vb, len(eb)):
                    row['state'] = 'not_jointly_reliably_visible'
                else:
                    covisible.append(j)
                    rows = np.asarray(obj.frame_rows.get(j, []), int)
                    row.update(core.frame_hypotheses(obj.H[rows, left], obj.H[rows, right], ih[rows],
                        int(obj.V[j, left]), int(obj.V[j, right]), int(iv[j]), obj.obs_sizes[rows],
                        obj.ids[rows], obj.owners[rows], obj.nodes[left]['source_track_ids'],
                        obj.nodes[right]['source_track_ids'], obj.families[rows]))
                fr.append(row)
            competitors = obj.competitors(left, right, ea, eb, union, covisible)
            pair.update(exclusive_A_points=len(ea), exclusive_B_points=len(eb), union_points=len(union),
                left_track_ids=obj.nodes[left]['final_track_ids'], right_track_ids=obj.nodes[right]['final_track_ids'],
                competitors=competitors, **core.decide(fr, (len(ea), len(eb)), competitors))
            frames_out.extend(fr)
        core.resolve_partner_competition(pairs)
        original_codes = {a.digest(pp) for pp in obj.nt.values()}
        emitted, used_endpoints = set(), set()
        for pair in pairs:
            digest = a.digest(arrays[pair['pair_id'] + '_union'])
            pair['union_sha256'] = digest
            pair['duplicate_original_NT'] = digest in original_codes
            pair['duplicate_accepted_union'] = pair['state'] == 'accept' and digest in emitted
            pair['emit'] = pair['state'] == 'accept' and digest not in original_codes and digest not in emitted
            if pair['state'] == 'accept':
                assert not used_endpoints & {pair['left'], pair['right']}
                used_endpoints.update((pair['left'], pair['right']))
                emitted.add(digest)
        dest = OUT / 'predictions' / scene
        dest.mkdir(parents=True)
        np.savez_compressed(dest / 'data.npz', **arrays)
        for name, value in [('nodes', obj.nodes), ('observations', obj.observations), ('pairs', pairs),
                             ('pair_frames', frames_out), ('source_dependence', obj.dependence),
                             ('lineage', dict(raw_tracks=obj.raw, final_tracks=obj.final)),
                             ('visibility', [dict(frame_index=fi, frame_id=obj.frame_ids[j], path=obj.visibility_paths[j]) for j, fi in enumerate(obj.fis)])]:
            a.seal(dest / (name + '.json'), value)
        a.seal(dest / 'receipt.json', dict(complete=True, GT_used=False, inputs=obj.used,
            counts=dict(final_T_records=len(obj.final), final_T_geometries=len(obj.track_nodes),
                raw_tracks=len(obj.raw), observations=len(obj.observations), source_frames=len(obj.fis),
                pairs=len(pairs), emitted=sum(p['emit'] for p in pairs)), proposal_census=census))
        outputs.update({str(p): a.sha(p) for p in dest.iterdir()})
        totals.update(scenes=1, final_T_records=len(obj.final), final_T_geometries=len(obj.track_nodes),
            raw_tracks=len(obj.raw), observations=len(obj.observations), proposed_pairs=len(pairs),
            emitted=sum(p['emit'] for p in pairs), frame_pair_rows=len(frames_out), **census)
        totals.update(Counter(p['state'] for p in pairs))
        print('scene_counts', len(pairs), dict(Counter(p['state'] for p in pairs)), flush=True)
        del obj, arrays, frames_out
    assert totals['scenes'] == 52 and totals['final_T_records'] == 2300 and totals['observations'] == 51463
    for p in (OUT / 'code_snapshot').rglob('*'):
        if p.is_file():
            outputs[str(p)] = a.sha(p)
    for name in ('contract.json', 'tests.json'):
        outputs[str(OUT / name)] = a.sha(OUT / name)
    a.seal(OUT / 'predictions_frozen.json', dict(complete=True, GT_used=False,
        time_unix=time.time(), counts=totals, outputs=outputs))
    print('PREDICTIONS SEALED', dict(totals), flush=True)


def csv_seal(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows([{k: str(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows])


def track_identity(pp, gt, ids, sizes):
    import evaluate_fi1_nt_region_identity as met
    meta = met.identity(pp, gt, ids, sizes)
    if len(pp) >= 100:
        return meta
    hits = np.array([int((gt[pp] == g).sum()) for g in ids])
    scores = hits / np.maximum(1, len(pp)+sizes-hits)
    idx = int(scores.argmax()) if len(scores) and scores.max() > 0 else None
    target = int(ids[idx]) if idx is not None else None
    hit = int(hits[idx]) if idx is not None else 0
    stable = hit >= 20 and hit/len(pp) >= .6
    meta.update(stable=stable, target=target, index=idx, hits=hit, purity=hit/len(pp),
                scope='stable' if stable else meta['scope'])
    return meta


def evaluate():
    guard(False)
    c = check_contract()
    frozen = a.read(OUT / 'predictions_frozen.json')
    assert frozen['complete'] and frozen['GT_used'] is False
    verify_map(frozen['outputs'])
    import evaluate_fi1_nt_region_identity as met
    old = a.read(a.OLD / 'contract.json')
    sc = a.read(SOURCE / 'contract.json')
    allpairs, allnodes, targets, masks, sampled, gtinputs = [], [], [], [], [], {}

    def composition(pp, gt, target=None):
        ids, ns = np.unique(gt[pp], return_counts=True)
        return dict(points=len(pp), distribution={str(g): int(n) for g, n in zip(ids, ns)},
                    known_instances=int((ids >= 1000).sum()), unknown=int(ns[ids < 1000].sum()),
                    target=int((gt[pp] == target).sum()) if target is not None else None)

    for i, (scene, spec) in enumerate(sc['jobs'].items(), 1):
        print('postseal_GT', i, 52, scene, flush=True)
        path = spec['job']['GT']
        assert a.sha(path) == old['GT_inputs'][path]
        gtinputs[path] = a.sha(path)
        gt = np.loadtxt(path, dtype=np.int64)
        ids, sizes = met.valid_targets(gt, old['class_ids'])
        _, nt = diag.load_nt(spec['job'])
        baseline = met.pool_base(nt, gt, ids, sizes)
        best = {arm: baseline.copy() for arm in ('proposed', 'accepted')}
        dest = OUT / 'predictions' / scene
        nodes = a.read(dest / 'nodes.json')
        pairs = a.read(dest / 'pairs.json')
        frames = a.read(dest / 'pair_frames.json')
        frameindex = defaultdict(list)
        for f in frames:
            frameindex[f['pair_id']].append(f)
        with np.load(dest / 'data.npz') as archive:
            z = {k: archive[k] for k in archive.files}
        meta = {n['node']: track_identity(z['n%04d' % n['node']], gt, ids, sizes) for n in nodes if n['final_track_ids']}
        for n in nodes:
            if n['final_track_ids']:
                allnodes.append(dict(scene=scene, node=n['node'], final_track_ids=n['final_track_ids'],
                    in_original_NT=n['in_original_NT'], **meta[n['node']], composition=composition(z['n%04d' % n['node']], gt)))
        scene_rows = []
        witness_ids = set()
        for p in pairs:
            left, right = p['left'], p['right']
            ml, mr = meta[left], meta[right]
            pa, pb = z['n%04d' % left], z['n%04d' % right]
            pp = z[p['pair_id'] + '_union']
            scores = met.ious(pp, gt, ids, sizes)
            relation = 'same' if ml['stable'] and mr['stable'] and ml['target'] == mr['target'] else 'different' if ml['stable'] and mr['stable'] else 'unresolved'
            target = ml['target'] if relation == 'same' else None
            target_index = ml['index'] if relation == 'same' else None
            union_gt = composition(pp, gt, target)
            purity = union_gt['target']/len(pp) if target is not None else None
            same_delta = float(scores[target_index]-baseline[target_index]) if target is not None else None
            loss = []
            for endpoint, m in [('left', ml), ('right', mr)]:
                if m['stable'] and m['parent_iou'] >= .9:
                    loss.append(dict(endpoint=endpoint, target=m['target'], parent_iou=m['parent_iou'],
                        union_iou=float(scores[m['index']]), loss=float(m['parent_iou']-scores[m['index']])))
            r = dict(scene=scene, pair_id=p['pair_id'], left=left, right=right,
                left_track_ids=p['left_track_ids'], right_track_ids=p['right_track_ids'],
                state=p['state'], reason=p['reason'], emit=p['emit'],
                left_scope=ml['scope'], right_scope=mr['scope'], relation=relation,
                left_target=ml['target'] if ml['stable'] else None,
                right_target=mr['target'] if mr['stable'] else None,
                same_target=target, union_GT=union_gt, union_target_purity=purity,
                same_target_pool_delta=same_delta,
                safe_gain02=bool(relation == 'same' and purity >= .9 and same_delta >= .02-1e-12),
                any_target_gain02=bool((scores-baseline >= .02-1e-12).any()),
                any_target_gain02_ids=ids[scores-baseline >= .02-1e-12].tolist(),
                ambiguous_endpoint=ml['scope'] == 'ambiguous' or mr['scope'] == 'ambiguous',
                mixed_endpoint=any(sum(n >= 20 for g, n in m['distribution'].items() if int(g) >= 1000) >= 2 for m in (ml, mr)),
                good_endpoint_losses=loss, good_endpoint_loss_gt02=any(x['loss'] > .02+1e-12 for x in loss),
                joint_source_families=p['joint_source_families'],
                joint_only_one_source_family=p['joint_only_one_source_family'],
                merge_winning_frames=p['merge_winning_frames'], separate_winning_frames=p['separate_winning_frames'],
                supported_competitor_nodes=p['supported_competitor_nodes'],
                duplicate_original_NT=p['duplicate_original_NT'])
            scene_rows.append(r)
            best['proposed'] = np.maximum(best['proposed'], scores)
            if p['emit']:
                best['accepted'] = np.maximum(best['accepted'], scores)
            for f in frameindex[p['pair_id']]:
                if f['state'] == 'not_jointly_reliably_visible':
                    continue
                witness_ids.update(f['joint_observation_ids'])
                witness_ids.update(x for x in f['separate_observation_ids'] if x is not None)
        observations = {r['observation_id']: r for r in a.read(dest / 'observations.json')}
        for oid in sorted(witness_ids):
            o = observations[oid]
            assert a.sha(o['point_indices_path']) == o['point_file_sha256']
            pp = a.points(o['point_indices_path'])
            masks.append(dict(scene=scene, observation_id=oid, raw_track_id=o['raw_track_id'],
                              composition=composition(pp, gt), cross_known_instances=len(np.unique(gt[pp][gt[pp] >= 1000])) > 1))
        for parent in spec['parents']:
            node = next(n['node'] for n in nodes if parent['track_id'] in n['final_track_ids'])
            m = meta[node]
            linked = [p for p in pairs if node in (p['left'], p['right'])]
            accepted = [p for p in linked if p['state'] == 'accept']
            assert len(accepted) <= 1
            row = dict(scene=scene, candidate_id=parent['candidate_id'], node=node,
                scope=m['scope'], stable=m['stable'], target=m['target'] if m['stable'] else None,
                proposed_partners=len(linked), accepted_pairs=[p['pair_id'] for p in accepted],
                emitted=bool(accepted and accepted[0]['emit']))
            if accepted and m['stable']:
                score = met.ious(z[accepted[0]['pair_id'] + '_union'], gt, ids, sizes)[m['index']]
                row.update(parent_delta=float(score-m['parent_iou']), complete_NT_delta=float(score-baseline[m['index']]))
            sampled.append(row)
        a.seal(OUT / 'evaluation' / (scene + '_pairs.json'), scene_rows)
        allpairs += scene_rows
        targets += met.pool_rows(scene, 'NT', baseline, baseline, ids)
        for arm in best:
            targets += met.pool_rows(scene, arm, baseline, best[arm], ids)
    accepted = [r for r in allpairs if r['state'] == 'accept']
    emitted = [r for r in accepted if r['emit']]
    same = [r for r in accepted if r['relation'] == 'same']
    safe = [r for r in emitted if r['safe_gain02']]
    unknown = sum(r['union_GT']['unknown'] for r in emitted)
    denom = sum(r['union_GT']['points'] for r in emitted)
    gates = dict(safe_gain_scenes=len({r['scene'] for r in safe}) >= 2,
        pair_identity_precision=bool(accepted) and len(same)/len(accepted) >= .9,
        no_confirmed_false_connection=not any(r['relation'] == 'different' for r in accepted),
        no_unresolved_accept=not any(r['relation'] == 'unresolved' for r in accepted),
        good_copy_safety=not any(r['good_endpoint_loss_gt02'] for r in accepted),
        annotation_unknown=(unknown/max(1, denom)) <= .1)
    pool = {arm: met.pool_summary([r for r in targets if r['mode'] == arm]) for arm in ('NT', 'proposed', 'accepted')}
    for arm in pool:
        pool[arm]['gain02_scenes'] = sorted({r['scene'] for r in targets if r['mode'] == arm and r['gain02']})
    confusion = {state: dict(Counter(r['relation'] for r in allpairs if r['state'] == state)) for state in ('accept', 'reject', 'unknown')}
    summary = dict(complete=True, prediction_seal_sha256=a.sha(OUT / 'predictions_frozen.json'), GT_inputs=gtinputs,
        counts=frozen['counts'], node_scopes=dict(Counter(r['scope'] for r in allnodes)),
        decision_relation_census=confusion, decision_reasons=dict(Counter(r['reason'] for r in allpairs)),
        accepted=len(accepted), emitted=len(emitted), accepted_resolved_same=len(same),
        accepted_confirmed_different=sum(r['relation'] == 'different' for r in accepted),
        accepted_unresolved=sum(r['relation'] == 'unresolved' for r in accepted),
        accepted_identity_precision=len(same)/len(accepted) if accepted else None,
        accepted_good_copy_loss_gt02=sum(r['good_endpoint_loss_gt02'] for r in accepted),
        accepted_unknown_point_fraction=unknown/max(1, denom),
        safe_gain02_pairs=len(safe), safe_gain02_targets=len({(r['scene'], r['same_target']) for r in safe}),
        safe_gain02_scenes=sorted({r['scene'] for r in safe}), pool=pool,
        proposed_safe_supply_pairs=sum(r['safe_gain02'] for r in allpairs),
        proposed_safe_supply_scenes=sorted({r['scene'] for r in allpairs if r['safe_gain02']}),
        proposed_safe_supply_by_state=dict(Counter(r['state'] for r in allpairs if r['safe_gain02'])),
        mixed_endpoint_pairs=sum(r['mixed_endpoint'] for r in allpairs),
        witness_masks=len(masks), witness_cross_known_masks=sum(r['cross_known_instances'] for r in masks),
        sampled_parent_actions=sum(r['emitted'] for r in sampled),
        gates=gates, pass_all=all(gates.values()),
        decision='eligible_for_separate_isolated_validation_proposal' if all(gates.values()) else 'stop_fixed_dual_hypothesis_rule',
        interpretation='IoU supply/identity diagnostics on exposed scenes. No AP, no replacement, no global-instance-tree or full-geometry upper-bound claim.')
    for name, rows in [('pairs', allpairs), ('nodes', allnodes), ('targets', targets), ('witness_masks', masks), ('sampled_parents', sampled)]:
        a.seal(OUT / 'evaluation' / (name + '.json'), rows)
        csv_seal(OUT / (name + '.csv'), rows)
    a.seal(OUT / 'evaluation' / 'summary.json', summary)
    outputs = {str(p): a.sha(p) for p in (OUT / 'evaluation').iterdir()}
    outputs.update({str(p): a.sha(p) for p in OUT.glob('*.csv')})
    a.seal(OUT / 'evaluation_frozen.json', dict(complete=True, GT_used=True, time_unix=time.time(),
        prediction_seal_sha256=summary['prediction_seal_sha256'], outputs=outputs))
    print('EVALUATED', {k: summary[k] for k in ('decision', 'accepted', 'emitted', 'accepted_resolved_same',
        'accepted_confirmed_different', 'accepted_unresolved', 'safe_gain02_scenes', 'gates', 'pool')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'generate', 'evaluate'))
    args = parser.parse_args()
    globals()[args.stage]()
