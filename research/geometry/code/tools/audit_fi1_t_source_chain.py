#!/usr/bin/env python3
"""Read-only upstream provenance reconstruction; evidence is sealed before GT."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / 'research/geometry/experiments'
OLD = EXP / 'fi1_nt_instance_ownership_prototype_exposed52_20261015_v1'
BLOCKS = EXP / 'fi1_local_block_attribution_selection_diagnostic_exposed52_20261015_v1'
PARENTS = EXP / 'fi1_query_relation_diagnostic_exposed52_20261014_v1'
OUT = EXP / 'fi1_t_source_chain_audit_exposed52_20261015_v1'
SOURCE_REV = '27b7becd33'
SOURCE_NAMES = ('run_fi1_frontend_comparison.py', 'build_details_frame_siou_tracks.py',
                'refine_details_automatic_tracks_consensus.py',
                'build_details_iterative_proposal_merges.py',
                'build_details_same_frame_hierarchy_preprocessor.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(points):
    return hashlib.sha256(np.asarray(points, dtype='<i8').tobytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [plain(v) for v in value]
    return value


def seal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(plain(value), ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    with path.open('x') as f:
        f.write(content)


def jsonlines(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def reserve():
    assert shutil.disk_usage(ROOT).free >= 50 * 2**30, '50GiB disk reserve'


def guard():
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        p = os.fsdecode(args[0])
        if any(word in p for word in ('ground_truth', '/evaluation/', '/diagnosis/',
                                     '/failure_diagnosis', 'parent_scopes.json')):
            raise PermissionError('GT forbidden before evidence seal: ' + p)
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if writing and not Path(p).resolve().is_relative_to(OUT.resolve()):
            raise PermissionError('Only new audit directory is writable: ' + p)
    sys.addaudithook(hook)


def points(path, key='point_indices'):
    with np.load(path, allow_pickle=False) as z:
        p = np.asarray(z[key], dtype=np.int64)
    assert p.ndim == 1 and len(p) == len(np.unique(p))
    return np.sort(p)


def union(chunks):
    chunks = list(chunks)
    return np.unique(np.concatenate(chunks)) if chunks else np.empty(0, np.int64)


def hit(a, b):
    if len(a) > len(b):
        a, b = b, a
    loc = np.searchsorted(b, a)
    ok = loc < len(b)
    ok[ok] &= b[loc[ok]] == a[ok]
    return a[ok]


def prepare():
    guard()
    reserve()
    assert not OUT.exists()
    c = read(OLD / 'contract.json')
    jobs = {}
    roots = set()
    for s, j in c['jobs'].items():
        geometry = {g['geometry_key']: g for g in read(j['binding']['geometry'])}
        parents = read(PARENTS / 'relations' / s / 'parents.json')
        rows = []
        for p in parents:
            g = geometry[p['parent_geometry']]
            aliases = [m for m in g['legacy_members'] if m['candidate_source'] == 'track']
            assert len(aliases) == 1, (s, p['candidate_id'], 'ambiguous locator')
            m = aliases[0]
            locator = m['geometry_locator']
            root = Path(locator['points_path']).parent.parent
            assert root.name == s and root.parent.name == 'merged'
            roots.add(str(root.parents[3]))
            rows.append(dict(candidate_id=p['candidate_id'], geometry_key=g['geometry_key'],
                             track_id=m['candidate_id'], locator=locator,
                             points_sha256=g['points_sha256'], source_root=str(root)))
        jobs[s] = dict(job=j, parents=rows)
    assert sum(len(j['parents']) for j in jobs.values()) == 207 and len(jobs) == 52
    assert len(roots) == 1
    frontend = Path(next(iter(roots)))
    original = read(frontend / 'contract.json')
    source_manifest = {}
    for name in SOURCE_NAMES:
        oldpath = str(ROOT / 'tools' / name)
        payload = subprocess.check_output(['git', 'show', SOURCE_REV + ':tools/' + name], cwd=ROOT)
        assert hashlib.sha256(payload).hexdigest() == original['assets'][oldpath]
        source_manifest[name] = dict(original_path=oldpath, original_sha256=original['assets'][oldpath],
                                     current_path=str(Path(oldpath).resolve()), current_sha256=sha(oldpath))
        dest = OUT / 'historical_source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write(payload)
    pilot = [next(s for s in sorted(jobs) if jobs[s]['job']['cohort'] == cohort)
             for cohort in ('fit12', 'exposed16', 'internal24')]
    seal(OUT / 'contract.json', dict(version=OUT.name, jobs=jobs, pilot=pilot,
        frontend=str(frontend), source_revision=SOURCE_REV, source_manifest=source_manifest,
        source_contract_sha256=sha(frontend / 'contract.json'),
        prior_contract_sha256=sha(OLD / 'contract.json'),
        counts=dict(scenes=52, parents=207, blocks=1308),
        scope='All original parents/blocks; no GT subset selection; no production writes.',
        observation_identity='Full path + scene + global arm observation ID + original frame ID/index + point file hash. Frame-cache local IDs linked by exact path, not ID.',
        replay='Existing cached visibility and masks only. Independently reconstruct D1 initial/kept SP, every logged merge input/output and frozen relation. Validate exact saved sets/counts before attributing loss.',
        reliability='Store exact raw point hits and hits intersecting cached depth visibility; existing lifted-SP r>=.1 and c>=.3 separately. No claim that mask existence establishes instance identity.',
        absence='No hit in existing G10 bank is not proof no useful unobserved frame exists. Occlusion versus bad depth is unresolved without corresponding records.',
        association='Saved track memberships and node/frame IDs observed. Pairwise greedy assignment rejection reasons were not logged; do not invent them.',
        references=['lineage_observed', 'd1_initial', 'merge_input', 'other_tracks', 'all_raw'],
        references_status='Diagnostic raw restorations in original editable domain, not proposed safe outputs. Seal before GT; report actual all-block unions and point-target upper bounds separately.',
        limits=dict(training=False, new_frames=False, threshold_search=False, AP=False,
                    production_modified=False, old_seals_modified=False)))
    seal(OUT / 'prepared.json', dict(complete=True, GT_used=False, pilot=pilot, parents=207))
    print('prepared', pilot, flush=True)


class Scene:
    def __init__(self, scene, spec, contract):
        self.s, self.spec = scene, spec
        self.j = spec['job']
        self.frontend = Path(contract['frontend'])
        self.arm = self.frontend / 'arms/D'
        self.used = {}
        self.receipts = {}

    def track(self, path, expected=None):
        path = Path(path)
        h = sha(path)
        if expected is not None:
            assert h == expected, ('hash mismatch', str(path), h, expected)
        self.used[str(path)] = h
        return path

    def saved(self, stage, name):
        if stage not in self.receipts:
            rp = self.arm / 'stage_receipts' / (self.s + '_' + stage + '.json')
            self.receipts[stage] = read(self.track(rp))
        path = self.arm / stage / self.s / name
        self.track(path, self.receipts[stage]['files'][name])
        return path

    def load(self):
        s = self.s
        self.manifest = read(self.track(self.frontend / 'manifests' / (s + '.json')))
        mf = self.manifest
        assert mf['scene_name'] == s
        proc = mf['files']['processed']
        assert Path(proc['path']).resolve() == Path(self.j['paths']['processed']).resolve()
        data = np.load(self.track(proc['path'], proc['sha256']), mmap_mode='r')
        assert len(data) == self.j['binding']['point_count'] and data.shape[1] >= 10
        self.sp = np.asarray(data[:, 9], dtype=np.int64)
        self.n, self.ns = len(data), int(self.sp.max()) + 1
        assert self.sp.min() >= 0 and np.array_equal(self.sp, data[:, 9])
        self.sizes = np.bincount(self.sp, minlength=self.ns)
        mesh = mf['files']['mesh']
        ply = PlyData.read(str(self.track(mesh['path'], mesh['sha256'])))['vertex']
        xyz = np.column_stack([ply[x] for x in ('x', 'y', 'z')])
        assert xyz.shape == data[:, :3].shape and np.allclose(xyz, data[:, :3], rtol=0, atol=1e-5)
        self.frames = {int(f['frame_index']): f for f in mf['frames']}
        assert list(self.frames) == list(range(len(self.frames)))
        self.allobs = jsonlines(self.saved('hierarchy', 'original_automatic_observations.jsonl'))
        kept = jsonlines(self.saved('hierarchy', 'automatic_observations.jsonl'))
        actions = jsonlines(self.saved('hierarchy', 'observation_action_ledger.jsonl'))
        self.hactions = {int(r['observation_id']): r for r in actions}
        self.keptids = {int(r['observation_id']) for r in kept}
        self.obs = {int(r['observation_id']): r for r in self.allobs}
        assert len(self.obs) == len(self.allobs) and set(self.obs) == set(self.hactions)
        assert self.allobs == jsonlines(self.track(self.arm / 'observations' / s / 'automatic_observations.jsonl'))
        assert all(self.obs[int(r['observation_id'])] == r for r in kept)
        self.rawtracks = read(self.saved('tracks', 'automatic_tracks.json'))['tracks']
        self.d1 = read(self.saved('consensus', 'automatic_tracks.json'))['tracks']
        self.final = read(self.saved('merged', 'automatic_tracks.json'))['tracks']
        self.actions = jsonlines(self.saved('merged', 'merge_actions.jsonl'))
        self.rounds = jsonlines(self.saved('merged', 'merge_rounds.jsonl'))
        self.relations = jsonlines(self.saved('merged', 'merge_round_relations.jsonl'))
        self.owner = {}
        for t in self.rawtracks:
            for oid in t['observation_ids']:
                assert oid not in self.owner and oid in self.keptids
                self.owner[oid] = int(t['track_id'])
        self.finalowner = {int(tid): int(t['track_id']) for t in self.final for tid in t['source_track_ids']}
        assert len(self.finalowner) == len(self.d1)
        self.obspts, self.visible, self.vcounts, self.ocols = {}, {}, {}, {}
        chosen = set(map(str, mf['arms']['D']))
        frames = [fi for fi, f in self.frames.items() if str(f['frame_id']) in chosen]
        self.frame_cache_receipts = {}
        for fi in frames:
            frame = self.frames[fi]
            fid = str(frame['frame_id'])
            cache = self.frontend / 'frame_cache/grounded_sam' / s / ('frame%06d' % int(fid))
            rc = read(self.track(cache / 'receipt.json'))
            local = read(self.track(cache / 'observations.json', rc['files']['observations.json']))
            bypath = {r['point_indices_path']: r for r in local}
            self.frame_cache_receipts[fi] = (cache, rc, bypath)
            vis = np.load(self.track(cache / 'visibility.npy', rc['files']['visibility.npy']), allow_pickle=False)
            assert vis.dtype == bool and vis.shape == (self.n,)
            self.visible[fi] = vis
            self.vcounts[fi] = np.bincount(self.sp[vis], minlength=self.ns)
        for r in self.allobs:
            oid, fi = int(r['observation_id']), int(r['frame_index'])
            assert r['scene_name'] == s and str(r['frame_id']) == str(self.frames[fi]['frame_id'])
            assert int(r['frame_id']) in mf['arms']['D']
            cache, rc, bypath = self.frame_cache_receipts[fi]
            path = Path(r['point_indices_path'])
            assert path.parent == cache
            local = bypath[str(path)]
            assert all(local[k] == r[k] for k in ('scene_name', 'frame_id', 'frame_index',
                                                  'mask_rle', 'predicted_iou', 'stability_score'))
            pp = points(self.track(path, rc['files'][path.name]))
            assert not len(pp) or (pp.min() >= 0 and pp.max() < self.n)
            assert self.visible[fi][pp].all(), ('mask contains nonvisible point', s, oid)
            self.obspts[oid] = pp
            self.ocols[oid] = np.bincount(self.sp[pp], minlength=self.ns)
        # Node IDs are local to post-hierarchy observations and exclude empty/zero-visible-SP masks.
        nodes = []
        for r in kept:
            fi, oid = int(r['frame_index']), int(r['observation_id'])
            if len(self.obspts[oid]) and np.any(self.vcounts[fi] / np.maximum(1, self.sizes) >= .1):
                nodes.append(oid)
        for t in self.rawtracks:
            ids = list(map(int, t['observation_ids']))
            assert [nodes[int(k)] for k in t['node_ids']] == ids
            assert [str(self.obs[k]['frame_id']) for k in ids] == list(map(str, t['frame_ids']))
            actual = points(self.saved('tracks', Path(t['points_path']).relative_to(self.arm / 'tracks' / s).as_posix()))
            assert np.array_equal(actual, union(self.obspts[k] for k in ids))
        self.initial, self.d1keep, self.diagnostics = {}, {}, {}
        saved_d1 = {int(t['track_id']): t for t in self.d1}
        for t in self.rawtracks:
            tid = int(t['track_id'])
            initial, keptsp, diag = self.consensus(t['observation_ids'])
            self.initial[tid], self.d1keep[tid], self.diagnostics[tid] = initial, keptsp, diag
            if len(keptsp):
                st = saved_d1[tid]
                assert set(st['superpoint_ids']) == set(keptsp)
                assert st['initial_superpoint_count'] == len(initial)
                assert st['removed_superpoint_count'] == len(set(initial) - set(keptsp))
                savedpts = points(self.saved('consensus', 'track_points/' + Path(st['points_path']).name))
                assert np.array_equal(savedpts, self.sppoints(keptsp))
            else:
                assert tid not in saved_d1
        assert {k for k, v in self.d1keep.items() if len(v)} == set(saved_d1)
        self.replay_merges()
        # Bound-domain identity of every selected parent, not just its integer track ID.
        self.track(self.j['binding']['geometry'])
        self.track(self.j['binding']['points'])
        self.parentmeta = read(self.track(PARENTS / 'relations' / s / 'parents.json'))
        with np.load(self.j['binding']['points']) as z:
            geom = {g['geometry_key']: g for g in read(self.j['binding']['geometry'])}
            for p in self.spec['parents']:
                loc = p['locator']
                pp = points(self.track(loc['points_path']), loc['array_key'])
                g = geom[p['geometry_key']]
                assert digest(pp) == p['points_sha256'] == g['points_sha256']
                assert np.array_equal(pp, z[g['array_key']])
                final = next(t for t in self.final if int(t['track_id']) == p['track_id'])
                assert Path(final['points_path']) == Path(loc['points_path'])
                assert np.array_equal(pp, self.sppoints(final['superpoint_ids']))
        assert len({r['identity'] for r in self.receipts.values()}) == 1
        return self

    def sppoints(self, ids):
        return np.flatnonzero(np.isin(self.sp, np.asarray(list(ids), np.int64)))

    def consensus(self, ids, candidates=None):
        byframe = defaultdict(list)
        for oid in sorted(set(map(int, ids))):
            byframe[int(self.obs[oid]['frame_index'])].append(oid)
        ins, vis = [], []
        for fi, oids in sorted(byframe.items()):
            pp = union(self.obspts[k] for k in oids)
            ins.append(np.bincount(self.sp[pp], minlength=self.ns))
            vis.append(self.vcounts[fi])
        inside, visible = np.asarray(ins), np.asarray(vis)
        ratio = inside / np.maximum(1, visible)
        if candidates is None:
            initial = np.flatnonzero(((visible / np.maximum(1, self.sizes) >= .1)
                                     & (visible > 0) & (ratio >= .3)).any(axis=0))
        else:
            initial = np.array(sorted(set(candidates)), np.int64)
        v = visible >= 3
        support = v & (ratio >= .5)
        vc, sc = v.sum(axis=0), support.sum(axis=0)
        cr = sc / np.maximum(1, vc)
        mean = (ratio * support).sum(axis=0) / np.maximum(1, sc)
        kept = initial[(sc[initial] >= 2) & (cr[initial] >= .3) & (mean[initial] >= .55)]
        diag = {int(k): dict(support_frames=int(sc[k]), visible_frames=int(vc[k]),
                            consensus_rate=float(cr[k]), mean_supported_coverage=float(mean[k]),
                            failed=[name for name, fail in [('support_frames', sc[k] < 2),
                                ('consensus_rate', cr[k] < .3), ('mean_supported_coverage', mean[k] < .55)] if fail])
                for k in initial}
        return initial, kept, diag

    def replay_merges(self):
        active = {int(t['track_id']): dict(sp=set(t['superpoint_ids']),
                  obs=set(t['observation_ids']), lineage={int(t['track_id'])}) for t in self.d1}
        self.events, self.roundstates = [], {}
        for rr in self.rounds:
            ri = int(rr['round_index'])
            self.roundstates[ri] = {k: dict(sp=set(v['sp']), lineage=set(v['lineage'])) for k, v in active.items()}
            relations = [r for r in self.relations if int(r['round_index']) == ri]
            assert len(relations) == len(active) * (len(active) - 1) // 2
            for r in relations:
                a, b = active[r['left_proposal_id']]['sp'], active[r['right_proposal_id']]['sp']
                inter = int(self.sizes[list(a & b)].sum())
                na, nb = int(self.sizes[list(a)].sum()), int(self.sizes[list(b)].sum())
                iou = inter / max(1, na + nb - inter)
                assert inter == r['point_intersection_count'] and na == r['left_point_count'] and nb == r['right_point_count']
                assert abs(iou - r['point_iou']) < 1e-12 and (iou > .3) == r['details_merge_eligible_observed']
            for a in [a for a in self.actions if a['round_index'] == ri]:
                left, right = a['anchor_proposal_id'], a['absorbed_proposal_id']
                aa, bb = active[left], active[right]
                assert aa['lineage'] == set(a['anchor_lineage_before']) and bb['lineage'] == set(a['absorbed_lineage'])
                initial = aa['sp'] | bb['sp']
                ids = aa['obs'] | bb['obs']
                _, kept, diag = self.consensus(ids, initial)
                ks = set(map(int, kept))
                assert len(initial) == a['union_superpoint_count'] and len(ks) == a['refined_superpoint_count']
                assert len(initial - ks) == a['removed_by_refinement_count']
                assert int(self.sizes[list(ks)].sum()) == a['refined_point_count']
                lineage = aa['lineage'] | bb['lineage']
                assert lineage == set(a['anchor_lineage_after'])
                self.events.append(dict(action=a, input=initial, output=ks, removed=initial - ks,
                                        lineage=lineage, diagnostics=diag))
                active[left] = dict(sp=ks, obs=ids, lineage=lineage)
                del active[right]
        assert set(active) == {int(t['track_id']) for t in self.final}
        for t in self.final:
            tid = int(t['track_id'])
            assert active[tid]['sp'] == set(t['superpoint_ids'])
            assert active[tid]['obs'] == set(t['observation_ids'])
            assert active[tid]['lineage'] == set(t['source_track_ids']) == set(t['lineage_proposal_ids'])

    def receipt(self):
        return dict(scene=self.s, complete=True, GT_used=False,
            checks=dict(processed_hash=True, mesh_xyz_same_order=True, point_index_domain=True,
                        global_to_cache_observation_by_path=True, frame_id_and_index=True,
                        point_files_and_visibility_hashes=True, node_to_observation=True,
                        raw_track_union_exact=True, all_D1_sets_replayed=True,
                        all_merge_actions_and_round_relations_replayed=True,
                        sampled_parent_locator_exact=True),
            counts=dict(parents=len(self.spec['parents']), original_observations=len(self.obs),
                        kept_observations=len(self.keptids), raw_tracks=len(self.rawtracks),
                        D1_tracks=len(self.d1), final_tracks=len(self.final),
                        merge_actions=len(self.events), merge_relations=len(self.relations)),
            inputs=self.used,
            limitations=['Association pairwise rejection/competition reasons not recorded.',
                         'Cached visibility cannot separate occlusion from depth/projection failure.',
                         'Raw observations and N are not instance truth.'])


def pilot():
    guard()
    reserve()
    c = read(OUT / 'contract.json')
    receipts = []
    for s in c['pilot']:
        print('pilot_alignment', s, flush=True)
        obj = Scene(s, c['jobs'][s], c).load()
        receipt = obj.receipt()
        seal(OUT / 'pilot' / (s + '.json'), receipt)
        receipts.append(receipt)
        del obj
    seal(OUT / 'pilot_passed.json', dict(complete=True, GT_used=False,
        scenes=c['pilot'], checks_passed=True, counts=dict(sum_counters(r['counts'] for r in receipts)),
        outputs={str(OUT / 'pilot' / (s + '.json')): sha(OUT / 'pilot' / (s + '.json')) for s in c['pilot']}))
    print('pilot_passed', flush=True)


def sum_counters(rows):
    total = Counter()
    for row in rows:
        total.update(row)
    return total


def export_scene(obj):
    s = obj.s
    d = OUT / 'evidence' / s
    parentrows, blockrows, obsrows, pairrows = [], [], [], []
    arrays = {}
    refs = read(obj.track(OLD / 'predictions' / s / 'blocks.json'))
    ftracks = {int(t['track_id']): t for t in obj.final}
    blockfile = obj.track(BLOCKS / 'predictions' / s / 'data.npz')
    predparents = {r['candidate_id']: r for r in obj.parentmeta}
    with np.load(blockfile) as z:
        for p in obj.spec['parents']:
            cid, tid = p['candidate_id'], int(p['track_id'])
            t = ftracks[tid]
            parent = obj.sppoints(t['superpoint_ids'])
            assert predparents[cid]['parent_geometry'] == p['geometry_key']
            lineage, oids = set(t['source_track_ids']), set(t['observation_ids'])
            raw = union(obj.obspts[k] for k in oids)
            initial = obj.sppoints(set().union(*(set(obj.initial[k]) for k in lineage)))
            kept = obj.sppoints(set().union(*(set(obj.d1keep[k]) for k in lineage)))
            events = [e for e in obj.events if e['lineage'] <= lineage]
            everinput = obj.sppoints(set().union(*(e['input'] for e in events))) if events else np.empty(0, np.int64)
            removed = obj.sppoints(set().union(*(e['removed'] for e in events))) if events else np.empty(0, np.int64)
            arrays[cid + '_parent'] = parent
            parentrows.append(dict(p, observation_ids=sorted(oids), source_track_ids=sorted(lineage),
                node_ids=t['node_ids'], frame_ids=t['frame_ids'],
                final_superpoint_ids=t['superpoint_ids'],
                original_track_records=[r for r in obj.rawtracks if int(r['track_id']) in lineage],
                D1_records=[r for r in obj.d1 if int(r['track_id']) in lineage],
                merge_actions=[e['action'] for e in events],
                raw_union_points=len(raw), D1_initial_points=len(initial), D1_kept_points=len(kept),
                exact_D1_initial={k:obj.initial[k] for k in sorted(lineage)},
                D1_removed_SP_diagnostics={k:{sp:v for sp,v in obj.diagnostics[k].items()
                                          if sp not in set(obj.d1keep[k])} for k in sorted(lineage)}))
            selected = [b for b in refs if b['candidate_id'] == cid]
            for b in selected:
                bid = b['block_id']
                block = z[bid + '_block']
                assert not len(hit(parent, block))
                arrays[bid + '_block'] = block
                arrays[bid + '_clip'] = z[bid + '_clipped_proposal']
                stages = dict(lineage_observed=hit(block, raw), d1_initial=hit(block, initial),
                              d1_kept=hit(block, kept), merge_input=hit(block, everinput),
                              merge_removed=hit(block, removed), final_T=hit(block, parent))
                allraw, posthierarchy, others, unassigned = [], [], [], []
                onblock, lineagehits, otherhits = [], [], []
                for oid, obs in obj.obs.items():
                    bp = hit(block, obj.obspts[oid])
                    if not len(bp):
                        continue
                    pp = hit(parent, obj.obspts[oid])
                    fi = int(obs['frame_index'])
                    spids = np.unique(obj.sp[bp])
                    vc = obj.vcounts[fi][spids]
                    lifted = spids[(vc / obj.sizes[spids] >= .1) & (vc > 0)
                                   & (obj.ocols[oid][spids] / np.maximum(1, vc) >= .3)]
                    owner = obj.owner.get(oid)
                    destination = obj.finalowner.get(owner)
                    row = dict(scene=s, candidate_id=cid, block_id=bid,
                        observation_id=oid, observation_namespace=str(obj.arm / 'observations' / s),
                        frame_id=obs['frame_id'], frame_index=fi,
                        point_indices_path=obs['point_indices_path'], point_file_sha256=obj.used[obs['point_indices_path']],
                        block_hits=len(bp), block_coverage=len(bp)/len(block),
                        block_visible=int(obj.visible[fi][block].sum()),
                        parent_hits=len(pp), parent_coverage=len(pp)/len(parent),
                        parent_visible=int(obj.visible[fi][parent].sum()),
                        lifted_block_superpoints=lifted,
                        raw_track_id=owner, final_track_id=destination,
                        in_parent_lineage=oid in oids, hierarchy_action=obj.hactions[oid]['action'],
                        representative_observation_id=obj.hactions[oid]['representative_observation_id'],
                        predicted_iou=obs['predicted_iou'], stability_score=obs['stability_score'])
                    obsrows.append(row)
                    onblock.append(oid)
                    allraw.append(bp)
                    if oid in obj.keptids:
                        posthierarchy.append(bp)
                    if owner is not None and owner not in lineage:
                        others.append(bp)
                        otherhits.append(oid)
                    if owner is None and oid in obj.keptids:
                        unassigned.append(bp)
                    if oid in oids:
                        lineagehits.append(oid)
                stages.update(all_raw=union(allraw), post_hierarchy=union(posthierarchy),
                              other_tracks=union(others), kept_unassigned=union(unassigned))
                for name, pp in stages.items():
                    arrays[bid + '_' + name] = pp
                # Every frozen-round relation touching this parent's evolving lineage and a block-bearing proposal.
                relevant_pairs = []
                for r in obj.relations:
                    state = obj.roundstates[r['round_index']]
                    left, right = r['left_proposal_id'], r['right_proposal_id']
                    a, bb = state[left], state[right]
                    for own, other in ((a, bb), (bb, a)):
                        if own['lineage'] & lineage and not other['lineage'] <= lineage:
                            hits = int(np.isin(obj.sp[block], list(other['sp'])).sum())
                            if hits:
                                pr = dict(r, scene=s, candidate_id=cid, block_id=bid,
                                          own_lineage=sorted(own['lineage']), other_lineage=sorted(other['lineage']),
                                          block_hits_other=hits)
                                pairrows.append(pr)
                                relevant_pairs.append(pr)
                removed_details = []
                for k in sorted(lineage):
                    dropped = set(map(int,obj.initial[k])) - set(map(int,obj.d1keep[k]))
                    for sp in sorted(dropped & set(map(int,obj.sp[block]))):
                        removed_details.append(dict(stage='D1', source_track_id=k, superpoint_id=sp,
                            block_hits=int((obj.sp[block]==sp).sum()), **obj.diagnostics[k][sp]))
                for e in events:
                    for sp in sorted(e['removed'] & set(map(int,obj.sp[block]))):
                        removed_details.append(dict(stage='merge_refine', action_index=e['action']['action_index'],
                            round_index=e['action']['round_index'], superpoint_id=sp,
                            block_hits=int((obj.sp[block]==sp).sum()), **e['diagnostics'][sp]))
                visible_frames = []
                for fi, vis in obj.visible.items():
                    bv, pv = int(vis[block].sum()), int(vis[parent].sum())
                    if bv:
                        visible_frames.append(dict(frame_index=fi, frame_id=obj.frames[fi]['frame_id'],
                                                   block_visible=bv, parent_visible=pv))
                flags = dict(raw_hit=bool(allraw), parent_lineage_observation=bool(len(stages['lineage_observed'])),
                    other_track_observation=bool(others), D1_removed=bool(removed_details and any(x['stage']=='D1' for x in removed_details)),
                    merge_removed=bool(len(stages['merge_removed'])),
                    no_raw_hit_in_cached_bank=not allraw,
                    no_cached_visibility=not visible_frames)
                blockrows.append(dict(scene=s, cohort=obj.j['cohort'], candidate_id=cid, block_id=bid,
                    region=b['region'], points=len(block), block_sha256=digest(block), track_id=tid,
                    prior_ownership_state=b['state'], stages={k:len(v) for k,v in stages.items()},
                    flags=flags, source_observation_ids=onblock, lineage_hit_observation_ids=lineagehits,
                    other_track_hit_observation_ids=otherhits, removed_superpoint_evidence=removed_details,
                    cached_visible_frames=visible_frames,
                    merge_pairs_with_block=len(relevant_pairs),
                    max_recorded_other_point_iou=max((r['point_iou'] for r in relevant_pairs), default=None),
                    recorded_eligible_other_pairs=sum(r['details_merge_eligible_observed'] for r in relevant_pairs),
                    association_rejection_reason='not_recorded',
                    original_observation_reliability='exact cached-visible point hits; no instance guarantee',
                    occlusion_or_depth_failure='unresolved_from_cached_visibility'))
            for mode in ('lineage_observed', 'd1_initial', 'merge_input', 'other_tracks', 'all_raw'):
                added = union(arrays[b['block_id'] + '_' + mode] for b in selected)
                arrays[cid + '_' + mode + '_union'] = np.union1d(parent, added)
    d.mkdir(parents=True)
    for name, rows in [('parents', parentrows), ('blocks', blockrows),
                       ('block_observations', obsrows), ('block_merge_relations', pairrows)]:
        seal(d / (name + '.json'), rows)
    np.savez_compressed(d / 'pointsets.npz', **arrays)
    receipt = obj.receipt()
    receipt['counts'].update(blocks=len(blockrows), block_observation_links=len(obsrows),
                             block_merge_relations=len(pairrows))
    seal(d / 'alignment.json', receipt)
    seal(d / 'receipt.json', dict(complete=True, GT_used=False, inputs=obj.used,
                                outputs={str(p):sha(p) for p in sorted(d.iterdir())}))
    return receipt['counts']


def generate():
    guard()
    reserve()
    c = read(OUT / 'contract.json')
    assert read(OUT / 'pilot_passed.json')['checks_passed']
    snapshot = OUT / 'code_snapshot' / Path(__file__).name
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open('xb') as f:
            f.write(Path(__file__).read_bytes())
    assert sha(snapshot) == sha(__file__)
    counts = Counter()
    for i, (s, spec) in enumerate(c['jobs'].items()):
        rp = OUT / 'evidence' / s / 'receipt.json'
        if rp.exists():
            rc = read(rp)
            for p, h in rc['outputs'].items():
                assert sha(p) == h
            counts.update(read(rp.parent / 'alignment.json')['counts'])
            continue
        print('GT_free_source_chain', i + 1, 52, s, flush=True)
        obj = Scene(s, spec, c).load()
        counts.update(export_scene(obj))
        del obj
        reserve()
    assert counts['parents'] == 207 and counts['blocks'] == 1308
    outputs = {str(p): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    seal(OUT / 'source_evidence_frozen.json', dict(complete=True, GT_used=False,
         counts=dict(counts), outputs=outputs, code_sha256=sha(__file__), time_unix=time.time()))
    print('all_source_evidence_sealed', dict(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'pilot', 'generate'))
    args = parser.parse_args()
    globals()[args.mode]()
