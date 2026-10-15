#!/usr/bin/env python3
"""One bounded D1 granularity comparison, with prediction-before-GT sealing."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import audit_fi1_t_source_chain as audit
import fi1_d1_subregion_core as core

ROOT = audit.ROOT
SOURCE = audit.OUT
OUT = audit.EXP / 'fi1_d1_subregion_minimal_exposed52_20261015_v1'
HERE = Path(__file__).resolve().parent
ARMS = ('initial', 'whole', 'sub')
VIEWS = ('full', 'old')
SCRIPTS = ('fi1_d1_subregion_core.py', 'run_fi1_d1_subregion_minimal.py',
           'test_fi1_d1_subregion.py', 'audit_fi1_t_source_chain.py',
           'evaluate_fi1_nt_region_identity.py', 'run_fi1_nt_region_identity.py',
           'analyze_fi1_t_source_chain.py')


def guard(no_gt=True):
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if no_gt and any(x in str(path) for x in ('ground_truth', '/evaluation/', '/diagnosis/',
                         'parent_scopes.json', 'evaluation_summary', 'supplementary_source_vote')):
            raise PermissionError('GT forbidden before prediction seal: ' + str(path))
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if writing and not path.is_relative_to(OUT):
            raise PermissionError('Only fresh experiment may be written: ' + str(path))
    sys.addaudithook(hook)


def checked(path, expected, used):
    path = Path(path)
    key = str(path)
    if key not in used:
        used[key] = audit.sha(path)
    assert used[key] == expected, ('input changed', key)
    return path


def verify_map(mapping):
    for path, digest in mapping.items():
        assert audit.sha(path) == digest, path


def frozen_contract():
    c = audit.read(OUT / 'contract.json')
    verify_map(c['code'])
    assert audit.sha(SOURCE / 'source_evidence_frozen.json') == c['source_seal_sha256']
    assert audit.sha(SOURCE / 'contract.json') == c['source_contract_sha256']
    return c


def prepare():
    guard()
    audit.reserve()
    assert not OUT.exists()
    seal = audit.read(SOURCE / 'source_evidence_frozen.json')
    assert seal['complete'] and seal['GT_used'] is False
    verify_map(seal['outputs'])
    code = {str(HERE / name): audit.sha(HERE / name) for name in SCRIPTS}
    hist = SOURCE / 'historical_source/refine_details_automatic_tracks_consensus.py'
    code[str(hist)] = audit.sha(hist)
    for path in code:
        dest = OUT / 'code_snapshot' / Path(path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    audit.seal(OUT / 'contract.json', dict(version=OUT.name, time_unix=time.time(),
        source_seal_sha256=audit.sha(SOURCE / 'source_evidence_frozen.json'),
        source_contract_sha256=audit.sha(SOURCE / 'contract.json'), code=code,
        counts=dict(scenes=52, parents=207, blocks=1308), parameters=core.PARAMETERS,
        arms=dict(NT='Complete original deduplicated NT retained in every comparison.',
                  initial='Parent plus union of original-track historical D1 initial SPs.',
                  whole='Parent plus independent original-track whole-SP D1 consensus readout.',
                  sub='Parent plus independent original-track exact mask-signature atom consensus readout.'),
        common_domain='Each original track uses its own frozen D1 initial SP domain; no extra initial SPs. All three restore arms use exactly these sources and this domain.',
        partition='Maximal equivalence classes (original SP ID, binary mask membership over unioned source frames); may be disconnected. Zero signature retained in evidence but never recovered. No spatial extrapolation.',
        signature_caveat='Positive-frame atom coverage is structurally 1, and negative-frame coverage 0 when visible; not independent identity evidence. Numeric thresholds fixed, their effective meaning changes with partition.',
        voting='Original tracks independent. Within each track union same-frame masks; one source frame one vote. Visibility only from existing source frames. Tiny atoms still require >=3 visible points in a vote and >=2 support frames.',
        outputs='After merge, preserve T; at most one parallel copy per parent/arm. Full outputs plus parent union (restored intersect original whole-block editable domain). Old domain is analysis-only, not query clipping.',
        gates=dict(safe_gain02_scenes_min=2, safe_added_target_fraction=.9,
                   stable_added_precision_min=.9, good_copy_loss_gt02_max=0,
                   unresolved_actions_max=0, annotation_unknown_fraction_max=.1),
        gate_scope='Full sub-SP outputs mandatory; old-domain diagnostic cannot rescue failure. Stable precision denominator includes unknown. Annotation unknown fraction across ALL parent additions; unresolved actions separate.',
        limits=dict(training=False, new_frames=False, query_clipping=False, N_gates=False,
                    single_frame_exception=False, threshold_search=False, AP=False,
                    production_modified=False, old_seals_modified=False),
        stop='One fixed run. Failure ends this scheme; success only permits proposing separate isolated validation, not automatic AP.'))
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE),
                            '-p', 'test_fi1_d1_subregion.py', '-v'], capture_output=True, text=True)
    assert tests.returncode == 0, tests.stderr
    audit.seal(OUT / 'tests.json', dict(passed=True, returncode=tests.returncode,
                                     stdout=tests.stdout, stderr=tests.stderr))
    print('prepared fixed contract', flush=True)


def historical():
    path = SOURCE / 'historical_source/refine_details_automatic_tracks_consensus.py'
    spec = importlib.util.spec_from_file_location('frozen_d1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate():
    guard()
    c = frozen_contract()
    assert c['parameters'] == core.PARAMETERS
    sc = audit.read(SOURCE / 'contract.json')
    seal = audit.read(SOURCE / 'source_evidence_frozen.json')
    hist = historical()
    outputs, total = {}, Counter()
    for ix, (scene, spec) in enumerate(sc['jobs'].items(), 1):
        audit.reserve()
        print('generate', ix, 52, scene, flush=True)
        ed = SOURCE / 'evidence' / scene
        used = {}
        alignment = audit.read(checked(ed / 'alignment.json', seal['outputs'][str(ed / 'alignment.json')], used))
        prior_inputs = alignment['inputs']

        def original(path):
            return checked(path, prior_inputs[str(path)], used)

        manifest_path = Path(sc['frontend']) / 'manifests' / (scene + '.json')
        manifest = audit.read(original(manifest_path))
        frame_meta = {int(f['frame_index']): f for f in manifest['frames']}
        data = np.load(original(spec['job']['paths']['processed']), mmap_mode='r')
        sp = np.asarray(data[:, 9], np.int64)
        assert len(sp) == spec['job']['binding']['point_count']
        parents = audit.read(checked(ed / 'parents.json', seal['outputs'][str(ed / 'parents.json')], used))
        blocks = audit.read(checked(ed / 'blocks.json', seal['outputs'][str(ed / 'blocks.json')], used))
        point_path = checked(ed / 'pointsets.npz', seal['outputs'][str(ed / 'pointsets.npz')], used)
        obs_path = Path(sc['frontend']) / 'arms/D/observations' / scene / 'automatic_observations.jsonl'
        observations = {int(o['observation_id']): o for o in audit.jsonlines(original(obs_path))}
        selected_ids = {int(oid) for p in parents for t in p['original_track_records'] for oid in t['observation_ids']}
        pointsets, visibility, provenance = {}, {}, {}
        for oid in sorted(selected_ids):
            o = observations[oid]
            fi = int(o['frame_index'])
            assert o['scene_name'] == scene and str(o['frame_id']) == str(frame_meta[fi]['frame_id'])
            assert int(o['frame_id']) in manifest['arms']['D']
            cache = Path(sc['frontend']) / 'frame_cache/grounded_sam' / scene / ('frame%06d' % int(o['frame_id']))
            path = Path(o['point_indices_path'])
            assert path.parent == cache
            if fi not in visibility:
                rc = audit.read(original(cache / 'receipt.json'))
                local = audit.read(original(cache / 'observations.json'))
                assert audit.sha(cache / 'observations.json') == rc['files']['observations.json']
                provenance[fi] = {r['point_indices_path']: r for r in local}
                vp = original(cache / 'visibility.npy')
                assert used[str(vp)] == rc['files']['visibility.npy']
                visibility[fi] = np.load(vp, allow_pickle=False)
                assert visibility[fi].dtype == bool and visibility[fi].shape == (len(sp),)
            local_o = provenance[fi][str(path)]
            assert all(local_o[k] == o[k] for k in ('scene_name', 'frame_id', 'frame_index', 'mask_rle', 'predicted_iou', 'stability_score'))
            pointsets[oid] = audit.points(original(path))
            assert visibility[fi][pointsets[oid]].all()
        arrays, track_rows, parent_rows = {}, [], []
        seen_tracks = {}
        with np.load(point_path) as old:
            for p in parents:
                cid = p['candidate_id']
                parent = old[cid + '_parent']
                assert np.array_equal(parent, audit.points(original(p['locator']['points_path']), p['locator']['array_key']))
                assert audit.digest(parent) == p['points_sha256']
                editable = audit.union(old[b['block_id'] + '_block'] for b in blocks if b['candidate_id'] == cid)
                arrays[cid + '_parent'], arrays[cid + '_editable'] = parent, editable
                candidates = {arm: [] for arm in ARMS}
                repeat_domain = []
                for t in p['original_track_records']:
                    tid = int(t['track_id'])
                    if tid not in seen_tracks:
                        key = 't%04d' % tid
                        domain = np.flatnonzero(np.isin(sp, p['exact_D1_initial'][str(tid)]))
                        frames = core.frame_unions(t['observation_ids'], observations, pointsets)
                        support = np.asarray([np.isin(domain, pts) for _, _, pts in frames])
                        vis = np.asarray([visibility[fi][domain] for fi, _, _ in frames])
                        sp_ids, sub_sp, whole, sub = core.track_readout(domain, sp[domain], support, vis)
                        assert set(sp_ids) == set(p['exact_D1_initial'][str(tid)])
                        expected = next(r for r in p['D1_records'] if int(r['track_id']) == tid)
                        assert set(sp_ids[whole['kept']]) == set(expected['superpoint_ids'])
                        frame_rows = [dict(frame_index=fi, inside_counts={int(s): int(n) for s, n in zip(sp_ids, whole['inside'][j])},
                                          visible_counts={int(s): int(n) for s, n in zip(sp_ids, whole['visible'][j])})
                                      for j, (fi, _, _) in enumerate(frames)]
                        kept, _ = hist.consensus_superpoints(set(map(int, sp_ids)), frame_rows, 3, .5, 2, .3, .55)
                        assert kept == set(expected['superpoint_ids'])
                        arrays[key + '_domain'] = domain
                        arrays[key + '_support'] = support
                        arrays[key + '_visibility'] = vis
                        arrays[key + '_sp'] = sp[domain]
                        arrays[key + '_sub_sp'] = sub_sp
                        for kind, result in [('whole', whole), ('sub', sub)]:
                            for name, value in result.items():
                                arrays[key + '_' + kind + '_' + name] = value
                        saved = dict(initial=domain, whole=domain[whole['kept'][whole['labels']]],
                                     sub=domain[sub['kept'][sub['labels']]], repeat=domain[support.sum(axis=0) >= 2])
                        seen_tracks[tid] = saved
                        frame_evidence = []
                        for fi, oids, pts in frames:
                            frame_evidence.append(dict(frame_index=fi, frame_id=frame_meta[fi]['frame_id'],
                                observation_ids=oids, observations=[dict(observation_id=oid,
                                    point_indices_path=observations[oid]['point_indices_path'],
                                    sha256=used[observations[oid]['point_indices_path']],
                                    cache_local_id=provenance[fi][observations[oid]['point_indices_path']]['observation_id']) for oid in oids],
                                visibility_path=str(Path(observations[oids[0]]['point_indices_path']).parent / 'visibility.npy'),
                                mask_union_points=len(pts)))
                        track_rows.append(dict(track_id=tid, key=key, original_record=t, frames=frame_evidence,
                            initial_SP_ids=sp_ids.tolist(), whole_kept_SP_ids=sp_ids[whole['kept']].tolist(),
                            sub_regions=len(sub['sizes']), sub_kept_regions=int(sub['kept'].sum()),
                            zero_signature_regions=int((sub['inside'].sum(axis=0) == 0).sum()),
                            tiny_regions=int((sub['sizes'] < 3).sum()),
                            source_frame_votes=len(frames), masks=len(t['observation_ids'])))
                        total['original_tracks'] += 1
                        total['sub_regions'] += len(sub['sizes'])
                    for arm in ARMS:
                        candidates[arm].append(seen_tracks[tid][arm])
                    repeat_domain.append(seen_tracks[tid]['repeat'])
                for arm in ARMS:
                    recovered = audit.union(candidates[arm])
                    arrays[cid + '_' + arm + '_full'] = np.union1d(parent, recovered)
                    arrays[cid + '_' + arm + '_old'] = np.union1d(parent, audit.hit(recovered, editable))
                arrays[cid + '_repeat_old_supply'] = audit.hit(audit.union(repeat_domain), editable)
                # Reproduce the earlier initial-domain control exactly in its old domain.
                assert np.array_equal(arrays[cid + '_initial_old'], old[cid + '_d1_initial_union'])
                parent_rows.append(dict(candidate_id=cid, track_id=p['track_id'], locator=p['locator'],
                    source_track_ids=p['source_track_ids'], blocks=sum(b['candidate_id'] == cid for b in blocks),
                    parent_points=len(parent), editable_points=len(editable),
                    outputs={arm: {view: len(arrays[cid + '_' + arm + '_' + view]) for view in VIEWS} for arm in ARMS}))
        dest = OUT / 'predictions' / scene
        dest.mkdir(parents=True)
        np.savez_compressed(dest / 'data.npz', **arrays)
        audit.seal(dest / 'tracks.json', track_rows)
        audit.seal(dest / 'parents.json', parent_rows)
        audit.seal(dest / 'receipt.json', dict(complete=True, GT_used=False, inputs=used,
                   historical_whole_replay=True, old_initial_restoration_exact=True))
        for path in dest.iterdir():
            outputs[str(path)] = audit.sha(path)
        total.update(scenes=1, parents=len(parents), blocks=len(blocks))
    assert [total[k] for k in ('scenes', 'parents', 'blocks')] == [52, 207, 1308]
    for path in (OUT / 'code_snapshot').iterdir():
        outputs[str(path)] = audit.sha(path)
    for name in ('contract.json', 'tests.json'):
        outputs[str(OUT / name)] = audit.sha(OUT / name)
    audit.seal(OUT / 'predictions_frozen.json', dict(complete=True, GT_used=False,
               time_unix=time.time(), counts=total, outputs=outputs))
    print('predictions sealed', dict(total), flush=True)


def evaluate():
    guard(False)
    c = frozen_contract()
    frozen = audit.read(OUT / 'predictions_frozen.json')
    assert frozen['complete'] and frozen['GT_used'] is False
    verify_map(frozen['outputs'])
    # This is deliberately imported only after complete prediction verification.
    import evaluate_fi1_nt_region_identity as met
    import run_fi1_nt_region_identity as diag
    from analyze_fi1_t_source_chain import summarize
    old = audit.read(audit.OLD / 'contract.json')
    sc = audit.read(SOURCE / 'contract.json')
    parents, targets, blocks, oracle, paired = [], [], [], [], []
    inputs = {}
    for ix, (scene, spec) in enumerate(sc['jobs'].items(), 1):
        print('postseal_GT', ix, 52, scene, flush=True)
        job = spec['job']
        checked(job['GT'], old['GT_inputs'][job['GT']], inputs)
        gt = np.loadtxt(job['GT'], dtype=np.int64)
        _, nt = diag.load_nt(job)
        alignment = audit.read(SOURCE / 'evidence' / scene / 'alignment.json')
        for field in ('geometry', 'points'):
            path = job['binding'][field]
            checked(path, alignment['inputs'][path], inputs)
        ids, sizes = met.valid_targets(gt, old['class_ids'])
        base = met.pool_base(nt, gt, ids, sizes)
        modes = [a + '_' + v for a in ARMS for v in VIEWS]
        best = {mode: base.copy() for mode in modes}
        refs = audit.read(OUT / 'predictions' / scene / 'parents.json')
        old_blocks = audit.read(SOURCE / 'evidence' / scene / 'blocks.json')
        with np.load(OUT / 'predictions' / scene / 'data.npz') as z, np.load(SOURCE / 'evidence' / scene / 'pointsets.npz') as bz:
            for p in refs:
                cid = p['candidate_id']
                parent, editable = z[cid + '_parent'], z[cid + '_editable']
                meta = met.identity(parent, gt, ids, sizes)
                target = meta['target'] if meta['stable'] else None
                bymode = {}
                for mode in modes:
                    pp = z[cid + '_' + mode]
                    add = np.setdiff1d(pp, parent)
                    row, score = met.outcome(parent, pp, gt, ids, sizes, meta, base)
                    if not meta['stable']:
                        row['target'] = None
                    outside = np.setdiff1d(add, editable)
                    row.update(scene=scene, candidate_id=cid, mode=mode,
                        annotation_unknown_added=int((gt[add] < 1000).sum()),
                        outside_added=len(outside), outside_unknown=int((gt[outside] < 1000).sum()),
                        outside_target=int((gt[outside] == target).sum()) if target else None,
                        outside_other=int(((gt[outside] >= 1000) & (gt[outside] != target)).sum()) if target else None,
                        added_target_fraction=row.get('added_target', 0)/len(add) if meta['stable'] and len(add) else None)
                    parents.append(row)
                    bymode[mode] = row
                    best[mode] = np.maximum(best[mode], score)
                for view in VIEWS:
                    w, s = bymode['whole_' + view], bymode['sub_' + view]
                    wa = np.setdiff1d(z[cid + '_whole_' + view], parent)
                    sa = np.setdiff1d(z[cid + '_sub_' + view], parent)
                    extra, lost = np.setdiff1d(sa, wa), np.setdiff1d(wa, sa)
                    paired.append(dict(scene=scene, candidate_id=cid, view=view, stable=meta['stable'], target=target,
                        sub_minus_whole_iou=(s['repaired_iou']-w['repaired_iou']) if meta['stable'] else None,
                        sub_only=len(extra), whole_only=len(lost),
                        sub_only_target=int((gt[extra] == target).sum()) if target else None,
                        sub_only_other=int(((gt[extra] >= 1000) & (gt[extra] != target)).sum()) if target else None,
                        sub_only_unknown=int((gt[extra] < 1000).sum())))
                for b in old_blocks:
                    if b['candidate_id'] != cid:
                        continue
                    bp = bz[b['block_id'] + '_block']
                    row = dict(scene=scene, candidate_id=cid, block_id=b['block_id'], stable=meta['stable'], target=target)
                    for arm in ARMS:
                        add = np.intersect1d(bp, np.setdiff1d(z[cid + '_' + arm + '_full'], parent))
                        row[arm] = dict(added=len(add), target=int((gt[add] == target).sum()) if target else None,
                            other=int(((gt[add] >= 1000) & (gt[add] != target)).sum()) if target else None,
                            unknown=int((gt[add] < 1000).sum()))
                    blocks.append(row)
                supply = z[cid + '_repeat_old_supply']
                pp = np.union1d(parent, supply[gt[supply] == target]) if target else parent
                r, _ = met.outcome(parent, pp, gt, ids, sizes, meta, base)
                oracle.append(dict(scene=scene, candidate_id=cid, mode='D1_initial_old_two_frame_no_extrapolation_target_oracle', **r))
        targets += met.pool_rows(scene, 'NT', base, base, ids)
        for mode in modes:
            targets += met.pool_rows(scene, mode, base, best[mode], ids)
    metrics = {}
    for mode in modes:
        rr = [r for r in parents if r['mode'] == mode]
        tr = [r for r in targets if r['mode'] == mode]
        m = summarize(rr)
        m['pool'] = met.pool_summary(tr)
        m['pool']['gain02_scenes'] = sorted({r['scene'] for r in tr if r['gain02']})
        m['same_parent_gain02_targets'] = len({(r['scene'], r['target']) for r in rr if r.get('pool_delta', -1) >= .02-1e-12})
        safe = [r for r in rr if r.get('pool_delta', -1) >= .02-1e-12 and r.get('added_target_fraction', 0) is not None and r['added_target_fraction'] >= .9]
        m['safe_gain02_scenes'] = sorted({r['scene'] for r in safe})
        m['safe_gain02_targets'] = len({(r['scene'], r['target']) for r in safe})
        m['all_added'] = sum(r['added'] for r in rr)
        m['annotation_unknown_added_all_parents'] = sum(r['annotation_unknown_added'] for r in rr)
        m['annotation_unknown_fraction_all_parents'] = m['annotation_unknown_added_all_parents']/m['all_added'] if m['all_added'] else 0
        m['outside_added_all_parents'] = sum(r['outside_added'] for r in rr)
        m['outside_target_stable'] = sum(r['outside_target'] for r in rr if r['stable'])
        m['outside_other_stable'] = sum(r['outside_other'] for r in rr if r['stable'])
        m['good_parents'] = sum(r.get('good_parent', False) for r in rr)
        m['gates'] = dict(safe_gain_scenes=len(m['safe_gain02_scenes']) >= 2,
            stable_precision=m['precision'] is not None and m['precision'] >= .9,
            good_copy_safety=m['good_copy_loss_gt02'] == 0,
            unresolved_safety=m['unresolved_actions'] == 0,
            annotation_unknown=m['annotation_unknown_fraction_all_parents'] <= .1)
        m['pass'] = all(m['gates'].values())
        metrics[mode] = m
    assert len(parents) == 207*6 and len(blocks) == 1308
    summary = dict(complete=True, prediction_seal_sha256=audit.sha(OUT / 'predictions_frozen.json'),
        GT_inputs=inputs, modes=metrics, baseline=met.pool_summary([r for r in targets if r['mode'] == 'NT']),
        narrow_oracle=summarize(oracle), narrow_oracle_gains=[r for r in oracle if r.get('pool_delta', -1) >= .02-1e-12],
        decision='stop_fixed_scheme' if not metrics['sub_full']['pass'] else 'eligible_for_separate_isolated_validation_proposal',
        scope='Exposed52 sampled parents. Pool IoU diagnostics, no AP or full-geometry claim.',
        signature_caveat=c['signature_caveat'])
    dest = OUT / 'evaluation'
    for name, value in [('parents', parents), ('targets', targets), ('blocks', blocks),
                         ('paired', paired), ('narrow_oracle', oracle), ('summary', summary)]:
        audit.seal(dest / (name + '.json'), value)
    audit.seal(OUT / 'evaluation_frozen.json', dict(complete=True,
        prediction_seal_sha256=summary['prediction_seal_sha256'], time_unix=time.time(),
        outputs={str(p): audit.sha(p) for p in dest.iterdir()}))
    print(audit.plain(dict(decision=summary['decision'], modes=metrics)), flush=True)


def verify():
    guard(False)
    frozen_contract()
    verify_map(audit.read(OUT / 'predictions_frozen.json')['outputs'])
    verify_map(audit.read(OUT / 'evaluation_frozen.json')['outputs'])
    sc = audit.read(SOURCE / 'contract.json')
    counts, inputs = Counter(), {}
    import evaluate_fi1_nt_region_identity as met
    import run_fi1_nt_region_identity as diag
    old = audit.read(audit.OLD / 'contract.json')
    evaluation = audit.read(OUT / 'evaluation/parents.json')
    target_rows = audit.read(OUT / 'evaluation/targets.json')
    indexed = {(r['scene'], r['candidate_id'], r['mode']): r for r in evaluation}

    def independent_iou(pp, gt, gid):
        if len(pp) < 100:
            return 0.
        hits = int((gt[pp] == gid).sum())
        return hits / (len(pp) + int((gt == gid).sum()) - hits)

    for scene, spec in sc['jobs'].items():
        print('verify', scene, flush=True)
        dest = OUT / 'predictions' / scene
        receipt = audit.read(dest / 'receipt.json')
        verify_map(receipt['inputs'])
        inputs.update(receipt['inputs'])
        gt = np.loadtxt(spec['job']['GT'], dtype=np.int64)
        _, nt = diag.load_nt(spec['job'])
        ids, _ = met.valid_targets(gt, old['class_ids'])
        gt_sizes = {int(g): int((gt == g).sum()) for g in ids}
        baseline = {int(g): 0. for g in ids}
        def scores(pp):
            if len(pp) < 100:
                return {g: 0. for g in baseline}
            gs, ns = np.unique(gt[pp], return_counts=True)
            hits = dict(zip(gs.tolist(), ns.tolist()))
            return {g: hits.get(g, 0)/(len(pp)+gt_sizes[g]-hits.get(g, 0)) for g in baseline}
        for pp in nt.values():
            for g, value in scores(pp).items():
                baseline[g] = max(baseline[g], value)
        expanded = {a+'_'+v: baseline.copy() for a in ARMS for v in VIEWS}
        with np.load(dest / 'data.npz') as z:
            track_sets = {}
            for t in audit.read(dest / 'tracks.json'):
                key = t['key']
                domain, support, vis, sp = [z[key + '_' + name] for name in ('domain', 'support', 'visibility', 'sp')]
                assert not (support & ~vis).any()
                assert [f['frame_index'] for f in t['frames']] == sorted({f['frame_index'] for f in t['frames']})
                # Independently reconstruct exact source-frame unions from raw files.
                for j, f in enumerate(t['frames']):
                    union_set = set()
                    for o in f['observations']:
                        union_set.update(audit.points(o['point_indices_path']).tolist())
                    np.testing.assert_array_equal(support[j], [int(p) in union_set for p in domain])
                    np.testing.assert_array_equal(vis[j], np.load(f['visibility_path'])[domain])
                track_sets[t['track_id']] = dict(initial=domain, repeat=domain[support.sum(axis=0) >= 2])
                for kind in ('whole', 'sub'):
                    labels = z[key + '_' + kind + '_labels']
                    nr = len(z[key + '_' + kind + '_sizes'])
                    selected = []
                    keys = set()
                    for rid in range(nr):
                        where = np.flatnonzero(labels == rid)
                        assert len(set(sp[where].tolist())) == 1
                        if kind == 'sub':
                            assert (support[:, where] == support[:, where[:1]]).all()
                            signature = (int(sp[where[0]]), tuple(support[:, where[0]].tolist()))
                            assert signature not in keys
                            keys.add(signature)
                        initial, votes, eligible, cov = False, [], [], []
                        for j in range(len(support)):
                            nvis, inside = int(vis[j, where].sum()), int(support[j, where].sum())
                            assert nvis == z[key + '_' + kind + '_visible'][j, rid]
                            assert inside == z[key + '_' + kind + '_inside'][j, rid]
                            ratio = inside / nvis if nvis else 0
                            initial |= nvis > 0 and nvis/len(where) >= .1 and ratio >= .3
                            if nvis >= 3:
                                eligible.append(j)
                                if ratio >= .5:
                                    votes.append(j)
                                    cov.append(ratio)
                        keep = initial and len(votes) >= 2 and len(votes)/max(1, len(eligible)) >= .3 and sum(cov)/max(1, len(votes)) >= .55
                        assert bool(keep) == bool(z[key + '_' + kind + '_kept'][rid])
                        if keep:
                            selected.extend(domain[where].tolist())
                        counts[kind + '_regions'] += 1
                    track_sets[t['track_id']][kind] = np.array(sorted(selected), np.int64)
                counts['original_tracks'] += 1
            for p in audit.read(dest / 'parents.json'):
                cid = p['candidate_id']
                parent, editable = z[cid + '_parent'], z[cid + '_editable']
                for arm in ARMS:
                    recovered = audit.union(track_sets[tid][arm] for tid in p['source_track_ids'])
                    for view in VIEWS:
                        expected = set(parent.tolist()) | (set(recovered.tolist()) if view == 'full' else set(recovered.tolist()) & set(editable.tolist()))
                        assert set(z[cid + '_' + arm + '_' + view].tolist()) == expected
                        mode = arm + '_' + view
                        output = z[cid + '_' + mode]
                        row = indexed[scene, cid, mode]
                        added = sorted(expected - set(parent.tolist()))
                        assert len(added) == row['added'] and row['deleted'] == 0
                        assert int((gt[added] < 1000).sum()) == row['annotation_unknown_added']
                        if row['stable']:
                            g = row['target']
                            assert int((gt[added] == g).sum()) == row['added_target']
                            assert int(((gt[added] >= 1000) & (gt[added] != g)).sum()) == row['added_other']
                            assert abs(independent_iou(output, gt, g)-row['repaired_iou']) < 1e-12
                            assert abs(row['pool_delta']-(row['repaired_iou']-baseline[g])) < 1e-12
                        for g, value in scores(output).items():
                            expanded[mode][g] = max(expanded[mode][g], value)
                        counts['parent_outputs'] += 1
                if not p['blocks']:
                    assert scene == 'scene0627_01' and cid == 'c006'
                    assert all(np.array_equal(parent, z[cid + '_' + a + '_old']) for a in ARMS)
                counts['parents'] += 1
                counts['blocks'] += p['blocks']
        for row in [r for r in target_rows if r['scene'] == scene]:
            g, mode = row['gt_id'], row['mode']
            assert abs(row['baseline']-baseline[g]) < 1e-12
            expected = baseline[g] if mode == 'NT' else expanded[mode][g]
            assert abs(row['expanded']-expected) < 1e-12
            counts['pool_target_rows'] += 1
    assert counts['parents'] == 207 and counts['blocks'] == 1308 and counts['parent_outputs'] == 1242
    audit.seal(OUT / 'verification.json', dict(complete=True, counts=counts,
        input_files_rehashed=len(inputs), prediction_files_rehashed=True, evaluation_files_rehashed=True,
        raw_frame_unions_reconstructed=True, independent_region_loop=True,
        independent_GT_counts_and_pool_IoU=True,
        original_track_outputs_union_exact=True, zero_block_parent_preserved=True,
        no_training=True, no_AP=True, no_production_or_old_seal_writes=True))
    print('verification passed', dict(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'generate', 'evaluate', 'verify'))
    args = parser.parse_args()
    globals()[args.stage]()
