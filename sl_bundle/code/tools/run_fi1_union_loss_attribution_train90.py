#!/usr/bin/env python3
"""U loss attribution on authenticated train30/train60 only, frozen B26/K800."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from tools import run_fi1_core_ablation_train90 as base
from tools.fi1_core_ablation import source_pool, features_and_members, learned_selection
from tools.fi1_r2_core import geometry_features
from tools.fi1_r3_core import pair_overlap, independent_context
from tools.fi1_union_loss_attribution import ARMS, THRESHOLDS, key, factorial_selections, member_diagnostics
from tools.fi1_frontend_protocol import sha256_file as sha, freeze_json, write_json as write

OLD = ROOT / '.local/fi1_core_ablation_train90_20261006_v2'
OUT = ROOT / '.local/fi1_union_loss_attribution_train90_20261007_v1'
read, rows = base.read, base.rows


class Guard(base.AccessGuard):
    def check(self, path):
        path = str(Path(path).resolve()); parts = set(Path(path).parts)
        if parts & {'test', 'scannet_test', 'scannet200_test', 'scannetv2_test'} or Path(path).name == 'scannetv2_test.txt':
            raise PermissionError('official final test prohibited')
        if '/.local/' in path and 'val312' in path:
            raise PermissionError('validation312 artifacts prohibited in training diagnostic')
        super().check(path)


def checked(out, contract):
    base.verify_files(contract['inputs']); base.verify_files(contract['code'])


def prepare(out):
    assert not out.exists(), 'new output required'
    assert sklearn.__version__ == '1.3.2' and shutil.disk_usage(ROOT).free >= 50 * 1024**3
    parent = read(OLD / 'contract.json'); accept = read(OLD / 'acceptance.json')
    assert accept['complete'] and accept['host_tmux_exited'] and len(parent['scenes']) == 90
    guard = Guard(parent['scenes'], [b['gt_path'] for b in parent['bindings'].values()]); sys.addaudithook(guard.hook)
    base.verify_files({str(ROOT / p): h for p, h in accept['artifacts'].items()})
    inputs = {str(OLD / n): sha(OLD / n) for n in ('contract.json', 'acceptance.json', 'predictions_frozen.json',
                'predictions_verified.json', 'evaluation/summary.json', 'completion_verified.json', 'final_verification.json')}
    inputs[str(base.MODEL)] = base.MODEL_HASH
    for scene in parent['scenes']:
        spec = parent['bindings'][scene]
        for name in ('geometry', 'points', 'evidence'):
            inputs[spec[name]] = parent['inputs'][spec[name]]
        d = OLD / 'scenes' / scene; r = read(d / 'receipt.json'); assert r['complete']
        inputs[str(d / 'receipt.json')] = sha(d / 'receipt.json')
        for n in ('nt_geometry.json', 'nt_members.jsonl', 'nt_features.npy', 'nt_scores.npy', 'nt_only_selected.json'):
            inputs[str(d / n)] = r['outputs'][str(d / n)]
    for group in (*parent['batches'], 'combined90'):
        p = OLD / 'evaluation' / (group + '_nt_only_AP.npy'); inputs[str(p)] = sha(p)
    code = base.code_paths()
    for name in ('tools/fi1_union_loss_attribution.py', 'tools/run_fi1_union_loss_attribution_train90.py', 'tools/verify_fi1_union_loss_attribution.py',
                 'tests/test_fi1_union_loss_attribution.py'):
        code[str(ROOT / name)] = sha(ROOT / name)
    contract = dict(version='fi1_union_loss_attribution_train90_20261007_v1', scenes=parent['scenes'],
        batches=parent['batches'], bindings=parent['bindings'], inputs=inputs, code=code,
        B_model=str(base.MODEL), B_model_sha256=base.MODEL_HASH, Dv3_training=parent['Dv3_training'], B_training=parent['B_training'],
        budget=800, arms=list(ARMS), definitions=dict(A='M0 NT scores s0', B='M0 scores s0 + all delta members scores s1',
        C='M0 only, scores s1 computed on full NTU before output restriction', D='M1 NTU scores s1'),
        member_identity='actual pointset SHA256 and class; delta=M1-M0, including newly eligible classes on an old pointset',
        tie_rule='descending score, original native-class alias preferred, geometry key, class index; shared tie flags invariant',
        NTU_rule='source-filter the frozen Full pool, removing only Uprime; preserve shared source aliases; own semantics unchanged',
        GT_policy='all360 selections independently checked and sealed before certified90 training GT; all heldout and final test forbidden',
        evaluation='original198-instance evaluator matching and AP AST, unchanged strict thresholds and ignore rules; pooled scenes',
        diagnostics='candidate descriptions and coverage are not original evaluator TP/FP or additive AP attributions',
        scope=parent['scope'], group_reporting='train30 and train60 separate, combined90 supplementary; no validation loss extrapolation',
        authorization='user requested the discussed four-group U loss attribution on existing authorized train90',
        new_training=False, new_visual_inference=False, production_modified=False, heldout_read=False, final_test_read=False,
        no_statistical_significance_claim=True)
    checked(out, contract); out.mkdir(parents=True); freeze_json(out / 'contract.json', contract)
    for p in code:
        dest = out / 'code_snapshot' / Path(p).relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    freeze_json(out / 'code_audit.json', dict(complete=True, no_parent_semantics_inheritance=True, all_prebudget_delta_members=True,
        C_context_not_recomputed_after_restriction=True, B_D_added_scores_identical=True, actual_training_path_guard=True,
        independent_predictions_before_GT=True, original_evaluator_AST=True, code=code))
    write(out / 'execution.json', dict(status='prepared', stage='tests'))
    print('Frozen train90 four-group contract prepared; no GT arrays opened.', flush=True)


def predict(out):
    c = read(out / 'contract.json'); guard = Guard(c['scenes'], [s['gt_path'] for s in c['bindings'].values()]); sys.addaudithook(guard.hook)
    checked(out, c); assert read(out / 'tests_verified.json')['passed']; model = joblib.load(base.MODEL)
    outputs = {}; ntrows = nturows = 0
    for n, scene in enumerate(c['scenes'], 1):
        d = out / 'scenes' / scene; spec = c['bindings'][scene]
        write(out / 'execution.json', dict(status='running', stage='predictions', scene=scene, completed=n-1, total=90))
        if (d / 'receipt.json').exists():
            r = read(d / 'receipt.json'); base.verify_files(r['outputs'])
            outputs.update(r['outputs']); outputs[str(d / 'receipt.json')] = sha(d / 'receipt.json'); ntrows += r['NT_members']; nturows += r['NTU_members']; continue
        gs, points, ev = base.load_geometry(spec); old = OLD / 'scenes' / scene
        nt = read(old / 'nt_geometry.json'); m0 = rows(old / 'nt_members.jsonl')
        x0 = np.load(old / 'nt_features.npy', allow_pickle=False); s0 = np.load(old / 'nt_scores.npy', allow_pickle=False)
        assert np.array_equal(np.clip(model.predict(x0), 0, 1), s0)
        ntu = source_pool(gs, ('native', 'track', 'pair_union')); keys = [g['geometry_key'] for g in ntu]
        assert all(a['source'] != 'refined_union' for g in ntu for a in base.original_aliases(g))
        m1, x1 = features_and_members(ntu, {k: points[k] for k in keys}, {k: ev[k] for k in keys}, spec['point_count'])
        normal, s1 = learned_selection(m1, x1, model)
        iou, cover = pair_overlap([points[k] for k in keys], spec['point_count'])
        assert np.allclose(independent_context(m1, keys, iou, cover), x1[:, 20:], atol=1e-14, rtol=0)
        own = {g['geometry_key']: geometry_features(g, ev[g['geometry_key']]) for g in ntu}
        assert np.array_equal(x1[:, :20], np.asarray([own[m['geometry_key']][m['class_index']] for m in m1]))
        expected = set()
        for g in ntu:
            aliases = base.original_aliases(g); classes = {a['class_index'] for a in aliases if 0 <= a['class_index'] < 198}
            if any(a['source'] != 'native' for a in aliases): classes |= set(np.flatnonzero(own[g['geometry_key']][:, 0] > 0).tolist())
            if g['point_count']: expected |= {(g['geometry_key'], cl) for cl in classes}
        assert expected == {key(m) for m in m1}
        selected = factorial_selections(m0, s0, m1, s1)
        assert selected['A'] == read(old / 'nt_only_selected.json') and selected['D'] == normal
        p0 = {key(m): i for i, m in enumerate(m0)}; p1 = {key(m): i for i, m in enumerate(m1)}
        aligned = np.asarray([p1[key(m)] for m in m0]); common_scores = s1[aligned]
        stable = [i for i in range(20) if i not in (14, 15, 16, 17)]
        assert np.array_equal(x0[:, stable], x1[aligned][:, stable]), 'shared own semantic/size features changed'
        # Independent scoring, member census and sort for all four interventions.
        for arm in ARMS:
            universe = m0 if arm in ('A', 'C') else m1; independently = []
            for m in universe:
                k = key(m); score = s0[p0[k]] if arm == 'A' or (arm == 'B' and k in p0) else s1[p1[k]]
                independently.append({**(m0[p0[k]] if arm in ('A', 'C') or (arm == 'B' and k in p0) else m), 'score': float(score)})
            independently.sort(key=lambda m: (-m['score'], not m['has_original_native_alias'], m['geometry_key'], m['class_index']))
            assert selected[arm] == independently[:800] and len(selected[arm]) == 800
        diff = common_scores - s0
        score_stats = dict(common_scores_changed=int((diff != 0).sum()), common_scores_increased=int((diff > 0).sum()),
            common_scores_decreased=int((diff < 0).sum()), max_abs_change=float(np.abs(diff).max(initial=0)),
            common_source_features_changed=int(np.any(x0[:, 14:18] != x1[aligned, 14:18], axis=1).sum()),
            common_context_features_changed=int(np.any(x0[:, 20:] != x1[aligned, 20:], axis=1).sum()))
        d.mkdir(parents=True); write(d / 'ntu_geometry.json', ntu)
        with (d / 'ntu_members.jsonl').open('w') as f:
            for m in m1: f.write(json.dumps(m, sort_keys=True) + '\n')
        np.save(d / 'ntu_features.npy', x1, allow_pickle=False); np.save(d / 'ntu_scores.npy', s1, allow_pickle=False)
        for arm in ARMS: write(d / (arm + '_selected.json'), selected[arm])
        counts = {a: dict(selected=800, unique_geometry=len({m['geometry_key'] for m in selected[a]}),
            original_members=sum(key(m) in p0 for m in selected[a]), added_members=sum(key(m) not in p0 for m in selected[a])) for a in ARMS}
        r = dict(complete=True, scene=scene, GT_read=False, NT_members=len(m0), NTU_members=len(m1), added_members=len(m1)-len(m0),
            NT_geometry=len(nt), NTU_geometry=len(ntu), score_changes=score_stats, selected_counts=counts,
            independent_member_feature_context_score_sorting=True, A_exact_parent=True, D_exact_normal_NTU=True,
            outputs={str(p): sha(p) for p in d.iterdir() if p.is_file()})
        freeze_json(d / 'receipt.json', r); outputs.update(r['outputs']); outputs[str(d / 'receipt.json')] = sha(d / 'receipt.json')
        ntrows += len(m0); nturows += len(m1)
        if n % 10 == 0: print(f'U attribution predictions and independent checks {n}/90', flush=True)
    checked(out, c)
    freeze_json(out / 'predictions_frozen.json', dict(complete=True, scene_arm_count=360, outputs=outputs, GT_read=False,
        NT_score_rows_checked=ntrows, NTU_feature_context_rows_checked=nturows, contract_sha256=sha(out / 'contract.json')))
    print('All360 selections independently checked and frozen before GT.', flush=True)


def evaluate(out):
    c = read(out / 'contract.json'); guard = Guard(c['scenes'], [s['gt_path'] for s in c['bindings'].values()]); sys.addaudithook(guard.hook)
    seal = read(out / 'predictions_frozen.json'); assert seal['complete'] and seal['scene_arm_count'] == 360 and not seal['GT_read']
    base.verify_files(seal['outputs']); checked(out, c); guard.gt_enabled = True
    ev = base.evaluator(); target = out / 'evaluation'; target.mkdir(exist_ok=True)
    matches = {a: {} for a in ARMS}; descriptions = {}; localchecks = GTtotal = 0
    for n, scene in enumerate(c['scenes'], 1):
        spec = c['bindings'][scene]; d = target / scene; source = out / 'scenes' / scene
        write(out / 'execution.json', dict(status='running', stage='AP_matches', scene=scene, completed=n-1, total=90))
        if (d / 'receipt.json').exists():
            r = read(d / 'receipt.json'); base.verify_files(r['outputs']); GTtotal += r['GT']; localchecks += r['intersection_checks']
            for a in ARMS: matches[a][scene] = read(d / (a + '_matches.json'))
            descriptions[scene] = read(d / 'diagnosis.json'); continue
        assert sha(spec['gt_path']) == spec['gt_sha256']; gt = np.loadtxt(spec['gt_path'], dtype=np.int64); assert len(gt) == spec['point_count']
        full, points, _ = base.load_geometry(spec); pool = read(source / 'ntu_geometry.json'); keys = [g['geometry_key'] for g in pool]
        sizes = np.asarray([g['point_count'] for g in pool]); ntkeys = {g['geometry_key'] for g in read(OLD / 'scenes' / scene / 'nt_geometry.json')}
        original = rows(OLD / 'scenes' / scene / 'nt_members.jsonl'); originalkeys = {key(m) for m in original}
        ids, counts = np.unique(gt, return_counts=True); good = (ids >= 1000) & (counts >= 100) & np.isin(ids // 1000, ev.VALID_CLASS_IDS)
        ids, counts = ids[good], counts[good]; GTtotal += len(ids)
        classes = [list(ev.VALID_CLASS_IDS).index(int(g) // 1000) for g in ids]
        iou, inter, psizes = base.geometry_overlaps([points[k] for k in keys], gt, ids, counts); assert np.array_equal(psizes, sizes)
        # Independently compare every geometric overlap against raw per-mask label counts.
        census = {}
        for i, k in enumerate(keys):
            gs, ns = np.unique(gt[points[k]], return_counts=True); cc = dict(zip(map(int, gs), map(int, ns))); census[k] = cc
            assert np.array_equal(inter[i], np.asarray([cc.get(int(g), 0) for g in ids]))
        chosen = {a: read(source / (a + '_selected.json')) for a in ARMS}; diag = {}; covered = {}; checks = len(keys) * len(ids)
        for arm in ARMS:
            diag[arm], covered[arm] = member_diagnostics(keys, sizes, iou, inter, classes, chosen[arm], original, ntkeys)
            ss = chosen[arm]; masks = np.zeros((len(gt), len(ss)), dtype=bool)
            for i, m in enumerate(ss): masks[points[m['geometry_key']], i] = True
            gm, pm = ev.assign_instances_for_scan(dict(pred_masks=masks,
                pred_classes=np.asarray([m['class_index'] for m in ss], dtype=np.int64),
                pred_scores=np.asarray([m['score'] for m in ss], dtype=np.float64)), spec['gt_path']); del masks
            bylabel = {l: [] for l in ev.CLASS_LABELS}
            for m in ss:
                if len(points[m['geometry_key']]) >= 100: bylabel[ev.CLASS_LABELS[m['class_index']]].append(m)
            for label, mr in bylabel.items():
                assert len(mr) == len(pm[label])
                for m, p in zip(mr, pm[label]):
                    cc = census[m['geometry_key']]; assert p['confidence'] == m['score'] and p['vert_count'] == len(points[m['geometry_key']])
                    assert p['void_intersection'] == sum(v for gid, v in cc.items() if gid // 1000 not in set(ev.VALID_CLASS_IDS))
                    expected = {g['instance_id']: cc[g['instance_id']] for g in gm[label] if cc.get(g['instance_id'], 0)}
                    assert {g['instance_id']: g['intersection'] for g in p['matched_gt']} == expected; checks += len(gm[label])
            for ps in pm.values():
                for p in ps:
                    for g in p['matched_gt']: g.pop('matched_pred', None)  # Unused recursive copy, all AP-used fields retained.
            matches[arm][scene] = base.serializable_matches(dict(gt=gm, pred=pm))
        comparisons = {}
        for name, before, after in [('B_minus_A', 'A', 'B'), ('C_minus_A', 'A', 'C'), ('D_minus_C', 'C', 'D'), ('D_minus_B', 'B', 'D'), ('D_minus_A', 'A', 'D')]:
            mb, ma = {key(m) for m in chosen[before]}, {key(m) for m in chosen[after]}
            entered, left = ma - mb, mb - ma
            def describe_members(mkeys, threshold):
                goodcount = wrongcount = 0
                for k, cl in mkeys:
                    i = keys.index(k); sm = np.asarray(classes) == cl
                    ok = bool((iou[i, sm] > threshold).any()); goodcount += ok
                    wrongcount += not ok and bool((iou[i] > threshold).any())
                return dict(members=len(mkeys), correct_range=int(goodcount), wrong_class_good_range=int(wrongcount))
            comparisons[name] = dict(entered_members=len(entered), left_members=len(left), entered_new_members=len(entered-originalkeys),
                left_original_members=len(left & originalkeys), entered_at75=describe_members(entered, .75), left_at75=describe_members(left, .75),
                coverage={str(t): dict(gained_GT=len(covered[after][str(t)]-covered[before][str(t)]),
                                      lost_GT=len(covered[before][str(t)]-covered[after][str(t)])) for t in THRESHOLDS})
        desc = dict(GT=len(ids), arms=diag, comparisons=comparisons,
            pool_coverage=dict(NT={str(t): int((iou[[k in ntkeys for k in keys]].max(axis=0, initial=0)>t).sum()) for t in THRESHOLDS},
                              NTU={str(t): int((iou.max(axis=0, initial=0)>t).sum()) for t in THRESHOLDS}),
            warning='correct_range, wrong_class_good_range and redundant_correct_members are descriptions, not evaluator TP/FP assignments')
        d.mkdir(); write(d / 'diagnosis.json', desc); descriptions[scene] = desc
        for a in ARMS: write(d / (a + '_matches.json'), matches[a][scene])
        r = dict(complete=True, GT=len(ids), GT_sha256=spec['gt_sha256'], intersection_checks=checks,
                 outputs={str(p): sha(p) for p in d.iterdir() if p.is_file()})
        freeze_json(d / 'receipt.json', r); localchecks += checks
        if n % 10 == 0: print(f'U attribution original AP matches and checks {n}/90', flush=True)
    assert GTtotal == 2001
    groups = {**c['batches'], 'combined90': c['scenes']}; AP = {}; arrays = {}; differences = {}; aggregates = {}; classchanges = {}
    def total_dict(values):
        if isinstance(values[0], dict): return {k: total_dict([v[k] for v in values]) for k in values[0]}
        return sum(values)
    for group, scans in groups.items():
        AP[group] = {}; classchanges[group] = {}; group_arrays = {}
        for a in ARMS:
            write(out / 'execution.json', dict(status='running', stage='pooled_AP', group=group, arm=a))
            aps, *_ = ev.evaluate_matches({scene: matches[a][scene] for scene in scans}); AP[group][a] = base.metrics(ev, aps); group_arrays[a] = aps
            if a == 'A': assert np.array_equal(aps, np.load(OLD / 'evaluation' / (group + '_nt_only_AP.npy'), allow_pickle=False), equal_nan=True)
            p = target / (group + '_' + a + '_AP.npy'); np.save(p, aps, allow_pickle=False); arrays[str(p)] = sha(p)
        differences[group] = {name: {k: 100 * (AP[group][after][k] - AP[group][before][k]) for k in ('ap', 'ap50', 'ap25')}
            for name, before, after in [('B_minus_A', 'A', 'B'), ('C_minus_A', 'A', 'C'), ('D_minus_C', 'C', 'D'), ('D_minus_B', 'B', 'D'), ('D_minus_A', 'A', 'D')]}
        differences[group]['interaction_D_minus_B_minus_C_plus_A'] = {
            k: 100 * (AP[group]['D'][k] - AP[group]['B'][k] - AP[group]['C'][k] + AP[group]['A'][k]) for k in ('ap', 'ap50', 'ap25')}
        for name, before, after in [('B_minus_A', 'A', 'B'), ('C_minus_A', 'A', 'C'), ('D_minus_A', 'A', 'D')]:
            av = np.nanmean(group_arrays[after][0, :, :9], axis=1); bv = np.nanmean(group_arrays[before][0, :, :9], axis=1)
            valid = np.isfinite(av) & np.isfinite(bv); delta = 100 * (av - bv)
            indices = np.flatnonzero(valid); indices = sorted(indices, key=lambda i: (delta[i], int(i)))
            classchanges[group][name] = dict(improved=int((delta[valid]>0).sum()), decreased=int((delta[valid]<0).sum()), equal=int((delta[valid]==0).sum()),
                all_valid=[dict(class_index=int(i), label=ev.CLASS_LABELS[i], delta_pp=float(delta[i])) for i in indices])
        aggregates[group] = dict(GT=sum(descriptions[s]['GT'] for s in scans),
            arms=total_dict([descriptions[s]['arms'] for s in scans]), comparisons=total_dict([descriptions[s]['comparisons'] for s in scans]),
            pool_coverage=total_dict([descriptions[s]['pool_coverage'] for s in scans]),
            score_changes={k: (max if k == 'max_abs_change' else sum)(read(out / 'scenes' / s / 'receipt.json')['score_changes'][k] for s in scans)
                           for k in read(out / 'scenes' / scans[0] / 'receipt.json')['score_changes']},
            selected_counts=total_dict([read(out / 'scenes' / s / 'receipt.json')['selected_counts'] for s in scans]),
            prebudget_counts={k: sum(read(out / 'scenes' / s / 'receipt.json')[k] for s in scans)
                             for k in ('NT_members', 'NTU_members', 'added_members', 'NT_geometry', 'NTU_geometry')})
        print(json.dumps(dict(group=group, AP=AP[group], differences_pp=differences[group])), flush=True)
    checked(out, c); base.verify_files(seal['outputs'])
    freeze_json(target / 'summary.json', dict(complete=True, AP_units='fraction', AP=AP, differences_pp=differences, AP_arrays=arrays,
        diagnostics=aggregates, per_class_changes=classchanges, GT=GTtotal, scenes=90, predictions=360, independent_GT_intersections=localchecks,
        A_parent_AP_arrays_exact=True, scope=c['scope'], no_statistical_significance_claim=True, new_training=False,
        new_visual_inference=False, heldout_read=False, final_test_read=False, production_modified=False,
        interpretation='two-by-two conditional diagnostic; nonlinear AP interaction, not an additive error budget; training cohorts are not validation loss proof'))
    freeze_json(out / 'completion_verified.json', dict(complete=True, predictions=360, all_predictions_before_GT=True,
        A_parent_AP_arrays_exact=True, tests=read(out / 'tests_verified.json'), NTU_feature_context_rows_checked=seal['NTU_feature_context_rows_checked'],
        independent_GT_intersections=localchecks, summary_sha256=sha(target / 'summary.json'), contract_sha256=sha(out / 'contract.json'),
        new_training=False, new_visual_inference=False, heldout_read=False, final_test_read=False, production_modified=False))
    write(out / 'execution.json', dict(status='complete', stage='complete', scenes=90))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=('prepare', 'predict', 'evaluate'))
    parser.add_argument('--output', type=Path, default=OUT); args = parser.parse_args(); out = args.output.resolve()
    assert out.is_relative_to(ROOT / '.local')
    with threadpool_limits(limits=1):
        try: globals()[args.command](out)
        except Exception as exc:
            if out.exists(): write(out / 'execution.json', dict(status='failed', error=repr(exc)))
            raise
