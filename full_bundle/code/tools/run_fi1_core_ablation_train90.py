#!/usr/bin/env python3
"""Three frozen core ablations on the existing train30/train60 cohorts only."""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from uuid import UUID

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from tools.fi1_core_ablation import ARMS, BUDGET, source_pool, features_and_members, learned_selection, direct_selection
from tools.fi1_frontend_protocol import digest, sha256_file as sha, write_json as write, freeze_json
from tools.fi1_r0_r1_protocol import original_aliases
from tools.fi1_r2_core import geometry_features
from tools.fi1_r3_core import independent_context, pair_overlap
from tools.fi1_r3b_delta_diagnostic_core import evaluator, metrics, rank
from tools.fi1_failure_attribution_core import geometry_overlaps

PRIOR = ROOT / '.local/fi1_current_failure_attribution90_20261001_v1'
INVENTORY = ROOT / '.local/fi1_g10_full597_closure_20260921_v1/inventory.json'
LEARN = Path('/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921/learning')
MODEL = LEARN / 'models/B.joblib'
MODEL_HASH = '48f1d79df57bd9e2ba70802ed18494d6390c5bbdfec6f990fc259449d84ea43a'
DV3 = Path('/home/jia/Wm/wm_open-yolo/OpenYOLO3D/pretrained/fi1_d_v3_official100_full_models_20260822')
THRESHOLDS = (.5, .75, .9)
read = lambda p: json.loads(Path(p).read_text())
rows = lambda p: [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
member_key = lambda m: (m['geometry_key'], m['class_index'])


def serializable_matches(value):
    """Serialize UUID identities losslessly as unique strings for resume; no match edits."""
    def convert(x):
        if isinstance(x, UUID): return str(x)
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def verify_files(files):
    for p, expected in files.items():
        if sha(p) != expected:
            raise ValueError('input/output hash changed: ' + p)


class AccessGuard:
    """Check actual opened paths, including symlinks; GT requires the sealed phase."""
    def __init__(self, scenes, gt_paths):
        self.scenes = set(scenes)
        self.gt_paths = {str(Path(p).resolve()) for p in gt_paths}
        self.gt_enabled = False
        self.gt_opened = set()

    def check(self, path):
        path = str(Path(path).resolve())
        if '/scannet_validation_stream/' in path or '/Dataset/scannet200/' in path:
            raise PermissionError('heldout asset access prohibited: ' + path)
        if set(re.findall(r'scene\d{4}_\d{2}', path)) - self.scenes:
            raise PermissionError('scene outside frozen train90: ' + path)
        if '/ground_truth/' in path:
            if not self.gt_enabled or path not in self.gt_paths:
                raise PermissionError('GT requires all270 sealed predictions and training whitelist')
            self.gt_opened.add(path)

    def hook(self, event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            self.check(os.fsdecode(args[0]))


def code_paths():
    names = ['tools/run_fi1_core_ablation_train90.py', 'tools/fi1_core_ablation.py',
             'tests/test_fi1_core_ablation.py', 'tools/fi1_frontend_protocol.py',
             'tools/fi1_r0_r1_protocol.py', 'tools/fi1_r2_core.py', 'tools/fi1_r3_core.py',
             'tools/fi1_r3b_delta_diagnostic_core.py', 'tools/fi1_failure_attribution_core.py',
             'evaluate/scannet200/eval_semantic_instance.py', 'evaluate/scannet200/util_3d.py',
             'evaluate/scannet200/scannet_constants.py']
    return {str(ROOT / n): sha(ROOT / n) for n in names}


def prepare(out):
    if out.exists():
        raise FileExistsError('new output required; do not overwrite prior contracts')
    assert sklearn.__version__ == '1.3.2', sklearn.__version__
    assert shutil.disk_usage(ROOT).free >= 50 * 1024**3
    prior, inv = read(PRIOR / 'contract.json'), read(INVENTORY)
    batches = {'train' + k: v for k, v in prior['scenes'].items()}
    scenes = sum(batches.values(), [])
    assert len(scenes) == len(set(scenes)) == len({s[:9] for s in scenes}) == 90
    assert set(scenes) <= set(inv['groups']['train'])
    assert not ({s[:9] for s in scenes} & {s[:9] for k in ('dev', 'val') for s in inv['groups'][k]})
    guard = AccessGuard(scenes, [])
    sys.addaudithook(guard.hook)
    assert read(PRIOR / 'completion_verified.json')['complete']
    assert sha(MODEL) == MODEL_HASH
    inputs = {str(p): sha(p) for p in [INVENTORY, PRIOR / 'contract.json', PRIOR / 'completion_verified.json', MODEL, DV3 / 'summary.json']}
    dv3 = read(DV3 / 'summary.json')
    assert dv3['training_dataset'] == 'official100' and dv3['model_count'] == 7
    for m in dv3['models']:
        p = DV3 / m['model_file']; assert sha(p) == m['model_sha256']
        inputs[str(p)] = m['model_sha256']
    bindings = {}
    for scene in scenes:
        drpath = LEARN / 'dataset' / scene / 'receipt.json'
        dr = read(drpath); assert dr['complete'] and dr['group'] == 'train'
        geompath, pointpath = Path(dr['geometry_file']), Path(dr['points_file'])
        own = geompath.parents[2]
        evidence = own / 'alpha_completed' / scene / 'evidence.jsonl'
        ep = own / 'alpha_completed' / scene / 'receipt.json'
        ei = own / 'alpha_completed/identity.json'
        vi = own / 'views/identity.json'
        gi = own / 'geometry_binding/identity.json'
        reference = PRIOR / 'predictions' / scene
        rr = read(reference / 'receipt.json')
        assert rr['complete'] and rr['model_sha256'] == MODEL_HASH
        paths = {str(drpath): sha(drpath), str(geompath): dr['inputs'][str(geompath)],
                 str(pointpath): dr['inputs'][str(pointpath)], str(evidence): dr['inputs'][str(evidence)],
                 str(ep): dr['inputs'][str(ep)], str(ei): dr['inputs'][str(ei)],
                 str(gi): dr['inputs'][str(gi)], str(vi): sha(vi),
                 str(reference / 'receipt.json'): sha(reference / 'receipt.json')}
        paths.update({str(LEARN / 'dataset' / scene / n): h for n, h in dr['outputs'].items()})
        paths.update({str(reference / n): h for n, h in rr['outputs'].items()})
        verify_files(paths)
        er = read(ep); assert er['identity_sha256'] == digest(read(ei))
        assert read(ei)['views_identity_sha256'] == digest(read(vi))
        assert read(vi)['binding_identity_sha256'] == digest(read(gi))
        # Bind GT hash from certified training receipt without opening its payload.
        gt = Path(inv['jobs'][scene]['data']) / 'ground_truth' / (scene + '.txt')
        assert inv['jobs'][scene]['group'] == 'train' and str(gt) in dr['inputs']
        inputs.update(paths)
        bindings[scene] = dict(geometry=str(geompath), points=str(pointpath), evidence=str(evidence),
                               dataset=str(drpath.parent), reference=str(reference),
                               point_count=dr['point_count'], gt_path=str(gt), gt_sha256=dr['inputs'][str(gt)])
    contract = dict(version='fi1_core_ablation_train90_20261006_v2', scenes=scenes, batches=batches,
                    arms=list(ARMS), bindings=bindings, inputs=inputs, code=code_paths(), budget=BUDGET,
                    B_model=str(MODEL), B_model_sha256=MODEL_HASH, Dv3_training='official100 frozen seven models',
                    B_training='597 GD10 matched', sklearn_version=sklearn.__version__,
                    nt_rule='retain actual geometries with N/T aliases; remove U/Uprime aliases, rebuild source flags and R1 class eligibility; recompute context and B predictions',
                    direct_rule='exact Full pre-budget members; original YOLO/Alpha direct_score; original tie rule',
                    GT_policy='seal all270 predictions before opening any training GT; no heldout access',
                    evaluation='original evaluator function ASTs, unchanged thresholds/ignore/matching; pooled scans',
                    class_agnostic_AP='not computed: no new frozen unified geometry-score protocol; best-IoU coverage reported',
                    scope='training diagnostic; both cohorts GT-exposed, B-seen; train60 cache availability bias; not generalization',
                    new_training=False, new_visual_inference=False, production_modified=False)
    out.mkdir(parents=True)
    freeze_json(out / 'contract.json', contract)
    (out / 'code_snapshot').mkdir()
    for p in contract['code']:
        target = out / 'code_snapshot' / Path(p).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    # Static audit before tests or execution; no fitting or asset controllers in inference core.
    tree = ast.parse((ROOT / 'tools/fi1_core_ablation.py').read_text())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not {'fit', 'loadtxt', 'imread', 'generate', 'train'} & set(calls)
    freeze_json(out / 'code_audit.json', dict(complete=True, code=contract['code'],
        no_GT_in_inference_core=True, source_filter_handles_shared_geometry=True,
        context_then_B_predict=True, direct_same_members=True, actual_path_guard=True,
        all_predictions_before_GT=True, evaluator_unmodified_AST=True))
    write(out / 'execution.json', dict(status='prepared', stage='tests', scenes=90, completed_scenes=[]))
    print('Prepared frozen train90 contract; no GT arrays opened.', flush=True)


def load_geometry(spec):
    gs = read(spec['geometry'])
    with np.load(spec['points'], allow_pickle=False) as z:
        points = {g['geometry_key']: z[g['array_key']].copy() for g in gs}
    evidence = {e['geometry_key']: e for e in rows(spec['evidence'])}
    assert len(gs) == len(points) == len(evidence)
    for g in gs:
        p = points[g['geometry_key']]
        assert p.ndim == 1 and p.dtype.kind in 'iu' and np.array_equal(p, np.unique(p))
        assert len(p) == g['point_count'] and (not len(p) or (p[0] >= 0 and p[-1] < spec['point_count']))
        assert hashlib.sha256(np.asarray(p, dtype='<i8').tobytes()).hexdigest() == g['points_sha256']
    return gs, points, evidence


def predict_scene(out, scene, spec, model):
    dest = out / 'scenes' / scene
    if (dest / 'receipt.json').exists():
        rc = read(dest / 'receipt.json'); verify_files(rc['outputs']); return rc
    gs, points, evidence = load_geometry(spec)
    members, x = features_and_members(gs, points, evidence, spec['point_count'])
    original = rows(Path(spec['dataset']) / 'members.jsonl')
    lookup = {member_key(m): i for i, m in enumerate(original)}
    ix = [lookup[member_key(m)] for m in members]
    assert len(ix) == len(set(ix)) == len(original)
    with np.load(Path(spec['dataset']) / 'features.npz', allow_pickle=False) as z:
        reference_x = z['x26']  # No supervision/weights arrays deserialized.
    assert np.array_equal(x, reference_x[ix]), ('Full X26 control mismatch', scene)
    full, scores = learned_selection(members, x, model)
    assert full == read(Path(spec['reference']) / 'selected.json'), ('Full selected mismatch', scene)
    assert np.array_equal(scores, np.load(Path(spec['reference']) / 'scores.npy', allow_pickle=False)[ix])
    nt = source_pool(gs); nkeys = {g['geometry_key'] for g in nt}
    nm, nx = features_and_members(nt, {k: points[k] for k in nkeys}, {k: evidence[k] for k in nkeys}, spec['point_count'])
    nselected, nscores = learned_selection(nm, nx, model)
    direct = direct_selection(members)
    assert {member_key(m) for m in nm} <= {member_key(m) for m in members}
    direct_lookup = {member_key(m): m['score'] for m in original}
    assert all(m['score'] == direct_lookup[member_key(m)] for m in members)
    dest.mkdir(parents=True)
    chosen = {'full': full, 'nt_only': nselected, 'direct_semantics': direct}
    for arm, selected in chosen.items(): write(dest / (arm + '_selected.json'), selected)
    write(dest / 'nt_geometry.json', nt)
    with (dest / 'nt_members.jsonl').open('w') as f:
        for m in nm: f.write(json.dumps(m, sort_keys=True) + '\n')
    np.save(dest / 'nt_features.npy', nx, allow_pickle=False)
    np.save(dest / 'nt_scores.npy', nscores, allow_pickle=False)
    np.save(dest / 'full_scores.npy', scores, allow_pickle=False)
    write(dest / 'full_member_order.json', [list(member_key(m)) for m in members])
    fullkeys = {member_key(m) for m in full}
    stats = {a: dict(geometry_count=len(nt if a == 'nt_only' else gs),
                    eligible_members=len(nm if a == 'nt_only' else members), selected_members=len(ss),
                    selected_unique_geometry=len({m['geometry_key'] for m in ss}),
                    entered_vs_full=len({member_key(m) for m in ss} - fullkeys),
                    left_vs_full=len(fullkeys - {member_key(m) for m in ss})) for a, ss in chosen.items()}
    removed = [g for g in gs if g['geometry_key'] not in nkeys]
    shared = [g for g in gs if g['geometry_key'] in nkeys and any(a['source'] in ('pair_union', 'refined_union') for a in original_aliases(g))]
    rc = dict(complete=True, scene=scene, GT_arrays_read=False, control_X26_exact=True,
              control_scores_exact=True, control_selection_exact=True, control_rows=len(members),
              direct_members_and_scores_exact=True, arms=stats,
              removed_unique_geometry=len(removed), retained_shared_geometry=len(shared),
              removed_members=len(members)-len(nm),
              removed_aliases=sum(len(original_aliases(g)) for g in gs)-sum(len(original_aliases(g)) for g in nt),
              outputs={str(p): sha(p) for p in dest.iterdir() if p.is_file()})
    freeze_json(dest / 'receipt.json', rc)
    return rc


def coverage(iou, keys, selected, gt_classes, ntkeys):
    result = {}
    for arm in ARMS:
        mask = np.asarray([k in ntkeys if arm == 'nt_only' else True for k in keys])
        best = iou[mask].max(axis=0, initial=0)
        sel = {(m['geometry_key'], m['class_index']) for m in selected[arm]}
        correct = np.zeros_like(iou, dtype=bool)
        for i, k in enumerate(keys):
            for j, c in enumerate(gt_classes): correct[i, j] = (k, c) in sel
        result[arm] = dict(GT=len(gt_classes), best_iou_sum=float(best.sum()),
            pool={str(t): int((best > t).sum()) for t in THRESHOLDS},
            selected_correct_class={str(t): int(((iou > t) & correct).any(axis=0).sum()) for t in THRESHOLDS})
    assert result['full']['pool'] == result['direct_semantics']['pool']
    return result


def evaluate(out, contract, guard):
    seal = read(out / 'predictions_frozen.json')
    assert seal['complete'] and seal['scene_arm_count'] == 270 and not seal['GT_arrays_read']
    verify_files(seal['outputs']); verify_files(contract['inputs']); verify_files(contract['code'])
    guard.gt_enabled = True
    target = out / 'evaluation'; target.mkdir(exist_ok=True)
    ev = evaluator()
    matches = {a: {} for a in ARMS}; coverage_by_scene = {}; bool_checks = 0
    for n, scene in enumerate(contract['scenes'], 1):
        spec = contract['bindings'][scene]; dest = target / scene
        write(out / 'execution.json', dict(status='running', stage='AP_matches', scene=scene, completed=n-1, total=90))
        if (dest / 'receipt.json').exists():
            rc = read(dest / 'receipt.json'); verify_files(rc['outputs'])
            for arm in ARMS: matches[arm][scene] = read(dest / (arm + '_matches.json'))
            coverage_by_scene[scene] = read(dest / 'coverage.json')
            bool_checks += rc['independent_boolean_intersections']
            continue
        assert sha(spec['gt_path']) == spec['gt_sha256']
        gt = np.loadtxt(spec['gt_path'], dtype=np.int64); assert len(gt) == spec['point_count']
        gs, points, _ = load_geometry(spec); keys = [g['geometry_key'] for g in gs]
        selected = {a: read(out / 'scenes' / scene / (a + '_selected.json')) for a in ARMS}
        ids, sizes = np.unique(gt, return_counts=True)
        valid = (ids >= 1000) & (sizes >= 100) & np.isin(ids // 1000, ev.VALID_CLASS_IDS)
        ids, sizes = ids[valid], sizes[valid]
        classes = [list(ev.VALID_CLASS_IDS).index(int(i)//1000) for i in ids]
        iou, inter, counts = geometry_overlaps([points[k] for k in keys], gt, ids, sizes)
        local_checks = 0
        # Independent bool intersections: hash-selected geometries, all valid GT.
        sample = sorted(range(len(gs)), key=lambda i: digest([scene, keys[i]]))[:8]
        for i in sample:
            mask = np.zeros(len(gt), dtype=bool); mask[points[keys[i]]] = True
            for j, gid in enumerate(ids):
                assert int(np.count_nonzero(mask & (gt == gid))) == int(inter[i, j])
                local_checks += 1
        ntkeys = {g['geometry_key'] for g in read(out / 'scenes' / scene / 'nt_geometry.json')}
        cov = coverage(iou, keys, selected, classes, ntkeys)
        for arm in ARMS:
            indices = [i for i, k in enumerate(keys) if arm != 'nt_only' or k in ntkeys]
            multi = (inter[indices] >= 100) & (inter[indices] >= .1 * counts[indices, None])
            cov[arm]['multi_significant_GT_geometry'] = int((multi.sum(axis=1) >= 2).sum())
        dest.mkdir()
        for arm in ARMS:
            ss = selected[arm]
            masks = np.zeros((len(gt), len(ss)), dtype=bool)
            for i, m in enumerate(ss): masks[points[m['geometry_key']], i] = True
            gm, pm = ev.assign_instances_for_scan(dict(pred_masks=masks,
                pred_classes=np.asarray([m['class_index'] for m in ss], dtype=np.int64),
                pred_scores=np.asarray([m['score'] for m in ss], dtype=np.float64)), spec['gt_path'])
            del masks
            matches[arm][scene] = serializable_matches(dict(gt=gm, pred=pm))
            write(dest / (arm + '_matches.json'), matches[arm][scene])
        coverage_by_scene[scene] = cov; bool_checks += local_checks
        write(dest / 'coverage.json', cov)
        freeze_json(dest / 'receipt.json', dict(complete=True, GT_sha256=spec['gt_sha256'],
            predictions_frozen_sha256=sha(out / 'predictions_frozen.json'),
            independent_boolean_intersections=local_checks,
            outputs={str(p): sha(p) for p in dest.iterdir() if p.is_file()}))
        print(f'AP matches {n}/90: {scene}', flush=True)
    groups = {**contract['batches'], 'combined90': contract['scenes']}
    pooled = {}; coverage_summary = {}; counts_summary = {}
    for group, scenes in groups.items():
        pooled[group] = {}; coverage_summary[group] = {}; counts_summary[group] = {}
        for arm in ARMS:
            write(out / 'execution.json', dict(status='running', stage='pooled_AP', group=group, arm=arm))
            aps, *_ = ev.evaluate_matches({s: matches[arm][s] for s in scenes})
            pooled[group][arm] = metrics(ev, aps)
            np.save(target / (group + '_' + arm + '_AP.npy'), aps, allow_pickle=False)
            covs = [coverage_by_scene[s][arm] for s in scenes]
            total = sum(c['GT'] for c in covs)
            coverage_summary[group][arm] = dict(GT=total,
                mean_best_iou=sum(c['best_iou_sum'] for c in covs)/total if total else None,
                pool={str(t): sum(c['pool'][str(t)] for c in covs) for t in THRESHOLDS},
                selected_correct_class={str(t): sum(c['selected_correct_class'][str(t)] for c in covs) for t in THRESHOLDS},
                multi_significant_GT_geometry=sum(c['multi_significant_GT_geometry'] for c in covs))
            stats = [read(out / 'scenes' / s / 'receipt.json')['arms'][arm] for s in scenes]
            counts_summary[group][arm] = {k: sum(c[k] for c in stats) for k in stats[0]}
        print(json.dumps(dict(group=group, AP=pooled[group])), flush=True)
    verify_files(seal['outputs']); verify_files(contract['inputs']); verify_files(contract['code'])
    summary = dict(complete=True, AP_units='fraction', pooled_AP=pooled, geometry_coverage=coverage_summary,
        candidate_counts=counts_summary, scenes=90, predictions=270,
        independent_boolean_intersections=bool_checks, scope=contract['scope'],
        evaluator='unchanged original evaluator AST; pooled matching, not mean per-scan AP',
        no_class_agnostic_AP=contract['class_agnostic_AP'], new_training=False, new_visual_inference=False,
        heldout_read=False, production_modified=False,
        nt_interpretation='conditional source-removal effect includes eligibility/source flags/context/B changes; not full removal of FI1',
        direct_interpretation='joint effect of learned scoring/context/competition supervision, not isolated internal contribution')
    freeze_json(target / 'summary.json', summary)
    return summary


def verify_predictions(out, contract):
    """Use independent loop context and independent sorting on every scene."""
    model = joblib.load(MODEL); checked = 0
    for scene in contract['scenes']:
        d = out / 'scenes' / scene; spec = contract['bindings'][scene]
        gs, points, evidence = load_geometry(spec)
        nt = read(d / 'nt_geometry.json'); nm = rows(d / 'nt_members.jsonl')
        # Independent source census, not source_pool().
        expected = {g['geometry_key'] for g in gs if any(m['candidate_source'] in ('native', 'track') for m in g['legacy_members'])}
        assert expected == {g['geometry_key'] for g in nt}
        assert all(a['source'] in ('native', 'track') for g in nt for a in original_aliases(g))
        nx = np.load(d / 'nt_features.npy', allow_pickle=False)
        keys = [g['geometry_key'] for g in nt]
        iou, cover = pair_overlap([points[k] for k in keys], spec['point_count'])
        independent = independent_context(nm, keys, iou, cover)
        assert np.allclose(independent, nx[:, 20:], atol=1e-14, rtol=0)
        own = {g['geometry_key']: geometry_features(g, evidence[g['geometry_key']]) for g in nt}
        assert np.array_equal(nx[:, :20], np.asarray([own[m['geometry_key']][m['class_index']] for m in nm]).reshape(-1,20))
        # Independently reconstruct the source-dependent class census.
        expected_members = set()
        for g in nt:
            old = {m['frozen_class_index'] for m in g['legacy_members'] if 0 <= m['frozen_class_index'] < 198}
            if any(m['candidate_source'] != 'native' for m in g['legacy_members']):
                old |= set(np.flatnonzero(own[g['geometry_key']][:, 0] > 0).tolist())
            if g['point_count']: expected_members |= {(g['geometry_key'], c) for c in old}
        assert expected_members == {member_key(m) for m in nm}
        scores = np.clip(model.predict(nx),0,1) if nm else np.empty(0)
        assert np.array_equal(scores, np.load(d / 'nt_scores.npy', allow_pickle=False))
        sorted_rows = sorted([{**m,'score':float(s)} for m,s in zip(nm,scores)],
                             key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:800]
        assert sorted_rows == read(d / 'nt_only_selected.json')
        original = rows(Path(spec['dataset']) / 'members.jsonl')
        assert sorted(original,key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:800] == read(d / 'direct_semantics_selected.json')
        assert read(d / 'full_selected.json') == read(Path(spec['reference']) / 'selected.json')
        checked += len(nm)
    freeze_json(out / 'predictions_verified.json', dict(complete=True, scenes=90,
        independent_NT_member_context_rows=checked, full_control_selection_exact=True,
        NT_alias_member_source_census=True, NT_scores_selection_replayed=True, direct_selection_replayed=True,
        contract_sha256=sha(out / 'contract.json')))


def run(out):
    contract = read(out / 'contract.json')
    assert sklearn.__version__ == contract['sklearn_version']
    guard = AccessGuard(contract['scenes'], [b['gt_path'] for b in contract['bindings'].values()])
    sys.addaudithook(guard.hook)
    verify_files(contract['inputs']); verify_files(contract['code'])
    assert read(out / 'tests_verified.json')['passed']
    assert read(out / 'code_audit.json')['complete']
    model = joblib.load(MODEL); completed = []
    for scene in contract['scenes']:
        write(out / 'execution.json', dict(status='running', stage='predictions', scene=scene, completed_scenes=completed))
        rc = predict_scene(out, scene, contract['bindings'][scene], model)
        completed.append(scene)
        print(f'predicted {len(completed)}/90: {scene} NT removed {rc["removed_unique_geometry"]} geometries', flush=True)
    outputs = {}
    for scene in contract['scenes']:
        p = out / 'scenes' / scene / 'receipt.json'
        outputs.update(read(p)['outputs']); outputs[str(p)] = sha(p)
    freeze_json(out / 'predictions_frozen.json', dict(complete=True, scene_arm_count=270,
        outputs=outputs, GT_arrays_read=False, contract_sha256=sha(out / 'contract.json')))
    write(out / 'execution.json', dict(status='running', stage='independent_prediction_verification'))
    verify_predictions(out, contract)
    summary = evaluate(out, contract, guard)
    assert len(guard.gt_opened) == 90 or all((out / 'evaluation' / s / 'receipt.json').exists() for s in contract['scenes'])
    freeze_json(out / 'completion_verified.json', dict(complete=True, scenes=90, predictions=270,
        all_predictions_before_GT=True, control_exact_replay=True, independent_prediction_verification=True,
        tests=read(out / 'tests_verified.json'), independent_boolean_intersections=summary['independent_boolean_intersections'],
        model_unchanged=True, new_training=False, new_visual_inference=False, heldout_read=False,
        production_modified=False, contract_sha256=sha(out / 'contract.json'),
        evaluation_sha256=sha(out / 'evaluation/summary.json'),
        predictions_frozen_sha256=sha(out / 'predictions_frozen.json')))
    write(out / 'execution.json', dict(status='complete', stage='complete', completed_scenes=completed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    assert out.is_relative_to(ROOT / '.local'), 'isolated project-local output required'
    os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    with threadpool_limits(limits=1):
        try:
            prepare(out) if args.command == 'prepare' else run(out)
        except Exception as exc:
            if out.exists(): write(out / 'execution.json', dict(status='failed', error=repr(exc), timestamp=time.time()))
            raise
