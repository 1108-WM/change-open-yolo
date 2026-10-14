#!/usr/bin/env python3
"""Incremental, CPU-only expansion of the audited R3b head from 200 to 597.

Reuse the original member, feature, context, target, weight and HGB kernels.
Old 200 assets keep their historical lineage. New 397 use audited A assets.
Never fit a partial set; never fit on development or diagnostic labels.
"""
import argparse
from collections.abc import Mapping
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from tools.prepare_fi1_r0_r1_inputs import read, sha, digest, load_partition, main_scene
from tools.fi1_r0_r1_protocol import PROTOCOL, build_pair
from tools.fi1_r2_core import FEATURES, MODEL_PARAMS, geometry_features, geometry_targets, valid_instances, scene_weights, select
from tools.fi1_r3_core import EXTRA_FEATURES, pair_overlap, context_features, independent_context, gt_iou, assign_competition, independent_assignment
from tools.run_fi1_r3 import arm_features, competition_reference
from tools.run_fi1_r2 import CLASS_IDS, DEV_GT, rows
from tools.audit_fi1_r2_dataset import features as independent_features
from tools.evaluate_fi1_r0_r1_gt import reconstruct_gt
from tools.prepare_scannet200_train_scene import DEFAULT_LABEL_MAP
from tools.analyze_fi1_r3 import configure_evaluator
from evaluate.scannet200 import eval_semantic_instance as evaluator

OLD = ROOT / '.local/fi1_r0_r4_preparation_20260913'
BASE = Path('/media/jia/软件1/scannet_additional_train_stream')
OUT = BASE / 'fi1_R3b_train597_20260917'
LOCAL = ROOT / '.local/fi1_train597_20260917'
MANIFEST = ROOT / '资料/训练597场景清单_20260917.json'
FIRST = ROOT / '资料/scannet_new397_20260916_first199.txt'
REST = ROOT / '资料/scannet_new397_20260916_remaining198.txt'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def freeze(path, value):
    if path.exists():
        require(read(path) == value, 'frozen identity changed: ' + str(path))
    else:
        write(path, value)


def verify(files):
    for p, h in files.items():
        require(sha(p) == h, 'bound file changed: ' + str(p))


def files_at(folder, outputs):
    return {str(folder / k): h for k, h in outputs.items()}


def evidence_compatibility(bi, ei, vi, reference_evidence, reference_views):
    require(vi['binding_identity_sha256'] == digest(bi) and ei['views_identity_sha256'] == digest(vi), 'own geometry/view chain differs')
    for k in ['class_names', 'protocol', 'config_sha256']:
        require(vi[k] == reference_views[k], 'semantic configuration differs: ' + k)
    # Cache folders contain different scene inventories. Compare model/text
    # content identity instead of requiring the same physical cache folder.
    required = ['model_provenance.json', 'text_features.npy', 'text_prompts.json']
    for name in required:
        p = str(Path(ei['cache'])/name)
        q = str(Path(reference_evidence['cache'])/name)
        require(ei['cache_inputs'][p] == reference_evidence['cache_inputs'][q], 'Alpha model/text differs: '+name)
        verify({p:ei['cache_inputs'][p],q:reference_evidence['cache_inputs'][q]})
    model = read(Path(ei['cache'])/'model_provenance.json')
    for k in ['alpha_clip_base', 'alpha_clip_checkpoint', 'sam_checkpoint']:
        p = model[k]
        require(ei['cache_inputs'][p] == reference_evidence['cache_inputs'][p] == model[k+'_sha256'], 'Alpha weight identity differs')


def expanded_roles(part, added, validation):
    old = part['roles']['train']
    require(len(old) == len(set(old)) == 200, 'old200 coverage differs')
    require(len(added) == len(set(added)) == 397, 'additional397 coverage differs')
    mains = lambda ss: {main_scene(s) for s in ss}
    require(len(mains(added)) == 397, 'additional main-scene duplicates')
    require(not mains(added) & mains(old + part['roles']['development_validation'] + part['roles']['diagnostic'] + validation), 'additional main-scene leak')
    train = sorted(old + added)
    require(len(train) == len(set(train)) == 597, 'full597 coverage differs')
    require(not mains(train) & mains(part['roles']['development_validation'] + validation), 'train/development/val main-scene leak')
    return {**part['roles'], 'train': train}


def initialize():
    part = load_partition(OLD / 'approved_split_v1')
    master = ROOT / '资料/scannet_new397_20260914_master397.txt'
    val = ROOT / 'models/Mask3D/third_party/ScanNet/Tasks/Benchmark/scannetv2_val.txt'
    added = master.read_text().split()
    require(FIRST.read_text().split() + REST.read_text().split() == added, 'new397 partition differs')
    roles = expanded_roles(part, added, val.read_text().split())
    manifest = {'version': 'R3b_train597_v1', 'roles': roles, 'old200': part['roles']['train'],
        'new397': added, 'first199': FIRST.read_text().split(), 'remaining198': REST.read_text().split(),
        'counts': {k: {'scans': len(v), 'main_scenes': len({main_scene(s) for s in v})} for k, v in roles.items()},
        'selection': 'previously frozen scene lists; no GT/AP selection',
        'inputs': {str(p): sha(p) for p in [master, FIRST, REST, val]},
        'development_is_blind': False, 'diagnostic_overlaps_old_train_mains': True}
    freeze(MANIFEST, manifest)
    audited = competition_reference(OLD)
    r2 = read(OLD / 'r2_v1/dataset/identity.json')
    r3 = read(OLD / 'r3_v1/identity.json')
    require(r2['class_index_to_dataset_id'] == CLASS_IDS, 'class order differs')
    for name in ['fi1_r0_r1_protocol.py', 'fi1_r2_core.py']:
        p = ROOT / 'tools' / name
        require(sha(p) == r2['code'][str(p)], 'old R2 kernel changed')
    p = ROOT / 'tools/fi1_r3_core.py'
    require(sha(p) == r3['code'][str(p)], 'old R3 kernel changed')
    r2s = read(OLD / 'r2_v1/dataset/summary.json')
    require(r2s['complete'] and r2s['identity_sha256'] == digest(r2), 'R2 summary binding differs')
    audit2 = read(OLD / 'r2_code_audit_v1/summary.json')
    ai2 = read(OLD / 'r2_code_audit_v1/identity.json')
    require(audit2['complete'] and audit2['identity_sha256'] == sha(OLD / 'r2_code_audit_v1/identity.json') and ai2['dataset_summary_sha256'] == sha(OLD / 'r2_v1/dataset/summary.json'), 'R2 audit binding differs')
    for p in [OLD/'r2_v1/dataset/summary.json', OLD/'r2_v1/dataset/identity.json', OLD/'r2_code_audit_v1/summary.json', OLD/'r2_code_audit_v1/identity.json']:
        audited[str(p)] = sha(p)
    code = {str(ROOT/'tools'/n): sha(ROOT/'tools'/n) for n in [
        'run_fi1_train597.py', 'fi1_r0_r1_protocol.py', 'fi1_r2_core.py', 'fi1_r3_core.py',
        'run_fi1_r2.py', 'run_fi1_r3.py', 'audit_fi1_r2_dataset.py', 'evaluate_fi1_r0_r1_gt.py',
        'prepare_scannet200_train_scene.py', 'analyze_fi1_r3.py']}
    identity = {'manifest_sha256': sha(MANIFEST), 'features': FEATURES + EXTRA_FEATURES,
        'model_parameters': MODEL_PARAMS, 'target': 'original R3b y_competition',
        'weights': 'original equal scene, geometry, class; mean row weight1 per fit',
        'class_ids': CLASS_IDS, 'code': code, 'old_audited_files': audited,
        'old200_lineage': 'reuse audited historical R2/R3 arrays; not rerendered fresh A',
        'new397_lineage': 'A autoSAM uniform30, frozen full D-v3, own YOLO/Alpha',
        'scope': 'full597 R3b head only; frozen visual and D-v3 models; no new OOF claim',
        'no_partial_fit': True, 'validation_is_blind': False, 'whole_system_oof': False,
        'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
        'evaluation': 'unchanged historical dev55 and separate diagnostic5; compare full200 head',
        'protocol': PROTOCOL}
    freeze(OUT/'identity.json', identity)
    return manifest, identity, {r['scene_name']: r for r in r2s['rows']}


def coverage(gt):
    instances = valid_instances(gt, CLASS_IDS)
    return {str(c): sum(i//1000 == c for i in instances) for c in CLASS_IDS}


def validated_arrays(data):
    x = arm_features(data, 'R3b')
    w, y = data['weights'], data['y_competition']
    require(w.shape == y.shape == (len(x),) and len(x) > 0, 'array alignment differs')
    require(np.isfinite(w).all() and (w > 0).all() and abs(w.sum()-1) < 1e-12, 'invalid scene weights')
    require(np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all(), 'invalid targets')
    return x, y, w


def old_data(scene):
    a = OLD/'r2_v1/dataset'/scene/'features.npz'
    b = OLD/'r3_v1/dataset'/scene/'context.npz'
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        return {**dict(x), **dict(y)}


def old_receipt(scene, identity, reports):
    folder = OUT/'dataset'/scene
    r2p = OLD/'r2_v1/dataset'/scene/'receipt.json'
    r3p = OLD/'r3_v1/dataset'/scene/'receipt.json'
    r2, r3 = read(r2p), read(r3p)
    require(r2['summary'] == reports[scene] and r3['summary']['role'] == 'train', 'old train role/report differs')
    require(r2['identity_sha256'] == digest(read(OLD/'r2_v1/dataset/identity.json')) and r3['identity_sha256'] == digest(read(OLD/'r3_v1/identity.json')), 'old scene identity differs')
    bound = {**files_at(r2p.parent, r2['outputs']), **files_at(r3p.parent, r3['outputs'])}
    verify(bound)
    gtpath = OLD/'r2_v1/dataset/ground_truth_train'/f'{scene}.txt'
    verify({str(gtpath): r2['inputs'][str(gtpath)]})
    data = old_data(scene)
    x, y, w = validated_arrays(data)
    bound.update({str(p): sha(p) for p in [r2p, r3p, gtpath]})
    result = {'complete': True, 'scene': scene, 'kind': 'old200_reference', 'identity_sha256': digest(identity),
        'inputs': bound, 'outputs': {}, 'members': len(x), 'positive_targets': int((y > 0).sum()),
        'class_valid_instances': coverage(np.loadtxt(gtpath, dtype=np.int64))}
    write(folder/'receipt.json', result)
    return result


def compute(geometry, points, evidence, gt):
    members = build_pair(geometry, evidence, PROTOCOL['scene_budget'])['R1']['eligible']
    require(bool(members), 'empty R1 members')
    keys = [g['geometry_key'] for g in geometry]
    require(len(keys) == len(set(keys)) == len(points) and set(keys) == set(evidence), 'geometry/evidence coverage differs')
    gx = {}
    max_feature_error = 0.
    for g, pts in zip(geometry, points):
        require(g['point_count'] == len(pts) and g['points_sha256'] == hashlib.sha256(np.asarray(pts, dtype='<i8').tobytes()).hexdigest(), 'point identity differs')
        key = g['geometry_key']
        gx[key] = geometry_features(g, evidence[key])
        max_feature_error = max(max_feature_error, float(np.max(np.abs(gx[key] - independent_features(g, evidence[key])))))
    require(max_feature_error <= 1e-12, 'independent own features differ')
    x = np.asarray([gx[m['geometry_key']][m['class_index']] for m in members])
    require(np.allclose(x[:, 0], [m['score'] for m in members], rtol=0, atol=1e-14), 'member score differs')
    iou, cover = pair_overlap(points, len(gt))
    cx = context_features(members, keys, iou, cover)
    require(np.allclose(cx, independent_context(members, keys, iou, cover), rtol=0, atol=1e-14), 'independent context differs')
    for i in np.unique(np.linspace(0, len(points)-1, min(8, len(points))).astype(int)):
        for j in range(len(points)):
            inter = len(np.intersect1d(points[i], points[j], assume_unique=True))
            union = len(points[i]) + len(points[j]) - inter
            require(iou[i, j] == (inter / union if union else 0), 'independent point overlap differs')
    overlap, ids = gt_iou(points, gt, CLASS_IDS)
    y, assignment = assign_competition(members, keys, overlap, ids, CLASS_IDS)
    yy, aa = independent_assignment(members, keys, overlap, ids, CLASS_IDS)
    require(np.array_equal(y, yy) and np.array_equal(assignment, aa), 'independent assignment differs')
    instances = valid_instances(gt, CLASS_IDS)
    targets = {k: geometry_targets(p, gt, instances, CLASS_IDS) for k, p in zip(keys, points)}
    ymax = np.array([targets[m['geometry_key']][m['class_index']] for m in members])
    require((y <= ymax).all() and len(set(assignment[assignment >= 0])) == int((assignment >= 0).sum()), 'competition target constraint failed')
    data = dict(X=x, context=cx, weights=scene_weights(members), y=ymax, y_competition=y, assigned_gt=assignment)
    validated_arrays(data)
    return data, members, max_feature_error


def new_receipt(scene, manifest, identity):
    batch = 'first199' if scene in manifest['first199'] else 'remaining198'
    root = BASE/f'fi1_A_{batch}_20260917'/ 'scenes'/scene
    ap = root/'audit.json'
    if not ap.exists():
        return None
    a = read(ap)
    require(a['complete'] and a['scene'] == scene and a['own_evidence_coverage_exact'], 'incomplete own evidence audit')
    own = root/'own/A'
    binding = own/'geometry_binding'
    evidence = own/'alpha_completed'
    br = read(binding/scene/'receipt.json')
    er = read(evidence/scene/'receipt.json')
    bi, ei, vi = [read(p/'identity.json') for p in [binding, evidence, own/'views']]
    reference = OLD/'alpha_completed_v1/ncs_validation60'
    rei = read(reference/'identity.json')
    rvi = read(Path(rei['views'])/'identity.json')
    evidence_compatibility(bi, ei, vi, rei, rvi)
    require(br['identity_sha256'] == digest(bi) and er['identity_sha256'] == digest(ei), 'new evidence/geometry identity differs')
    require(vi['class_names'] == [evaluator.ID_TO_LABEL[c] for c in CLASS_IDS], 'new class mapping differs')
    bound = {**files_at(binding/scene, br['output_files']), **files_at(evidence/scene, er['outputs'])}
    verify(bound)
    # Original upstream audit has already checked all RGB-D and model files.
    # Bind receipts and their exact consumed outputs; do not reread every frame.
    for p in [ap, binding/scene/'receipt.json', evidence/scene/'receipt.json', binding/'identity.json', evidence/'identity.json', own/'views/identity.json']:
        bound[str(p)] = sha(p)
    for cache in [ei['cache'], rei['cache']]:
        for name in ['model_provenance.json','text_features.npy','text_prompts.json']:
            p=Path(cache)/name;bound[str(p)]=sha(p)
    gs = read(binding/scene/'geometry.json')
    evrows = rows(evidence/scene/'evidence.jsonl')
    ev = {r['geometry_key']: r for r in evrows}
    require(len(ev) == len(evrows), 'duplicate evidence keys')
    with np.load(binding/scene/'points.npz', allow_pickle=False) as p:
        points = [p[g['array_key']] for g in gs]
    prepared = BASE/'prepared'/scene/f'{scene[5:]}.npy'
    gtfile = BASE/'prepared/ground_truth'/f'{scene}.txt'
    gtcheck = reconstruct_gt(scene, BASE/'scans', prepared, DEFAULT_LABEL_MAP, gtfile)
    bound.update(gtcheck['files'])
    data, members, error = compute(gs, points, ev, np.loadtxt(gtfile, dtype=np.int64))
    verify(bound)
    folder = OUT/'dataset'/scene
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'features.npz', **data)
    # Member ordering is fixed by the original builder; a compact identity
    # avoids a second multi-GB copy of verbose eligible-member JSONL files.
    result = {'complete': True, 'scene': scene, 'kind': 'new397_A', 'identity_sha256': digest(identity),
        'inputs': bound, 'outputs': {'features.npz': sha(folder/'features.npz')},
        'members_sha256': digest(members), 'members': len(members), 'positive_targets': int((data['y_competition'] > 0).sum()),
        'class_valid_instances': coverage(np.loadtxt(gtfile, dtype=np.int64)),
        'raw_gt_reconstruction_equal': True, 'max_independent_feature_error': error,
        'context_and_assignment_independently_replayed': True, 'prepared_at': now()}
    write(folder/'receipt.json', result)
    return result


def now():
    return datetime.datetime.now().astimezone().isoformat()


def resources():
    mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    return mem['MemAvailable'], shutil.disk_usage(BASE).free


def training_gate(manifest, reports):
    wanted = manifest['roles']['train']
    require(len(wanted) == 597 and set(reports) == set(wanted), 'refusing partial/nontraining fit: require all597')
    require(all(r['complete'] and s == r['scene'] for s, r in reports.items()), 'incomplete/mismatched scene receipt')
    require({s for s,r in reports.items() if r['kind']=='old200_reference'} == set(manifest['old200']), 'old200 membership differs')
    require({s for s,r in reports.items() if r['kind']=='new397_A'} == set(manifest['new397']), 'new397 membership differs')


def train_and_evaluate(manifest, identity, receipts, reports):
    training_gate(manifest, receipts)
    verify(identity['code']); verify(identity['old_audited_files'])
    for s, r in receipts.items():
        verify(r['inputs']); verify(files_at(OUT/'dataset'/s, r['outputs']))
    ds = {s: sha(OUT/'dataset'/s/'receipt.json') for s in manifest['roles']['train']}
    freeze(OUT/'training_identity.json', {'identity_sha256': sha(OUT/'identity.json'), 'scenes': ds,
        'training_scenes': manifest['roles']['train'], 'no_validation_labels_in_fit': True})
    mp = OUT/'models/R3b_full597.joblib'
    fr = OUT/'models/receipt.json'
    if fr.exists():
        f = read(fr)
        require(f['training_identity_sha256'] == sha(OUT/'training_identity.json'), 'fit binding differs')
        verify({str(mp): f['model_sha256']})
        model = joblib.load(mp)
    else:
        count = sum(r['members'] for r in receipts.values())
        required = max(24*1024**3, count*26*8*5 + 12*1024**3)
        while resources()[0] < required:
            write(LOCAL/'execution.json', {'status':'waiting_for_fit_memory', 'required_available_bytes': required, 'completed':597, 'total':597, 'updated_at':now()})
            time.sleep(30)
        x = np.empty((count, 26)); y = np.empty(count); w = np.empty(count)
        start = 0
        for s in manifest['roles']['train']:
            if s in manifest['old200']:
                data = old_data(s)
            else:
                with np.load(OUT/'dataset'/s/'features.npz', allow_pickle=False) as z:
                    data = dict(z)
            xx, yy, ww = validated_arrays(data)
            end = start + len(xx); x[start:end], y[start:end], w[start:end] = xx, yy, ww; start = end
        require(start == count, 'fit row coverage differs')
        w *= len(w)/w.sum()
        write(LOCAL/'execution.json', {'status':'fitting_full597', 'completed':597, 'total':597, 'fit_rows':count, 'updated_at':now()})
        started = time.monotonic()
        model = HistGradientBoostingRegressor(**MODEL_PARAMS).fit(x, y, sample_weight=w)
        require(model.n_iter_ == 100 and model.n_features_in_ == 26, 'incomplete/wrong-dimensional fit')
        mp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, mp)
        saved = joblib.load(mp)
        # Complete row-wise persistence check in bounded chunks.
        for i in range(0, count, 100000):
            require(np.array_equal(model.predict(x[i:i+100000]), saved.predict(x[i:i+100000])), 'persisted model replay differs')
        write(fr, {'complete':True, 'training_identity_sha256':sha(OUT/'training_identity.json'), 'model_sha256':sha(mp),
            'fit_scans':597, 'fit_main_scenes':manifest['counts']['train']['main_scenes'], 'fit_rows':count,
            'seconds':time.monotonic()-started, 'full_training_row_persistence_replay_exact':True,
            'independent_refit_performed':False})
        del x, y, w, data, xx, yy, ww
    evaluate(model, manifest, reports)


def evaluate(model, manifest, reports):
    configure_evaluator()
    selected = {}; geometry = {}; counts = {}; inputs = {}
    gt_binding = read(OLD/'r2_v1/evaluation/gt_binding.json')
    gt_records = {r['scene_name']:r for r in gt_binding['rows']}
    require(sha(OLD/'r2_v1/evaluation/gt_binding.json') == read(OLD/'r2_v1/evaluation/summary.json')['gt_binding_sha256'], 'development GT binding differs')
    old_model_path = OLD/'r3_v1/models/R3b_full200.joblib'
    fit = read(OLD/'r3_v1/fit_summary.json')
    require(sha(old_model_path) == next(r['model_sha256'] for r in fit['rows'] if r['name']=='R3b_full200'), 'old model differs')
    old_model = joblib.load(old_model_path)
    inputs[str(old_model_path)] = sha(old_model_path)
    dest = OUT/'evaluation'; dest.mkdir(exist_ok=True)
    for s in manifest['roles']['development_validation'] + manifest['roles']['diagnostic']:
        r = reports[s]
        for p, h in [(r['R1_eligible'],r['R1_eligible_sha256']), (r['geometry_file'],r['geometry_sha256']), (r['points_file'],r['points_sha256'])]:
            inputs[p] = h
        for sub, name in [('r2_v1','features.npz'), ('r3_v1','context.npz')]:
            rr = read(OLD/sub/'dataset'/s/'receipt.json')
            inputs.update(files_at(OLD/sub/'dataset'/s, rr['outputs']))
        gtpath = DEV_GT/f'{s}.txt'
        inputs[str(gtpath)] = gt_records[s]['files'][str(gtpath)]
        verify(inputs)
        members = rows(r['R1_eligible']); data = old_data(s); x = arm_features(data, 'R3b')
        geometry[s] = {g['geometry_key']:g for g in read(r['geometry_file'])}
        counts[s] = gt_records[s]['point_count']
        for name, m in [('full597',model), ('full200',old_model)]:
            scores = np.clip(m.predict(x), 0, 1)
            chosen = select(members, scores)
            expected = sorted([{**v,'score':float(score)} for v,score in zip(members,scores)], key=lambda v:(-v['score'],not v['has_original_native_alias'],v['geometry_key'],v['class_index']))[:600]
            require(chosen == expected, 'independent top600 differs')
            if name == 'full200':
                require(chosen == rows(OLD/'r3_v1/predictions'/s/'R3b_selected.jsonl'), 'old200 selected baseline changed')
            selected[s, name] = chosen
            write(dest/'predictions'/s/f'{name}.json', chosen)
    class Predictions(Mapping):
        def __init__(self, scenes, name): self.scenes, self.name = scenes, name
        def __iter__(self): return iter(self.scenes)
        def __len__(self): return len(self.scenes)
        def __getitem__(self, s):
            if s not in self.scenes: raise KeyError(s)
            rr = selected[s, self.name]
            mask = np.zeros((counts[s],len(rr)), dtype=bool)
            with np.load(reports[s]['points_file'], allow_pickle=False) as pts:
                for i,r in enumerate(rr): mask[pts[geometry[s][r['geometry_key']]['array_key']],i] = True
            return {'pred_masks':mask,'pred_classes':np.array([r['class_index'] for r in rr],dtype=np.int64), 'pred_scores':np.array([r['score'] for r in rr])}
    ap = {}
    for role in ['development_validation','diagnostic']:
        ap[role] = {}
        for name in ['full200','full597']:
            with (dest/f'{role}_{name}.log').open('w') as log, contextlib.redirect_stdout(log):
                av,*_ = evaluator.evaluate(Predictions(manifest['roles'][role],name),str(DEV_GT),str(dest/f'{role}_{name}.csv'),dataset='scannet200')
            ap[role][name] = {k: float(av[v]) for k,v in {'ap':'all_ap','ap50':'all_ap_50%','ap25':'all_ap_25%'}.items()}
            write(dest/'progress.json',ap)
    write(dest/'summary.json', {'complete':True,'AP':ap,'inputs':inputs,'same_historical_development_geometry':True,
        'validation_is_blind':False,'diagnostic_is_independent':False,'AP_units':'fraction'})


def smoke_old(scene, reports):
    r = reports[scene]
    rr = read(OLD/'r2_v1/dataset'/scene/'receipt.json')
    evidence = Path(read(Path(r['R1_eligible']).parent.parent/'summary.json')['evidence_root'])/scene/'evidence.jsonl'
    verify({r['R1_eligible']:r['R1_eligible_sha256'],r['geometry_file']:r['geometry_sha256'],r['points_file']:r['points_sha256']})
    er = read(evidence.parent/'receipt.json'); verify({str(evidence):er['outputs']['evidence.jsonl']})
    gtfile = OLD/'r2_v1/dataset/ground_truth_train'/f'{scene}.txt'
    verify({str(gtfile):rr['inputs'][str(gtfile)]})
    gs = read(r['geometry_file'])
    with np.load(r['points_file'],allow_pickle=False) as p: points = [p[g['array_key']] for g in gs]
    actual,members,error = compute(gs,points,{e['geometry_key']:e for e in rows(evidence)},np.loadtxt(gtfile,dtype=np.int64))
    require(members == rows(r['R1_eligible']), 'old member replay differs')
    old = old_data(scene)
    for k in actual: require(np.array_equal(actual[k],old[k]), 'old full-array replay differs: '+k)
    return {'complete':True,'scene':scene,'all_arrays_exact':True,'members_exact':True,'max_feature_error':error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','smoke','run'])
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--fit-when-ready',action='store_true')
    args = parser.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    with (LOCAL/'run.lock').open('a') as lock, threadpool_limits(limits=2):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,identity,reports = initialize()
        if args.action == 'init': print(manifest['counts']); return
        if args.action == 'smoke':
            old = smoke_old(manifest['old200'][0],reports)
            new = new_receipt(manifest['first199'][0],manifest,identity)
            require(new is not None,'pilot A scene unavailable')
            write(LOCAL/'smoke.json',{'complete':True,'old':old,'new':new,'identity_sha256':sha(OUT/'identity.json')})
            print('SMOKE PASSED',old['scene'],new['scene'],flush=True); return
        smoke = read(LOCAL/'smoke.json')
        require(smoke['complete'] and smoke['identity_sha256']==sha(OUT/'identity.json'),'require smoke for current identity')
        done = {}
        for s in manifest['roles']['train']:
            rp = OUT/'dataset'/s/'receipt.json'
            if rp.exists():
                r=read(rp);require(r['identity_sha256']==digest(identity) and r['scene']==s and r['complete'],'resume identity differs')
                verify(r['inputs']);verify(files_at(rp.parent,r['outputs']));done[s]=r
        while True:
            verify(identity['code'])
            for s in manifest['old200'] + manifest['new397']:
                if s in done: continue
                if s not in manifest['old200']:
                    batch='first199' if s in manifest['first199'] else 'remaining198'
                    if not (BASE/f'fi1_A_{batch}_20260917/scenes'/s/'audit.json').exists(): continue
                if resources()[0]<12*1024**3 or resources()[1]<50*1024**3: break
                write(LOCAL/'execution.json',{'status':'preparing','pid':os.getpid(),'current':s,'completed':len(done),'total':597,'updated_at':now()})
                r = old_receipt(s,identity,reports) if s in manifest['old200'] else new_receipt(s,manifest,identity)
                if r is not None:
                    done[s]=r;print('PREPARED',len(done),597,s,r['members'],flush=True)
            summary={'complete':len(done)==597,'prepared':len(done),'total':597,'old200_ready':sum(s in done for s in manifest['old200']),
                'new397_ready':sum(s in done for s in manifest['new397']),'missing':[s for s in manifest['roles']['train'] if s not in done],
                'rows':[done[s] for s in sorted(done)],'updated_at':now(),'identity_sha256':digest(identity)}
            write(OUT/'dataset/summary.json',summary)
            groups={k:[done[s] for s in ss if s in done] for k,ss in [('old200',manifest['old200']),('new397_ready',manifest['new397']),('all_ready',manifest['roles']['train'])]}
            write(OUT/'dataset/class_coverage.json',{'prepared_scans':len(done),'target_scans':597,'complete':len(done)==597,
                'classes':[{ 'dataset_id':c,'name':evaluator.ID_TO_LABEL[c],**{g:{'valid_instances':sum(r['class_valid_instances'][str(c)] for r in rs),
                    'scans_with_valid_instances':sum(r['class_valid_instances'][str(c)]>0 for r in rs)} for g,rs in groups.items()}} for c in CLASS_IDS]})
            if len(done)==597:
                if args.fit_when_ready:
                    train_and_evaluate(manifest,identity,done,reports)
                write(LOCAL/'execution.json',{'status':'complete' if args.fit_when_ready else 'dataset_ready','completed':597,'total':597,'new_training':args.fit_when_ready,'updated_at':now()})
                return
            write(LOCAL/'execution.json',{'status':'waiting_for_assets_or_resources','pid':os.getpid(),'completed':len(done),'total':597,'old200_ready':summary['old200_ready'],'new397_ready':summary['new397_ready'],'updated_at':now()})
            if not args.watch:return
            time.sleep(30)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        write(LOCAL/'failure.json',{'error':repr(exc),'time':now(),'pid':os.getpid()})
        raise
