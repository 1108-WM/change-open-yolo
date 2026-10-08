#!/usr/bin/env python3
"""Replay R2/R3 AP and audit class/source/fold/duplicate diagnostics in a new directory."""
import argparse
from collections import Counter
import csv
import fcntl
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_fi1_r0_r1_inputs import read, write, sha, digest
from tools.run_fi1_r2 import rows, DEV_GT
from tools.fi1_r3_core import pair_overlap
from tools.fi1_r3_analysis_core import (SOURCES, geometry_sources, duplicate_counts,
    aggregate_duplicates, selection_change, filter_matches, weighted_errors, member_key)
from evaluate.scannet200 import eval_semantic_instance as ev

ARMS = ('R2', 'R3a', 'R3b')


def configure_evaluator():
    # Same 198 instance classes as evaluate(dataset='scannet200'). Only skip
    # head/common/tail presentation; AP matching and averaging are unchanged.
    ev.VALID_CLASS_IDS = np.array([ev.PRED_ID_TO_ID[i] for i in range(198)])
    ev.CLASS_LABELS = tuple(ev.ID_TO_LABEL[int(i)] for i in ev.VALID_CLASS_IDS)
    ev.ID_TO_LABEL = dict(zip(ev.VALID_CLASS_IDS, ev.CLASS_LABELS))
    ev.LABEL_TO_ID = {v: k for k, v in ev.ID_TO_LABEL.items()}
    ev.DATASET_NAME = 'scannet'
    assert list(ev.opt['min_region_sizes']) == [100]


def check(path, expected):
    if sha(Path(path)) != expected:
        raise ValueError(f'input identity changed: {path}')


def prediction_dir(base, scene, arm):
    return base / ('r2_v1/fit/predictions' if arm == 'R2' else 'r3_v1/predictions') / scene


def compact_metrics(aps):
    av = ev.compute_averages(aps)
    return {'ap': float(av['all_ap']), 'ap50': float(av['all_ap_50%']), 'ap25': float(av['all_ap_25%'])}


def public_json(value):
    if isinstance(value, dict):
        return {str(k): public_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def evaluate_group(out, scenes, arm, name, removed_source=None):
    from tools.run_fi1_r4_qwen import atomic
    if arm not in ARMS or removed_source not in (None, *SOURCES):
        raise ValueError('unknown arm/source')
    scenes = list(scenes)
    if not scenes or len(set(scenes)) != len(scenes):raise ValueError('empty/duplicate scenes')
    if any(Path(s).name != s for s in [name, *scenes]):raise ValueError('invalid group/scene name')
    dest = out/'evaluations'/f'{name}_{arm}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = {}
        for scene in scenes:
            rp = out/'scenes'/scene/'receipt.json'
            cache = rp.parent/f'{arm}_matches.pkl'
            check(cache, read(rp)['outputs'][cache.name])
            inputs[str(rp)] = sha(rp)
            inputs[str(cache)] = sha(cache)
        identity = {'scenes': scenes, 'arm': arm, 'name': name, 'removed_source': removed_source,
                    'inputs': inputs, 'class_labels': list(ev.CLASS_LABELS),
                    'class_ids': ev.VALID_CLASS_IDS.tolist(),
                    # repr preserves +inf/-inf thresholds without non-JSON numbers.
                    'options': {k: repr(np.asarray(v).tolist()) for k,v in ev.opt.items()},
                    'code': {str(p):sha(p) for p in [Path(__file__), Path(ev.__file__), ROOT/'tools/fi1_r3_analysis_core.py']}}
        receipt = dest.with_suffix('.receipt.json')
        if any(p.exists() for p in (dest, dest.with_suffix('.npy'), receipt)):
            if not receipt.exists():raise ValueError('unbound/partial AP cache; preserve it and use a new output directory')
            old = read(receipt)
            if old['identity'] != identity:raise ValueError('AP cache identity differs')
            for filename,h in old['outputs'].items():check(dest.parent/filename, h)
            if set(old['outputs']) != {dest.name, dest.with_suffix('.npy').name}:raise ValueError('AP cache output coverage differs')
            result = read(dest)
            aps = np.load(dest.with_suffix('.npy'), allow_pickle=False)
            validate_ap_array(aps)
            if result['scans'] != len(scenes) or result['scene_names'] != scenes or result['removed_source'] != removed_source or result['AP'] != public_json(compact_metrics(aps)):
                raise ValueError('AP cache grouping/metrics differ')
            classes = ev.compute_averages(aps)['classes']
            if set(result['classes']) != set(classes):raise ValueError('AP cache class coverage differs')
            for label,metrics in public_json(classes).items():
                if any(result['classes'][label][k] != v for k,v in metrics.items()):raise ValueError('AP cache class metrics differ')
            return result
        result = _evaluate_group(out, scenes, arm, name, removed_source)
        for path,h in inputs.items():check(path,h)
        for path,h in identity['code'].items():check(path,h)
        atomic(receipt, {'identity': identity, 'outputs': {p.name:sha(p) for p in (dest, dest.with_suffix('.npy'))}})
        return result


def validate_ap_array(aps):
    shape = (len(ev.opt['min_region_sizes']), len(ev.CLASS_LABELS), len(ev.opt['overlaps']))
    if aps.shape != shape or not np.issubdtype(aps.dtype, np.floating) or np.isinf(aps).any():
        raise ValueError('invalid AP array shape/type/value')
    finite = aps[np.isfinite(aps)]
    if np.any((finite < 0) | (finite > 1)):raise ValueError('AP outside [0,1]')


def _evaluate_group(out, scenes, arm, name, removed_source):
    dest = out/'evaluations'/f'{name}_{arm}.json'
    matches = {}
    for scene in scenes:
        with (out/'scenes'/scene/f'{arm}_matches.pkl').open('rb') as f:
            match = pickle.load(f)
        if removed_source:
            keep = {p['uuid'] for pp in match['pred'].values() for p in pp if removed_source not in p['analysis_sources']}
            match = filter_matches(match, keep)
        matches[scene] = match
    aps, *_ = ev.evaluate_matches(matches)
    validate_ap_array(aps)
    av = ev.compute_averages(aps)
    counts = Counter()
    scene_counts = Counter()
    for match in matches.values():
        for label, instances in match['gt'].items():
            n = sum(g['instance_id'] >= 1000 and g['vert_count'] >= 100 for g in instances)
            counts[label] += n
            scene_counts[label] += n > 0
    result = {'scans': len(scenes), 'scene_names': scenes, 'removed_source': removed_source,
              'AP': compact_metrics(aps), 'classes': {k: {**v, 'gt_instances': counts[k], 'gt_scans': scene_counts[k]} for k, v in av['classes'].items()}}
    np.save(dest.with_suffix('.npy'), aps)
    write(dest, public_json(result))
    print('AP', name, arm, result['AP'], flush=True)
    return public_json(result)


def prepare_scene(base, out, report, partition, dev_binding):
    scene = report['scene_name']; role = report['role']; dest = out/'scenes'/scene
    if (dest/'receipt.json').exists():
        old = read(dest/'receipt.json')
        for p, h in old['inputs'].items(): check(p, h)
        for p, h in old['outputs'].items(): check(dest/p, h)
        return old['summary']
    dest.mkdir(exist_ok=True)
    inputs = {}
    def bind(path, expected=None):
        path = Path(path)
        h = sha(path)
        if expected is not None and h != expected: raise ValueError(f'changed input {path}')
        inputs[str(path)] = h
    bind(report['R1_eligible'], report['R1_eligible_sha256'])
    bind(report['geometry_file'], report['geometry_sha256'])
    bind(report['points_file'], report['points_sha256'])
    for subroot, fname in [('r2_v1/dataset', 'features.npz'), ('r3_v1/dataset', 'context.npz')]:
        rp = base/subroot/scene/'receipt.json'; rec = read(rp); bind(rp)
        bind(rp.parent/fname, rec['outputs'][fname])
    members = rows(report['R1_eligible']); geoms = read(report['geometry_file'])
    keys = [g['geometry_key'] for g in geoms]; sources = {g['geometry_key']: geometry_sources(g) for g in geoms}
    if len(members) != report['member_count']: raise ValueError('member count differs')
    gtfile = base/'r2_v1/dataset/ground_truth_train'/f'{scene}.txt' if role == 'train' else DEV_GT/f'{scene}.txt'
    binding = report['gt_binding'] if role == 'train' else dev_binding[scene]
    bind(gtfile, binding['files'][str(gtfile)])
    gt = np.loadtxt(gtfile, dtype=np.int64)
    if len(gt) != binding['point_count']: raise ValueError('GT point domain differs')
    with np.load(report['points_file'], allow_pickle=False) as p:
        points = [p[g['array_key']] for g in geoms]
    iou, _ = pair_overlap(points, len(gt))
    point_lookup = dict(zip(keys, points))
    pre = duplicate_counts(members, keys, iou)
    with np.load(base/'r3_v1/dataset'/scene/'context.npz') as c:
        cx = c['context']; competition = c['y_competition'] if role == 'train' else None
    # Independent equality against every previously audited context row.
    hist = Counter(np.rint(np.expm1(cx[:,1])).astype(int).tolist())
    if dict(hist) != pre['degree_histogram'] or int((cx[:,0]>.5).sum()) != pre['members_with_neighbor']:
        raise ValueError('duplicate statistics disagree with frozen context')
    with np.load(base/'r2_v1/dataset'/scene/'features.npz') as a:
        maximum = a['y'] if role == 'train' else None
        weights = a['weights']
        for i, m in enumerate(members):
            if list(a['X'][i,14:18].astype(bool)) != [s in sources[m['geometry_key']] for s in SOURCES]:
                raise ValueError('geometry source differs from frozen features')
    summary = {'scene_name': scene, 'role': role, 'dataset_source': report['source'],
               'before_budget': pre, 'before_sources_exclusive': dict(Counter('+'.join(sources[m['geometry_key']]) for m in members)),
               'arms': {}, 'target_statistics': None}
    if role == 'train':
        summary['target_statistics'] = {'independent_positive': int((maximum>0).sum()), 'competition_positive': int((competition>0).sum()),
            'positive_removed': int(((maximum>0)&(competition==0)).sum()), 'positive_lowered_nonzero': int(((competition>0)&(competition<maximum)).sum())}
    selected = {}
    for arm in ARMS:
        pd = prediction_dir(base, scene, arm); rp = pd/('receipt.json' if arm == 'R2' else f'{arm}_receipt.json')
        pr = read(rp); bind(rp)
        expected_model = ('full200' if role != 'train' else f"fold{next(i for i, scans in enumerate(partition['training_heldout_folds']) if scene in scans)}")
        if Path(pr['model']).stem != (expected_model if arm == 'R2' else f'{arm}_{expected_model}'):
            raise ValueError('wrong downstream fold model')
        for name, h in pr['outputs'].items(): bind(pd/name, h)
        scores = np.load(pd/f'{arm}_eligible_scores.npy')
        chosen = rows(pd/f'{arm}_selected.jsonl')
        # Replay independently, not through selection implementation under test.
        order = sorted(range(len(members)), key=lambda i: (-float(scores[i]), not members[i]['has_original_native_alias'], members[i]['geometry_key'], members[i]['class_index']))[:600]
        if chosen != [{**members[i], 'score': float(scores[i])} for i in order]: raise ValueError('selected member replay differs')
        selected[arm] = chosen
        mask = np.zeros((len(gt), len(chosen)), dtype=bool)
        for j, m in enumerate(chosen): mask[point_lookup[m['geometry_key']], j] = True
        pred = {'pred_masks': mask, 'pred_classes': np.array([m['class_index'] for m in chosen]), 'pred_scores': np.array([m['score'] for m in chosen])}
        gg, pp = ev.assign_instances_for_scan(pred, str(gtfile))
        valid_chosen = [m for m in chosen if len(point_lookup[m['geometry_key']])>=100]
        uuid_sources = {}
        for preds in pp.values():
            for p in preds:
                m = valid_chosen[p['pred_id']]
                if ev.PRED_ID_TO_ID[m['class_index']] != p['label_id'] or m['score'] != p['confidence']: raise ValueError('source prediction alignment differs')
                p['analysis_sources'] = sources[m['geometry_key']]
                uuid_sources[p['uuid']] = p['analysis_sources']
        for inst in gg.values():
            for g in inst:
                for p in g['matched_pred']: p['analysis_sources'] = uuid_sources[p['uuid']]
        with (dest/f'{arm}_matches.pkl').open('wb') as f: pickle.dump({'gt': gg, 'pred': pp}, f, protocol=4)
        del mask, pred, gg, pp
        summary['arms'][arm] = {'after_budget': duplicate_counts(chosen, keys, iou),
            'after_sources_exclusive': dict(Counter('+'.join(sources[m['geometry_key']]) for m in chosen)),
            'unique_geometries': len({m['geometry_key'] for m in chosen}),
            'weighted_errors': weighted_errors(scores, maximum, competition, weights) if role=='train' else None}
    summary['R3b_vs_R2_selection'] = selection_change(selected['R2'], selected['R3b'], sources)
    write(dest/'receipt.json', {'inputs': inputs, 'outputs': {f'{a}_matches.pkl': sha(dest/f'{a}_matches.pkl') for a in ARMS}, 'summary': summary})
    print('scene audited', scene, role, flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=ROOT/'.local/fi1_r0_r4_preparation_20260913')
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args();base=args.base.resolve();out=args.output.resolve()
    configure_evaluator()
    identity_paths = [base/f for f in ['r2_v1/dataset/identity.json', 'r2_v1/dataset/summary.json', 'r2_v1/fit/summary.json', 'r2_v1/evaluation/summary.json', 'r2_v1/evaluation/gt_binding.json', 'r3_v1/summary.json', 'r3_v1/fit_summary.json', 'r3_v1/dataset/summary.json', 'r3_code_audit_v1/summary.json']]
    identity_paths += [Path(__file__), ROOT/'tools/fi1_r3_analysis_core.py', ROOT/'tools/fi1_r3_core.py', Path(ev.__file__)]
    identity = {'files': {str(x):sha(x) for x in identity_paths}, 'kind': 'posthoc_explanation_no_tuning',
        'source_ablation': 'remove all selected members whose geometry has source; no refill; mixed aliases nonadditive',
        'whole_system_oof':False, 'validation_is_blind':False}
    if out.exists():
        if read(out/'identity.json') != identity: raise ValueError('refusing changed run identity')
    else:
        out.mkdir(parents=True);write(out/'identity.json', identity)
    for name in ['scenes','evaluations']: (out/name).mkdir(exist_ok=True)
    ds=read(base/'r2_v1/dataset/summary.json');part=read(base/'r2_v1/dataset/identity.json')['partition']
    reports=ds['rows'];scene_names=[r['scene_name'] for r in reports]
    if len(scene_names)!=260 or len(set(scene_names))!=260 or set(scene_names)!=set(s for ss in part['roles'].values() for s in ss): raise ValueError('scan accounting differs')
    bindings={r['scene_name']:r for r in read(base/'r2_v1/evaluation/gt_binding.json')['rows']}
    summaries=[prepare_scene(base,out,r,part,bindings) for r in reports]
    groups={**part['roles'], **{f'fold{i}':scans for i,scans in enumerate(part['training_heldout_folds'])}}
    groups.update({f'source_{source}':scans for source,scans in part['source_scans'].items() if source!='ncs_validation60'})
    aggregate={}
    for name,scans in groups.items():
        rs=[r for r in summaries if r['scene_name'] in scans]
        aggregate[name]={'scan_count':len(rs),'before_budget':aggregate_duplicates([r['before_budget'] for r in rs]),'arms':{}}
        for arm in ARMS:
            aggregate[name]['arms'][arm]={'after_budget':aggregate_duplicates([r['arms'][arm]['after_budget'] for r in rs])}
            if all(r['role']=='train' for r in rs):
                aggregate[name]['arms'][arm]['mean_scene_weighted_errors']={k:float(np.mean([r['arms'][arm]['weighted_errors'][k] for r in rs])) for k in ['mse_max_iou','mse_competition']}
        aggregate[name]['selection_changes']={k:sum(r['R3b_vs_R2_selection'][k] for r in rs) for k in ['retained','added','removed']}
        if all(r['role']=='train' for r in rs):
            aggregate[name]['targets']={k:sum(r['target_statistics'][k] for r in rs) for k in rs[0]['target_statistics']}
    write(out/'statistics.json', {'groups':aggregate,'scenes':summaries,'total_members':sum(r['before_budget']['members'] for r in summaries)})
    results={}
    old2=read(base/'r2_v1/evaluation/summary.json')['AP'];old3=read(base/'r3_v1/summary.json')['AP']
    for name,scans in groups.items():
        if name=='train': continue  # folds and training sources retain their own AP denominators
        results[name]={}
        for arm in ARMS:
            result=evaluate_group(out,scans,arm,name);results[name][arm]=result
            if name in ('development_validation','diagnostic'):
                expected=old2[name] if arm=='R2' else old3[name][arm]
                if any(abs(result['AP'][k]-expected[k])>1e-12 for k in expected): raise ValueError('existing AP replay changed')
                csvpath=base/(f'r2_v1/evaluation/{name}.csv' if arm=='R2' else f'r3_v1/evaluation/{name}_{arm}.csv')
                for row in csv.DictReader(csvpath.open()):
                    actual=result['classes'][row['class']]
                    for k,v in [('ap','ap'),('ap50','ap50%'),('ap25','ap25%')]:
                        val=float(row[k]);new=actual[v]
                        if not ((np.isnan(val) and new is None) or (new is not None and abs(new-val)<=1e-12)): raise ValueError('class AP replay differs')
    for role in ('development_validation','diagnostic'):
        for source in SOURCES:
            name=f'{role}_without_{source}';results[name]={a:evaluate_group(out,part['roles'][role],a,name,source) for a in ('R2','R3b')}
    # Per-main-scene AP describes breadth; do not average it to recover global AP.
    for main in sorted({s[:9] for s in part['roles']['development_validation']}):
        scans=[s for s in part['roles']['development_validation'] if s[:9]==main];name=f'dev_main_{main}'
        results[name]={a:evaluate_group(out,scans,a,name) for a in ('R2','R3b')}
    write(out/'summary.json', {'complete':True, 'identity_sha256':sha(out/'identity.json'),
        'statistics_sha256':sha(out/'statistics.json'), 'all_existing_class_AP_replayed':True,
        'new_training':False,'evaluations':{g:{a:r['AP'] for a,r in arms.items()} for g,arms in results.items()},
        'scene_count':260,'whole_system_oof':False,'validation_is_blind':False})
    print('analysis complete',out,flush=True)


if __name__=='__main__': main()
