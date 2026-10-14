#!/usr/bin/env python3
"""Independently rebuild class membership/scores/budget from frozen evidence.

Does not import the production score, member builder, or budget selector.
Checks both eligible and selected predictions, actual point sets, artifact
paths and identity links. Does not read labels or compute AP.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.prepare_fi1_r0_r1_inputs import load_partition, read, sha, digest, write


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_paths(receipt, binding, scene):
    """The files verified must be the files subsequently consumed."""
    expected = {'geometry_file': binding/scene/'geometry.json', 'points_file': binding/scene/'points.npz'}
    for key, path in expected.items():
        require(Path(receipt[key]).resolve() == path.resolve(), f'{scene}: audited/consumed {key} path differs')
    return expected


def rebuild(geometry, evidence):
    require(geometry['geometry_key'] == evidence['geometry_key'], 'evidence belongs to a different geometry')
    require(geometry['points_sha256'] == evidence['points_sha256'], 'evidence points differ')
    distributions = []
    for name in ['yolo', 'alpha']:
        item = evidence[name]
        require(item['acquisition_complete'] is True, f'{name} acquisition incomplete')
        require(type(item['available']) is bool, 'modality availability must be explicit')
        if not item['available']:
            require(bool(item.get('missing_reason')), 'missing evidence without reason')
            continue
        vector = np.array(item['votes' if name == 'yolo' else 'cosine'], dtype=np.float64)
        require(vector.shape == (198,) and np.isfinite(vector).all(), 'invalid class evidence')
        if name == 'yolo':
            require((vector >= 0).all() and (vector == np.floor(vector)).all() and vector.sum() > 0, 'invalid votes')
            distributions.append(vector/vector.sum())
        else:
            require((np.abs(vector) <= 1.00001).all(), 'invalid cosine')
            distributions.append(softmax((vector-vector.max())*100.))
    scores = sum(distributions)/len(distributions) if distributions else np.zeros(198)
    aliases = []
    for m in geometry['legacy_members']:
        aliases.append({'alias_key': f"{m['candidate_source']}:{m['candidate_id']}",
                        'source': m['candidate_source'], 'class_index': m['frozen_class_index']})
    for p in geometry['dv3_plans']:
        if p['append_only']:
            aliases.append({'alias_key': p['plan_key'], 'source': 'refined_union', 'class_index': p['frozen_class_index']})
    require(len({m['alias_key'] for m in aliases}) == len(aliases), 'duplicate alias')
    native = {m['class_index'] for m in aliases if m['source'] == 'native' and 0 <= m['class_index'] < 198}
    original = {m['class_index'] for m in aliases if 0 <= m['class_index'] < 198}
    expanded = original | ({int(c) for c in np.where(scores > 0)[0]} if any(m['source'] != 'native' for m in aliases) else set())
    return scores, original, expanded, native, aliases


def audit_members(geometry, evidence, actual):
    expected = {'R0': {}, 'R1': {}}
    stats = {'geometry_count':len(geometry), 'scored_geometry_count':0, 'alpha_positive_all198_count':0,
             'nonnative_geometry_count':0, 'max_score_abs_error':0.}
    all_native, background, empty = set(), 0, 0
    for g in geometry:
        score, original, expanded, native, aliases = rebuild(g, evidence[g['geometry_key']])
        key = g['geometry_key']
        background += sum(a['class_index'] in [-1,198] for a in aliases)
        if g['point_count'] == 0:
            empty += 1
            continue
        all_native.update((key,a['alias_key']) for a in aliases if a['source']=='native' and a['class_index'] in native)
        stats['scored_geometry_count'] += 1
        nonnative = any(a['source'] != 'native' for a in aliases)
        stats['nonnative_geometry_count'] += nonnative
        stats['alpha_positive_all198_count'] += nonnative and evidence[key]['alpha']['available'] and len(expanded)==198
        for arm, classes in [('R0',original),('R1',expanded)]:
            for c in classes:
                own = [a for a in aliases if a['class_index']==c]
                native_aliases = [a['alias_key'] for a in own if a['source']=='native']
                expected[arm][key,c] = {'geometry_key':key,'class_index':c,'score':float(score[c]),
                    'has_original_native_alias':c in native,'native_alias_keys':native_aliases,
                    'original_aliases':own,'expanded_class':c not in original}
    for arm in ['R0','R1']:
        ranked = sorted(expected[arm].values(), key=lambda r:(-r['score'],not r['has_original_native_alias'],r['geometry_key'],r['class_index']))
        require(len(ranked)==len(actual[arm]['eligible']), f'{arm}: eligible member count differs')
        for part, expected_rows in [('eligible',ranked),('selected',ranked[:600])]:
            require(len(actual[arm][part])==len(expected_rows), f'{arm}: {part} count differs')
            for want, got in zip(expected_rows, actual[arm][part]):
                require(set(want)==set(got), f'{arm}: member schema differs')
                for key in set(want)-{'score'}:
                    require(want[key]==got[key], f'{arm}: {part} {key} differs from independent reconstruction')
                error = abs(want['score']-got['score'])
                require(np.isfinite(got['score']) and error <= 1e-14, f'{arm}: class score differs')
                stats['max_score_abs_error'] = max(stats['max_score_abs_error'], error)
        selected = ranked[:600]
        kept_native = {(r['geometry_key'],a) for r in selected for a in r['native_alias_keys']}
        native_pre = {k for k,_ in all_native}
        native_post = {k for k,_ in kept_native}
        expected_audit = {'eligible_class_members':len(ranked),'selected_class_members':len(selected),
            'eligible_geometry_count':len({r['geometry_key'] for r in ranked}),
            'selected_geometry_count':len({r['geometry_key'] for r in selected}),
            'eligible_native_alias_count':len(all_native),'selected_native_alias_count':len(kept_native),
            'budget_evicted_native_alias_count':len(all_native-kept_native),
            'eligible_native_geometry_count':len(native_pre),'selected_native_geometry_count':len(native_post),
            'budget_evicted_native_geometry_count':len(native_pre-native_post),
            'legacy_background_sentinel_alias_count':background,'empty_geometry_count':empty,
            'selected_expanded_classes':sum(r['expanded_class'] for r in selected),
            'selected_original_alias_sources':dict(Counter(a['source'] for r in selected for a in r['original_aliases']))}
        require(expected_audit==actual[arm]['audit'], f'{arm}: budget audit differs')
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    pred = args.predictions.resolve()
    partition = load_partition(args.split)
    summary = read(pred/'summary.json')
    require(summary['complete'] and not summary['smoke'],'full prediction run required')
    evidence_root = Path(summary['evidence_root'])
    ei = read(evidence_root/'identity.json')
    require(digest(ei)==summary['evidence_identity_sha256'],'evidence identity mismatch')
    vi = read(Path(ei['views'])/'identity.json')
    require(digest(vi)==ei['views_identity_sha256'],'view identity mismatch')
    binding = Path(vi['binding'])
    bi = read(binding/'identity.json')
    require(digest(bi)==vi['binding_identity_sha256'],'binding identity mismatch')
    require(digest(partition)==bi['partition_id'],'partition mismatch')
    scenes = partition['source_scans'][bi['source']]
    require(scenes==bi['scenes']==vi['scenes']==ei['scenes']==sorted(r['scene_name'] for r in summary['rows']), 'source/scene coverage mismatch')
    require(digest(read(pred/'protocol.json'))==summary['protocol_sha256']==vi['protocol_sha256'],'protocol identity mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for scene in scenes:
        receipt = read(pred/scene/'receipt.json')
        require(receipt['evidence_receipt_sha256']==sha(evidence_root/scene/'receipt.json'),'prediction/evidence receipt link mismatch')
        er = read(evidence_root/scene/'receipt.json')
        require(er['identity_sha256']==digest(ei),'evidence scene identity mismatch')
        paths = checked_paths(receipt,binding,scene)
        files={str(pred/scene/k):v for k,v in receipt['files'].items()}
        files.update({str(paths['geometry_file']):receipt['geometry_sha256'],str(paths['points_file']):receipt['points_sha256']})
        files.update({str(evidence_root/scene/k):v for k,v in er['outputs'].items()})
        for path,h in files.items():
            require(sha(path)==h,f'{scene}: artifact hash changed: {path}')
        gs = read(paths['geometry_file'])
        ev = rows(evidence_root/scene/'evidence.jsonl')
        lookup = {r['geometry_key']:r for r in ev}
        require(len(lookup)==len(ev)==len(gs) and set(lookup)=={g['geometry_key'] for g in gs},'geometry/evidence coverage mismatch')
        with np.load(paths['points_file'],allow_pickle=False) as points:
            require(set(points.files)=={g['array_key'] for g in gs},'packed geometry coverage differs')
            for g in gs:
                p=points[g['array_key']]
                require(p.ndim==1 and p.dtype.kind in 'iu' and len(p)==g['point_count'] and np.array_equal(p,np.unique(p)),'invalid packed point set')
                h=hashlib.sha256(np.asarray(p,dtype='<i8').tobytes()).hexdigest()
                require(h==g['points_sha256'] and g['geometry_key']==f'{scene}:geometry:{h}','actual packed point hash mismatch')
        budget = read(pred/scene/'audit.json')
        actual = {arm:{**{part:rows(pred/scene/f'{arm}_{part}.jsonl') for part in ['eligible','selected']},
                       'audit':budget[arm]} for arm in ['R0','R1']}
        report = {'scene_name':scene,**audit_members(gs,lookup,actual),'verified_file_count':len(files)}
        reports.append(report)
        write(args.output/f'{scene}.json',report)
        print(scene, 'independent members/scores/budget/points passed',flush=True)
    write(args.output/'summary.json',{'audit_valid':True,'scene_count':len(reports),'rows':reports,
        'prediction_summary_path':str(pred/'summary.json'),'prediction_summary_sha256':sha(pred/'summary.json'),
        'partition_id':digest(partition),'independent_score_and_member_implementation':True,
        'new_ap':False,'ground_truth_read':False,'new_training':False,
        'code_sha256':sha(__file__), 'max_score_abs_error':max(r['max_score_abs_error'] for r in reports),
        'limitations':['Visual inference and raw frame projection are not replayed here.',
                       'This is not a second implementation of the ScanNet evaluator.',
                       'Historical supervision and RGB provenance limitations remain.']})


if __name__=='__main__':
    main()
