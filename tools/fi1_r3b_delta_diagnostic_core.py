"""Fixed-budget interventions and original ScanNet AP kernels without Torch."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import types
from uuid import uuid4
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 'fi1-r3b-member-contribution-pilot-20260915-v1|'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def key(row):
    return row['geometry_key'], row['class_index']


def rank(rows):
    return sorted(rows, key=lambda r: (-r['score'], not r['has_original_native_alias'], *key(r)))


def action_hash(scene, row):
    return hashlib.sha256((SEED + scene + '|' + row['geometry_key'] + '|' + str(row['class_index'])).encode()).hexdigest()


def select_actions(scene, members, scores, y, competition):
    require(scores.shape == y.shape == competition.shape == (len(members),), 'label/score shape')
    require(np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all(), 'invalid scores')
    require(np.isfinite(y).all() and np.isfinite(competition).all()
            and ((0 <= competition) & (competition <= y) & (y <= 1)).all(), 'invalid labels')
    require(len({key(r) for r in members}) == len(members), 'duplicate members')
    scored = [{**r, 'score': float(s)} for r, s in zip(members, scores)]
    baseline = rank(scored)[:600]
    selected = {key(r) for r in baseline}
    strata = {f'{s}_{c}': [] for s in ('selected', 'unselected') for c in ('zero', 'positive')}
    for i, row in enumerate(scored):
        if y[i] <= .5:
            continue
        group = ('selected' if key(row) in selected else 'unselected') + '_' + ('zero' if competition[i] == 0 else 'positive')
        strata[group].append((i, row))
    actions = []
    for group, candidates in strata.items():
        for i, row in sorted(candidates, key=lambda pair: action_hash(scene, pair[1]))[:2]:
            inside = key(row) in selected
            new_score = None if inside or len(baseline) < 600 else float(np.nextafter(baseline[-1]['score'], np.inf))
            reason = None if inside else ('fewer_than_600' if len(baseline) < 600 else
                                         'threshold_at_one' if new_score > 1 else None)
            action = {'action_id': action_hash(scene, row), 'scene': scene, 'member_index': i,
                'geometry_key': row['geometry_key'], 'class_index': row['class_index'], 'stratum': group,
                'action': 'disable_and_refill' if inside else 'promote_and_reselect',
                'score_before': row['score'], 'new_score': new_score if reason is None else None,
                'maximum_IoU': float(y[i]), 'competition_IoU': float(competition[i]),
                'available': reason is None, 'unavailable_reason': reason}
            if action['available']:
                after = intervene(scored, [action])
                before_keys, after_keys = selected, {key(r) for r in after}
                require(len(before_keys - after_keys) == len(after_keys - before_keys) == 1,
                        'single action must replace exactly one budget member')
            actions.append(action)
    return sorted(actions, key=lambda a: a['action_id']), {g: len(v) for g, v in strata.items()}


def intervene(scored, actions):
    lookup = {key(r): r for r in scored}
    require(len(lookup) == len(scored), 'duplicate original members')
    removed, updates = set(), {}
    for a in sorted(actions, key=lambda a: a['action_id']):
        require(a['available'], 'unavailable action')
        k = key(a)
        require(k in lookup and lookup[k]['score'] == a['score_before'], 'action identity/score differs')
        require(k not in removed and k not in updates, 'duplicate intervention')
        if a['action'] == 'disable_and_refill':
            removed.add(k)
        else:
            require(a['action'] == 'promote_and_reselect' and np.isfinite(a['new_score'])
                    and lookup[k]['score'] < a['new_score'] <= 1, 'invalid promotion')
            updates[k] = a['new_score']
    return rank([{**r, 'score': updates.get(key(r), r['score'])} for r in scored if key(r) not in removed])[:600]


def load_definitions(path, names, namespace):
    nodes = [n for n in ast.parse(Path(path).read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    require({n.name for n in nodes} == set(names), 'evaluator definition set differs')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def evaluator():
    """Execute unmodified function ASTs; no custom matching or AP equations."""
    constants = runpy.run_path(str(ROOT / 'evaluate/scannet200/scannet_constants.py'))
    util = {'np': np, 'json': json}
    load_definitions(ROOT / 'evaluate/scannet200/util_3d.py', {'Instance', 'get_instances', 'load_ids'}, util)
    names = dict(zip(constants['VALID_CLASS_IDS_200'], constants['CLASS_LABELS_200']))
    ids = np.asarray(constants['VALID_CLASS_IDS_200_INST'])
    ns = dict(constants, np=np, copy=copy, deepcopy=copy.deepcopy, uuid4=uuid4,
        util_3d=types.SimpleNamespace(**util), DATASET_NAME='scannet200', VALID_CLASS_IDS=ids,
        CLASS_LABELS=tuple(names[i] for i in ids), ID_TO_LABEL={int(i): names[i] for i in ids},
        PRED_ID_TO_ID={**dict(enumerate(ids)), 198: -1},
        opt={'overlaps': np.append(np.arange(.5, .95, .05), .25), 'min_region_sizes': np.array([100]),
             'distance_threshes': np.array([float('inf')]), 'distance_confs': np.array([-float('inf')])})
    def fail(message, **kwargs):
        raise ValueError(message)
    ns['util'] = types.SimpleNamespace(print_error=fail)
    load_definitions(ROOT / 'evaluate/scannet200/eval_semantic_instance.py',
        {'evaluate_matches', 'compute_averages', 'make_pred_info', 'assign_instances_for_scan'}, ns)
    return types.SimpleNamespace(**ns)


def strip_analysis(value):
    if isinstance(value, dict):
        return {k: strip_analysis(v) for k, v in value.items() if k != 'analysis_sources'}
    if isinstance(value, list):
        return [strip_analysis(v) for v in value]
    return value


def metrics(ev, array):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        av = ev.compute_averages(array)
    return {name: float(av[k]) if np.isfinite(av[k]) else None
            for name, k in [('ap', 'all_ap'), ('ap50', 'all_ap_50%'), ('ap25', 'all_ap_25%')]}


def delta_metrics(before, after):
    return {k: after[k] - before[k] if after[k] is not None and before[k] is not None else None for k in before}


def sign(delta):
    return 'undefined' if delta is None else 'zero' if abs(delta) <= 1e-12 else 'positive' if delta > 0 else 'negative'
