"""Independent R2 + Qwen experiment; retain all selected geometry-class members."""
from copy import deepcopy
import math

CONTRACT = {
    'version': 'fi1_r4_qwen_score_exchange_v1',
    'baseline': 'frozen_R2_full200_top600',
    'reference_only': 'R3b',
    'max_geometries_per_scan': 4,
    'views_per_geometry': 3,
    'candidate_pair': 'highest_R2_score_class_vs_full_Alpha_argmax_both_already_selected',
    'route_order': 'current_R2_score_desc_then_geometry_key_asc',
    'view_order': 'visible_ratio_desc_then_numeric_frame_id_asc',
    'crop': 'own_geometry_crop_scales_scale_index0_no_extra_resize',
    'action': 'exchange_two_existing_member_scores_only_if_AB_B_and_BA_A',
    'fallback': 'unchanged_on_invalid_abstain_order_disagreement_or_runtime_failure',
    'geometry_mutation': False, 'member_mutation': False, 'training': False,
    'model': 'local_Qwen2.5-VL-7B-Instruct_SHA256_bound',
    'dtype': 'bfloat16', 'attention': 'sdpa', 'device': 'cuda',
    'min_pixels': 100352, 'max_pixels': 200704, 'max_new_tokens': 8,
    'do_sample': False, 'seed': 20260914, 'max_attempts_per_order': 1,
}


def prompt(a, b):
    return (
        'The three images are fixed views of the same 3D object crop. '
        'Choose whether the visible object is better described by Option A or Option B. '
        'If unclear, partly absent, or neither option is reliably supported, abstain.\n'
        f'Option A: {a}\nOption B: {b}\n'
        'Return exactly one token: A, B, or ABSTAIN.'
    )


def parse(raw):
    if not isinstance(raw, str): return 'ABSTAIN', False
    value = raw.strip()
    return (value, True) if value in ('A', 'B', 'ABSTAIN') else ('ABSTAIN', False)


def decision(raw_ab, raw_ba):
    a, valid_a = parse(raw_ab); b, valid_b = parse(raw_ba)
    if not (valid_a and valid_b): return 'KEEP', 'invalid_output'
    if (a, b) == ('B', 'A'): return 'EXCHANGE', 'symmetric_proposed'
    if (a, b) == ('A', 'B'): return 'KEEP', 'symmetric_current'
    return 'KEEP', 'abstain_or_order_disagreement'


def validate_selected(rows):
    keys = [(r['geometry_key'], r['class_index']) for r in rows]
    if len(keys) != len(set(keys)) or len(keys) > 600:
        raise ValueError('invalid baseline member identities/budget')
    if any(type(r['class_index']) is not int or not 0<=r['class_index']<198 or
           not math.isfinite(r['score']) or not 0<=r['score']<=1 for r in rows):
        raise ValueError('invalid baseline scores/classes')


def route_pairs(selected, evidence):
    validate_selected(selected)
    groups = {}
    for r in selected: groups.setdefault(r['geometry_key'], []).append(r)
    pairs = []
    reasons = {}
    for key, rows in groups.items():
        alpha = evidence[key]['alpha']
        reason = None
        if not alpha['available']: reason = 'alpha_unavailable'
        else:
            cosine = alpha['cosine']
            if len(cosine)!=198 or not all(math.isfinite(v) for v in cosine): raise ValueError('invalid Alpha vector')
            proposed = min(range(198), key=lambda c: (-cosine[c], c))
            current = min(rows, key=lambda r: (-r['score'], not r['has_original_native_alias'], r['class_index']))
            alt = next((r for r in rows if r['class_index']==proposed), None)
            if proposed==current['class_index']: reason='current_Alpha_agree'
            elif alt is None: reason='Alpha_class_not_in_selected'
            elif current['score']<=alt['score']: reason='equal_scores_no_action'
            else:
                pairs.append({'geometry_key':key,'current_class_index':current['class_index'],
                    'proposed_class_index':proposed,'current_score':current['score'],'proposed_score':alt['score']})
        if reason: reasons[reason]=reasons.get(reason,0)+1
    return sorted(pairs, key=lambda r: (-r['current_score'],r['geometry_key'])), reasons


def choose_views(request):
    views = request['alpha']['view_requests']
    selected=[]; frames=set()
    for view in sorted(views,key=lambda v:(-v['visible_ratio'],int(v['frame_id']))):
        if view['visible_point_count']<=0 or view['visible_ratio']<=0: continue
        frame=int(view['frame_id'])
        if frame in frames: continue
        crops=[c for c in view['crop_scales'] if c['scale_index']==0]
        if len(crops)!=1:raise ValueError('missing/ambiguous scale zero crop')
        box=crops[0]['crop_xyxy'];h,w=view['image_shape']
        if len(box)!=4 or any(type(v) is not int for v in box) or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
            raise ValueError('invalid own-geometry crop bounds')
        selected.append({'frame_id':str(frame),'image_path':view['image_path'],'image_sha256':view['image_sha256'],
            'image_shape':view['image_shape'],'bbox_xyxy':box,'visible_ratio':view['visible_ratio']})
        frames.add(frame)
        if len(selected)==CONTRACT['views_per_geometry']:return selected
    return []


def apply_decisions(selected, tasks, outputs):
    """Keep list order, masks, class identities, aliases and per-geometry score multiset."""
    validate_selected(selected)
    expected={t['task_id']:t for t in tasks}
    if len(expected)!=len(tasks) or len({t['geometry_key'] for t in tasks})!=len(tasks):raise ValueError('duplicate task geometry/id')
    found={r['task_id']:r for r in outputs}
    if len(found)!=len(outputs) or set(found)!=set(expected):raise ValueError('missing/extra/duplicate outputs')
    result=deepcopy(selected); lookup={(r['geometry_key'],r['class_index']):i for i,r in enumerate(selected)}
    ledger=[]
    for key,t in expected.items():
        r=found[key]
        i=lookup[(t['geometry_key'],t['current_class_index'])];j=lookup[(t['geometry_key'],t['proposed_class_index'])]
        if i==j or selected[i]['score']!=t['current_score'] or selected[j]['score']!=t['proposed_score']:
            raise ValueError('task baseline scores or classes differ')
        if r.get('runtime_error'):
            action,reason='KEEP','runtime_error'
        else:action,reason=decision(r.get('raw_ab'),r.get('raw_ba'))
        if action=='EXCHANGE':result[i]['score'],result[j]['score']=result[j]['score'],result[i]['score']
        ledger.append({'task_id':key,'action':action,'reason':reason})
    validate_selected(result)
    return result,ledger
