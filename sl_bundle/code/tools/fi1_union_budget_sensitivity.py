"""Prefix-budget diagnostics using frozen multiclass predictions and matches."""
from copy import deepcopy
from tools.fi1_union_loss_attribution import key


def prefix_matches(template, selected, sizes, labels, valid_ids, budget):
    """Project both directions of original matches without changing AP inputs.

    Class-list order is the original scored prediction order. Masks smaller than
    100 points are absent from matches, but still consume an output budget slot.
    """
    if type(budget) is not int or not 1 <= budget <= len(selected):
        raise ValueError('invalid prefix budget')
    if len({key(m) for m in selected}) != len(selected):
        raise ValueError('duplicate selected member')
    byclass = {label: [] for label in labels}
    for rank, m in enumerate(selected):
        if sizes[m['geometry_key']] >= 100:
            byclass[labels[m['class_index']]].append((rank, m))
    keep = set(); seen = set(); pred_id = {}
    for rank, m in enumerate(selected):
        if sizes[m['geometry_key']] >= 100:
            pred_id[rank] = len(pred_id)
    for ci, label in enumerate(labels):
        expected, actual = byclass[label], template['pred'][label]
        if len(expected) != len(actual):
            raise ValueError('cached class alignment differs')
        for (rank, m), p in zip(expected, actual):
            if (p['confidence'] != m['score'] or p['label_id'] != int(valid_ids[ci])
                    or p['vert_count'] != sizes[m['geometry_key']] or p['pred_id'] != pred_id[rank]):
                raise ValueError('cached member identity differs')
            if p['uuid'] in seen:
                raise ValueError('duplicate cached prediction UUID')
            seen.add(p['uuid'])
            if rank < budget:
                keep.add(p['uuid'])
    result = deepcopy(template)
    for label in labels:
        result['pred'][label] = [p for p in result['pred'][label] if p['uuid'] in keep]
        for g in result['gt'][label]:
            g['matched_pred'] = [p for p in g['matched_pred'] if p['uuid'] in keep]
        for p in result['pred'][label]:
            for g in p['matched_gt']:
                g.pop('matched_pred', None)  # Unused recursive field.
    return result


def conditional_effects(ap):
    pairs = [('B_minus_A', 'A', 'B'), ('C_minus_A', 'A', 'C'),
             ('D_minus_C', 'C', 'D'), ('D_minus_B', 'B', 'D'), ('D_minus_A', 'A', 'D')]
    result = {name: {m: 100 * (ap[after][m] - ap[before][m]) for m in ('ap', 'ap50', 'ap25')}
              for name, before, after in pairs}
    result['interaction_D_minus_B_minus_C_plus_A'] = {
        m: 100 * (ap['D'][m] - ap['B'][m] - ap['C'][m] + ap['A'][m])
        for m in ('ap', 'ap50', 'ap25')}
    return result
