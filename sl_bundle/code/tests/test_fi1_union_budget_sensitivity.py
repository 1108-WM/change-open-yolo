from copy import deepcopy
import numpy as np
import pytest
from tools.fi1_union_budget_sensitivity import prefix_matches
from tools.fi1_union_loss_attribution import factorial_selections
from tools.fi1_r3b_delta_diagnostic_core import evaluator


def fixture(tmp_path):
    ev = evaluator(); ids = ev.VALID_CLASS_IDS
    gt = np.r_[np.full(120, int(ids[0])*1000+1), np.full(120, int(ids[1])*1000+1),
               np.full(40, int(ids[0])*1000+2), np.zeros(20, dtype=int)]
    path = tmp_path/'synthetic_gt.txt'; np.savetxt(path, gt, fmt='%d')
    masks = np.zeros((300, 6), dtype=bool)
    masks[:120, 0] = True; masks[:120, 1] = True; masks[240:, 2] = True
    masks[120:240, 3] = True; masks[:120, 4] = True; masks[120:, 5] = True
    cls = np.array([0, 0, 0, 1, 1, 1]); scores = np.array([.9, .9, .85, .8, .7, .6])
    selected = [dict(geometry_key=str(i), class_index=int(cls[i]), score=float(scores[i])) for i in range(6)]
    sizes = {str(i): int(masks[:, i].sum()) for i in range(6)}
    gm, pm = ev.assign_instances_for_scan(dict(pred_masks=masks, pred_classes=cls, pred_scores=scores), str(path))
    return ev, path, masks, cls, scores, selected, sizes, dict(gt=gm, pred=pm)


@pytest.mark.parametrize('budget', [1, 3, 4, 6])
def test_projection_equals_fresh_original_matching_and_AP(tmp_path, budget):
    ev, path, masks, cls, scores, selected, sizes, template = fixture(tmp_path)
    original = deepcopy(template)
    projected = prefix_matches(template, selected, sizes, ev.CLASS_LABELS, ev.VALID_CLASS_IDS, budget)
    gm, pm = ev.assign_instances_for_scan(dict(pred_masks=masks[:, :budget], pred_classes=cls[:budget],
                                             pred_scores=scores[:budget]), str(path))
    a, *_ = ev.evaluate_matches({'scan': projected})
    b, *_ = ev.evaluate_matches({'scan': dict(gt=gm, pred=pm)})
    assert np.array_equal(a, b, equal_nan=True)
    assert template == original
    # Tiny masks consume a slot but are absent from both matching directions.
    assert sum(map(len, projected['pred'].values())) == sum(sizes[str(i)] >= 100 for i in range(budget))
    for label in ev.CLASS_LABELS:
        assert len(projected['gt'][label]) == len(template['gt'][label])
        for p in projected['pred'][label]:
            assert p['void_intersection'] == template['pred'][label][projected['pred'][label].index(p)]['void_intersection']


def test_wrong_cached_alignment_rejected(tmp_path):
    ev, _, _, _, _, selected, sizes, template = fixture(tmp_path)
    selected[0]['score'] = .91
    with pytest.raises(ValueError, match='identity'):
        prefix_matches(template, selected, sizes, ev.CLASS_LABELS, ev.VALID_CLASS_IDS, 4)


def test_budget_prefix_preserves_all_four_frozen_score_interventions():
    old = [dict(geometry_key=k, class_index=0, has_original_native_alias=(k=='z')) for k in ('x', 'y', 'z')]
    new = old + [dict(geometry_key='u', class_index=1, has_original_native_alias=False)]
    full = factorial_selections(old, [.8, .7, .7], new, [.6, .5, .6, .9], 3)
    small = factorial_selections(old, [.8, .7, .7], new, [.6, .5, .6, .9], 2)
    assert all(small[a] == full[a][:2] for a in full)
