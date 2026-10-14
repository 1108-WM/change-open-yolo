import numpy as np
from tools.fi1_s_member_quality_competition import own_labels, competition_labels, compete


def member(k, c=0):
    return dict(geometry_key=k, class_index=c, has_original_native_alias=False)


def test_good_duplicate_quality_not_zero_and_direction_per_class():
    rows = [member('N'), member('T'), member('N', 1)]
    q, t = own_labels(rows, ['N', 'T'], np.array([[.76, .02], [.95, .01]]),
                      np.array([1001, 2001]), np.array([1, 2]))
    assert np.allclose(q, [.76, .95, .02])
    assert list(t) == [1001, 1001, -1]
    assert list(competition_labels(q, t, np.array([[0, 1], [1, 0], [2, 1]]))) == [2, 1, 0]


def test_distinct_targets_and_tied_identity_abstain():
    q, t = own_labels([member('N')], ['N'], np.array([[.7, .7]]), np.array([1001, 1002]), np.array([1]))
    assert t[0] == -1
    assert competition_labels(np.array([.9, .8]), np.array([1001, 1002]), np.array([[0, 1]]))[0] == 2


def test_witness_no_chain_and_budget_undo():
    rows = [member('a'), member('b'), member('c'), member('d')]
    scores, audit = compete(rows, [.9, .8, .7, .95], np.array([[0, 1], [1, 2]]), [.99, .99], budget=2)
    assert scores[1] < .8 and scores[2] == .7
    assert len(audit['actions']) == 1
    scores, audit = compete(rows, [.9, .8, .7, .95], np.array([[0, 1]]), [.99], budget=1)
    assert np.array_equal(scores, [.9, .8, .7, .95]) and audit['restored'] == 1


def test_class_specific_action_preserves_other_class():
    rows = [member('a'), member('b'), member('b', 1)]
    scores, audit = compete(rows, [.9, .8, .8], np.array([[0, 1]]), [.99])
    assert scores[1] < .8 and scores[2] == .8
