import numpy as np

from tools.fi1_anchor_local_decontamination import crop, positive_prompt_ids, local_domain


def view(group, reliable=(1,1,1,1), inside=(1,0,0,0), excluded=(0,1,1,1), valid=True):
    return dict(group=group, reliable=np.array(reliable, bool), inside=np.array(inside, bool),
                excluded=np.array(excluded, bool), valid=valid)


def run(views, core=(0,), local=(1,1,1,1), budget=1.):
    from tools.fi1_anchor_local_decontamination import PARAMETERS
    return crop(np.arange(4), np.array(core), np.array(local, bool), views,
                {**PARAMETERS, 'max_deleted_fraction':budget})


def test_occlusion_missing_and_unseen_never_supply_deletion():
    out, _ = run([view(0, reliable=(1,1,0,0)), view(1, reliable=(1,1,1,0))])
    assert out.tolist() == [0,2,3]


def test_same_pose_repeated_frames_are_one_vote():
    out, receipt = run([view(0), view(0)])
    assert out.tolist() == [0,1,2,3] and not receipt['accepted']


def test_any_positive_observation_vetoes_deletion():
    out, _ = run([view(0), view(1), view(2, inside=(1,1,0,0), excluded=(0,0,1,1))])
    assert out.tolist() == [0,1]


def test_core_and_fixed_local_domain_protected():
    out, _ = run([view(0, inside=(0,0,0,0), excluded=(1,1,1,1)),
                  view(1, inside=(0,0,0,0), excluded=(1,1,1,1))], core=(0,1), local=(1,1,1,0))
    assert out.tolist() == [0,1,3]


def test_large_action_abstains_instead_of_clipping_to_budget():
    out, receipt = run([view(0),view(1)], budget=.2)
    assert out.tolist() == [0,1,2,3] and receipt['reason'] == 'deletion_budget_abstain'


def test_identity_invalid_views_cannot_remove_points():
    out, receipt = run([view(0),view(1, valid=False)])
    assert out.tolist() == [0,1,2,3] and not receipt['accepted']


def test_prompt_points_stay_in_core_and_are_deterministic():
    xyz=np.column_stack((np.arange(5),np.zeros((5,2))))
    assert positive_prompt_ids(np.arange(5),xyz).tolist() == [2,0,4]


def test_local_domain_has_no_background_or_mask_dependency():
    xyz=np.array([[0,0,0],[1,0,0],[1.1,0,0]])
    assert local_domain(np.array([0,1]),xyz,.15).tolist() == [False,True]
    assert not local_domain(np.arange(3),xyz,.15).any()
