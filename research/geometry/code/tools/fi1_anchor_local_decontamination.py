"""One fixed, GT-free local deletion operator for paired mask evidence."""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

PARAMETERS = dict(
    scenes_per_fold_per_group=1, candidates_per_source=4,
    min_parent_points=100, max_parent_points=100000,
    min_core_points=20, core_parent_fraction=.95, max_core_trials=5,
    min_core_visible_points=10, min_core_visible_fraction=.25,
    min_parent_visible_points=20, min_parent_visible_fraction=.10,
    core_mask_coverage=.95, views=3, min_pose_groups=2,
    measured_depth_tolerance=.02, pixel_exclusion_margin=2,
    local_boundary_radius=.15, max_deleted_fraction=.20,
    sam_quality_min=.88, sam_stability_min=.95,
    old_replay_iou_min=.999, prompt_points=3,
)
GATE = dict(min_new_actions=5, min_action_scenes=3,
            min_value_targets=3, min_value_scenes=2,
            pool_iou_improvement=.02, paired_iou_improvement=.01,
            deletion_pollution_precision=.90, max_target_loss_fraction=.01,
            max_per_action_target_loss_fraction=.05,
            max_unknown_action_fraction=.10, max_identity_drift=0,
            max_parent_high_iou_losses=0)
THRESHOLDS = (.50, .75, .80, .85, .90, .95)


def local_domain(parent, xyz, radius=PARAMETERS['local_boundary_radius']):
    """Freeze the editable boundary from input geometry, shared by both arms."""
    parent = np.asarray(parent, dtype=np.int64)
    outside = np.ones(len(xyz), dtype=bool)
    outside[parent] = False
    if not outside.any():
        return np.zeros(len(parent), dtype=bool)
    distances, _ = cKDTree(xyz[outside]).query(xyz[parent], distance_upper_bound=radius)
    return np.isfinite(distances)


def positive_prompt_ids(core, xyz, count=3):
    """Deterministic center-first farthest sampling within one visible core."""
    core = np.asarray(core, dtype=np.int64)
    if not len(core):
        return core.copy()
    cloud = xyz[core]
    first = int(np.argmin(np.sum((cloud-cloud.mean(axis=0))**2, axis=1)))
    selected = [first]
    distance = np.sum((cloud-cloud[first])**2, axis=1)
    while len(selected) < min(count, len(core)):
        distance[selected] = -1
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.sum((cloud-cloud[index])**2, axis=1))
    return core[selected]


def crop(parent, core, local, views, parameters=PARAMETERS):
    """Two independent exclusions, no positive veto, local points only.

    A view carries reliable visibility, mask inclusion and margin-qualified
    exclusion for aligned parent points. Unknown or occluded pixels cannot
    supply exclusion. Invalid identity views are omitted, never background.
    """
    parent = np.asarray(parent, dtype=np.int64)
    local = np.asarray(local, dtype=bool)
    if local.shape != parent.shape or not np.array_equal(parent, np.unique(parent)):
        raise ValueError('unaligned parent/local domain')
    groups = defaultdict(lambda: np.zeros(len(parent), dtype=bool))
    supported = np.zeros(len(parent), dtype=bool)
    reliable = np.zeros(len(parent), dtype=bool)
    for view in views:
        if not view['valid']:
            continue
        known = np.asarray(view['reliable'], dtype=bool)
        inside = np.asarray(view['inside'], dtype=bool)
        excluded = np.asarray(view['excluded'], dtype=bool)
        if any(a.shape != parent.shape for a in (known, inside, excluded)):
            raise ValueError('unaligned observation')
        if np.any(inside & excluded):
            raise ValueError('mask support and exclusion contradict')
        groups[int(view['group'])] |= known & excluded
        supported |= known & inside
        reliable |= known
    if len(groups) < parameters['min_pose_groups']:
        return parent.copy(), dict(accepted=False, reason='insufficient_identity_views',
                                  valid_groups=len(groups), deleted=0)
    votes = np.sum(list(groups.values()), axis=0)
    protected = np.isin(parent, core)
    remove = local & ~protected & ~supported & (votes >= parameters['min_pose_groups'])
    proposed = int(remove.sum())
    reason = ('no_reliable_local_exclusion' if proposed == 0 else
              'deletion_budget_abstain' if proposed > parameters['max_deleted_fraction']*len(parent) else
              'accepted')
    accepted = reason == 'accepted'
    kept = parent[~remove] if accepted else parent.copy()
    return kept, dict(accepted=accepted, reason=reason, valid_groups=len(groups),
                     proposed_deleted=proposed, deleted=len(parent)-len(kept),
                     local_points=int(local.sum()), protected_core_points=int(protected.sum()),
                     reliable_observed_points=int(reliable.sum()),
                     unseen_or_unreliable_retained=int((~reliable).sum()))
