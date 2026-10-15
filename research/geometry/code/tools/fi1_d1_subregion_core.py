"""Frozen post-merge D1 readouts; no labels, reference gates, or mask inference."""
from collections import defaultdict

import numpy as np


PARAMETERS = dict(min_visible_ratio=.10, min_mask_support=.30,
                  min_visible_points=3, frame_coverage=.50,
                  min_support_frames=2, min_consensus_rate=.30,
                  mean_supported_coverage=.55)


def frame_unions(observation_ids, observations, pointsets):
    grouped = defaultdict(list)
    for oid in sorted(set(map(int, observation_ids))):
        grouped[int(observations[oid]['frame_index'])].append(oid)
    return [(fi, ids, np.unique(np.concatenate([pointsets[k] for k in ids])))
            for fi, ids in sorted(grouped.items())]


def partition(superpoints, support):
    """Maximal exact mask-signature classes inside each original SP, including zero."""
    packed = np.packbits(support.T, axis=1, bitorder='little')
    keys = np.column_stack((superpoints, packed))
    unique, labels = np.unique(keys, axis=0, return_inverse=True)
    return labels, unique[:, 0].astype(np.int64)


def readout(labels, support, visibility):
    assert support.shape == visibility.shape
    assert not (support & ~visibility).any()
    sizes = np.bincount(labels)
    nr = len(sizes)
    inside = np.asarray([np.bincount(labels[row], minlength=nr) for row in support])
    visible = np.asarray([np.bincount(labels[row], minlength=nr) for row in visibility])
    coverage = inside / np.maximum(1, visible)
    initial = ((visible > 0) & (visible / sizes >= PARAMETERS['min_visible_ratio'])
               & (coverage >= PARAMETERS['min_mask_support'])).any(axis=0)
    eligible = visible >= PARAMETERS['min_visible_points']
    votes = eligible & (coverage >= PARAMETERS['frame_coverage'])
    sc, vc = votes.sum(axis=0), eligible.sum(axis=0)
    rate = sc / np.maximum(1, vc)
    mean = (coverage * votes).sum(axis=0) / np.maximum(1, sc)
    kept = (initial & (sc >= PARAMETERS['min_support_frames'])
            & (rate >= PARAMETERS['min_consensus_rate'])
            & (mean >= PARAMETERS['mean_supported_coverage']))
    return dict(labels=labels, sizes=sizes, inside=inside, visible=visible,
                initial=initial, kept=kept, support_frames=sc, visible_frames=vc,
                consensus_rate=rate, mean_supported_coverage=mean)


def track_readout(domain, superpoints, support, visibility):
    assert len(domain) and support.shape[1] == len(domain)
    sp_ids, whole_labels = np.unique(superpoints, return_inverse=True)
    sub_labels, sub_sp = partition(superpoints, support)
    whole = readout(whole_labels, support, visibility)
    sub = readout(sub_labels, support, visibility)
    assert whole['initial'].all(), 'common domain must be historical D1 initial SPs'
    # A retained exact-signature atom cannot contain any unobserved point.
    assert (support[:, sub['kept'][sub_labels]].sum(axis=0) >= 2).all()
    return sp_ids, sub_sp, whole, sub
