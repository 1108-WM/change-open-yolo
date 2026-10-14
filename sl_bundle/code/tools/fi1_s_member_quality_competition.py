"""Class-member quality and directed competition; inference uses no GT."""
import numpy as np

THRESHOLDS = np.asarray([.5, .55, .6, .65, .7, .75, .8, .85, .9])
ANCHOR = .5
CONFIDENCE = .9
ATTENUATION = .5
NEIGHBORS = 3


def candidate_edges(members, keys, iou, sizes):
    """Top three overlapping same-class neighbors, both directions, any source.

    Topology is fixed from point IoU before any labels or learned scores.
    Small masks remain in the output pool, but cannot witness replacement.
    """
    index = {k: i for i, k in enumerate(keys)}
    by_class = {}
    for i, m in enumerate(members):
        if sizes[index[m['geometry_key']]] >= 100:
            by_class.setdefault(m['class_index'], []).append(i)
    pairs = set()
    for rows in by_class.values():
        for a in rows:
            ga = index[members[a]['geometry_key']]
            eligible = [b for b in rows if b != a and iou[ga, index[members[b]['geometry_key']]] > 0]
            eligible.sort(key=lambda b: (-iou[ga, index[members[b]['geometry_key']]], members[b]['geometry_key']))
            for b in eligible[:NEIGHBORS]:
                pairs.add((min(a, b), max(a, b)))
    return np.asarray(sorted((a, b) for a, b in pairs for a, b in ((a, b), (b, a))), dtype=np.int64).reshape(-1, 2)


def edge_features(x26, frozen_scores, edges, geometry_index, iou, coverage):
    """Winner own20, loser own20, differences, frozen scores and point relation."""
    a, b = edges.T
    ga, gb = geometry_index[a], geometry_index[b]
    return np.column_stack((x26[a, :20], x26[b, :20], x26[a, :20] - x26[b, :20],
                            frozen_scores[a], frozen_scores[b], frozen_scores[a] - frozen_scores[b],
                            iou[ga, gb], coverage[ga, gb], coverage[gb, ga]))


def own_labels(members, keys, overlaps, gt_ids, class_ids):
    """Independent same-class maximum IoU, allowing several good duplicates."""
    lookup = {k: i for i, k in enumerate(keys)}
    quality = np.zeros(len(members))
    target = np.full(len(members), -1, dtype=np.int64)
    for i, m in enumerate(members):
        cols = np.flatnonzero(gt_ids // 1000 == class_ids[m['class_index']])
        if not len(cols):
            continue
        values = overlaps[lookup[m['geometry_key']], cols]
        best = float(values.max())
        quality[i] = best
        # Ambiguous identity is abstention, not a tie broken using GT IDs.
        winners = cols[values == best]
        if best > .5 and len(winners) == 1:
            target[i] = int(gt_ids[winners[0]])
    return quality, target


def competition_labels(quality, targets, edges):
    """0 unknown/abstain; 1 safe directed representative; 2 unsafe.

    Safe requires reliable same-class same-target identity, no lost supported
    evaluator threshold, and no continuous range-quality regression. Ties use
    the already frozen member order, never source direction or GT ID.
    """
    a, b = edges.T
    known = (targets[a] >= 0) & (targets[b] >= 0)
    same = targets[a] == targets[b]
    support_a = quality[a, None] > THRESHOLDS
    support_b = quality[b, None] > THRESHOLDS
    lost = np.any(support_b & ~support_a, axis=1)
    better = (quality[a] > quality[b]) | ((quality[a] == quality[b]) & (a < b))
    safe = known & same & ~lost & better
    labels = np.zeros(len(edges), dtype=np.int64)
    labels[known] = 2
    labels[safe] = 1
    return labels


def order(members, scores):
    return sorted(range(len(members)), key=lambda i: (-float(scores[i]), not members[i]['has_original_native_alias'],
                                                     members[i]['geometry_key'], members[i]['class_index']))


def compete(members, base_scores, edges, safe_probability, budget=800):
    """Soft rescore with direct protected witnesses and final-budget restoration.

    A witness cannot itself be attenuated. Every active witness must remain in
    final K800; absent witnesses cause monotone undo, not a deletion or chain.
    No action expands to another class on the same geometry.
    """
    base = np.asarray(base_scores, dtype=float)
    probabilities = np.asarray(safe_probability)
    if base.shape != (len(members),) or probabilities.shape != (len(edges),):
        raise ValueError('member/edge prediction alignment differs')
    if not np.isfinite(base).all() or not np.isfinite(probabilities).all():
        raise ValueError('nonfinite learned scores')
    priority = {i: r for r, i in enumerate(order(members, base))}
    candidates = []
    for j, (a, b) in enumerate(edges):
        if members[a]['class_index'] != members[b]['class_index']:
            raise ValueError('cross-class competition prohibited')
        if probabilities[j] > CONFIDENCE and priority[a] < priority[b]:
            candidates.append((priority[a], priority[b], int(a), int(b), float(probabilities[j])))
    witnesses, actions = set(), {}
    for _, _, a, b, probability in sorted(candidates):
        if a in actions or b in witnesses or b in actions:
            continue
        witnesses.add(a)
        actions[b] = (a, probability)
    proposed = len(actions)
    while True:
        scores = base.copy()
        for b, (_, probability) in actions.items():
            scores[b] *= 1 - ATTENUATION * probability
        chosen = set(order(members, scores)[:budget])
        invalid = [b for b, (a, _) in actions.items() if a not in chosen]
        if not invalid:
            break
        for b in invalid:
            del actions[b]
    return scores, dict(proposed=proposed, restored=proposed-len(actions),
                        actions=[dict(winner=a, loser=b, probability=p) for b, (a, p) in sorted(actions.items())])
