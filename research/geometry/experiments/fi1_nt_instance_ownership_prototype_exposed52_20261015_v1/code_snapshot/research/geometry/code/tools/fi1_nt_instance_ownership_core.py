"""GT/query-free N/T relation groups and tri-state block ownership prototype."""
import numpy as np

PARAMETERS = {'majority': .5, 'query_clip': .5, 'min_reference_points': 100}


def duplicate_groups(intersection, sizes, keys):
    """Complete-link, not connected components: a broad bridge cannot join disjoint seeds.

    Greedy order is fixed by geometry key. Groups are relation hypotheses, not GT
    identities. Reciprocal half coverage is inclusive. No semantic/query evidence.
    """
    intersection = np.asarray(intersection)
    sizes = np.asarray(sizes)
    majority = PARAMETERS['majority']
    compatible = ((intersection >= majority * sizes[:, None]) &
                  (intersection >= majority * sizes[None, :]))
    groups = []
    for node in sorted(range(len(keys)), key=lambda i: keys[i]):
        eligible = [g for g, members in enumerate(groups)
                    if all(compatible[node, member] for member in members)]
        if eligible:
            groups[eligible[0]].append(node)
        else:
            groups.append([node])
    return groups


def credible_groups(groups, sizes, sources):
    """An anchor group needs distinct N and T geometries with >=100 points each.

    A single deduplicated mask with N/T aliases is not two independent witnesses.
    'Credible' is a prototype certificate, not a claim of correctness.
    """
    result = []
    for gid, group in enumerate(groups):
        native = [n for n in group if sizes[n] >= PARAMETERS['min_reference_points'] and 'native' in sources[n]]
        track = [n for n in group if sizes[n] >= PARAMETERS['min_reference_points'] and 'track' in sources[n]]
        if any(n != t for n in native for t in track):
            result.append(gid)
    return result


def group_support(hits, count, groups, credible):
    if count == 0:
        return []
    return [g for g in credible
            if any(hits[n] >= PARAMETERS['majority'] * count for n in groups[g])]


def ownership(parent_hits, parent_size, block_hits, block_size, groups, credible):
    """Unique same reference accepts; unique different reference rejects; else unknown."""
    parent_owners = group_support(parent_hits, parent_size, groups, credible)
    block_owners = group_support(block_hits, block_size, groups, credible)
    if len(parent_owners) == len(block_owners) == 1:
        same = parent_owners[0] == block_owners[0]
        return dict(state='accept' if same else 'reject',
                    reason='unique_shared_instance_reference' if same else 'unique_different_instance_references',
                    parent_groups=parent_owners, block_groups=block_owners)
    return dict(state='unknown', reason='no_reference' if not parent_owners or not block_owners else 'competing_references',
                parent_groups=parent_owners, block_groups=block_owners)
