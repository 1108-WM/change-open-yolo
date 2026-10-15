"""Frozen, GT-free primitives for the N/T region identity diagnostic."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

PARAMETERS = dict(seed=20261014, voxel_size=.02, contact_radius=.15,
                  edge_radius=.05, max_neighbors=4, probability=.5,
                  visible_groups=2, min_points=100, topk=2,
                  ambiguity_gap=.01, ambiguity_ratio=.9)


def safe_input(raw):
    x = np.array(raw[:, :9], dtype=np.float32, copy=True)
    if x.ndim != 2 or x.shape[1] != 9 or not np.isfinite(x).all():
        raise ValueError('invalid XYZ/RGB/normal input')
    return x


def coordinate_rows(source, destination):
    """Destination row -> source row, using coordinates, never row assumptions."""
    lookup = {tuple(map(int, row)): i for i, row in enumerate(source)}
    if len(lookup) != len(source):
        raise ValueError('duplicate sparse input coordinate')
    indices = np.array([lookup[tuple(map(int, row))] for row in destination], np.int64)
    if len(indices) != len(source) or len(np.unique(indices)) != len(source):
        raise ValueError('non-bijective sparse coordinate map')
    assert np.array_equal(source[indices], destination)
    return indices


def partition(xyz, neighbors, membership, radius=.05):
    n = len(xyz)
    src = np.repeat(np.arange(n), neighbors.shape[1])
    dst = neighbors.ravel()
    distance = np.linalg.norm(xyz[src]-xyz[dst], axis=1)
    close = (distance <= radius) & (src != dst)
    same = np.all(membership[src] == membership[dst], axis=1)
    graph = csr_matrix((np.ones(int((close & same).sum())),
                        (src[close & same], dst[close & same])), shape=(n, n))
    count, labels = connected_components(graph, directed=False)
    # Stable region IDs follow the first original point in each component.
    first = np.full(count, n, np.int64)
    np.minimum.at(first, labels, np.arange(n))
    remap = np.empty(count, np.int32); remap[np.argsort(first)] = np.arange(count)
    labels = remap[labels]
    cross = close & (labels[src] != labels[dst])
    pairs = np.sort(np.column_stack((labels[src[cross]], labels[dst[cross]])), axis=1)
    edges, inverse = np.unique(pairs, axis=0, return_inverse=True)
    edge_distance = np.full(len(edges), np.inf)
    np.minimum.at(edge_distance, inverse, distance[cross])
    order = np.argsort(labels, kind='stable').astype(np.int32)
    offsets = np.r_[0, np.cumsum(np.bincount(labels))].astype(np.int64)
    return labels, order, offsets, edges.astype(np.int32), edge_distance


def region_stats(probability, labels):
    count = np.bincount(labels)
    sums = np.zeros((len(count), probability.shape[1]), np.float64)
    square = np.zeros_like(sums)
    np.add.at(sums, labels, probability)
    np.add.at(square, labels, np.square(probability.astype(np.float64)))
    mean = sums / count[:, None]
    std = np.sqrt(np.maximum(square/count[:, None]-mean**2, 0))
    return mean, std


def association(probability, full_volume, indices):
    intersection = probability[indices].sum(0, dtype=np.float64)
    score = intersection / np.maximum(len(indices)+full_volume-intersection, 1e-12)
    top = np.lexsort((np.arange(len(score)), -score))[:2]
    gap = float(score[top[0]]-score[top[1]]) if len(top) > 1 else None
    ratio = float(score[top[1]]/max(score[top[0]], 1e-12)) if len(top) > 1 else None
    return dict(top=top.tolist(), scores=score[top].tolist(), gap=gap, ratio=ratio,
                missing=not len(top) or float(score[top[0]]) <= 0,
                ambiguous=len(top) < 2 or gap <= PARAMETERS['ambiguity_gap'] or ratio >= PARAMETERS['ambiguity_ratio'])


def probe(parent, halo, probability, visible):
    member = np.isin(halo, parent); editable = np.asarray(visible) >= 2
    mask = np.where(editable, np.asarray(probability) >= .5, member)
    result = halo[mask]
    return result if len(result) >= 100 else parent.copy()


def feasible(parent, halo, gt, target, visible, regions=None, deletion_only=False):
    """Post-hoc feasible construction, never a GT-free output or optimum."""
    member = np.isin(halo, parent); seen = np.asarray(visible) >= 2
    labels = gt[halo]; known = labels >= 1000
    add = ~member & seen & (labels == target) & (not deletion_only)
    remove = member & seen & known & (labels != target)
    if regions is not None:
        keep_add = np.zeros(len(halo), bool); keep_remove = keep_add.copy()
        for region in regions:
            domain = region[member[region] & seen[region]] if deletion_only else region
            if not len(domain): continue
            if np.all(labels[domain] == target): keep_add[region] = True
            if np.all((labels[domain] >= 1000) & (labels[domain] != target)):
                keep_remove[region] = True
        add &= keep_add; remove &= keep_remove
    result = halo[(member & ~remove) | add]
    return result if len(result) >= 100 else parent.copy()
