"""Symmetric pair evidence and nonrecursive decisions, independent of masks/GT."""
import numpy as np
from scipy.spatial import cKDTree


def sample(x, limit=256):
    return x[np.linspace(0, len(x)-1, min(len(x), limit), dtype=int)] if len(x) else x


def distribution(v):
    v = np.asarray(v)
    return np.r_[np.mean(v, axis=0), np.std(v, axis=0)] if len(v) else None


def appearance(raw, ids):
    x = raw[sample(ids)]
    if not len(x):
        return np.full(36, np.nan)
    rgb = np.clip(x[:, 3:6]/255., 0, 1)
    normals = x[:, 6:9]
    length = np.linalg.norm(normals, axis=1)
    n = normals[length > 1e-8]/length[length > 1e-8, None]
    hist = np.concatenate([np.histogram(rgb[:, j], bins=8, range=(0, 1))[0]
                           for j in range(3)])/len(rgb)
    return np.r_[rgb.mean(0), rgb.std(0), hist,
                 np.mean(np.abs(n), axis=0) if len(n) else [np.nan]*3,
                 np.std(n, axis=0) if len(n) else [np.nan]*3]


def surface(raw, left, right):
    shared = np.intersect1d(left, right, assume_unique=True)
    aa = np.setdiff1d(left, shared, assume_unique=True)
    bb = np.setdiff1d(right, shared, assume_unique=True)
    sizes = sorted([len(left), len(right)])
    exclusive_sizes = sorted([len(aa), len(bb)])
    union = np.union1d(left, right)
    xyz = raw[:, :3]
    extent = lambda p: np.ptp(xyz[p], axis=0)
    ea, eb = extent(left), extent(right)
    ca, cb = xyz[left].mean(0), xyz[right].mean(0)
    geometry = np.r_[np.log1p(sizes), np.log1p(exclusive_sizes),
                     len(shared)/len(union), np.linalg.norm(ca-cb),
                     np.sort((ea+eb)/2), np.sort(np.abs(ea-eb)),
                     np.sort(extent(union))]
    contacts = np.empty((0, 2), dtype=np.int64)
    if len(aa) and len(bb):
        # Bidirectional nearest contacts avoid an arbitrary endpoint direction.
        da, ia = cKDTree(xyz[aa]).query(xyz[bb])
        db, ib = cKDTree(xyz[bb]).query(xyz[aa])
        edges = np.r_[np.c_[aa[ia], bb], np.c_[aa, bb[ib]]]
        distance = np.r_[da, db]
        near = distance <= .10
        contacts = np.unique(edges[near], axis=0)
        geometry = np.r_[geometry, distance.min(), np.quantile(distance, [.1, .5, .9]),
                         np.mean(distance <= .05), near.mean()]
    else:
        geometry = np.r_[geometry, [np.nan]*4, 0., 0.]
    sa, sb = appearance(raw, aa), appearance(raw, bb)
    app = np.r_[(sa+sb)/2, np.abs(sa-sb), len(contacts) > 0]
    if len(contacts):
        a, b = sample(contacts).T
        rgb = np.clip(raw[:, 3:6]/255., 0, 1)
        color = np.linalg.norm(rgb[a]-rgb[b], axis=1)/np.sqrt(3)
        na, nb = raw[a, 6:9], raw[b, 6:9]
        la, lb = np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1)
        valid = (la > 1e-8) & (lb > 1e-8)
        nd = 1-np.abs(np.sum(na[valid]*nb[valid], axis=1)/(la[valid]*lb[valid]))
        app = np.r_[app, np.quantile(color, [.1, .5, .9]), valid.mean(),
                    np.quantile(nd, [.1, .5, .9]) if len(nd) else [np.nan]*3]
    else:
        app = np.r_[app, [np.nan]*3, 0., [np.nan]*3]
    return geometry, app, aa, bb, contacts


def adapt_intrinsic(intrinsic, rgb_size, depth_size):
    w, h = rgb_size
    dw, dh = depth_size
    k = intrinsic.copy()
    if (w, h) != (dw, dh):
        resized = int(np.floor(dw*h/w))
        k[0, 0] *= resized/h
        k[1, 1] *= dw/w
        k[0, 2] *= (dh-1)/(h-1)
        k[1, 2] *= (dw-1)/(w-1)
    return k


def project(xyz, pose, k):
    camera = np.c_[xyz, np.ones(len(xyz))] @ np.linalg.inv(pose).T
    projected = camera @ k.T
    valid = np.isfinite(projected).all(axis=1) & (camera[:, 2] > 0)
    pixels = np.zeros((len(xyz), 2), dtype=np.int64)
    pixels[valid] = np.trunc(projected[valid, :2]/projected[valid, 2, None]).astype(int)
    return pixels, camera[:, 2], valid


def distinct_pose(a, b):
    angle = np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3])-1)/2, -1, 1))
    return np.linalg.norm(a[:3, 3]-b[:3, 3]) >= .20 or angle >= np.pi/12


def depth_path_evidence(depth, pixa, pixb):
    """Measured depth jumps on eight nearest projected inter-region paths."""
    if not len(pixa) or not len(pixb):
        return np.full(8, np.nan)
    a, b = sample(pixa, 128), sample(pixb, 128)
    _, ia = cKDTree(a).query(b)
    _, ib = cKDTree(b).query(a)
    edges = np.unique(np.r_[np.c_[a[ia], b], np.c_[a, b[ib]]], axis=0)
    distance = np.linalg.norm(edges[:, :2]-edges[:, 2:], axis=1)
    jumps, missing, lengths = [], [], []
    for j in np.argsort(distance, kind='stable')[:8]:
        line = np.rint(np.linspace(edges[j, :2], edges[j, 2:], 32)).astype(int)
        z = depth[line[:, 1], line[:, 0]]
        ok = np.isfinite(z) & (z > 0)
        valid_edges = ok[1:] & ok[:-1]
        v = np.abs(np.diff(z))[valid_edges]
        jumps.append([np.median(v), np.quantile(v, .9), np.max(v)] if len(v) else [np.nan]*3)
        missing.append(float((~ok).mean()))
        lengths.append(distance[j])
    v = np.asarray(jumps)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(v, axis=0)
    return np.r_[mean, np.mean(missing), np.max(missing), np.median(lengths),
                 np.max(lengths), len(edges[:8])/8]


def labels(left, right):
    if left['kind'] == 'clear' and right['kind'] == 'clear':
        return 'same' if left['target'] == right['target'] else 'different'
    return 'mixed' if 'mixed' in (left['kind'], right['kind']) else 'unknown'


def endpoint_label(ids, counts):
    counts, ids = np.asarray(counts), np.asarray(ids)
    total = int(counts.sum())
    known = ids >= 1000
    mixed = int(np.sum(known & (counts >= 20))) >= 2
    j = int(np.argmax(np.where(known, counts, 0))) if known.any() else None
    target = int(ids[j]) if j is not None else None
    purity = float(counts[j]/total) if j is not None else 0.
    clear = not mixed and j is not None and counts[j] >= 20 and purity >= .9
    return dict(kind='mixed' if mixed else 'clear' if clear else 'unknown',
                target=target, purity=purity, points=total,
                distribution={str(i): int(n) for i, n in zip(ids, counts)})


def decide(rows, scores, eligible):
    result = [dict(**r, score=float(s), state='unknown', reason='intermediate_score', emit=False)
              for r, s in zip(rows, scores)]
    for r, ok in zip(result, eligible):
        if not ok:
            r.update(reason='insufficient_measured_relation_evidence')
        elif r['score'] <= .1:
            r.update(state='reject', reason='low_relation_probability_not_a_cannot_link_label')
        elif r['score'] >= .9:
            r.update(reason='high_score_competing_partner')
    used = set()
    for i in sorted(range(len(rows)), key=lambda j: (-scores[j], rows[j]['pair_id'])):
        r = result[i]
        if not eligible[i] or scores[i] < .9:
            continue
        if r['left'] in used or r['right'] in used:
            continue
        used.update([r['left'], r['right']])
        r.update(state='accept', reason='high_relation_probability_disjoint_pair')
    return result
