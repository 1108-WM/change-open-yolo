"""Parent-conditioned U admission; reuse the frozen NTU member/scoring pipeline.

This is a development experiment, not a replacement for production Full.
Only admission changes. Neither class IDs nor training labels are features.
"""
from collections import Counter, defaultdict
import hashlib

import numpy as np
from scipy.spatial import cKDTree

from tools.fi1_core_ablation import source_pool
from tools.fi1_split_grow_core import point_indices, unit
from tools.fi1_r0_r1_protocol import direct_score
from tools.train_fi1_action_contribution import PARAMS as ACTION_PARAMS

PARAMS = dict(ACTION_PARAMS, min_samples_leaf=40, random_state=20261007)
TAUS = np.arange(50, 91, 5) / 100.
ARMS = ('NT', 'NTU', 'existing_gate', 'existing_oof', 'region_oof')
LEARNED = ('existing_oof', 'region_oof')
SEED = 'fi1-union-admission-development200-v1'


def scene_folds(scenes, count=5):
    """Balanced fixed folds of physical scenes, independent of labels/source."""
    physical = sorted({s[:9] for s in scenes},
                      key=lambda s: hashlib.sha256((SEED+s).encode()).hexdigest())
    assignment = {s: i % count for i, s in enumerate(physical)}
    return {s: assignment[s[:9]] for s in scenes}


def admitted_pool(geometry, keep):
    """Drop rejected U aliases BEFORE R1/context; protect shared N/T geometry."""
    pool = source_pool(geometry, ('native', 'track', 'pair_union'))
    available = {g['geometry_key'] for g in pool
                 if any(a['candidate_source'] == 'pair_union' for a in g['legacy_members'])}
    if not set(keep) <= available:
        raise ValueError('unknown U admission key')
    result = []
    for g in pool:
        if g['geometry_key'] in available and g['geometry_key'] not in keep:
            result.extend(source_pool([g]))
        else:
            result.append(g)
    return result


def anchor_targets(parent_iou, other_iou, union_iou, gt_ids):
    """Fixed parent anchor; signed gain against BOTH parents on that same GT.

    A parent with IoU<.5 is excluded from gain fitting, not assigned a fake
    negative gain. All rows supervise anchored absolute union IoU (zero when
    the parent has no valid overlap). Prediction never reads this validity.
    The two directions are fitted separately with equal proposal weight.
    """
    p, o, u = [np.asarray(v, dtype=float) for v in (parent_iou, other_iou, union_iou)]
    ids = np.asarray(gt_ids)
    if p.shape != o.shape or p.shape != u.shape or p.shape != ids.shape:
        raise ValueError('anchor alignment')
    if not len(p) or p.max() <= 0:
        return dict(anchor=None, reliable=False, gain=None, absolute_iou=0.,
                    delta_iou=None, delta_threshold=None, parent_iou=0.)
    index = int(np.lexsort((ids, -p))[0])
    before = max(p[index], o[index]); after = u[index]
    dq = float(np.mean(after > TAUS) - np.mean(before > TAUS))
    delta = float(after-before)
    return dict(anchor=int(ids[index]), reliable=bool(p[index] >= .5),
                gain=.5*(delta+dq), absolute_iou=float(after), delta_iou=delta,
                delta_threshold=dq, parent_iou=float(p[index]))


def sample_weights(records):
    """Equal scene, then U proposal, then valid direction (never 198 classes)."""
    tree = defaultdict(lambda: Counter())
    for r in records:
        tree[r['scene'][:9]][r['union_key']] += 1
    return np.asarray([len(records)/(len(tree)*len(tree[r['scene'][:9]])*
                        tree[r['scene'][:9]][r['union_key']]) for r in records])


def admission_keys(records, gain, quality):
    """One prespecified operating point; no AP-driven threshold search."""
    gain, quality = np.asarray(gain), np.asarray(quality)
    if gain.shape != (len(records),) or quality.shape != gain.shape:
        raise ValueError('prediction alignment')
    if not np.isfinite(gain).all() or not np.isfinite(quality).all():
        raise ValueError('nonfinite admission prediction')
    return {r['union_key'] for r, g, q in zip(records, gain, quality) if g > 0 and q > .5}


def semantic_contrast(parent, other, union):
    """Distribution comparisons, retaining uncertainty; no class-name gates."""
    result = {}
    vectors = [direct_score(e) for e in (parent, other, union)]
    for name, a, b in [('parent_other', vectors[0], vectors[1]),
                       ('parent_union', vectors[0], vectors[2])]:
        result[name+'_l1'] = float(np.abs(a-b).sum())
        result[name+'_dot'] = float(a @ b)
        result[name+'_top_agrees'] = float(a.sum()>0 and b.sum()>0 and np.argmax(a)==np.argmax(b))
    for name, e in zip(('parent','other','union'), (parent,other,union)):
        result[name+'_modalities_available'] = sum(e[m]['available'] for m in ('yolo','alpha'))
    return result


def boundary_features(raw9, parent, extra):
    """Directional point evidence relative to the parent's own local variation.

    Nearest-neighbor surface distances are descriptors, not hard ownership
    decisions. Input is exactly XYZ/RGB/normals; annotation columns forbidden.
    """
    raw = np.asarray(raw9)
    if raw.ndim != 2 or raw.shape[1] != 9 or not np.isfinite(raw).all():
        raise ValueError('only finite XYZ RGB normals accepted')
    parent, extra = [point_indices(p, len(raw)) for p in (parent,extra)]
    names = ('distance_median','distance_p90','distance_min','color_mean','color_p90',
             'normal_mean','normal_valid_fraction','internal_distance','internal_color',
             'internal_normal','color_excess','normal_excess','added_fraction')
    result = {n:0. for n in names};result['empty_increment'] = float(not len(extra))
    if not len(parent) or not len(extra):
        return result
    rgb = np.clip(raw[:,3:6] / (255. if raw[:,3:6].max()>1.5 else 1.),0,1)
    normal = unit(raw[:,6:9]); valid = np.linalg.norm(raw[:,6:9],axis=1)>1e-12
    tree = cKDTree(raw[parent,:3]); dist, idx = tree.query(raw[extra,:3], k=1)
    opposite = parent[idx]; colors = np.linalg.norm(rgb[extra]-rgb[opposite],axis=1)
    nv = valid[extra]&valid[opposite]
    nd = 1-np.abs(np.sum(normal[extra]*normal[opposite],axis=1))
    if len(parent)>1:
        dd, ii = tree.query(raw[parent,:3], k=2); neighbor=parent[ii[:,1]]
        innerv = valid[parent]&valid[neighbor]
        innerc = float(np.mean(np.linalg.norm(rgb[parent]-rgb[neighbor],axis=1)))
        innern = float(np.mean(1-np.abs(np.sum(normal[parent[innerv]]*normal[neighbor[innerv]],axis=1)))) if innerv.any() else 0.
        innerd = float(np.median(dd[:,1]))
    else:
        innerc=innern=innerd=0.
    result.update(distance_median=float(np.median(dist)),distance_p90=float(np.quantile(dist,.9)),
        distance_min=float(dist.min()),color_mean=float(colors.mean()),color_p90=float(np.quantile(colors,.9)),
        normal_mean=float(nd[nv].mean()) if nv.any() else 0.,normal_valid_fraction=float(nv.mean()),
        internal_distance=innerd,internal_color=innerc,internal_normal=innern,
        color_excess=float(colors.mean()-innerc),normal_excess=float(nd[nv].mean()-innern) if nv.any() else 0.,
        added_fraction=float(len(extra)/len(parent)))
    return result


def region_view_features(parent, extra, frames, excluded_groups):
    """Continuous joint/separate evidence; one observation vote per pose group.

    Frame bank uses existing physical visibility and raw point masks. A broad
    joint mask and separating masks may coexist: retain BOTH measurements.
    Unknown is explicit; no source-frame vote can become independent evidence.
    """
    bygroup = defaultdict(list)
    for f in frames:
        if f['group'] in excluded_groups or not len(parent) or not len(extra):
            continue
        vis=f['visible']; a=parent[vis[parent]]; b=extra[vis[extra]]
        if len(a)<3 or len(b)<3 or len(a)/len(parent)<.1 or len(b)/len(extra)<.1:
            continue
        # Distinct observation IDs are not necessarily distinct masks: the
        # frozen GD-SAM bank contains exact-pointset aliases. They cannot be
        # two independent separating witnesses in the same view.
        unique = {np.asarray(p,dtype='<i8').tobytes():p for p in f['masks']}
        cov = np.asarray([(np.intersect1d(a,p,assume_unique=True).size/len(a),
                           np.intersect1d(b,p,assume_unique=True).size/len(b)) for p in unique.values()],dtype=float).reshape(-1,2)
        joint = float(np.minimum(cov[:,0],cov[:,1]).max(initial=0))
        # Max product of distinct-mask directional isolation. Same mask can
        # never stand as both separating witnesses.
        separation=0.
        if len(cov)>1:
            l=cov[:,0]*(1-cov[:,1]);r=cov[:,1]*(1-cov[:,0])
            table=np.minimum(l[:,None],r[None,:]);np.fill_diagonal(table,0)
            separation=float(table.max())
        bygroup[f['group']].append((joint,separation,len(a)/len(parent),len(b)/len(extra)))
    rows=np.asarray([np.max(v,axis=0) for v in bygroup.values()],dtype=float).reshape(-1,4)
    result=dict(log1p_independent_groups=float(np.log1p(len(rows))),unknown=float(not len(rows)))
    for i,n in enumerate(('joint','separate','parent_visible','extra_visible')):
        result[n+'_mean']=float(rows[:,i].mean()) if len(rows) else 0.
        result[n+'_max']=float(rows[:,i].max()) if len(rows) else 0.
    return result
