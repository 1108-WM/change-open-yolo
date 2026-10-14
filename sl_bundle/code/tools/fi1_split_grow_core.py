"""GT-free split/grow/merge kernels for an isolated FI1 research branch.

This is a project adaptation, not a reproduction of SGS-3D or GeoOpen3D.
Affinity uses observed point coverage of a frontier superpoint rather than IoU
between disjoint superpoint closures. Missing depth is unknown, never a vote.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from collections import defaultdict

import numpy as np


@dataclass(frozen=True)
class SplitGrowConfig:
    split_method: str = "hdbscan"
    feature_mode: str = "mask3d"
    min_seed_points: int = 100
    min_samples: int = 10
    seed_superpoint_fraction: float = 0.6
    min_support_groups: int = 2
    min_visible_ratio: float = 0.1
    min_mask_support: float = 0.5
    min_support_rate: float = 0.5
    max_contact_distance: float = 0.05
    max_normal_difference: float = 0.5
    max_color_difference: float = 0.4
    max_depth_jump: float = 0.05
    min_feature_cosine: float = 0.6
    min_affinity: float = 0.25
    max_growth_steps: int = 256
    growth_hops: int = 1
    merge_thresholds: tuple[float, ...] = (0.9, 0.75, 0.6)
    merge_feature_cosine: float = 0.8
    min_parent_coverage: float = 0.9
    max_outside_fraction: float = 0.25

    def __post_init__(self):
        if self.split_method not in ("hdbscan", "graph"):
            raise ValueError("unknown split method")
        if self.feature_mode not in ("mask3d", "geometry_color_ablation"):
            raise ValueError("feature mode must be explicit")
        for k in ("min_seed_points", "min_samples", "min_support_groups", "max_growth_steps"):
            v = getattr(self, k)
            if type(v) is not int or v < 1:
                raise ValueError(f"invalid {k}")
        if type(self.growth_hops) is not int or not 0 <= self.growth_hops <= 2:
            raise ValueError("growth must have a bounded local domain")
        fractions = ("seed_superpoint_fraction", "min_visible_ratio", "min_mask_support",
                     "min_support_rate", "min_feature_cosine", "min_affinity",
                     "merge_feature_cosine", "min_parent_coverage", "max_outside_fraction",
                     "max_normal_difference", "max_color_difference")
        for k in fractions:
            if not np.isfinite(getattr(self, k)) or not 0 < getattr(self, k) <= 1:
                raise ValueError(f"invalid {k}")
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.max_contact_distance, self.max_depth_jump)):
            raise ValueError("invalid geometric distance")
        if (not self.merge_thresholds or any(not 0 < v <= 1 for v in self.merge_thresholds)
                or tuple(sorted(set(self.merge_thresholds), reverse=True)) != self.merge_thresholds):
            raise ValueError("merge thresholds must strictly decrease")

    def record(self):
        return asdict(self)


def point_indices(values, domain):
    p = np.asarray(values)
    if (p.ndim != 1 or p.dtype.kind not in "iu" or not np.array_equal(p, np.unique(p))
            or (len(p) and (p[0] < 0 or p[-1] >= domain))):
        raise ValueError("invalid sorted unique point indices")
    return p.astype(np.int64, copy=False)


def point_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


class SuperpointGraph:
    """Point-domain-preserving mesh adjacency; no annotations are accepted."""
    def __init__(self, xyz, rgb, normals, superpoints, faces, features=None):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        rgb = np.asarray(rgb, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        self.superpoints = np.asarray(superpoints)
        n = len(self.xyz)
        if (n == 0 or self.xyz.shape != (n, 3) or rgb.shape != (n, 3)
                or normals.shape != (n, 3) or self.superpoints.shape != (n,)
                or self.superpoints.dtype.kind not in "iu" or self.superpoints.min() < 0
                or not all(np.isfinite(a).all() for a in (self.xyz, rgb, normals))):
            raise ValueError("invalid geometry-only arrays")
        rgb = np.clip(rgb / (255. if rgb.max() > 1.5 else 1.), 0, 1)
        normals = unit(normals)
        self.ids, self.inverse = np.unique(self.superpoints, return_inverse=True)
        self.lookup = {int(s): i for i, s in enumerate(self.ids)}
        self.sizes = np.bincount(self.inverse)
        order = np.argsort(self.inverse, kind="stable")
        chunks = np.split(order, np.cumsum(self.sizes)[:-1])
        self.points = {int(s): np.sort(p) for s, p in zip(self.ids, chunks)}
        self.centers = self.aggregate(self.xyz)
        self.colors = self.aggregate(rgb)
        self.normals = unit(self.aggregate(normals))
        if features is None:
            self.features = None
        else:
            f = np.asarray(features, dtype=np.float64)
            if (f.ndim != 2 or len(f) != len(self.ids) or f.shape[1] < 1
                    or not np.isfinite(f).all() or np.any(np.linalg.norm(f, axis=1) < 1e-12)):
                raise ValueError("invalid aligned superpoint features")
            self.features = unit(f)
        faces = np.asarray(faces)
        if (faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
                or (faces.size and (faces.min() < 0 or faces.max() >= n))):
            raise ValueError("mesh faces outside original point domain")
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        a, b = edges[:, 0], edges[:, 1]
        different = self.inverse[a] != self.inverse[b]
        a, b = a[different], b[different]
        low = np.minimum(self.inverse[a], self.inverse[b])
        high = np.maximum(self.inverse[a], self.inverse[b])
        pairs, inv = np.unique(np.column_stack((low, high)), axis=0, return_inverse=True)
        counts = np.bincount(inv, minlength=len(pairs))
        distances = np.linalg.norm(self.xyz[a] - self.xyz[b], axis=1)
        nd = 1 - np.abs(np.sum(normals[a] * normals[b], axis=1))
        cd = np.linalg.norm(rgb[a] - rgb[b], axis=1) / np.sqrt(3)
        distance_sums = np.bincount(inv, weights=distances, minlength=len(pairs))
        normal_sums = np.bincount(inv, weights=nd, minlength=len(pairs))
        color_sums = np.bincount(inv, weights=cd, minlength=len(pairs))
        self.edges, self.neighbors = {}, defaultdict(set)
        self.boundary_points = {}
        pair_order = np.argsort(inv, kind="stable")
        pair_chunks = np.split(pair_order, np.cumsum(counts)[:-1]) if len(counts) else []
        for i, (u, v) in enumerate(pairs):
            key = int(self.ids[u]), int(self.ids[v])
            self.edges[key] = {"contact_count": int(counts[i]),
                              "distance": float(distance_sums[i] / counts[i]),
                              "normal_difference": float(normal_sums[i] / counts[i]),
                              "color_difference": float(color_sums[i] / counts[i])}
            indexes = pair_chunks[i]
            self.boundary_points[key] = np.column_stack((a[indexes], b[indexes]))
            self.neighbors[key[0]].add(key[1])
            self.neighbors[key[1]].add(key[0])

    def aggregate(self, values):
        out = np.zeros((len(self.ids), values.shape[1]), dtype=np.float64)
        np.add.at(out, self.inverse, values)
        return out / self.sizes[:, None]

    def point_set(self, ids):
        ids = sorted(set(map(int, ids)))
        if any(s not in self.points for s in ids):
            raise ValueError("unknown superpoint")
        return np.sort(np.concatenate([self.points[s] for s in ids])) if ids else np.empty(0, np.int64)

    def count(self, ids):
        return sum(int(self.sizes[self.lookup[s]]) for s in set(ids))

    def feature(self, ids):
        if self.features is None:
            raise ValueError("Mask3D features absent; choose a named ablation explicitly")
        indexes = [self.lookup[s] for s in sorted(set(ids))]
        return unit(np.average(self.features[indexes], axis=0, weights=self.sizes[indexes]))

    def connected(self, ids, config):
        remaining, parts = set(map(int, ids)), []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            component, stack = {start}, [start]
            while stack:
                u = stack.pop()
                for v in sorted(self.neighbors[u] & remaining):
                    if edge_allowed(self.edges[tuple(sorted((u, v)))], config):
                        remaining.remove(v)
                        component.add(v)
                        stack.append(v)
            parts.append(component)
        return sorted(parts, key=lambda s: (-self.count(s), sorted(s)))


def edge_allowed(edge, config):
    return (edge["distance"] <= config.max_contact_distance
            and edge["normal_difference"] <= config.max_normal_difference
            and edge["color_difference"] <= config.max_color_difference)


def pose_groups(poses, translation=0.1, rotation_degrees=15.):
    representatives, groups = [], {}
    for frame in sorted(poses):
        pose = np.asarray(poses[frame])
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or abs(np.linalg.det(pose)) < 1e-12:
            raise ValueError("invalid pose")
        group = None
        for i, prior in enumerate(representatives):
            distance = np.linalg.norm(pose[:3, 3] - prior[:3, 3])
            angle = np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T @ prior[:3, :3])-1)/2, -1, 1)))
            if distance < translation and angle < rotation_degrees:
                group = i
                break
        if group is None:
            group = len(representatives)
            representatives.append(pose)
        groups[frame] = group
    return groups


def pixel_containment(observations):
    """Strict pixel containment, not the historical approximate 0.95 relation."""
    from tools.build_details_same_frame_hierarchy_preprocessor import decode_binary_mask_rle
    by_frame = defaultdict(list)
    for row in observations:
        by_frame[int(row["frame_index"])].append(row)
    relations = []
    for frame, rows in sorted(by_frame.items()):
        entries = [(row, decode_binary_mask_rle(row["mask_rle"])) for row in rows]
        if any(int(mask.sum()) != row["area"] for row, mask in entries):
            raise ValueError("RLE area mismatch")
        for i, (a, ma) in enumerate(entries):
            for b, mb in entries[:i]:
                if ma.shape != mb.shape:
                    raise ValueError("same-frame pixel domains differ")
                intersection = int(np.count_nonzero(ma & mb))
                if intersection == 0:
                    continue
                small, large = (a, b) if a["area"] < b["area"] else (b, a)
                kind = ("duplicate" if intersection == a["area"] == b["area"] else
                        "containment" if intersection == small["area"] else "partial")
                relations.append({"frame_index": frame, "kind": kind,
                                  "small_observation_id": int(small["observation_id"]),
                                  "large_observation_id": int(large["observation_id"]),
                                  "intersection_pixels": intersection})
    return relations


def split_observation(graph, observed_points, allowed_ids, config):
    p = point_indices(observed_points, len(graph.xyz))
    p = p[np.isin(graph.superpoints[p], list(allowed_ids))]
    if len(p) < config.min_seed_points:
        return [], {"noise_points": len(p), "clusters": 0,
                    "observed_points_in_parent": len(p), "density_clusters": 0,
                    "assigned_superpoint_count": 0, "small_components_rejected": 0,
                    "unseeded_observed_points": len(p), "zero_seed_reason": "too_few_observed_points"}
    if config.split_method == "hdbscan":
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=config.min_seed_points, min_samples=config.min_samples,
                         allow_single_cluster=True, n_jobs=1).fit_predict(graph.xyz[p])
    else:
        labels = np.zeros(len(p), dtype=np.int64)
    seeds, assigned_all, small_components = [], set(), 0
    all_ids, totals = np.unique(graph.superpoints[p], return_counts=True)
    total = dict(zip(map(int, all_ids), map(int, totals)))
    density_labels = sorted(set(labels) - {-1})
    for label in density_labels:
        cluster = p[labels == label]
        ids, counts = np.unique(graph.superpoints[cluster], return_counts=True)
        # SP ownership is based on the cluster's fraction of observed points;
        # seeds remain whole superpoints and never fracture an existing atom.
        assigned = {int(s) for s, count in zip(ids, counts)
                    if count / total[int(s)] >= config.seed_superpoint_fraction}
        assigned_all |= assigned
        for component in graph.connected(assigned, config):
            if graph.count(component) >= config.min_seed_points:
                seeds.append(component)
            else:
                small_components += 1
    # The grow kernel uses this order for action.seed and proposal.seed_tokens.
    # Persist the same order, so the saved split seed index identifies that seed.
    seeds.sort(key=lambda s: tuple(sorted(s)))
    seeded = set().union(*seeds) if seeds else set()
    reason = (None if seeds else "density_all_noise" if not density_labels else
              "superpoint_majority_rejected" if not assigned_all else "components_too_small")
    return seeds, {"noise_points": int(np.sum(labels < 0)), "clusters": len(seeds),
                   "observed_points_in_parent": len(p), "density_clusters": len(density_labels),
                   "assigned_superpoint_count": len(assigned_all),
                   "small_components_rejected": small_components,
                   "unseeded_observed_points": int(np.count_nonzero(~np.isin(graph.superpoints[p], list(seeded)))),
                   "zero_seed_reason": reason}


def support_table(graph, observation_points, visible_points, group_ids, config):
    """Aggregate once per pose group; unseen SPs are omitted from denominators."""
    if not (len(observation_points) == len(visible_points) == len(group_ids)):
        raise ValueError("support rows differ")
    groups = {}
    for points, visible, group in zip(observation_points, visible_points, group_ids):
        p, v = point_indices(points, len(graph.xyz)), point_indices(visible, len(graph.xyz))
        if not np.isin(p, v).all():
            raise ValueError("observation points are not physically visible")
        vc = np.bincount(graph.inverse[v], minlength=len(graph.ids))
        pc = np.bincount(graph.inverse[p], minlength=len(graph.ids))
        valid = (vc >= 3) & (vc / graph.sizes >= config.min_visible_ratio)
        fraction = np.divide(pc, vc, out=np.zeros(len(vc), float), where=vc > 0)
        if group not in groups:
            groups[group] = (valid.copy(), np.where(valid, fraction, 0))
        else:
            prior_valid, prior_fraction = groups[group]
            groups[group] = (prior_valid | valid, np.maximum(prior_fraction, np.where(valid, fraction, 0)))
    if not groups:
        return {int(s): {"support_groups": 0, "visible_groups": 0, "rate": 0., "coverage": 0.}
                for s in graph.ids}
    valid = np.stack([g[0] for _, g in sorted(groups.items())])
    fractions = np.stack([g[1] for _, g in sorted(groups.items())])
    votes = valid & (fractions >= config.min_mask_support)
    return {int(s): {"support_groups": int(votes[:, i].sum()), "visible_groups": int(valid[:, i].sum()),
                     "rate": float(votes[:, i].sum() / max(1, valid[:, i].sum())),
                     "coverage": float(fractions[:, i].sum() / max(1, valid[:, i].sum()))}
            for i, s in enumerate(graph.ids)}


def grow_seeds(graph, seeds, support, allowed_ids, config, depth_jumps=None, seed_support=None):
    """Joint competition prevents one seed consuming another seed's frontier."""
    if config.feature_mode == "mask3d" and graph.features is None:
        raise ValueError("Mask3D feature mode requires real aligned features")
    seeds = [set(s) for s in sorted(seeds, key=lambda s: tuple(sorted(s)))]
    if seed_support is not None and set(seed_support) != {tuple(sorted(s)) for s in seeds}:
        raise ValueError("seed support must be keyed by exact canonical seed identity")
    owner = {}
    for i, seed in enumerate(seeds):
        if not seed <= set(allowed_ids):
            raise ValueError("seed outside growth domain")
        for s in seed:
            if s in owner:
                raise ValueError("seed ownership conflict")
            owner[s] = i
    current, actions = [s.copy() for s in seeds], []
    depth_jumps = depth_jumps or {}
    for step in range(config.max_growth_steps):
        choices = []
        for i, region in enumerate(current):
            mean_feature = graph.feature(region) if config.feature_mode == "mask3d" else None
            seed_feature = graph.feature(seeds[i]) if config.feature_mode == "mask3d" else None
            frontier = set().union(*(graph.neighbors[s] for s in region)) - owner.keys()
            for v in sorted(frontier & set(allowed_ids)):
                evidence = (seed_support[tuple(sorted(seeds[i]))][v]
                            if seed_support is not None else support[v])
                if (evidence["support_groups"] < config.min_support_groups
                        or evidence["rate"] < config.min_support_rate):
                    continue
                contacts = []
                for u in sorted(graph.neighbors[v] & region):
                    pair = tuple(sorted((u, v)))
                    if not edge_allowed(graph.edges[pair], config):
                        continue
                    depth = depth_jumps.get(pair)
                    if depth is not None and depth > config.max_depth_jump:
                        continue
                    contacts.append(pair)
                if not contacts:
                    continue
                if config.feature_mode == "mask3d":
                    feature = graph.features[graph.lookup[v]]
                    cosine = float(mean_feature @ feature)
                    anchor_cosine = float(seed_feature @ feature)
                    if min(cosine, anchor_cosine) < config.min_feature_cosine:
                        continue
                    similarity = max(0., cosine)
                else:
                    cosine = None
                    similarity = 1 - min(graph.edges[p]["color_difference"] for p in contacts)
                affinity = similarity * evidence["coverage"]
                if affinity >= config.min_affinity:
                    choices.append((-affinity, v, i, cosine, contacts))
        if not choices:
            break
        minus_affinity, v, i, cosine, contacts = min(choices, key=lambda x: x[:3])
        owner[v] = i
        current[i].add(v)
        actions.append({"step": step, "seed": i, "added_superpoint": v,
                        "affinity": -minus_affinity, "feature_cosine": cosine,
                        "contacts": [list(p) for p in contacts], "depth_unknown": all(p not in depth_jumps for p in contacts)})
    return current, actions


def proposal_iou(graph, a, b):
    return graph.count(a & b) / max(1, graph.count(a | b))


def progressive_merge(graph, proposals, config):
    """Recheck current unions; prevent transitive bridging across seed siblings."""
    current = [dict(p, superpoint_ids=set(p["superpoint_ids"]),
                    lineage=list(p["lineage"]), seed_tokens=set(map(tuple, p["seed_tokens"])))
               for p in proposals]
    actions = []
    for threshold in config.merge_thresholds:
        while True:
            choices = []
            for i, a in enumerate(current):
                for j in range(i):
                    b = current[j]
                    # Distinct siblings originating from one observation cannot merge.
                    ta, tb = defaultdict(set), defaultdict(set)
                    for obs, seed in a["seed_tokens"]: ta[obs].add(seed)
                    for obs, seed in b["seed_tokens"]: tb[obs].add(seed)
                    if any(ta[o] != tb[o] for o in ta.keys() & tb.keys()):
                        continue
                    if set(a["frames"]) == set(b["frames"]) and len(set(a["frames"])) == 1:
                        continue
                    overlap = proposal_iou(graph, a["superpoint_ids"], b["superpoint_ids"])
                    if overlap < threshold:
                        continue
                    if (config.feature_mode == "mask3d" and
                            float(graph.feature(a["superpoint_ids"]) @ graph.feature(b["superpoint_ids"]))
                            < config.merge_feature_cosine):
                        continue
                    if len(graph.connected(a["superpoint_ids"] | b["superpoint_ids"], config)) != 1:
                        continue
                    choices.append((-overlap, j, i))
            if not choices:
                break
            negative, j, i = min(choices)
            a, b = current[j], current[i]
            union = a["superpoint_ids"] | b["superpoint_ids"]
            merged = {"superpoint_ids": union, "frames": sorted(set(a["frames"]) | set(b["frames"])),
                      "pose_groups": sorted(set(a["pose_groups"]) | set(b["pose_groups"])),
                      "lineage": sorted(set(a["lineage"]) | set(b["lineage"])),
                      "seed_tokens": a["seed_tokens"] | b["seed_tokens"]}
            actions.append({"threshold": threshold, "iou": -negative,
                            "left": sorted(a["superpoint_ids"]), "right": sorted(b["superpoint_ids"]),
                            "union": sorted(union), "lineage": merged["lineage"]})
            current[j] = merged
            del current[i]
    return sorted(current, key=lambda p: tuple(sorted(p["superpoint_ids"]))), actions


def family_decision(graph, parent_ids, children, config):
    """Select a complete, supported, conflict-free child set; otherwise parent."""
    reasons, sets = [], [set(c["superpoint_ids"]) for c in children]
    if len(sets) < 2:
        reasons.append("fewer_than_two_children")
    for child, ids in zip(children, sets):
        if graph.count(ids) < config.min_seed_points:
            reasons.append("small_child")
        if len(set(child["pose_groups"])) < config.min_support_groups:
            reasons.append("insufficient_pose_groups")
        if len(graph.connected(ids, config)) != 1:
            reasons.append("disconnected_child")
    union = set().union(*sets) if sets else set()
    if sum(graph.count(s) for s in sets) != graph.count(union):
        reasons.append("child_ownership_conflict")
    coverage = graph.count(union & set(parent_ids)) / max(1, graph.count(parent_ids))
    outside = graph.count(union - set(parent_ids)) / max(1, graph.count(union))
    if coverage < config.min_parent_coverage:
        reasons.append("incomplete_parent_coverage")
    if outside > config.max_outside_fraction:
        reasons.append("excessive_outside_growth")
    return {"choice": "parent" if reasons else "children", "reasons": sorted(set(reasons)),
            "parent_coverage": coverage, "outside_fraction": outside}
