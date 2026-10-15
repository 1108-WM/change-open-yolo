"""Training-only duplicate relationship balancing with fixed class masses."""
from collections import Counter, defaultdict
import numpy as np


def group_keys(rows, scene_labels):
    keys, y = [], []
    for row in rows:
        scene = row['scene']
        nodes = scene_labels[scene]['nodes']
        left, right = nodes[str(row['left'])], nodes[str(row['right'])]
        if left['kind'] != 'clear' or right['kind'] != 'clear':
            raise ValueError('Only clear fitting rows may define physical relations')
        a, b = sorted([int(left['target']), int(right['target'])])
        same = int(a == b)
        keys.append((scene, same, a, b))
        y.append(same)
    return keys, np.asarray(y, dtype=np.int64)


def balance(keys, reference_weights):
    reference = np.asarray(reference_weights, dtype=float)
    if len(keys) != len(reference) or not np.isfinite(reference).all() or (reference <= 0).any():
        raise ValueError('Positive finite reference weight per fitting row required')
    mass, groups = defaultdict(float), defaultdict(set)
    counts = Counter(keys)
    for key, weight in zip(keys, reference):
        category = key[:2]
        mass[category] += float(weight)
        groups[category].add(key)
    result = np.array([mass[key[:2]]/len(groups[key[:2]])/counts[key] for key in keys])
    return result
