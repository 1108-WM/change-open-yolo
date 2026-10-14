"""Read-only coverage attribution; duplicate hits remain covered GT instances."""
import numpy as np


def geometry_overlaps(points, gt, ids, gt_sizes, minimum=100):
    sizes = np.asarray([len(p) for p in points], dtype=np.int64)
    inter = np.zeros((len(points), len(ids)), dtype=np.int64)
    lookup = {int(g): i for i, g in enumerate(ids)}
    for row, p in enumerate(points):
        p = np.asarray(p)
        if p.ndim != 1 or p.dtype.kind not in 'iu' or not np.array_equal(p, np.unique(p)):
            raise ValueError('point indices must be sorted unique integers')
        if len(p) and (p[0] < 0 or p[-1] >= len(gt)):
            raise ValueError('point indices outside scene')
        labels, counts = np.unique(gt[p], return_counts=True)
        for label, count in zip(labels, counts):
            if int(label) in lookup:
                inter[row, lookup[int(label)]] = count
    iou = inter / np.maximum(sizes[:, None] + np.asarray(gt_sizes)[None, :] - inter, 1)
    iou[sizes < minimum] = 0
    return iou, inter, sizes


def coverage_funnel(iou, geometry_classes, selected_classes, gt_classes, threshold):
    """Sets contain actual member classes, with selected classes counted per geometry."""
    out = []
    for j, cls in enumerate(gt_classes):
        hits = np.flatnonzero(iou[:, j] > threshold)
        eligible = [int(i) for i in hits if int(cls) in geometry_classes[int(i)]]
        chosen = [int(i) for i in hits if int(cls) in selected_classes[int(i)]]
        state = ('geometry_missing' if not len(hits) else
                 'correct_class_missing' if not eligible else
                 'budget_excluded' if not chosen else 'selected_covered')
        out.append({'state': state, 'matching_geometries': len(hits),
                    'correct_members': len(eligible), 'selected_correct_members': len(chosen),
                    'duplicate_correct_members': max(0, len(chosen) - 1)})
    return out
