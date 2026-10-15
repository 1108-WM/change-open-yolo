"""Scene-controlled relationship weights and fixed diagnostic metrics."""
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def weights(scenes, overlap, y, balanced=False):
    scenes = list(scenes)
    keys = [(s, bool(o), int(label)) for s, o, label in zip(scenes, overlap, y)]
    if balanced:
        count = Counter(keys)
        cells = Counter(s for s, _, _ in set(keys))
        w = np.array([1/(cells[s]*count[key]) for key in keys for s in [key[0]]])
    else:
        count = Counter(scenes)
        w = np.array([1/count[s] for s in scenes])
    return w*(len(w)/w.sum())


def cell(overlap, label):
    return ('overlap' if overlap else 'disjoint')+'_'+('same' if label else 'different')


def metrics(y, score, scenes):
    y, score = np.asarray(y, int), np.asarray(score, float)
    if not len(y):
        return dict(rows=0)
    p = np.clip(score, 1e-12, 1-1e-12)
    loss = -(y*np.log(p)+(1-y)*np.log1p(-p))
    positive, negative = y == 1, y == 0
    def describe(mask):
        v = score[mask]
        return dict(rows=len(v), mean=float(v.mean()), q10=float(np.quantile(v, .1)),
                    median=float(np.median(v)), q90=float(np.quantile(v, .9)),
                    high90=int((v >= .9).sum()), high90_fraction=float((v >= .9).mean()),
                    low10=int((v <= .1).sum()), low10_fraction=float((v <= .1).mean())) if len(v) else dict(rows=0)
    scene_losses = [float(loss[np.array(scenes) == s].mean()) for s in sorted(set(scenes))]
    return dict(rows=len(y), scenes=len(set(scenes)), positive=describe(positive), negative=describe(negative),
        logloss=float(loss.mean()), macro_scene_logloss=float(np.mean(scene_losses)),
        balanced_logloss=float((loss[positive].mean()+loss[negative].mean())/2) if positive.any() and negative.any() else None,
        brier=float(np.mean((score-y)**2)),
        balanced_accuracy50=float(((score[positive] >= .5).mean()+(score[negative] < .5).mean())/2) if positive.any() and negative.any() else None,
        ROC_AUC=float(roc_auc_score(y, score)) if positive.any() and negative.any() else None,
        pair_average_precision=float(average_precision_score(y, score)) if positive.any() else None)


def improvement(base, variant):
    both = base.get('balanced_logloss') is not None and variant.get('balanced_logloss') is not None
    return dict(balanced_logloss_lower=bool(both and variant['balanced_logloss'] < base['balanced_logloss']),
        positive_high90_not_lower=bool(variant.get('positive', {}).get('high90_fraction', -1) >= base.get('positive', {}).get('high90_fraction', 0)),
        negative_high90_not_higher=bool(variant.get('negative', {}).get('high90_fraction', 1) <= base.get('negative', {}).get('high90_fraction', 0)),
        ROC_AUC_higher=bool(base.get('ROC_AUC') is not None and variant.get('ROC_AUC') is not None and variant['ROC_AUC'] > base['ROC_AUC']))
