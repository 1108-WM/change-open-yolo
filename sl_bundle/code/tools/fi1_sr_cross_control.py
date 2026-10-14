"""Fixed-member S/R score swap and independent cached-match replay."""
import copy
import numpy as np


def key(row):
    return row['geometry_key'], row['class_index']


def fixed_rescore(selected, pool, scores):
    scores=np.asarray(scores)
    if scores.shape!=(len(pool),) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('invalid full-pool score vector')
    index={key(m):i for i,m in enumerate(pool)}
    if len(index)!=len(pool) or len({key(m) for m in selected})!=len(selected):
        raise ValueError('duplicate members')
    result=[]
    for m in selected:
        if key(m) not in index:
            raise ValueError('selected member absent from score pool')
        source=pool[index[key(m)]]
        if {k:v for k,v in m.items() if k!='score'}!={k:v for k,v in source.items() if k!='score'}:
            raise ValueError('member metadata differs')
        result.append({**m,'score':float(scores[index[key(m)]])})
    return sorted(result,key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))


def independent_rescore_match(template, old_rows, new_rows, sizes, agnostic=False):
    """Rebuild both matching directions without projection helpers or raw GT.

    Matching is score-independent; keep every recorded geometry intersection,
    ignored instance and void count, change only global pred_id/confidence/order.
    """
    identity=(lambda m:m['geometry_key']) if agnostic else key
    before=[m for m in old_rows if sizes[m['geometry_key']]>=100]
    after=[m for m in new_rows if sizes[m['geometry_key']]>=100]
    assert {identity(m) for m in before}=={identity(m) for m in after}
    old_id={i:identity(m) for i,m in enumerate(before)}
    new_id={identity(m):i for i,m in enumerate(after)}
    score={identity(m):m['score'] for m in after}
    by_key={};class_label={}
    for label,ps in template['pred'].items():
        for p in ps:
            k=old_id[p['pred_id']];assert k not in by_key
            assert p['confidence']==before[p['pred_id']]['score']
            assert p['vert_count']==sizes[before[p['pred_id']]['geometry_key']]
            by_key[k]=p;class_label[k]=label
    assert set(by_key)==set(new_id)
    result=dict(gt={},pred={label:[] for label in template['pred']})
    lookup={}
    for label,gs in template['gt'].items():
        result['gt'][label]=[]
        for g in gs:
            row=copy.deepcopy({k:v for k,v in g.items() if k!='matched_pred'})
            row['matched_pred']=[];result['gt'][label].append(row)
            lookup[label,g['instance_id']]=row
    for m in after:
        k=identity(m);p=copy.deepcopy(by_key[k]);p['confidence']=score[k];p['pred_id']=new_id[k]
        label=class_label[k];result['pred'][label].append(p)
        for g in p['matched_gt']:
            lookup[label,g['instance_id']]['matched_pred'].append(
                {**{k:v for k,v in p.items() if k!='matched_gt'},'intersection':g['intersection']})
    return result
