"""Reuse frozen U interventions and original matching projections on new cohorts."""
from pathlib import Path
from tools.run_fi1_union_loss_attribution_train90 import Guard as ParentGuard
from tools.fi1_class_agnostic_ablation import project_matches
from tools.fi1_union_loss_attribution import key


class Guard(ParentGuard):
    def __init__(self, contract, raw_GT_allowed=False):
        super().__init__(contract['scenes'], [])
        self.raw_GT_allowed=raw_GT_allowed
        self.training_gt={str(Path(b['gt_path']).resolve()) for b in contract['bindings'].values()}
        self.labels={str(Path(p).resolve()) for p in contract.get('label_inputs',{})}
    def check(self,path):
        p=str(Path(path).resolve())
        if any(part.startswith('ground_truth') for part in Path(p).parts):
            if not self.raw_GT_allowed or not self.gt_enabled or p not in self.training_gt:
                raise PermissionError('GT requires sealed development200 predictions and exact whitelist')
            self.gt_opened.add(p)
            # Parent only recognizes /ground_truth/, while legacy uses mirrors.
            original=self.gt_paths;self.gt_paths=self.training_gt
            try:super().check(p)
            finally:self.gt_paths=original
            return
        if p in self.labels and not self.gt_enabled:
            raise PermissionError('cached labels require all800 development predictions sealed')
        if '/learning/dataset/' in p and Path(p).name in ('features.npz','members.jsonl'):
            raise PermissionError('cached training targets not used in this replay')
        super().check(p)


def project_multiclass(template, universe, selected, sizes, labels, valid_ids):
    """Reuse existing projection per class; recover original global pred_id order.

    One original assignment of the union of all arm outputs suffices because
    geometric matching does not depend on confidence. Both matching directions
    are reconstructed in the arm's own prediction order and own scores.
    """
    if len({key(m) for m in universe})!=len(universe) or len({key(m) for m in selected})!=len(selected):
        raise ValueError('duplicate multiclass member')
    if not {key(m) for m in selected}<={key(m) for m in universe}:
        raise ValueError('projection member outside frozen universe')
    result=dict(gt={},pred={});uuid_key={}
    for ci,label in enumerate(labels):
        members=[m for m in universe if m['class_index']==ci and sizes[m['geometry_key']]>=100]
        expected=template['pred'][label]
        if len(members)!=len(expected):raise ValueError('original cached class alignment differs')
        for m,p in zip(members,expected):
            if p['vert_count']!=sizes[m['geometry_key']] or p['label_id']!=int(valid_ids[ci]) or p['confidence']!=m['score']:
                raise ValueError('original cached member identity differs')
            uuid_key[p['uuid']]=key(m)
        subset=[m for m in selected if m['class_index']==ci]
        projected=project_matches(template,[m['geometry_key'] for m in members],subset,label)
        result['gt'].update(projected['gt']);result['pred'].update(projected['pred'])
    pred_ids={key(m):i for i,m in enumerate(m for m in selected if sizes[m['geometry_key']]>=100)}
    for label in labels:
        for p in result['pred'][label]:p['pred_id']=pred_ids[uuid_key[p['uuid']]]
        for g in result['gt'][label]:
            for p in g['matched_pred']:p['pred_id']=pred_ids[uuid_key[p['uuid']]]
    return result
