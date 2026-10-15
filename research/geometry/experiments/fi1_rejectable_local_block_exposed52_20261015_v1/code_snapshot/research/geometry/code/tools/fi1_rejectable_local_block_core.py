"""Two-stage local attribution and point readout, no GT or learned parameters."""
import numpy as np

def frame_witness(edit,reference,visible,masks):
    """A witness requires a unique best automatic mask on BOTH visible subsets.

    Coverage .5 reuses the existing descriptive majority definition. Ties remain
    ambiguous; missing data is distinct from observed zero support. Two masks
    with different IDs are observational conflict, not proof of different GT.
    """
    e=np.asarray(edit)[visible[edit]];r=np.asarray(reference)[visible[reference]]
    row=dict(edit_visible=len(e),reference_visible=len(r),state='missing',edit_best_mask=None,reference_best_mask=None,edit_best_coverage=None,reference_best_coverage=None,edit_ties=0,reference_ties=0)
    if not len(e) or not len(r) or not masks.shape[1]:return row
    ec=masks[e].mean(0);rc=masks[r].mean(0);em=float(ec.max());rm=float(rc.max());ei=np.flatnonzero(ec==em);ri=np.flatnonzero(rc==rm)
    row.update(edit_best_coverage=em,reference_best_coverage=rm,edit_ties=len(ei),reference_ties=len(ri))
    if em<=0 or rm<=0:row['state']='observed_no_support';return row
    if len(ei)!=1 or len(ri)!=1:row['state']='ambiguous_tie';return row
    row.update(edit_best_mask=int(ei[0]),reference_best_mask=int(ri[0]))
    if em<.5 or rm<.5:row['state']='observed_weak';return row
    row['state']='common' if ei[0]==ri[0] else 'conflict'
    return row

def attribution(witnesses,query_mean):
    """A fixed conservative certificate; evidence absence is abstention.

    Not inferred from AUC. The >=2 support count is the previous two-view
    convention, .5 is the existing query/mask support definition. This rule is
    a diagnostic hypothesis, not an assertion that mask consistency is GT.
    """
    states=[r['state'] for r in witnesses]
    if query_mean<.5:return 'abstain_no_query_support'
    if 'conflict' in states:return 'reject_observation_conflict'
    if states.count('common')>=2:return 'accept_shared_mask_certificate'
    return 'abstain_insufficient_observation'

def supporter(probability,indices,query_ids):
    q=np.asarray(sorted(set(map(int,query_ids))),dtype=np.int64)
    if not len(q):raise ValueError('empty frozen query set')
    means=probability[indices][:,q].mean(0,dtype=np.float64)
    k=int(np.lexsort((q,-means))[0]);return int(q[k]),float(means[k])

def readout(block,probability,query,accepted):
    block=np.asarray(block,np.int64)
    if not accepted:return block[:0],block[:0]
    return block,block[probability[block,query]>=.5]
