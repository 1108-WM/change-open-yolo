"""GT-free, fixed-budget selection from existing projected RGB-D observations."""
import numpy as np

BUDGET = 6
ROLE_WEIGHTS = (.20, .25, .25, .30)


def region_weights(parent, halo, boundary, disagreements):
    """Outside/contact and N/T disagreement are evidence needs, not negatives."""
    roles = [np.isin(halo,parent), np.isin(halo,boundary),
             ~np.isin(halo,parent), np.isin(halo,disagreements)]
    weights=np.zeros(len(halo),float)
    for role,weight in zip(roles,ROLE_WEIGHTS):
        if role.any():weights[role]+=weight/int(role.sum())
    return roles,weights


def choose_views(frame_ids,groups,reliable,covered,has_masks,roles,weights,budget=BUDGET):
    """First preserve subject observation, then gain complementary two-group cover.

    No instance identity or masks are accepted/rejected here. Mask coverage
    mildly rewards available image evidence; unknown pixels never count as a
    negative. Only existing nonempty mask frames may enter the six-view bank.
    """
    if reliable.shape!=covered.shape or reliable.shape[1]!=len(weights):
        raise ValueError('unaligned candidate observation domain')
    parent=roles[0];min_body=min(20,int(parent.sum()))
    eligible=np.asarray(has_masks,bool)&(reliable[:,parent].sum(axis=1)>=min_body)
    chosen=[];used_groups=set();count=np.zeros(len(weights),np.uint8);trace=[]
    for step in range(budget):
        ranked=[]
        for i in np.flatnonzero(eligible):
            if int(groups[i]) in used_groups:continue
            novelty=reliable[i]&(count<2)
            gain=float(weights[novelty].sum())
            mask_gain=float(weights[novelty&covered[i]].sum())
            body=float(reliable[i,parent].mean()) if parent.any() else 0.
            priority=body if step==0 else gain+.25*mask_gain
            ranked.append((-priority,-gain,-mask_gain,int(frame_ids[i]),int(i),body))
        if not ranked:break
        priority,gain,mask_gain,_,i,body=min(ranked)
        if step and gain==0:break
        chosen.append(i);used_groups.add(int(groups[i]));count+=reliable[i]
        trace.append(dict(frame=int(frame_ids[i]),group=int(groups[i]),
                          priority=-priority,weighted_new_coverage=-gain,
                          weighted_mask_coverage=-mask_gain,subject_fraction=body))
    return chosen,trace
