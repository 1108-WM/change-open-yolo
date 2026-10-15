"""Two fixed whole-region T-add rules; no GT, deletion or extra gates."""
import numpy as np

def blocks(parent,halo,labels,visible,probability,query):
    member=np.isin(halo,parent)
    for rid in range(int(labels.max())+1):
        ix=np.flatnonzero((labels==rid)&~member&(visible>=2))
        if not len(ix):continue
        # Member regions must not cross the original parent boundary.
        assert not member[labels==rid].any()
        means=probability[ix].mean(0,dtype=np.float64);own=float(means[query]);means[query]=-1
        competitor=int(np.lexsort((np.arange(len(means)),-means))[0]);other=float(means[competitor]);margin=own-other
        yield dict(region=rid,indices=ix,query_mean=own,competitor_id=competitor,competitor_mean=other,margin=margin,A=own>=.5,B=own>=.5 and margin>=.1)

def union(parent,selected):
    added=np.unique(np.concatenate(selected)) if selected else np.empty(0,np.int64)
    assert not np.intersect1d(parent,added).size
    return np.union1d(parent,added),added
