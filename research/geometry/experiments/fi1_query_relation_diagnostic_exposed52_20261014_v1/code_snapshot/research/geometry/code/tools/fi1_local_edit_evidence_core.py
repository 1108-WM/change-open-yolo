"""GT-free proposal domains and evidence, no inference acceptance rule."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def pieces(labels, member, visible, probability, query, neighbors, xyz):
    """Every nonempty editable region and query-boundary connected alternative."""
    pq=probability[:,query];others=probability.copy();others[:,query]=-1
    compete=others.max(1)
    state=(pq>=.5).astype(np.int8)+2*(compete>pq).astype(np.int8)
    for region in range(int(labels.max())+1):
        domain=np.flatnonzero((labels==region)&(visible>=2))
        if not len(domain):continue
        assert np.all(member[domain]==member[domain[0]])
        yield region,'region',0,domain
        if len(np.unique(state[domain]))==1:continue
        lookup=np.full(len(labels),-1,np.int32);lookup[domain]=np.arange(len(domain))
        dest=lookup[neighbors[domain]];src,col=np.nonzero(dest>=0);tar=dest[src,col]
        near=neighbors[domain][src,col];valid=(state[domain[src]]==state[near])&(np.linalg.norm(xyz[domain[src]]-xyz[near],axis=1)<=.05)
        graph=csr_matrix((np.ones(valid.sum()),(src[valid],tar[valid])),shape=(len(domain),len(domain)))
        count,cc=connected_components(graph,directed=False)
        for i in range(count):yield region,'query_boundary',i,domain[cc==i]

def query_evidence(p,q):
    own=p[:,q];means=p.mean(0,dtype=np.float64);means[q]=-1
    competitor=int(np.lexsort((np.arange(len(means)),-means))[0]);other=p[:,competitor]
    clipped=np.clip(own,1e-6,1-1e-6);entropy=-(clipped*np.log(clipped)+(1-clipped)*np.log(1-clipped))
    winner=p.argmax(1)
    return dict(query_mean=float(own.mean()),query_std=float(own.std()),query_min=float(own.min()),query_max=float(own.max()),query_entropy=float(entropy.mean()),query_ge05_fraction=float((own>=.5).mean()),query_ge08_fraction=float((own>=.8).mean()),competitor_id=competitor,competitor_mean=float(other.mean()),query_margin_mean=float((own-other).mean()),competitor_winner_fraction=float((winner!=q).mean()),distinct_query_winners=len(np.unique(winner)))

def frame_evidence(edit,reference,visible,mask):
    """Mask IDs local to each frame. Remove edited points from deletion reference."""
    es=edit[visible[edit]];rs=reference[visible[reference]]
    out=dict(edit_visible=len(es),reference_visible=len(rs),missing=True,common=False,conflict=False,edit_reference_mask_coverage=None,edit_best_mask_coverage=None,reference_best_mask_coverage=None,common_best_mask=False)
    if not len(es) or not len(rs) or not mask.shape[1]:return out
    ec=mask[es].mean(0);rc=mask[rs].mean(0);e=int(ec.argmax());r=int(rc.argmax())
    if ec[e]<=0 or rc[r]<=0:return out
    out.update(missing=False,common=bool(ec[r]>=.5 and rc[r]>=.5),conflict=bool(e!=r and ec[e]>=.5 and rc[r]>=.5 and ec[r]<.5),edit_reference_mask_coverage=float(ec[r]),edit_best_mask_coverage=float(ec[e]),reference_best_mask_coverage=float(rc[r]),common_best_mask=e==r)
    return out

def screens(row):
    """Predeclared diagnostic bins, not tuned or authorized execution rules."""
    add=row['action']=='add';q=row['query_mean'];margin=row['query_margin_mean']
    aligned=q>=.5 if add else q<.5
    contrast=aligned and (margin>=.1 if add else margin<=-.1)
    confident=q>=.8 and margin>=.1 if add else q<=.2 and row['competitor_mean']>=.8 and margin<=-.1
    views=row['common_frames']>=2 if add else row['conflict_frames']>=2
    return dict(all=True,query_direction=aligned,query_contrast=contrast,query_confident_contrast=confident,multiview_direction=views,query_contrast_multiview=contrast and views)
