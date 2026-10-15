"""GT-free class context and deterministic one-to-one competition supervision."""
import numpy as np
from scipy import sparse

EXTRA_FEATURES=['same_class_max_iou','log1p_neighbors_iou_gt_half',
    'max_iou_times_neighbor_direct','direct_gap_to_strong_neighbor',
    'strong_neighbor_higher_score_fraction','max_own_point_coverage']


def point_matrix(points,domain):
    for p in points:
        if p.ndim!=1 or p.dtype.kind not in 'iu' or not np.array_equal(p,np.unique(p)) or (len(p) and (p[0]<0 or p[-1]>=domain)):
            raise ValueError('invalid point index domain/order')
    sizes=np.asarray([len(p) for p in points],dtype=np.int64)
    indices=np.concatenate(points) if points else np.empty(0,dtype=np.int64)
    return sparse.csr_matrix((np.ones(len(indices),dtype=np.int64),indices,np.r_[0,np.cumsum(sizes)]),shape=(len(points),domain))


def pair_overlap(points,domain):
    masks=point_matrix(points,domain)
    intersection=(masks@masks.T).toarray()
    sizes=np.array([len(p) for p in points])
    union=sizes[:,None]+sizes[None,:]-intersection
    iou=np.divide(intersection,union,out=np.zeros_like(union,dtype=float),where=union>0)
    coverage=np.divide(intersection,sizes[:,None],out=np.zeros_like(intersection,dtype=float),where=sizes[:,None]>0)
    return iou,coverage


def context_features(members,geometry_keys,iou,coverage):
    lookup={k:i for i,k in enumerate(geometry_keys)}
    if len({(r['geometry_key'],r['class_index']) for r in members})!=len(members):raise ValueError('duplicate members')
    out=np.zeros((len(members),6),dtype=np.float64)
    classes=np.array([r['class_index'] for r in members]);indexes=np.array([lookup[r['geometry_key']] for r in members])
    scores=np.array([r['score'] for r in members],dtype=np.float64)
    for c in np.unique(classes):
        rr=np.flatnonzero(classes==c);ii=indexes[rr];ss=scores[rr]
        overlap=iou[np.ix_(ii,ii)].copy();cover=coverage[np.ix_(ii,ii)].copy()
        np.fill_diagonal(overlap,0);np.fill_diagonal(cover,0)
        strong=overlap>.5;counts=strong.sum(axis=1)
        out[rr,0]=overlap.max(axis=1);out[rr,1]=np.log1p(counts)
        out[rr,2]=(overlap*ss[None,:]).max(axis=1)
        out[rr,3]=np.where(counts>0,ss-np.where(strong,ss[None,:],-np.inf).max(axis=1),0)
        out[rr,4]=np.divide((strong&(ss[None,:]>ss[:,None])).sum(axis=1),counts,out=np.zeros(len(rr)),where=counts>0)
        out[rr,5]=cover.max(axis=1)
    if not np.isfinite(out).all():raise ValueError('nonfinite context')
    return out


def gt_iou(points,gt,class_ids):
    ids,counts=np.unique(gt,return_counts=True)
    valid=(ids>=1000)&(counts>=100)&np.isin(ids//1000,class_ids)
    ids=ids[valid];counts=counts[valid]
    if not len(ids):return np.zeros((len(points),0)),ids
    truth=sparse.csr_matrix(np.column_stack([gt==i for i in ids]),dtype=np.int64)
    inter=(point_matrix(points,len(gt))@truth).toarray()
    sizes=np.array([len(p) for p in points]);union=sizes[:,None]+counts[None,:]-inter
    iou=inter/union;iou[sizes<100]=0
    return iou,ids


def assign_competition(members,geometry_keys,overlap,gt_ids,class_ids):
    lookup={k:i for i,k in enumerate(geometry_keys)}
    classes=np.array([r['class_index'] for r in members]);indexes=np.array([lookup[r['geometry_key']] for r in members])
    scores=np.array([r['score'] for r in members]);native=np.array([r['has_original_native_alias'] for r in members])
    y=np.zeros(len(members));assignment=np.full(len(members),-1,dtype=np.int64)
    for c in np.unique(classes):
        rr=np.flatnonzero(classes==c);gg=np.flatnonzero(gt_ids//1000==class_ids[c])
        table=overlap[np.ix_(indexes[rr],gg)]
        a,b=np.nonzero(table>0)
        if not len(a):continue
        row=rr[a];col=gg[b]
        keys=np.array([members[i]['geometry_key'] for i in row])
        order=np.lexsort((gt_ids[col],keys,~native[row],-scores[row],-table[a,b]))
        used=set()
        for edge in order:
            i=int(row[edge]);j=int(col[edge])
            if assignment[i]<0 and j not in used:
                assignment[i]=int(gt_ids[j]);y[i]=overlap[indexes[i],j];used.add(j)
    return y,assignment


def independent_context(members,geometry_keys,iou,coverage):
    lookup={k:i for i,k in enumerate(geometry_keys)};by_class={}
    for i,r in enumerate(members):by_class.setdefault(r['class_index'],[]).append(i)
    out=np.zeros((len(members),6))
    for i,r in enumerate(members):
        neighbors=[j for j in by_class[r['class_index']] if j!=i]
        if not neighbors:continue
        own=lookup[r['geometry_key']];js=[lookup[members[j]['geometry_key']] for j in neighbors]
        ov=iou[own,js];ss=np.array([members[j]['score'] for j in neighbors]);strong=ov>.5
        out[i]=[max(ov),np.log1p(sum(strong)),max(ov*ss),r['score']-max(ss[strong]) if strong.any() else 0,
            np.mean(ss[strong]>r['score']) if strong.any() else 0,max(coverage[own,js])]
    return out


def independent_assignment(members,geometry_keys,overlap,gt_ids,class_ids):
    lookup={k:i for i,k in enumerate(geometry_keys)};edges=[]
    for i,r in enumerate(members):
        g=lookup[r['geometry_key']]
        for j,gt in enumerate(gt_ids):
            if int(gt)//1000==class_ids[r['class_index']] and overlap[g,j]>0:
                edges.append((-float(overlap[g,j]),-r['score'],not r['has_original_native_alias'],r['geometry_key'],int(gt),i))
    y=np.zeros(len(members));assignment=np.full(len(members),-1,dtype=np.int64);used=set()
    for minus_iou,_,_,_,gt,i in sorted(edges):
        if assignment[i]==-1 and gt not in used:
            y[i]=-minus_iou;assignment[i]=gt;used.add(gt)
    return y,assignment
