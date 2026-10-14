"""Geometry-only multiscale inspection regions on the frozen point domain."""
from dataclasses import replace
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tools.fi1_split_grow_core import SplitGrowConfig,unit

PARAMETERS=dict(scales_m=(.05,.12,.30),offsets=(0.,.5),
                color_limits=(.12,.22,.35),normal_limits=(.15,.30,.50),
                edge_distance=.05,budgets=(48,96,192),primary_budget=96,
                size_edges=(16,64,256),disagreement_bonus=.25,groups=2,
                min_mask_points=100,probability_threshold=.8)
SIZE_NAMES=('1_15','16_63','64_255','256_plus')
SUPPLY_GATE=dict(min_regions=1000,min_safe_actions=250,min_safe_parents=40,
                 min_safe_scenes=20,min_safe64=25,min_safe256=8,
                 min_mixed_scenes=15,min_unknown_regions=100,
                 min_unknown_parents=20,min_unknown_scenes=10,
                 min_pool_gain02_targets=10,min_pool_gain02_scenes=5)


def components(indices,neighbors,allowed_edges=None):
    """Frozen directed kNN becomes undirected without dropping asymmetric edges."""
    indices=np.asarray(indices,np.int32)
    if not len(indices):return []
    lookup=np.full(len(neighbors),-1,np.int32);lookup[indices]=np.arange(len(indices))
    dst=lookup[neighbors[indices]];src=np.broadcast_to(np.arange(len(indices))[:,None],dst.shape)
    keep=dst>=0
    if allowed_edges is not None:keep&=allowed_edges
    a,b=src[keep],dst[keep];graph=csr_matrix((np.ones(len(a),np.uint8),(a,b)),shape=(len(indices),len(indices)))
    n,labels=connected_components(graph,directed=False);order=np.argsort(labels,kind='stable');cuts=np.cumsum(np.bincount(labels))[:-1]
    return [np.sort(indices[k]) for k in np.split(order,cuts)]


def probability_blocks(parent,halo,probability,visible,neighbors):
    # Explicit float32 fixes the old float16 0.8 boundary, only in this new arm.
    p=np.asarray(probability,dtype=np.float32)
    idx=np.flatnonzero(np.isin(halo,parent)&(visible>=2)&(p[:,1]>=np.float32(.8)))
    return components(idx,neighbors)


def size_bins(counts):return np.searchsorted(np.asarray(PARAMETERS['size_edges']),counts,side='right')


def inspection_order(blocks,xyz,disagreement,limit=32):
    """Size strata plus alternating coverage/space; no disagreement eligibility."""
    if not blocks:return []
    count=np.asarray([len(b) for b in blocks]);bins=size_bins(count)
    centers=np.stack([xyz[b].mean(0) for b in blocks]);fraction=np.asarray([disagreement[b].mean() for b in blocks])
    available=np.ones(len(blocks),bool);center=centers.mean(0);distance=np.sum((centers-center)**2,1);chosen=[]
    for step in range(min(limit,len(blocks))):
        category=step%4
        eligible=np.flatnonzero(available&(bins==category))
        if not len(eligible):eligible=np.flatnonzero(available)
        score=(np.sqrt(count)*(1+.25*fraction)) if (step//4)%2==0 else (distance+.000001*fraction)
        index=min(eligible,key=lambda i:(-float(score[i]),-int(count[i]),int(blocks[i][0])))
        chosen.append(int(index));available[index]=False
        distance=np.minimum(distance,np.sum((centers-centers[index])**2,1))
    return chosen


def propose(xyz,rgb,normals,parent,halo,visible,neighbors,disagreement):
    """Partition visible parent points using spatial tiles AND point-edge changes.

    Tiles bound a region, but are never indivisible decisions: edges split them.
    Original superpoint and annotation columns are not accepted.
    """
    xyz=np.asarray(xyz,np.float32);rgb=np.asarray(rgb,np.float32);normals=np.asarray(normals,np.float32)
    n=len(halo)
    if xyz.shape!=(n,3) or rgb.shape!=(n,3) or normals.shape!=(n,3) or neighbors.shape!=(n,8):raise ValueError('point evidence shape')
    if not all(np.isfinite(a).all() for a in (xyz,rgb,normals)):raise ValueError('nonfinite raw geometry')
    if not np.array_equal(halo,np.unique(halo)) or not np.isin(parent,halo).all():raise ValueError('point domain')
    rgb=np.clip(rgb/(255. if rgb.max(initial=0)>1.5 else 1.),0,1)
    lengths=np.linalg.norm(normals,axis=1);normals=unit(normals)
    member=np.isin(halo,parent);indices=np.flatnonzero(member&(visible>=2)).astype(np.int32)
    if not len(indices):return [],[],{str(b):np.empty(0,np.int32) for b in PARAMETERS['budgets']},dict(visible_parent=0,family_regions=[0]*6)
    lookup=np.full(n,-1,np.int32);lookup[indices]=np.arange(len(indices));dst=lookup[neighbors[indices]];src=np.broadcast_to(np.arange(len(indices))[:,None],dst.shape)
    eligible=dst>=0;a=src[eligible];b=dst[eligible];pa=indices[a];pb=indices[b]
    dd=np.linalg.norm(xyz[pa]-xyz[pb],axis=1)
    nd=1-np.abs((normals[pa]*normals[pb]).sum(1));nd[(lengths[pa]<1e-6)|(lengths[pb]<1e-6)]=0
    cd=np.linalg.norm(rgb[pa]-rgb[pb],axis=1)/np.sqrt(3)
    origin=xyz[member].min(0);registry={};regions=[];metadata=[];family_orders=[];family_counts=[]
    for scale,(width,color,normal) in enumerate(zip(PARAMETERS['scales_m'],PARAMETERS['color_limits'],PARAMETERS['normal_limits'])):
        cfg=replace(SplitGrowConfig(),max_contact_distance=.05,max_color_difference=color,max_normal_difference=normal)
        # Reuse the project's sign-invariant normal/color affinity conventions.
        affinity=(dd<=cfg.max_contact_distance)&(nd<=cfg.max_normal_difference)&(cd<=cfg.max_color_difference)
        for shift in PARAMETERS['offsets']:
            family=len(family_orders);cells=np.floor((xyz[indices]-origin)/width+shift).astype(np.int64)
            _,cell_ids=np.unique(cells,axis=0,return_inverse=True);keep=affinity&(cell_ids[a]==cell_ids[b])
            graph=csr_matrix((np.ones(int(keep.sum()),np.uint8),(a[keep],b[keep])),shape=(len(indices),len(indices)))
            count,labels=connected_components(graph,directed=False);order=np.argsort(labels,kind='stable');cuts=np.cumsum(np.bincount(labels))[:-1]
            blocks=[np.sort(indices[k]).astype(np.int32) for k in np.split(order,cuts)];family_counts.append(len(blocks));ids=[]
            for block in blocks:
                key=block.tobytes();rid=registry.get(key)
                if rid is None:
                    rid=len(regions);registry[key]=rid;regions.append(block)
                    extent=np.ptp(xyz[block],axis=0);metadata.append(dict(region=rid,points=len(block),size_bin=SIZE_NAMES[int(size_bins(len(block)))],families=[],extent=extent.tolist(),NT_disagreement_fraction=float(disagreement[block].mean())))
                metadata[rid]['families'].append(family);ids.append(rid)
            family_orders.append([ids[k] for k in inspection_order(blocks,xyz,disagreement,32)])
    selected={}
    for budget in PARAMETERS['budgets']:
        # Six families share the fixed budget; exact duplicate sets count once.
        ids=[];seen=set()
        for family in family_orders:
            for rid in family[:budget//6]:
                if rid not in seen:ids.append(rid);seen.add(rid)
        selected[str(budget)]=np.asarray(ids,np.int32)
    trace=dict(visible_parent=len(indices),family_regions=family_counts,family_orders=family_orders,
               unique_raw_regions=len(regions),budget_counts={b:len(v) for b,v in selected.items()},missing_normals=int((lengths[indices]<1e-6).sum()))
    return regions,metadata,selected,trace


def union(blocks):return np.unique(np.concatenate(blocks)) if blocks else np.empty(0,np.int32)


def remove(parent,halo,indices):
    result=np.setdiff1d(parent,halo[indices],assume_unique=True)
    return result if len(result)>=100 else parent.copy()
