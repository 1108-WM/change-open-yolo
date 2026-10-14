"""One-hop local completion from NT, witnessed by independent raw mask groups."""
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
from tools.fi1_split_grow_core import point_indices

PARAMETERS=dict(min_visible_points=3,min_visible_fraction=.1,region_mask_coverage=.5,
    isolation_max_coverage=.1,min_independent_joint_groups=2,min_point_joint_groups=2,
    min_joint_support_fraction=.5,max_boundary_distance=.05,min_boundary_contacts=3,
    max_added_fraction=1.,min_pair_iou=.05,max_pair_iou=.9)


def witnessed_addition(parent,donor,frames,xyz,parameters=PARAMETERS):
    """Never change parent, never add unsupported/unknown points, no GT/classes.

    Pose repeats cast one vote; distinct exact-mask aliases cannot manufacture
    separation. A broad joint mask does not override a separate-mask witness.
    """
    parent=point_indices(parent,len(xyz));donor=point_indices(donor,len(xyz));extra=np.setdiff1d(donor,parent,assume_unique=True)
    result=dict(accepted=False,reason=None,joint_groups=[],separated_groups=[],proposed_extra_points=len(extra))
    if not len(parent) or not len(extra):return parent.copy(),dict(result,reason='empty_increment')
    bygroup=defaultdict(list);separated=set();witnesses=[]
    for f in frames:
        pa=parent[f['visible'][parent]];ex=extra[f['visible'][extra]]
        if len(pa)<parameters['min_visible_points'] or len(ex)<parameters['min_visible_points'] or len(pa)/len(parent)<parameters['min_visible_fraction'] or len(ex)/len(extra)<parameters['min_visible_fraction']:continue
        cov=np.array([(len(np.intersect1d(pa,m,assume_unique=True))/len(pa),len(np.intersect1d(ex,m,assume_unique=True))/len(ex)) for m in f['masks']]).reshape(-1,2)
        threshold=parameters['region_mask_coverage'];iso=parameters['isolation_max_coverage'];group=f['group']
        l=np.flatnonzero((cov[:,0]>=threshold)&(cov[:,1]<=iso));r=np.flatnonzero((cov[:,1]>=threshold)&(cov[:,0]<=iso))
        if any(a!=b for a in l for b in r):
            separated.add(group);witnesses.append(dict(frame=f['frame'],group=group,kind='separated'));continue
        joint=np.flatnonzero(np.minimum(cov[:,0],cov[:,1])>=threshold)
        if len(joint):
            supported=np.zeros(len(extra),bool)
            for m in joint:supported|=np.isin(extra,f['masks'][m],assume_unique=True)
            bygroup[group].append((f['visible'][extra],supported))
            witnesses.append(dict(frame=f['frame'],group=group,kind='joint',mask_indices=list(map(int,joint))))
    jointgroups=set(bygroup)-separated
    result.update(joint_groups=sorted(jointgroups),separated_groups=sorted(separated),witnesses=witnesses)
    if separated:return parent.copy(),dict(result,reason='separation_counterevidence')
    if len(jointgroups)<parameters['min_independent_joint_groups']:return parent.copy(),dict(result,reason='insufficient_independent_joint')
    visible=np.zeros(len(extra),int);support=np.zeros(len(extra),int)
    for group in sorted(jointgroups):
        vv,ss=zip(*bygroup[group]);visible+=np.logical_or.reduce(vv);support+=np.logical_or.reduce(ss)
    good=(support>=parameters['min_point_joint_groups'])&(support>=parameters['min_joint_support_fraction']*visible)
    addition=extra[good];result.update(supported_extra_points=len(addition),support_fraction=float(len(addition)/len(extra)))
    if not len(addition):return parent.copy(),dict(result,reason='no_witnessed_points')
    if len(addition)>parameters['max_added_fraction']*len(parent):return parent.copy(),dict(result,reason='anchor_identity_budget')
    dist,_=cKDTree(xyz[parent]).query(xyz[addition],k=1)
    contacts=int(np.sum(dist<=parameters['max_boundary_distance']));result['boundary_contacts']=contacts
    if contacts<parameters['min_boundary_contacts']:return parent.copy(),dict(result,reason='no_local_contact')
    return np.union1d(parent,addition),dict(result,accepted=True,reason='accepted',added_points=len(addition))
