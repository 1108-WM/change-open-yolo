"""GT-free complementary-fragment retrieval and explicit association hypotheses.

Spatial/projection proximity retrieves pairs, never establishes identity.
Exclusive regions prevent shared points from manufacturing bridge evidence.
"""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

PARAMETERS=dict(spatial_radius=.10,minimum_contacts=3,projection_bbox_gap_pixels=3.,
    projection_min_groups=2,minimum_visible_points=3,minimum_visible_fraction=.1,
    joint_coverage=.5,separation_other_max=.1,compact_mask_precision=.8,
    minimum_joint_groups=2,identity_contact_radius=.05,maximum_color_difference=.4,
    maximum_normal_difference=.5,maximum_depth_jump=.05,maximum_pair_iou=.9,depth_contact_sample_limit=128)
SIGNATURES=('joint_only','compact_no_separation','compact_boundary_depth')

def bbox_gap(a,b):
    """Euclidean gap between closed axis-aligned boxes, zero when intersecting."""
    return float(np.linalg.norm(np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0)))

def project_points(xyz,pose,intrinsic):
    xyz=np.asarray(xyz);pose=np.asarray(pose);intrinsic=np.asarray(intrinsic)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or pose.shape!=(4,4) or intrinsic.shape!=(4,4):raise ValueError('projection domain')
    homogeneous=np.column_stack((xyz,np.ones(len(xyz))));camera=homogeneous@np.linalg.inv(pose).T;projected=camera@intrinsic.T
    pixels=np.zeros((len(xyz),2),np.int64);valid=np.isfinite(projected).all(axis=1)&(projected[:,2]>0)
    pixels[valid]=np.trunc(projected[valid,:2]/projected[valid,2,None]).astype(np.int64)
    return pixels,camera[:,2],valid

def boundary(a,b,raw,parameters=PARAMETERS):
    """Evidence between exclusive regions; unknown normals remain unknown."""
    a=np.asarray(a);b=np.asarray(b)
    if not len(a) or not len(b):return dict(contacts10=0,contacts05=0,contact_pairs=[],color_difference=None,normal_difference=None,normal_valid_fraction=0.,distance_min=None)
    tree=cKDTree(raw[a,:3]);dist,index=tree.query(raw[b,:3],k=1,distance_upper_bound=parameters['spatial_radius']);ok=np.isfinite(dist);near=ok&(dist<=parameters['identity_contact_radius'])
    left=a[index[near]];right=b[near];rgb=np.clip(raw[:,3:6]/255.,0,1);color=np.linalg.norm(rgb[left]-rgb[right],axis=1)/np.sqrt(3)
    normals=raw[:,6:9];norm=np.linalg.norm(normals,axis=1);nv=(norm[left]>1e-12)&(norm[right]>1e-12)
    nd=1-np.abs(np.sum(normals[left[nv]]/norm[left[nv],None]*normals[right[nv]]/norm[right[nv],None],axis=1))
    selected=np.linspace(0,len(left)-1,min(len(left),parameters['depth_contact_sample_limit']),dtype=int) if len(left) else np.empty(0,int)
    return dict(contacts10=int(ok.sum()),contacts05=int(near.sum()),distance_min=float(dist[ok].min()) if ok.any() else None,
        color_difference=float(np.median(color)) if len(color) else None,normal_difference=float(np.median(nd)) if len(nd) else None,
        normal_valid_fraction=float(nv.mean()) if len(nv) else 0.,contact_pairs=np.column_stack((left[selected],right[selected])).tolist())

def view_evidence(a,b,frames,contact_pairs,parameters=PARAMETERS,endpoints=None,shared=None):
    groups=defaultdict(lambda:dict(eligible=False,joint=False,compact=False,separated=False,projection=False,depth=False))
    witnesses=[];cp=np.asarray(contact_pairs,dtype=int).reshape(-1,2)
    for f in frames:
        g=groups[f['group']]
        fast=endpoints is not None and shared is not None and not len(shared)
        if fast:
            na,nb=map(int,f['candidate_visible'][list(endpoints)]);boxa,boxb=[f['candidate_boxes'][i] for i in endpoints]
        else:
            aa=a[f['visible'][a]];bb=b[f['visible'][b]];na,nb=len(aa),len(bb)
            boxa=(f['pixels'][aa].min(axis=0),f['pixels'][aa].max(axis=0)) if na else None;boxb=(f['pixels'][bb].min(axis=0),f['pixels'][bb].max(axis=0)) if nb else None
        if na<parameters['minimum_visible_points'] or nb<parameters['minimum_visible_points'] or na/len(a)<parameters['minimum_visible_fraction'] or nb/len(b)<parameters['minimum_visible_fraction']:continue
        g['eligible']=True;gap=bbox_gap(boxa,boxb)
        if gap<=parameters['projection_bbox_gap_pixels']:g['projection']=True
        if endpoints is not None:
            sharedcounts=np.array([len(np.intersect1d(shared,m,assume_unique=True)) for m in f['masks']]) if len(shared) else 0
            ia=f['candidate_intersections'][endpoints[0]]-sharedcounts;ib=f['candidate_intersections'][endpoints[1]]-sharedcounts;sizes=f['mask_sizes']
        else:
            ia=np.array([len(np.intersect1d(aa,m,assume_unique=True)) for m in f['masks']]);ib=np.array([len(np.intersect1d(bb,m,assume_unique=True)) for m in f['masks']]);sizes=np.array([len(m) for m in f['masks']])
        ca=ia/na;cb=ib/nb
        joint=(ca>=parameters['joint_coverage'])&(cb>=parameters['joint_coverage']);tight=joint&((ia+ib)>=parameters['compact_mask_precision']*sizes)
        separate=bool(np.any((ca>=parameters['joint_coverage'])&(cb<=parameters['separation_other_max'])) and np.any((cb>=parameters['joint_coverage'])&(ca<=parameters['separation_other_max'])))
        g['joint']|=bool(joint.any());g['compact']|=bool(tight.any());g['separated']|=separate
        depth_known=False;depth_ok=False
        if len(cp):
            visible=f['visible'][cp].all(axis=1);pairs=cp[visible]
            if len(pairs)>=parameters['minimum_contacts']:
                jump=np.abs(f['depth'][pairs[:,0]]-f['depth'][pairs[:,1]]);depth_known=True;depth_ok=float(np.median(jump))<=parameters['maximum_depth_jump'];g['depth']|=depth_ok
        witnesses.append(dict(frame=f['frame'],group=f['group'],visible_a=na,visible_b=nb,projection_bbox_gap=gap,joint=bool(joint.any()),compact=bool(tight.any()),separated=separate,depth_known=depth_known,depth_continuous=depth_ok))
    result={name+'_groups':sorted(k for k,g in groups.items() if g[name]) for name in ('eligible','joint','compact','separated','projection','depth')}
    result['joint_and_compact_depth_groups']=sorted(k for k,g in groups.items() if g['compact'] and g['depth'])
    result['witnesses']=witnesses;result['unknown_identity']=not result['joint_groups'] and not result['separated_groups']
    return result

def signatures(boundary_row,views,parameters=PARAMETERS):
    joint=len(views['joint_groups'])>=parameters['minimum_joint_groups']
    compact=len(views['compact_groups'])>=parameters['minimum_joint_groups'] and not views['separated_groups']
    bd=boundary_row;surface=bd['contacts05']>=parameters['minimum_contacts'] and bd['normal_difference'] is not None and bd['color_difference'] is not None and bd['normal_valid_fraction']>=.5 and bd['normal_difference']<=parameters['maximum_normal_difference'] and bd['color_difference']<=parameters['maximum_color_difference']
    depth=len(views['joint_and_compact_depth_groups'])>=parameters['minimum_joint_groups']
    return dict(joint_only=joint,compact_no_separation=compact,compact_boundary_depth=compact and surface and depth)
