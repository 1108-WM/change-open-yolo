"""Measured depth and projected exclusive-region frozen DINO evidence.

No GT, class labels, parent semantic scores, or output selection are inputs.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
from PIL import Image
from tools import ncs5_proxyclip_core as dino
from tools import run_fi1_complementary_fragment_diagnostic as prior

def load_model(c):
    import torch
    assert torch.cuda.is_available(),'GPU required; no hidden CPU fallback'
    assert prior.sha(c['model_checkpoint'])==c['model_sha256']==dino.EXPECTED_DINO_SHA256
    return dino.load_dinov2_model(Path(c['model_checkpoint']),device='cuda')

def normalized(v):
    n=np.linalg.norm(v);return v/max(n,1e-12)

def pool_tokens(tokens,pixels,width,height):
    # Match frontend depth->RGB proportional coordinates, then resized token grid.
    px=np.clip(pixels[:,0]*dino.TOKEN_WIDTH/width,0,dino.TOKEN_WIDTH-1).astype(int)
    py=np.clip(pixels[:,1]*dino.TOKEN_HEIGHT/height,0,dino.TOKEN_HEIGHT-1).astype(int)
    bins,count=np.unique(py*dino.TOKEN_WIDTH+px,return_counts=True)
    return normalized(np.average(tokens.reshape(-1,384)[bins],axis=0,weights=count)).astype(np.float32),len(bins)

def depth_path_evidence(depth,pixa,pixb):
    """Actual sensor depth along projected inter-region paths, not 3D delta-z."""
    if not len(pixa) or not len(pixb):return np.zeros(8)
    # Deterministic sparse paths between nearest projected region samples.
    a=pixa[np.linspace(0,len(pixa)-1,min(128,len(pixa)),dtype=int)]
    b=pixb[np.linspace(0,len(pixb)-1,min(128,len(pixb)),dtype=int)]
    da,ia=cKDTree(a).query(b);db,ib=cKDTree(b).query(a)
    edges=np.r_[np.column_stack((a[ia],b)),np.column_stack((a,b[ib]))]
    edges=np.unique(edges,axis=0);distance=np.linalg.norm(edges[:,:2]-edges[:,2:],axis=1)
    order=np.argsort(distance,kind='stable')[:8]
    values=[];missing=[];length=[]
    for j in order:
        line=np.rint(np.linspace(edges[j,:2],edges[j,2:],32)).astype(int)
        z=depth[line[:,1],line[:,0]];ok=np.isfinite(z)&(z>0)
        missing.append(float((~ok).mean()));length.append(float(distance[j]))
        valid_edges=ok[1:]&ok[:-1];jump=np.abs(np.diff(z))[valid_edges]
        values.append([np.median(jump) if len(jump) else 0,np.quantile(jump,.9) if len(jump) else 0,np.max(jump,initial=0)])
    v=np.asarray(values)
    return np.r_[v.mean(0),np.mean(missing),np.max(missing),np.median(length),np.max(length),len(order)/8]

def region_depth_evidence(depth,pixels,camera_depth,indices):
    h,w=depth.shape;pp=pixels[indices];z=camera_depth[indices]
    ok=(z>0)&(pp[:,0]>=0)&(pp[:,0]<w)&(pp[:,1]>=0)&(pp[:,1]<h)
    pp=pp[ok];z=z[ok]
    if not len(pp):return np.zeros(8)
    sensor=depth[pp[:,1],pp[:,0]];known=sensor>0;res=z-sensor
    # Sensor shallower than candidate = foreground occlusion; missing kept explicit.
    mask=np.zeros(depth.shape,bool);mask[pp[:,1],pp[:,0]]=True
    rim=binary_dilation(mask,iterations=1)&~mask
    horizontal=np.abs(depth[:,1:]-depth[:,:-1]);hx=(rim[:,1:]|rim[:,:-1])&(depth[:,1:]>0)&(depth[:,:-1]>0)
    vertical=np.abs(depth[1:]-depth[:-1]);hy=(rim[1:]|rim[:-1])&(depth[1:]>0)&(depth[:-1]>0)
    jumps=np.r_[horizontal[hx],vertical[hy]]
    return np.asarray([float((~known).mean()),float((known&(res>.05)).mean()),float((known&(res<-.05)).mean()),float(np.median(np.abs(res[known]))) if known.any() else 0,float(np.quantile(jumps,.5)) if len(jumps) else 0,float(np.quantile(jumps,.9)) if len(jumps) else 0,float((depth[rim]<=0).mean()) if rim.any() else 0,float(known.mean())])

def chosen_views(aa,bb,frames):
    ranked=[]
    for i,f in enumerate(frames):
        a=f['visible'][aa].sum();b=f['visible'][bb].sum()
        if min(a,b)<3 or a/len(aa)<.1 or b/len(bb)<.1:continue
        ranked.append((-min(a/len(aa),b/len(bb)),-min(a,b),f['frame'],i))
    selected=[];groups=set()
    for _,_,_,i in sorted(ranked):
        g=frames[i]['group']
        if g in groups:continue
        groups.add(g);selected.append(i)
        if len(selected)==3:break
    return selected

def visual_scene(c,parent,s,raw,pts,rows,model):
    import torch
    keys=list(pts);frames,_,_=prior.load_frames(parent,s,raw,[pts[k] for k in keys])
    lookup={int(f['frame_id']):f for f in prior.read(parent['jobs'][s]['paths']['manifest'])['frames']}
    d=Path(c['version']) # output location resolved by caller module to avoid model semantics dependency.
    out=prior.ROOT/'.local'/c['version']/'scenes'/s;out.mkdir(parents=True,exist_ok=True)
    # Dense forward passes batched only for selected fixed30 frames.
    tensors=[]
    for f in frames:
        im=np.asarray(Image.open(lookup[f['frame']]['color']['path']).convert('RGB'));_,t=dino.preprocess_full_rgb(im);tensors.append(t)
        f['sensor_depth']=np.asarray(Image.open(lookup[f['frame']]['depth']['path']),dtype=np.float32)/1000
    maps=[]
    with torch.inference_mode():
        for start in range(0,len(tensors),4):
            t=torch.stack(tensors[start:start+4]).cuda();v=model.get_intermediate_layers(t,n=1,reshape=True)[0]
            assert tuple(v.shape[1:])==(384,24,32)
            v=v.permute(0,2,3,1).float().cpu().numpy();maps.extend(v)
    result=[];receipts=[];emb=np.zeros((len(rows),3,2,384),np.float16)
    region_cache={};pool_cache={}
    for k,row in enumerate(rows):
        pa,pb=pts[row['left']],pts[row['right']];common=np.intersect1d(pa,pb,assume_unique=True);aa=np.setdiff1d(pa,common,assume_unique=True);bb=np.setdiff1d(pb,common,assume_unique=True);chosen=chosen_views(aa,bb,frames);views=[];same=[];depth_features=[];tokens=[]
        for vi,fi in enumerate(chosen):
            f=frames[fi];a=aa[f['visible'][aa]];b=bb[f['visible'][bb]];h,w=f['sensor_depth'].shape
            # Cache exact exclusive regions, preserving all evidence identities.
            cache_a=(row['left'],common.tobytes(),fi);cache_b=(row['right'],common.tobytes(),fi)
            if cache_a not in pool_cache:pool_cache[cache_a]=pool_tokens(maps[fi],f['pixels'][a],w,h)
            if cache_b not in pool_cache:pool_cache[cache_b]=pool_tokens(maps[fi],f['pixels'][b],w,h)
            va,na=pool_cache[cache_a];vb,nb=pool_cache[cache_b];emb[k,vi,0]=va;emb[k,vi,1]=vb;tokens.append((va,vb));same.append(float(va@vb))
            if cache_a not in region_cache:region_cache[cache_a]=region_depth_evidence(f['sensor_depth'],f['pixels'],f['depth'],aa)
            if cache_b not in region_cache:region_cache[cache_b]=region_depth_evidence(f['sensor_depth'],f['pixels'],f['depth'],bb)
            path=depth_path_evidence(f['sensor_depth'],f['pixels'][a],f['pixels'][b]);ra=region_cache[cache_a];rb=region_cache[cache_b];df=np.r_[path,(ra+rb)/2,np.abs(ra-rb)];depth_features.append(df)
            views.append(dict(frame=f['frame'],group=f['group'],visible_left=len(a),visible_right=len(b),left_token_cells=na,right_token_cells=nb,dino_cosine=float(va@vb),sensor_depth= df.tolist(),rgb_path=lookup[f['frame']]['color']['path'],depth_path=lookup[f['frame']]['depth']['path']))
        def stats(values):
            return [np.mean(values),np.min(values),np.max(values),np.std(values)] if len(values) else [0]*4
        cross=[float(a@d) for i,(a,b) in enumerate(tokens) for j,(e,d) in enumerate(tokens) if i!=j]
        consistency_a=[float(tokens[i][0]@tokens[j][0]) for i in range(len(tokens)) for j in range(i+1,len(tokens))]
        consistency_b=[float(tokens[i][1]@tokens[j][1]) for i in range(len(tokens)) for j in range(i+1,len(tokens))]
        # Symmetric endpoints. Missing evidence never becomes a negative identity.
        consistency=[(a+b)/2 for a,b in zip(consistency_a,consistency_b)]
        df=np.asarray(depth_features);means=df.mean(0) if len(df) else np.zeros(24);stds=df.std(0) if len(df) else np.zeros(24)
        result.append(np.r_[len(chosen)/3,stats(same),stats(cross),stats(consistency),means,stds]);receipts.append(views)
    np.savez_compressed(out/'region_embeddings.npz',embeddings=emb)
    del model # caller still holds frozen model, no training or deletion.
    return result,receipts
