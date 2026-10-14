#!/usr/bin/env python3
"""Real point-mask learning on original physical development200 folds."""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from io import BytesIO
import argparse,hashlib,json,sys,shutil,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
from tools import diagnose_fi1_nt_geometry_gaps as source
from tools import audit_fi1_gd10_candidate_observations as audit
from tools import fi1_local_repair_capacity as capacity
from tools import fi1_candidate_observation_selection as views
from tools import fi1_candidate_local_segmenter as core
from tools import ncs5_proxyclip_core as dino
from tools.fi1_complementary_fragment_evidence import project_points
from tools.fi1_r0_r1_protocol import original_aliases
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha,digest

OUT=ROOT/'.local/fi1_candidate_local_repair_learning_development200_20261012_v1'
SEED=audit.prior.old.SEED
read=source.read


def stage(name,**kwargs):
    write_json(OUT/'execution.json',dict(stage=name,updated=time.time(),**kwargs));print(name,kwargs,flush=True)


class Guard:
    def __init__(self,c,mode,fold=None):self.c,self.mode,self.fold=c,mode,fold
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(Path(args[0]).resolve())
        if Path(p).suffix in ('.py','.pyc'):return
        if any(q in p for q in ('/val312/','/reserve100/','fi1_sl_single200_dev55')):raise PermissionError('outside original development200')
        if any(q.startswith('ground_truth') for q in Path(p).parts):
            allowed=self.mode=='labels' and (OUT/'features_frozen.json').exists() or self.mode=='evaluate' and (OUT/'predictions_frozen.json').exists()
            if not allowed or p not in self.c['GT_inputs']:raise PermissionError('GT outside supervised-label or post-prediction evaluation stage')
        if str(OUT/'labels')+'/' in p:
            if self.mode not in ('train','evaluate','labels'):raise PermissionError('prediction/extraction cannot read training labels')
            if self.mode=='train':
                scene=next((s for s in self.c['scenes'] if '/'+s+'/' in p),None)
                if scene is None or self.c['folds'][scene]==self.fold:raise PermissionError('heldout supervision forbidden in training')
        if self.mode in ('plan','features','predict') and (Path(p).name in ('summary.json','geometry_quality.json','target_census.json','overlap.npz') or '/diagnosis/' in p):
            raise PermissionError('GT diagnosis forbidden in inference features and actions')


def prepare():
    assert not OUT.exists();assert shutil.disk_usage(ROOT).free>=50*1024**3
    original=read(source.OUT/'contract.json');fixed=read(audit.OUT/'contract.json')
    physical=defaultdict(set)
    for s,f in original['folds'].items():physical[s[:9]].add(f)
    assert len(original['scenes'])==200 and all(len(v)==1 for v in physical.values())
    jobs={s:dict(binding=j['binding'],paths={k:j['paths'][k] for k in ('processed','manifest','observations')},views=j['views'],GT=j['GT']) for s,j in original['jobs'].items()}
    inputs={}
    for s,j in jobs.items():
        for path in [j['binding'][k] for k in ('geometry','points')]+list(j['paths'].values())+[j['views']]:inputs[path]=original['inputs'][path]
    files=[Path(__file__).resolve(),ROOT/'tools/fi1_candidate_local_segmenter.py',
           ROOT/'tools/verify_fi1_candidate_local_repair.py',ROOT/'tools/evaluate_fi1_candidate_local_repair.py',ROOT/'tools/run_fi1_candidate_local_repair.sh',
           ROOT/'tests/test_fi1_candidate_local_segmenter.py',ROOT/'docs/FI1_CANDIDATE_LOCAL_REPAIR_LEARNING_20261012.md',
           ROOT/'tools/fi1_candidate_observation_selection.py',ROOT/'tools/fi1_local_repair_capacity.py',
           ROOT/'tools/ncs5_proxyclip_core.py',ROOT/'tools/fi1_complementary_fragment_evidence.py',
           ROOT/'tools/audit_fi1_gd10_candidate_observations.py',ROOT/'tools/run_fi1_anchor_local_decontamination.py']
    c=dict(version=OUT.name,scenes=original['scenes'],folds=original['folds'],groups=original['groups'],
           class_ids=original['class_ids'],jobs=jobs,GT_inputs=original['GT_inputs'],inputs=inputs,
           frozen_SL_models=fixed['frozen_SL_models'],code={str(p):sha(p) for p in files},
           source_contract=str(source.OUT/'contract.json'),source_contract_sha256=sha(source.OUT/'contract.json'),
           fixed10=fixed['scenes'],fixed10_seal=str(audit.OUT/'observations_frozen.json'),fixed10_seal_sha256=sha(audit.OUT/'observations_frozen.json'),
           dino_checkpoint=str(dino.DINO_CHECKPOINT),dino_sha256=sha(dino.DINO_CHECKPOINT),parameters=core.PARAMETERS,
           arms=list(core.ARMS),selection_seed=SEED,view_budget=6,radius=.15,
           sampling='Same GT-free hash rule:4 native and4 track unique pointsets per scene,100..100000 parent points. Reuse all80 frozen parents/domains/views in fixed10. No GT quality sampling.',
           representation='64 point features:16 geometry/membership/NT disagreement +32 frozen orthogonal sketch of per-point projected DINOv2 +8 visual std +3 image RGB +5 visibility/mask evidence. 8-neighbor raw point graph; no atomic voxel decisions.',
           training='One shared64-width two-message point graph per physical fold, trained on both observation arms from four folds,12epochs AdamW. Unknown=third class, supervised target=best valid parent GT IoU with>=20 intersections. Good IoU>.95 gets .25 keep consistency. No heldout labels, no validation checkpoint choice, no threshold sweep.',
           action='Target/other probability>=.8 and>=2 measured-depth-visible selected physical groups permits additions/deletions. Unknown/occluded/unseen keep parent state. Predicted mask<100 abstains. No positive-mask veto/core/budget; old masks are continuous references only.',
           gate=dict(min_value_targets=3,min_value_scenes=2,min_actions=5,min_action_scenes=3,
                     min_add_target_precision=.9,min_delete_pollution_precision=.9,max_target_loss=.01,
                     max_good_degradation_rate=.02,max_drift=0,max_unknown_action_fraction=.1),
           scope='Real OOF geometry only, original NT retained. All predictions sealed before GT evaluation. Exposed development cohort, upstream/B26 saw these scenes; not whole-system blind generalization.',
           new_training=True,new_segmentation_model=True,new_SAM_inference=False,SL_modified=False,production_modified=False)
    assert c['dino_sha256']==dino.EXPECTED_DINO_SHA256
    audit.prior.old.verify_files(c['frozen_SL_models']);audit.prior.old.verify_files(c['code'])
    OUT.mkdir();freeze_json(OUT/'contract.json',c)
    rng=np.random.default_rng(core.PARAMETERS['seed']);q=np.linalg.qr(rng.normal(size=(384,32)))[0].astype(np.float32)
    np.save(OUT/'visual_projection.npy',q)
    for p in files:
        target=OUT/'code_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    stage('prepared',scenes=200,folds={f:sum(v==f for v in c['folds'].values()) for f in range(5)},max_parents=1600)


def bank_points(bank,point_count):
    receipt=read(bank/'receipt.json');rows=read(bank/'observations.json');assert len(rows)==receipt['kept_observation_count']
    assert sha(bank/'observations.json')==receipt['files']['observations.json']
    cache=np.load(bank/'visibility.npy');assert cache.shape==(point_count,) and cache.dtype==bool
    assert sha(bank/'visibility.npy')==receipt['files']['visibility.npy']
    points=[]
    for o in rows:
        path=Path(o['point_indices_path']);assert path.parent==bank
        data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==receipt['files'][path.name]
        with np.load(BytesIO(data)) as z:p=z['point_indices'].copy()
        assert np.array_equal(p,np.unique(p)) and len(p)>=20 and cache[p].all()
        assert p[0]>=0 and p[-1]<point_count and o['predicted_iou']>.88 and o['stability_score']>=.95
        counts=o['mask_rle']['counts'];h,w=o['mask_rle']['size']
        assert sum(counts)==h*w and sum(counts[1::2])==o['area']
        points.append(p)
    return rows,points,receipt


def plan_scene(s):
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'plan').hook);j=c['jobs'][s];dest=OUT/'plan'/s
    if (dest/'receipt.json').exists():r=read(dest/'receipt.json');audit.prior.old.verify_files(r['outputs']);return r
    raw=np.load(j['paths']['processed'],mmap_mode='r');xyz=np.asarray(raw[:,:3]);geometry,points=audit.prior.old.load_nt(j)
    used=set();chosen=[]
    for name in ('native','track'):
        eligible=[g for g in geometry if 100<=g['point_count']<=100000 and g['geometry_key'] not in used and any(a['source']==name for a in original_aliases(g))]
        for g in sorted(eligible,key=lambda g:(digest([SEED,s,name,g['geometry_key']]),g['geometry_key']))[:4]:used.add(g['geometry_key']);chosen.append((name,g))
    arrays={};candidate_rows=[]
    for i,(name,g) in enumerate(chosen):
        cid=f'c{i:03d}';parent=points[g['geometry_key']];hh=capacity.halo(parent,xyz)
        arrays[cid+'_parent']=parent;arrays[cid+'_halo']=hh
        boundary=parent[audit.prior.old.core.local_domain(parent,xyz)]
        other='track' if name=='native' else 'native';pairs=[]
        for otherg in geometry:
            key=otherg['geometry_key']
            if key==g['geometry_key'] or not any(a['source']==other for a in original_aliases(otherg)):continue
            hit=len(np.intersect1d(parent,points[key],assume_unique=True));iou=hit/(len(parent)+len(points[key])-hit)
            if iou>.05:pairs.append((-iou,key))
        pairs=sorted(pairs)[:4]
        disagreement=np.unique(np.concatenate([np.setxor1d(parent,points[key],assume_unique=True) for _,key in pairs])) if pairs else np.empty(0,int)
        roles,weight=views.region_weights(parent,hh,boundary,disagreement)
        arrays[cid+'_roles']=np.array(roles);arrays[cid+'_weights']=weight
        candidate_rows.append(dict(candidate_id=cid,source=name,parent=g['geometry_key'],parent_points=len(parent),halo_points=len(hh),views={}))
    roi=np.unique(np.concatenate([arrays[r['candidate_id']+'_halo'] for r in candidate_rows]))
    m=read(j['paths']['manifest']);fm={int(r['frame_id']):r for r in m['frames']};obs=[json.loads(l) for l in Path(j['paths']['observations']).read_text().splitlines()]
    bankroot=Path(obs[0]['point_indices_path']).parent.parent
    cached=set(read(j['views'])['raw_frames']);source_hashes={}
    if s in c['fixed10']:
        old=read(audit.OUT/'plan'/s/'selected.json');original=read(audit.prior.OUT/'plan'/s/'candidates.json')
        assert [(r['source'],r['parent']) for r in candidate_rows]==[(r['source'],r['parent']) for r in original]
        with np.load(audit.OUT/'plan'/s/'evidence.npz') as z:
            assert np.array_equal(roi,z['roi']);frame_ids=z['frames'];group_ids=z['groups'];visible=audit.unpack(z,'reliable');covered=audit.unpack(z,'covered');has_masks=z['has_masks']
        for r,o in zip(candidate_rows,old):r['views']=o['views']
        # Exact domain/selection replay proves new code has not retuned fixed10.
        for q in (audit.OUT/'plan'/s).iterdir():
            if q.is_file():source_hashes[str(q)]=sha(q)
        poses={r['frame']:np.array(r['pose']) for r in read(audit.prior.OUT/'plan'/s/'frames.json')['records'] if r['frame']%10==0}
    else:
        base=Path(j['paths']['processed']).parent
        complete=set.intersection(*[{int(p.stem) for p in (base/k).iterdir() if p.stem.isdigit()} for k in ('color','depth','poses')])
        poses={}
        for f in sorted(complete):
            p=np.fromstring((base/'poses'/f'{f}.txt').read_text(),sep=' ')
            if len(p)!=16:continue
            p=p.reshape(4,4)
            if np.isfinite(p).all() and abs(np.linalg.det(p))>=1e-12:poses[f]=p
        groups=capacity.physical_groups(poses);frame_ids=np.array(sorted(set(m['arms']['D'])&set(poses)),np.int64)
        assert set(frame_ids)=={f for f in poses if f%10==0}
        group_ids=np.array([groups[f] for f in frame_ids],np.int32);has_masks=[]
        visible=np.zeros((len(frame_ids),len(roi)),bool);covered=visible.copy();intrinsic=np.loadtxt(m['files']['intrinsics']['path'])
        for i,f in enumerate(frame_ids):
            bank=bankroot/f'frame{f:06d}';rows,pp,receipt=bank_points(bank,len(raw))
            source_hashes[str(bank/'receipt.json')]=sha(bank/'receipt.json')
            entry=fm[int(f)];data=Path(entry['depth']['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==entry['depth']['sha256']
            sensor=np.asarray(Image.open(BytesIO(data)),float)/1000.
            with Image.open(entry['color']['path']) as image:w,h=image.size
            k=capacity.adapt_intrinsic(intrinsic,(h,w),sensor.shape)
            visible[i]=capacity.reliable_visibility(xyz[roi],poses[int(f)],k,sensor)
            if pp:covered[i]=visible[i]&np.isin(roi,np.unique(np.concatenate(pp)))
            has_masks.append(bool(rows))
        has_masks=np.array(has_masks,bool)
    old_indices=np.flatnonzero(np.isin(frame_ids,list(cached)));assert len(old_indices)==len(cached)
    for r in candidate_rows:
        cid=r['candidate_id'];ix=np.searchsorted(roi,arrays[cid+'_halo'])
        for arm,available in ((core.ARMS[0],old_indices),(core.ARMS[1],np.arange(len(frame_ids)))):
            selected,trace=views.choose_views(frame_ids[available],group_ids[available],visible[available][:,ix],covered[available][:,ix],has_masks[available],arrays[cid+'_roles'],arrays[cid+'_weights'])
            record=dict(frames=frame_ids[available[selected]].tolist(),trace=trace)
            if arm in r['views']:assert record==r['views'][arm],(s,cid,arm,'fixed10 replay mismatch')
            r['views'][arm]=record
    dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'domains.npz',**arrays)
    np.savez_compressed(dest/'evidence.npz',roi=roi,frames=frame_ids,groups=group_ids,has_masks=has_masks,
                        reliable=np.packbits(visible,axis=1,bitorder='little'),covered=np.packbits(covered,axis=1,bitorder='little'),
                        poses=np.stack([poses[int(f)] for f in frame_ids]))
    freeze_json(dest/'candidates.json',candidate_rows)
    receipt=dict(complete=True,GT_read=False,scene=s,parents=len(candidate_rows),GD10_frames=len(frame_ids),source_banks=source_hashes,
                 outputs={str(q):sha(q) for q in dest.iterdir() if q.is_file()})
    freeze_json(dest/'receipt.json',receipt);print('planned',s,len(candidate_rows),len(frame_ids),flush=True);return receipt


def plan():
    c=read(OUT/'contract.json');audit.prior.old.verify_files(c['code']);audit.prior.old.verify_files(c['inputs'])
    receipts=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(plan_scene,s) for s in c['scenes']]):
            receipts.append(f.result());stage('GT_free_planning',scenes=len(receipts),total=200)
    outputs={};banks={}
    for r in receipts:outputs.update(r['outputs']);banks.update(r['source_banks'])
    freeze_json(OUT/'plan_frozen.json',dict(complete=True,GT_read=False,parents=sum(r['parents'] for r in receipts),outputs=outputs,source_banks=banks,visual_projection_sha256=sha(OUT/'visual_projection.npy')))


def static_features(raw,parent,halo,roles):
    xyz=raw[halo,:3];own=raw[parent,:3];center=own.mean(0);scale=max(.05,float(np.linalg.norm(own.max(0)-own.min(0))))
    rel=xyz-center;normal=raw[halo,6:9];normal=normal/np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    near,_=cKDTree(own).query(xyz)
    x=np.c_[np.clip(rel,-2,2),np.clip(raw[halo,3:6]/255.,0,1),normal,np.isin(halo,parent),
            np.minimum(near/.15,1),roles[3],np.minimum(np.linalg.norm(rel,axis=1)/scale,2),np.clip(rel/scale,-2,2)]
    assert x.shape==(len(halo),16) and np.isfinite(x).all();return x.astype(np.float32)


def extract_features():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'features').hook)
    assert torch.cuda.is_available();torch.set_num_threads(4)
    audit.prior.old.verify_files(c['code']);audit.prior.old.verify_files(read(OUT/'plan_frozen.json')['outputs'])
    model=dino.load_dinov2_model(Path(c['dino_checkpoint']),device='cuda');projection=np.load(OUT/'visual_projection.npy')
    all_outputs={};frame_count=0
    for scene_index,s in enumerate(c['scenes']):
        dest=OUT/'features'/s
        if (dest/'receipt.json').exists():
            receipt=read(dest/'receipt.json');audit.prior.old.verify_files(receipt['outputs']);all_outputs.update(receipt['outputs']);frame_count+=receipt['image_frames'];continue
        stage('frozen_DINO_point_features',scene=s,completed=scene_index,total=200)
        j=c['jobs'][s];raw=np.asarray(np.load(j['paths']['processed'],mmap_mode='r')[:,:9]);m=read(j['paths']['manifest']);fm={int(r['frame_id']):r for r in m['frames']}
        plans=read(OUT/'plan'/s/'candidates.json');needed=sorted({f for r in plans for a in core.ARMS for f in r['views'][a]['frames']})
        obs=[json.loads(l) for l in Path(j['paths']['observations']).read_text().splitlines()];bankroot=Path(obs[0]['point_indices_path']).parent.parent
        with np.load(OUT/'plan'/s/'evidence.npz') as evidence,np.load(OUT/'plan'/s/'domains.npz') as domains:
            roi=evidence['roi'];vis=audit.unpack(evidence,'reliable');frame_index={int(f):i for i,f in enumerate(evidence['frames'])}
            cloud={};inputs={}
            for start in range(0,len(needed),4):
                ids=needed[start:start+4];images=[];tensors=[]
                for f in ids:
                    path=fm[f]['color']['path'];assert sha(path)==fm[f]['color']['sha256'];inputs[path]=fm[f]['color']['sha256']
                    rgb=np.asarray(Image.open(path).convert('RGB'));images.append(rgb);tensors.append(dino.preprocess_full_rgb(rgb)[1])
                with torch.inference_mode():
                    tokens=model.get_intermediate_layers(torch.stack(tensors).cuda(),n=1,reshape=True)[0].permute(0,2,3,1).float()
                    tokens=torch.nn.functional.normalize(tokens,dim=-1)
                    sketch=(tokens@torch.tensor(projection,device='cuda'))*np.sqrt(384/32)
                    sketch=sketch.cpu().numpy()
                for f,image,tokens in zip(ids,images,sketch):
                    i=frame_index[f];sensor=np.asarray(Image.open(fm[f]['depth']['path']),float)/1000.
                    k=capacity.adapt_intrinsic(np.loadtxt(m['files']['intrinsics']['path']),image.shape[:2],sensor.shape)
                    pixel,depth,finite=project_points(raw[roi,:3],evidence['poses'][i],k)
                    dh,dw=sensor.shape;h,w=image.shape[:2]
                    rgbpix=np.rint(pixel*np.array([w/dw,h/dh])).astype(int);rgbpix[:,0]=np.clip(rgbpix[:,0],0,w-1);rgbpix[:,1]=np.clip(rgbpix[:,1],0,h-1)
                    tx=np.clip((rgbpix[:,0]*32/w).astype(int),0,31);ty=np.clip((rgbpix[:,1]*24/h).astype(int),0,23)
                    rows,pp,receipt=bank_points(bankroot/f'frame{f:06d}',len(raw))
                    cloud[f]=dict(embedding=tokens[ty,tx],rgb=image[rgbpix[:,1],rgbpix[:,0]]/255.,masks=pp,quality=[o['predicted_iou'] for o in rows],visible=vis[i])
            outputs={};arrays={}
            for r in plans:
                cid=r['candidate_id'];parent=domains[cid+'_parent'];halo=domains[cid+'_halo'];ix=np.searchsorted(roi,halo)
                base=static_features(raw,parent,halo,domains[cid+'_roles']);nn,weights=core.graph(raw[halo,:3])
                arrays[cid+'_halo']=halo;arrays[cid+'_parent']=parent;arrays[cid+'_neighbors']=nn;arrays[cid+'_neighbor_weights']=weights
                for arm in core.ARMS:
                    selected=r['views'][arm]['frames'];embedding=np.zeros((len(halo),32));square=np.zeros((len(halo),8));rgb=np.zeros((len(halo),3));count=np.zeros(len(halo));anymask=count.copy();bestmask=count.copy();quality=count.copy()
                    for f in selected:
                        frame=cloud[f];seen=frame['visible'][ix];count+=seen
                        embedding[seen]+=frame['embedding'][ix][seen];square[seen]+=frame['embedding'][ix][seen,:8]**2;rgb[seen]+=frame['rgb'][ix][seen]
                        all_union=np.unique(np.concatenate(frame['masks'])) if frame['masks'] else np.empty(0,int)
                        anymask+=seen&np.isin(halo,all_union)
                        ranking=[]
                        for k,mask in enumerate(frame['masks']):
                            hit=len(np.intersect1d(parent,mask,assume_unique=True));iou=hit/(len(parent)+len(mask)-hit)
                            ranking.append((-iou,k))
                        if ranking:
                            _,best=min(ranking);supported=seen&np.isin(halo,frame['masks'][best]);bestmask+=supported;quality+=supported*frame['quality'][best]
                    norm=np.maximum(count[:,None],1);mean=embedding/norm;std=np.sqrt(np.maximum(0,square/norm-mean[:,:8]**2))
                    # Selected frames have distinct physical groups by contract.
                    visual=np.c_[mean,std,rgb/norm,count/6,anymask/np.maximum(count,1),bestmask/np.maximum(count,1),quality/np.maximum(count,1),count/6]
                    x=np.c_[base,visual].astype(np.float16);assert x.shape==(len(halo),64) and np.isfinite(x).all()
                    arrays[cid+'_'+arm+'_x']=x;arrays[cid+'_'+arm+'_groups']=count.astype(np.uint8)
            dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'data.npz',**arrays)
            receipt=dict(complete=True,GT_read=False,image_frames=len(needed),parents=len(plans),image_inputs=inputs,
                         outputs={str(dest/'data.npz'):sha(dest/'data.npz')})
            freeze_json(dest/'receipt.json',receipt);all_outputs.update(receipt['outputs']);frame_count+=len(needed)
        assert shutil.disk_usage(ROOT).free>=50*1024**3
    audit.prior.old.verify_files(c['frozen_SL_models']);freeze_json(OUT/'features_frozen.json',dict(complete=True,GT_read=False,outputs=all_outputs,image_frames=frame_count,DINO_frozen=True,feature_dimension=64))
    stage('features_sealed',scenes=200,image_frames=frame_count)


def labels():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'labels').hook);outputs={};counts=Counter()
    assert read(OUT/'features_frozen.json')['complete']
    for s in c['scenes']:
        j=c['jobs'][s];assert sha(j['GT'])==c['GT_inputs'][j['GT']];gt=np.loadtxt(j['GT'],dtype=np.int64)
        ids,sizes=audit.prior.valid_targets(gt,c['class_ids']);arrays={};rows=[]
        with np.load(OUT/'features'/s/'data.npz') as data:
            for r in read(OUT/'plan'/s/'candidates.json'):
                cid=r['candidate_id'];parent=data[cid+'_parent'];halo=data[cid+'_halo'];scores=audit.prior.ious(parent,gt,ids,sizes)
                index=int(np.argmax(scores)) if len(ids) and scores.max()>0 else None
                gid=None if index is None else int(ids[index]);hit=0 if gid is None else int(np.sum(gt[parent]==gid))
                reliable=gid is not None and hit>=20;target=gid if reliable else None
                y=np.full(len(halo),2,np.uint8)
                if reliable:y[gt[halo]>=1000]=1;y[gt[halo]==gid]=0
                good=bool(reliable and scores[index]>.95);arrays[cid+'_y']=y
                rows.append(dict(candidate_id=cid,target=target,good=good,parent_iou=float(scores.max(initial=0)),parent_target_points=hit))
                counts['parents']+=1;counts['known_identity']+=reliable;counts['unknown_identity']+=not reliable;counts['good_keep_samples']+=good
        d=OUT/'labels'/s;d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'labels.npz',**arrays);freeze_json(d/'meta.json',rows)
        outputs.update({str(q):sha(q) for q in d.iterdir()})
    freeze_json(OUT/'labels_frozen.json',dict(complete=True,outputs=outputs,counts=dict(counts),used_only_in_training_folds=True));stage('supervision_ready',**dict(counts))


def load_example(s,cid,arm,with_labels=False):
    with np.load(OUT/'features'/s/'data.npz') as z:
        result=dict(x=z[cid+'_'+arm+'_x'].astype(np.float32),neighbors=z[cid+'_neighbors'].astype(np.int64),weights=z[cid+'_neighbor_weights'].astype(np.float32))
    if with_labels:
        with np.load(OUT/'labels'/s/'labels.npz') as z:result['y']=z[cid+'_y'].astype(np.int64)
        result['good']=next(r['good'] for r in read(OUT/'labels'/s/'meta.json') if r['candidate_id']==cid)
    return result


def collate(examples,device='cuda'):
    parts=[];offset=0;nn=[]
    for e in examples:
        parts.append((offset,offset+len(e['x'])));nn.append(e['neighbors']+offset);offset+=len(e['x'])
    x=torch.tensor(np.concatenate([e['x'] for e in examples]),device=device)
    neighbors=torch.tensor(np.concatenate(nn),device=device,dtype=torch.long);weights=torch.tensor(np.concatenate([e['weights'] for e in examples]),device=device)
    return x,neighbors,weights,parts


def train():
    c=read(OUT/'contract.json');guard=Guard(c,'train',fold=0);sys.addaudithook(guard.hook)
    assert torch.cuda.is_available();torch.set_num_threads(4);audit.prior.old.verify_files(c['code'])
    frozen={}
    for fold in range(5):
        guard.fold=fold;d=OUT/'models'/f'fold{fold}';d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():r=read(d/'receipt.json');audit.prior.old.verify_files(r['outputs']);frozen.update(r['outputs']);continue
        torch.manual_seed(core.PARAMETERS['seed']+fold);rng=np.random.default_rng(core.PARAMETERS['seed']+fold)
        model=core.CandidateLocalSegmenter().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        train_scenes=[s for s in c['scenes'] if c['folds'][s]!=fold];test_scenes=[s for s in c['scenes'] if c['folds'][s]==fold]
        items=[(s,r['candidate_id'],a) for s in train_scenes for r in read(OUT/'plan'/s/'candidates.json') for a in core.ARMS]
        examples={};scene_counts=Counter(s for s,_,_ in items)
        # Features/labels are cached for this fold only; no heldout supervision.
        for s in train_scenes:
            with np.load(OUT/'features'/s/'data.npz') as z,np.load(OUT/'labels'/s/'labels.npz') as y:
                meta={r['candidate_id']:r for r in read(OUT/'labels'/s/'meta.json')}
                for r in read(OUT/'plan'/s/'candidates.json'):
                    cid=r['candidate_id']
                    for arm in core.ARMS:
                        examples[(s,cid,arm)]=dict(x=z[cid+'_'+arm+'_x'].astype(np.float32),neighbors=z[cid+'_neighbors'].astype(np.int64),weights=z[cid+'_neighbor_weights'].astype(np.float32),y=y[cid+'_y'].astype(np.int64),good=meta[cid]['good'])
        epoch_start=0;history=[]
        if (d/'checkpoint.pt').exists():
            state=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);epoch_start=state['epoch'];history=state['history'];rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        for epoch in range(epoch_start,12):
            model.train();order=rng.permutation(len(items));losses=[];start_time=time.time()
            for start in range(0,len(order),4):
                batch_keys=[items[i] for i in order[start:start+4]];batch=[examples[key] for key in batch_keys]
                # Split large graphs into single-candidate batches to bound VRAM.
                batches=[list(zip(batch_keys,batch))] if sum(len(e['x']) for e in batch)<=60000 else [[(k,e)] for k,e in zip(batch_keys,batch)]
                for pairs in batches:
                    keys,ee=zip(*pairs);x,nn,w,parts=collate(ee);optimizer.zero_grad(set_to_none=True)
                    logits=model(x,nn,w,parts);loss=0
                    for key,e,(lo,hi) in zip(keys,ee,parts):
                        y=torch.tensor(e['y'],device='cuda');count=min(4096,hi-lo);indices=torch.tensor(rng.choice(hi-lo,count,replace=False),device='cuda')
                        value=core.segmentation_loss(logits[lo:hi],y,x[lo:hi,9]>.5,e['good'],indices)
                        loss=loss+value*(len(items)/len(train_scenes)/scene_counts[key[0]])/len(ee)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            entry=dict(epoch=epoch+1,mean_training_loss=float(np.mean(losses)),seconds=time.time()-start_time,steps=len(losses));history.append(entry)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),d/'checkpoint.pt')
            stage('training',fold=fold,**entry)
        torch.save(model.cpu().state_dict(),d/'model.pt')
        freeze_json(d/'history.json',dict(train_scenes=train_scenes,test_scenes=test_scenes,epochs=history,model_parameters=sum(p.numel() for p in model.parameters()),shared_between_arms=True))
        outputs={str(d/n):sha(d/n) for n in ('model.pt','history.json')};freeze_json(d/'receipt.json',dict(complete=True,outputs=outputs,fold=fold,heldout_labels_read=False));frozen.update(outputs)
        del examples,model,optimizer;torch.cuda.empty_cache()
    freeze_json(OUT/'models_frozen.json',dict(complete=True,outputs=frozen,models=5,shared_between_arms=True,checkpoint_selected_by_fixed_epoch=True))
    stage('fivefold_models_sealed')


def predict():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'predict').hook)
    assert torch.cuda.is_available();torch.set_num_threads(4);audit.prior.old.verify_files(read(OUT/'models_frozen.json')['outputs'])
    outputs={};actions=Counter()
    for fold in range(5):
        model=core.CandidateLocalSegmenter().cuda();model.load_state_dict(torch.load(OUT/'models'/f'fold{fold}'/'model.pt',map_location='cuda',weights_only=True));model.eval()
        for s in c['scenes']:
            if c['folds'][s]!=fold:continue
            dest=OUT/'predictions'/s
            if (dest/'receipt.json').exists():r=read(dest/'receipt.json');audit.prior.old.verify_files(r['outputs']);outputs.update(r['outputs']);actions.update(r['actions']);continue
            stage('GT_free_heldout_prediction',fold=fold,scene=s);arrays={};records=[]
            with np.load(OUT/'features'/s/'data.npz') as z:
                for r in read(OUT/'plan'/s/'candidates.json'):
                    cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];record=dict(candidate_id=cid,source=r['source'],arms={})
                    for arm in core.ARMS:
                        e=dict(x=z[cid+'_'+arm+'_x'].astype(np.float32),neighbors=z[cid+'_neighbors'].astype(np.int64),weights=z[cid+'_neighbor_weights'].astype(np.float32))
                        x,nn,w,parts=collate([e])
                        with torch.inference_mode():p=model(x,nn,w,parts).softmax(1).cpu().numpy().astype(np.float16)
                        repaired,decision=core.repair(parent,halo,p,z[cid+'_'+arm+'_groups'])
                        arrays[cid+'_'+arm+'_points']=repaired;arrays[cid+'_'+arm+'_probability']=p
                        decision['points_sha256']=hashlib.sha256(repaired.astype('<i8').tobytes()).hexdigest();record['arms'][arm]=decision
                        actions[arm+'_accepted']+=decision['accepted'];actions[arm+'_parents']+=1
                    records.append(record)
            dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'predictions.npz',**arrays);freeze_json(dest/'decisions.json',records)
            oo={str(dest/n):sha(dest/n) for n in ('predictions.npz','decisions.json')};stat=Counter()
            for r in records:
                for arm in core.ARMS:stat[arm+'_accepted']+=r['arms'][arm]['accepted'];stat[arm+'_parents']+=1
            freeze_json(dest/'receipt.json',dict(complete=True,fold=fold,GT_read=False,outputs=oo,actions=dict(stat)));outputs.update(oo)
        del model;torch.cuda.empty_cache()
    freeze_json(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,GT_read=False,all_test_predictions_OOF=True,original_NT_retained=True));stage('all_predictions_sealed')


def evaluate():
    from tools.evaluate_fi1_candidate_local_repair import evaluate as run
    run()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','plan','features','labels','train','predict','evaluate']);a=p.parse_args()
    globals()[a.mode]()


if __name__=='__main__':main()
