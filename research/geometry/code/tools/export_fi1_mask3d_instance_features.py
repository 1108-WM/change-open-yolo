#!/usr/bin/env python3
"""Read-only Mask3D head hook, explicit sparse-coordinate to original-point mapping."""
from pathlib import Path
import gc,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'models/Mask3D'))
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
from tools.fi1_mask3d_feature_ab_core import safe_columns
OUT=ROOT/'.local/fi1_mask3d_feature_ab_internal_20261013_v1'
SOURCE=ROOT/'.local/fi1_candidate_local_repair_learning_development200_20261012_v1'

def read(p):return json.loads(Path(p).read_text())
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def reserve():
    import shutil
    for p in (ROOT,OUT.resolve()):
        if shutil.disk_usage(p).free<50*2**30:raise RuntimeError('50GiB reserve reached: '+str(p))

class Guard:
    def __call__(self,event,args):
        if event=='open' and args and isinstance(args[0],(str,Path)):
            p=Path(args[0]);s=str(p)
            if any(x.startswith('ground_truth') for x in p.parts) or '/labels/' in s or '/diagnosis/' in s or 'independent_analysis' in s:raise PermissionError('feature export cannot read GT/labels/diagnoses')

def run():
    import MinkowskiEngine as ME
    import albumentations as A
    from models.Mask3D.mask3d import get_model,map_output_to_pointcloud
    c=read(OUT/'contract.json');sys.addaudithook(Guard());torch.set_num_threads(4);assert torch.cuda.is_available();reserve()
    checkpoint=ROOT/'pretrained/checkpoints/scannet200_val.ckpt';assert sha(checkpoint)==c['mask3d_checkpoint_sha256']
    torch.manual_seed(c['seed']);net=get_model(str(checkpoint)).cuda().eval()
    for p in net.parameters():p.requires_grad_(False)
    normalize=A.Normalize(mean=(.47793125906962,.4303257521323044,.3749598901421883),std=(.2834475483823543,.27566157565723015,.27018971370874995))
    outputs={};receipts=[]
    for index,s in enumerate(c['scenes']):
        reserve();dest=OUT/'instance_features'/s;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'receipt.json').exists():
            r=read(dest/'receipt.json')
            for p,h in r['outputs'].items():assert sha(p)==h
            outputs.update(r['outputs']);receipts.append(r);continue
        tic=time.time();path=c['jobs'][s]['paths']['processed'];assert sha(path)==c['inputs'][path]
        raw=np.load(path,mmap_mode='r');allowed=safe_columns(raw);xyz=allowed[:,:3]
        colors=normalize(image=allowed[:,3:6].astype(np.uint8)[None])['image'][0]
        coords=np.floor(xyz/.02)
        _,_,unique,inverse=ME.utils.sparse_quantize(coordinates=coords,features=colors,return_index=True,return_inverse=True)
        unique=np.asarray(unique,dtype=np.int64);inverse=np.asarray(inverse,dtype=np.int64)
        coordinate,feature=ME.utils.sparse_collate(coords=[torch.from_numpy(coords[unique]).int()],feats=[torch.from_numpy(colors[unique]).float()])
        data=ME.SparseTensor(coordinates=coordinate,features=feature,device='cuda')
        # Preserve the registered wrapper's call convention; do not change positional inputs.
        raw_coordinates=feature.cuda()
        torch.manual_seed(c['seed']+index);np.random.seed(c['seed']+index)
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all();numpy_rng=np.random.get_state()
        with torch.no_grad():old=net(data,raw_coordinates=raw_coordinates,point2segment=None)
        old_masks=old['pred_masks'][0].detach().cpu();old_logits=old['pred_logits'].detach().cpu()
        legacy_masks,legacy_scores=map_output_to_pointcloud(old,inverse,None,None)
        old_hard=ahash(legacy_masks.detach().cpu().numpy());old_scores=ahash(legacy_scores.detach().cpu().numpy())
        del old,legacy_masks,legacy_scores;torch.cuda.empty_cache()
        captured=[]
        def capture(module,arguments,value):captured.append((value.C.detach().cpu().numpy(),value.F.detach().cpu().numpy()))
        hook=net.model.mask_features_head.register_forward_hook(capture)
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng);np.random.set_state(numpy_rng)
        try:
            with torch.no_grad():new=net(data,raw_coordinates=raw_coordinates,point2segment=None)
        finally:hook.remove()
        assert torch.equal(old_masks,new['pred_masks'][0].detach().cpu()),'mask logits changed'
        assert torch.equal(old_logits,new['pred_logits'].detach().cpu()),'class logits changed'
        masks,scores=map_output_to_pointcloud(new,inverse,None,None)
        assert ahash(masks.detach().cpu().numpy())==old_hard and ahash(scores.detach().cpu().numpy())==old_scores
        assert len(captured)==1
        sparse_coords,features=captured[0];assert features.shape==(len(unique),128)
        lookup={tuple(map(int,q)):i for i,q in enumerate(sparse_coords[:,1:])}
        alignment=np.asarray([lookup[tuple(map(int,q))] for q in coords[unique]],np.int64)
        assert len(np.unique(alignment))==len(unique)
        assert np.array_equal(sparse_coords[alignment,1:],coords[unique])
        assert np.array_equal(coords[unique][inverse],coords) and len(inverse)==len(xyz)
        with np.load(SOURCE/'features'/s/'data.npz') as z:
            roi=np.unique(np.concatenate([z[k] for k in z.files if k.endswith('_halo')])).astype(np.int64)
        prior=features[alignment[inverse[roi]]].astype(np.float16)
        assert prior.shape==(len(roi),128) and np.isfinite(prior).all()
        # Only local ROI needs feature storage. inverse maps every original point, including outside ROI.
        np.savez_compressed(dest/'data.npz',point_ids=roi,features=prior,unique_map=unique.astype(np.int32),inverse_map=inverse.astype(np.int32),feature_row_for_unique=alignment.astype(np.int32))
        r=dict(complete=True,scene=s,GT_read=False,columns_used=list(range(9)),network_columns=list(range(6)),superpoint_used=False,
            point_count=len(xyz),voxel_count=len(unique),local_point_count=len(roi),feature_dimension=128,
            original_point_coordinate_mapping_exact=True,mask_logits_bitwise_unchanged=True,class_logits_bitwise_unchanged=True,
            hard_masks_bitwise_unchanged=True,scores_bitwise_unchanged=True,hard_masks_sha256=old_hard,scores_sha256=old_scores,
            input_sha256=sha(path),allowed_columns_sha256=ahash(allowed),feature_sha256=ahash(prior),seconds=time.time()-tic,
            outputs={str(dest/'data.npz'):sha(dest/'data.npz')},voxel_is_not_output_unit=True)
        freeze_json(dest/'receipt.json',r);outputs.update(r['outputs']);outputs[str(dest/'receipt.json')]=sha(dest/'receipt.json');receipts.append(r)
        write_json(OUT/'execution.json',dict(stage='Mask3D_GT_free_export',completed=index+1,total=len(c['scenes']),scene=s,seconds=r['seconds'],updated=time.time()))
        print('Mask3D export',index+1,len(c['scenes']),s,r['seconds'],flush=True)
        del allowed,raw,data,feature,raw_coordinates,new,masks,scores,old_masks,old_logits,captured,features,prior;gc.collect();torch.cuda.empty_cache()
    assert sha(checkpoint)==c['mask3d_checkpoint_sha256']
    freeze_json(OUT/'instance_features_frozen.json',dict(complete=True,outputs=outputs,scenes=len(receipts),GT_read=False,dimension=128,interface_only=True,geometry_improvement_claimed=False,
        all_mask_outputs_unchanged=all(r['hard_masks_bitwise_unchanged'] for r in receipts),all_original_index_maps_exact=True,checkpoint_unchanged=True))

if __name__=='__main__':run()
