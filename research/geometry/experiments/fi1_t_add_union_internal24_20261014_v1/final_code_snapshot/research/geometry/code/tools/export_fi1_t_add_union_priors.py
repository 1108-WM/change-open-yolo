#!/usr/bin/env python3
"""Existing frozen XYZ P1 exporter; no paired RGB diagnostic or model training.

Derived from sealed run_fi1_nt_region_identity.export; sparse map, checkpoint,
normalization, hooks and original point multiplicity are unchanged.
"""
from run_fi1_nt_region_identity import *
import run_fi1_t_add_union as task
OUT=task.OUT
guard=task.guard

def export():
    # This entry runs only in the registered Mask3D environment on host GPU.
    import torch
    import MinkowskiEngine as ME
    import albumentations as A
    sys.path.append(str(ROOT/'models/Mask3D'))
    from models.Mask3D.mask3d import get_model
    c=read(OUT/'contract.json');guard();reserve();verify(c['code']);verify(read(OUT/'structure_frozen.json')['outputs'])
    assert torch.cuda.is_available();torch.set_num_threads(4)
    assert sha(c['checkpoint'])==c['checkpoint_sha256']
    torch.manual_seed(core.PARAMETERS['seed']);net=get_model(c['checkpoint']).cuda().eval()
    for p in net.parameters():p.requires_grad_(False)
    normalize=A.Normalize(mean=(.47793125906962,.4303257521323044,.3749598901421883),std=(.2834475483823543,.27566157565723015,.27018971370874995))
    files=[]
    for si,s in enumerate(c['scenes']):
        d=OUT/'priors'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():verify(read(d/'receipt.json')['outputs']);files+=list(d.glob('*'));continue
        reserve();raw=np.load(c['jobs'][s]['paths']['processed'],mmap_mode='r');allowed=core.safe_input(raw)
        # Poison forbidden columns without ever inspecting their real values.
        poisoned=np.zeros((len(raw),max(12,raw.shape[1])),np.float32);poisoned[:,:9]=allowed;poisoned[:,9:]=np.nan
        assert np.array_equal(core.safe_input(poisoned),allowed);del poisoned,raw
        xyz=allowed[:,:3]; colors=normalize(image=allowed[:,3:6].astype(np.uint8)[None])['image'][0]
        coords=np.floor(xyz/.02).astype(np.int32)
        _,_,unique,inverse=ME.utils.sparse_quantize(coordinates=coords,features=colors,return_index=True,return_inverse=True)
        unique=np.asarray(unique,np.int64);inverse=np.asarray(inverse,np.int64)
        coordinate,feature=ME.utils.sparse_collate(coords=[torch.from_numpy(coords[unique])],feats=[torch.from_numpy(colors[unique]).float()])
        data=ME.SparseTensor(coordinates=coordinate,features=feature,device='cuda')
        sparse_coords=data.C.cpu().numpy();src_coords=coordinate.cpu().numpy()
        row_from_unique=core.coordinate_rows(src_coords,sparse_coords)
        raw_xyz=torch.from_numpy(xyz[unique][row_from_unique].copy()).cuda()
        rgb=torch.from_numpy(colors[unique][row_from_unique].copy()).cuda()
        assert torch.equal(data.F,rgb), 'RGB backbone sparse rows differ'
        unique_from_row=np.argsort(row_from_unique)
        original_to_row=unique_from_row[inverse]
        assert np.array_equal(sparse_coords[original_to_row,1:],coords)
        with np.load(OUT/'structure'/s/'data.npz') as z:
            roi=np.unique(np.concatenate([z[k] for k in z.files if k.endswith('_halo')]))
        torch.manual_seed(core.PARAMETERS['seed']+si);np.random.seed(core.PARAMETERS['seed']+si)
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();numpy_rng=np.random.get_state()
        arrays=dict(point_ids=roi,unique_map=unique,inverse_map=inverse,input_sparse_coordinates=sparse_coords,
                    row_from_unique=row_from_unique,original_to_sparse_row=original_to_row)
        audits={};previous={};captured={};embeds=[]
        def head_hook(module,args,value):captured['head']=(value.C.detach().cpu().numpy(),value.F.detach().cpu().numpy())
        def embedding_hook(module,args,value):embeds.append(value.detach().cpu().numpy())
        def backbone_rows_hook(module,args,value):
            assert np.array_equal(value[1][-1].C.detach().cpu().numpy(),sparse_coords), 'raw XYZ rows differ from positional-coordinate map'
        hooks=[net.model.mask_features_head.register_forward_hook(head_hook),net.model.mask_embed_head.register_forward_hook(embedding_hook),
               net.model.backbone.register_forward_hook(backbone_rows_hook)]
        try:
            for mode,position in [('P1',raw_xyz)]:
                captured.clear();embeds.clear();torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);np.random.set_state(numpy_rng)
                tic=time.time()
                with torch.no_grad():output=net(data,raw_coordinates=position,point2segment=None)
                pc=output['backbone_features'];bc=pc.C.detach().cpu().numpy();bf=pc.F.detach().cpu().numpy()
                hc,hf=captured['head']; masks=output['pred_masks'][0]
                assert masks.ndim==2 and len(masks)==len(hf)
                # pred_masks rows follow mask_features.decomposed_features for single batch.
                map_input_to_head=core.coordinate_rows(hc,sparse_coords)
                rows=map_input_to_head[original_to_row]
                probability=masks.sigmoid().detach().cpu().numpy()
                assert np.isfinite(probability).all() and np.isfinite(hf).all() and np.isfinite(bf).all()
                multiplicity=np.bincount(rows,minlength=len(hf))
                volume=multiplicity.astype(np.float64)@probability.astype(np.float64)
                arrays[mode+'_probability']=probability[rows[roi]].astype(np.float32)
                arrays[mode+'_full_volume']=volume
                arrays[mode+'_hard_volume']=(multiplicity[:,None]*(probability>=.5)).sum(0)
                arrays[mode+'_class_logits']=output['pred_logits'].detach().cpu().numpy()
                arrays[mode+'_mask_embedding']=embeds[-1]
                arrays[mode+'_head_roi']=hf[rows[roi]].astype(np.float16)
                audits[mode]=dict(queries=probability.shape[1],seconds=time.time()-tic,GT_read=False,
                    full_point_count=len(xyz),voxel_count=len(hf),roi_points=len(roi),finite=True,
                    head_sha256=hashlib.sha256(hf.tobytes()).hexdigest(),backbone_sha256=hashlib.sha256(bf.tobytes()).hexdigest())
                del output,pc,masks,probability;gc.collect();torch.cuda.empty_cache()
        finally:
            for hook in hooks:hook.remove()
        np.savez_compressed(d/'data.npz',**arrays)
        seal(d/'receipt.json',dict(complete=True,GT_read=False,safe_columns_poison_test=True,
            color_features_and_XYZ_position_separate=True,sparse_original_mapping_exact=True,modes=audits,
            outputs={str(d/'data.npz'):sha(d/'data.npz')}));files+=list(d.glob('*'))
        stage('stage_P1_prior',scene=s,index=si+1,prior=audits['P1'])
        del arrays,previous,captured,embeds,data,raw_xyz,rgb,allowed;gc.collect();torch.cuda.empty_cache()
    verify(c['prior_inputs']);assert sha(c['checkpoint'])==c['checkpoint_sha256']
    stage_seal('priors',files,GT_read=False,original_prior_unchanged=True,checkpoint_unchanged=True)

if __name__=='__main__':export()
