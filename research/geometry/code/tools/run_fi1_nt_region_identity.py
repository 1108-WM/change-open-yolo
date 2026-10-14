#!/usr/bin/env python3
"""Stages 0--3: immutable preparation, regions, paired Mask3D and GT-free probes."""
from pathlib import Path
import argparse, gc, hashlib, json, os, shutil, sys, time
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
import fi1_nt_region_identity_core as core
from tools.fi1_frontend_protocol import freeze_json, write_json, sha256_file as sha
from tools.fi1_core_ablation import source_pool
from tools.fi1_r0_r1_protocol import original_aliases

OUT = ROOT/'research/geometry/experiments/fi1_nt_region_identity_diagnostic_20261014_v1'
AB = ROOT/'.local/fi1_mask3d_feature_ab_internal_20261013_v1'
SOURCE = ROOT/'.local/fi1_candidate_local_repair_learning_development200_20261012_v1'
REGIONS = ROOT/'.local/fi1_local_region_supply_development200_20261013_v1'
OBS = 'gd10_selected6'


def read(p): return json.loads(Path(p).read_text())
def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x
def seal(p, value): freeze_json(p, plain(value))
def stage(name, **kw):
    write_json(OUT/'execution.json', plain(dict(stage=name, updated=time.time(), **kw)))
    print(name, plain(kw), flush=True)
def reserve():
    for p in (ROOT, AB.resolve()):
        if shutil.disk_usage(p).free < 50*2**30: raise RuntimeError('50GiB reserve: '+str(p))
def verify(mapping):
    for p,h in mapping.items():
        if sha(p) != h: raise ValueError('identity changed: '+p)
def stage_seal(name, files, **kw):
    seal(OUT/(name+'_frozen.json'), dict(complete=True, time=time.time(),
                                      outputs={str(p):sha(p) for p in files}, **kw))
def guard():
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)): return
        p = os.fsdecode(args[0])
        if any(v in p for v in ('ground_truth', '/labels/', '/diagnosis/', '/structure_diagnosis/',
                                '/prior_diagnosis/', '/evaluation/', 'dev55', 'val312')):
            raise PermissionError('GT/diagnosis forbidden in construction/export/association: '+p)
    sys.addaudithook(hook)


def load_nt(job):
    geometry = source_pool(read(job['binding']['geometry']))
    with np.load(job['binding']['points']) as z:
        points = {g['geometry_key']: z[g['array_key']].astype(np.int64) for g in geometry}
    for g in geometry:
        p = points[g['geometry_key']]
        assert len(p) == g['point_count'] and np.array_equal(p, np.unique(p))
        assert hashlib.sha256(p.astype('<i8').tobytes()).hexdigest() == g['points_sha256']
    return geometry, points


def prepare():
    assert not OUT.exists(), 'new root already exists; inspect before reuse'
    reserve(); guard(); old = read(AB/'contract.json')
    legal = {}; strata = {}
    for group in ('historical90', 'new110'):
        pool = {}
        for s in sorted(set(old['splits']['fit']) & set(old['groups'][group])):
            j = old['jobs'][s]
            required = [j['paths']['processed'], j['binding']['geometry'], j['binding']['points'],
                        j['paths']['observations'], SOURCE/'features'/s/'data.npz',
                        SOURCE/'plan'/s/'candidates.json', SOURCE/'plan'/s/'evidence.npz',
                        REGIONS/'regions'/s/'regions.npz']
            if all(Path(p).is_file() for p in required):
                pool.setdefault(s.split('_')[0], s); legal[s] = required
        strata[group] = sorted(pool.values(), key=lambda s:
            (hashlib.sha256(('nt-region-identity-20261014:'+s.split('_')[0]).encode()).hexdigest(), s))
    chosen = {g:v[:6] for g,v in strata.items()}
    short = 12-sum(map(len, chosen.values()))
    for g,v in strata.items():
        if short:
            extra=v[len(chosen[g]):len(chosen[g])+short]; chosen[g]+=extra; short-=len(extra)
    scenes = [s for g in chosen for s in chosen[g]]
    assert len(scenes) == 12 and len({s.split('_')[0] for s in scenes}) == 12
    OUT.mkdir(parents=True)
    inputs = {str(AB/'contract.json'):sha(AB/'contract.json')}
    parents = {}; prior_inputs = {}
    for s in scenes:
        for p in legal[s]: inputs[str(p)] = sha(p)
        parents[s] = read(SOURCE/'plan'/s/'candidates.json')
        p=AB/'instance_features'/s/'data.npz'; prior_inputs[str(p)]=sha(p)
    files=[Path(__file__).resolve(), Path(__file__).with_name('fi1_nt_region_identity_core.py'),
           Path(__file__).with_name('evaluate_fi1_nt_region_identity.py'),
           Path(__file__).with_name('verify_fi1_nt_region_identity.py'),
           Path(__file__).with_name('run_fi1_nt_region_identity.sh'),
           ROOT/'research/geometry/code/tests/test_fi1_nt_region_identity.py',
           ROOT/'docs/FI1_NT_REGION_IDENTITY_NEXT_STEP_20261014.md',
           ROOT/'full_bundle/vendor/Mask3D/mask3d/models/mask3d.py',
           ROOT/'full_bundle/vendor/Mask3D/mask3d/__init__.py',
           ROOT/'full_bundle/vendor/Mask3D/mask3d/datasets/semseg.py',
           ROOT/'full_bundle/vendor/Mask3D/mask3d/trainer/trainer.py',
           ROOT/'related papers/Clutt3R-Seg_2026_arXiv2602.11660.txt']
    checkpoint=ROOT/'pretrained/checkpoints/scannet200_val.ckpt'
    c=dict(version=OUT.name, scenes=scenes, strata=chosen, legal_counts={g:len(v) for g,v in strata.items()},
           sampling='fit96 only; SHA256(nt-region-identity-20261014:physical_id), first legal scan; no replacement after freeze',
           jobs={s:old['jobs'][s] for s in scenes}, parents=parents, class_ids=old['class_ids'],
           GT_inputs={old['jobs'][s]['GT']:old['GT_inputs'][old['jobs'][s]['GT']] for s in scenes},
           parameters=core.PARAMETERS, inputs=inputs, prior_inputs=prior_inputs,
           code={str(p):sha(p) for p in files}, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
           frozen_SL_models=old['frozen_SL_models'], storage=str(OUT.resolve()),
           neighbors='full NT exact dedup; overlap or minimum distance<=.15m; sort -intersection/parent size, distance, stable geometry ID; max4',
           regions='full membership vector incl zero, original8NN undirected edges <=.05m; no small-region removal',
           association='full-scene soft IoU; stable query ID tie; parent and every region top2, gap<=.01 or ratio>=.9 marks ambiguity only; no GT or exclusion',
           query_diagnostic='all150 queries evaluated as local visible原点 probes against every valid GT instance; oracle best readout only, never GT replacement',
           region_readout='parent top1 query region mean>=.5, identical visibility>=2/min100 fallback; regions remain contextual hypotheses',
           structure_feasible='stable parent target only; point GT known edits; new region pure over complete halo region, unknown retained; deletion comparison purity over same parent-visible subset; old96 union feasible',
           identity='same best valid target matching >=20 hits/purity>=.6 as AB; mixed parents list all GT instance identities, no unique correctness assignment',
           modes=['P0_point','P0_region','P1_point','P1_region'],
           structure_GT_after='structure_frozen', prior_GT_after='priors_frozen AND associations_frozen',
           training=False, AP_run=False, SL_modified=False, production_modified=False,
           disk_initial={str(p):shutil.disk_usage(p).free for p in (ROOT,AB.resolve())})
    verify(inputs); verify(prior_inputs); verify(c['frozen_SL_models'])
    seal(OUT/'contract.json', c)
    for p in files:
        q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    seal(OUT/'scenes.json',dict(scenes=scenes,strata=chosen));stage('stage0_frozen',scenes=scenes)


def structures():
    c=read(OUT/'contract.json'); guard(); verify(c['code']); verify(c['inputs']); reserve(); outputs=[]
    for s in c['scenes']:
        dest=OUT/'structure'/s; dest.mkdir(parents=True,exist_ok=True)
        if (dest/'receipt.json').exists(): verify(read(dest/'receipt.json')['outputs']);outputs+=list(dest.glob('*'));continue
        j=c['jobs'][s];xyz=core.safe_input(np.load(j['paths']['processed'],mmap_mode='r'))[:,:3]
        geom,nt=load_nt(j); by_key={g['geometry_key']:g for g in geom}
        # Source loader already deduplicates pointsets; validate exact identity once more.
        unique={}; aliases={}
        for key in sorted(nt):
            blob=nt[key].astype('<i8').tobytes()
            if blob in unique: aliases[unique[blob]].append(key)
            else: unique[blob]=key; aliases[key]=[key]
        arrays={}; metadata=[]; support=[]; contacts=[]
        obs=[json.loads(l) for l in Path(j['paths']['observations']).read_text().splitlines()]
        bankroot=Path(obs[0]['point_indices_path']).parent.parent
        needed=sorted({f for r in c['parents'][s] for f in r['views'][OBS]['frames']})
        banks={};bank_inputs={}
        for f in needed:
            p=bankroot/f'frame{f:06d}'/'observations.json';rows=read(p);bank_inputs[str(p)]=sha(p)
            masks=[]
            for row in rows:
                p=Path(row['point_indices_path']);bank_inputs[str(p)]=sha(p)
                with np.load(p) as z: masks.append(z['point_indices'].copy())
            banks[f]=masks
        with np.load(SOURCE/'features'/s/'data.npz') as z, np.load(SOURCE/'plan'/s/'evidence.npz') as ev:
            roi=ev['roi']; frame_lookup={int(f):i for i,f in enumerate(ev['frames'])}
            visible=np.unpackbits(ev['reliable'],axis=1,bitorder='little')[:,:len(roi)].astype(bool)
            for r in c['parents'][s]:
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];nn=z[cid+'_neighbors']; tree=cKDTree(xyz[parent]); candidates=[]
                for key in unique.values():
                    p=nt[key]
                    if np.array_equal(p,parent):continue
                    hit=len(np.intersect1d(parent,p,assume_unique=True))
                    # Exact minimum distance, evaluated in bounded chunks.
                    distance=0. if hit else float(tree.query(xyz[p],k=1)[0].min())
                    if hit or distance<=.15:
                        candidates.append(dict(key=key, overlap=hit/len(parent), hit=hit, distance=distance,
                            sources=sorted({a['source'] for a in original_aliases(by_key[key])}), aliases=aliases[key]))
                candidates.sort(key=lambda q:(-q['overlap'],q['distance'],q['key'])); kept=candidates[:4]
                membership=np.column_stack([np.isin(halo,parent)]+[np.isin(halo,nt[q['key']]) for q in kept])
                labels,order,offset,edges,dist=core.partition(xyz[halo],nn,membership)
                arrays.update({cid+'_parent':parent,cid+'_halo':halo,cid+'_labels':labels,cid+'_order':order,
                               cid+'_offsets':offset,cid+'_membership':membership,cid+'_edges':edges,
                               cid+'_edge_distance':dist,cid+'_visible':z[cid+'_'+OBS+'_groups']})
                regions=[order[lo:hi] for lo,hi in zip(offset[:-1],offset[1:])]
                evidence=[]
                for f in r['views'][OBS]['frames']:
                    seen=visible[frame_lookup[f],np.searchsorted(roi,halo)]
                    count=np.bincount(labels[seen],minlength=len(regions));cov=np.zeros((len(regions),len(banks[f])),np.int32)
                    for k,p in enumerate(banks[f]): cov[:,k]=np.bincount(labels[seen & np.isin(halo,p)],minlength=len(regions))
                    best=cov.argmax(1) if cov.shape[1] else np.full(len(regions),-1)
                    hit=cov[np.arange(len(regions)),best] if cov.shape[1] else np.zeros(len(regions),int)
                    best=np.where(hit>0,best,-1)
                    arrays[cid+f'_frame{f}_mask_hits']=cov;arrays[cid+f'_frame{f}_visible']=count
                    evidence.append((f,best,count,hit))
                    for rid in range(len(regions)):
                        support.append(dict(scene=s,candidate_id=cid,region=rid,frame=f,visible=int(count[rid]),
                            best_mask=int(best[rid]),best_mask_hits=int(hit[rid]),missing=best[rid]<0))
                for edge,(u,v) in enumerate(edges):
                    same=different=missing=0
                    for f,best,count,hit in evidence:
                        if count[u]==0 or count[v]==0 or best[u]<0 or best[v]<0: missing+=1
                        elif best[u]==best[v]:same+=1
                        else:different+=1
                    contacts.append(dict(scene=s,candidate_id=cid,u=int(u),v=int(v),minimum_distance=float(dist[edge]),
                        membership_changes=int(np.count_nonzero(membership[regions[u][0]]!=membership[regions[v][0]])),
                        same_best_mask_frames=same,different_best_mask_frames=different,missing_frames=missing))
                omitted=candidates[4:];covered=np.zeros(len(halo),bool)
                for q in omitted:covered |= np.isin(halo,nt[q['key']])
                pairs=[]
                for q in kept:
                    if ('track' if r['source']=='native' else 'native') in q['sources']:
                        m=np.isin(halo,nt[q['key']]);own=membership[:,0]
                        pairs.append(dict(neighbor=q['key'],intersection=int((own&m).sum()),parent_only=int((own&~m).sum()),neighbor_only=int((~own&m).sum())))
                metadata.append(dict(r,neighbors=kept,neighbors_before_truncation=len(candidates),omitted=candidates[4:],
                    omitted_covered_halo_points=int(covered.sum()),regions=len(regions),edges=len(edges),pair_explanation=pairs,
                    region_member_vectors=[membership[b[0]].astype(int).tolist() for b in regions]))
                print('regions',s,cid,len(halo),len(regions),flush=True)
        np.savez_compressed(dest/'data.npz',**arrays);seal(dest/'candidates.json',metadata)
        seal(dest/'frame_support.json',support);seal(dest/'contacts.json',contacts)
        files=list(dest.glob('*'));seal(dest/'receipt.json',dict(complete=True,GT_read=False,bank_inputs=bank_inputs,
            outputs={str(p):sha(p) for p in files}));outputs+=list(dest.glob('*'));reserve()
        stage('stage1_structure',scene=s)
    stage_seal('structure',outputs,GT_read=False,contract_sha256=sha(OUT/'contract.json'))


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
            for mode,position in [('P0',rgb),('P1',raw_xyz)]:
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
                if mode=='P0':previous=dict(hc=hc.copy(),hf=hf.copy(),bc=bc.copy(),bf=bf.copy())
                else:
                    audits['paired']=dict(head_coordinate_identical=np.array_equal(previous['hc'],hc),
                        head_bitwise_identical=np.array_equal(previous['hf'],hf),
                        backbone_coordinate_identical=np.array_equal(previous['bc'],bc),
                        backbone_bitwise_identical=np.array_equal(previous['bf'],bf),
                        ROI_mean_probability_absolute_change=float(np.abs(arrays['P0_probability']-arrays['P1_probability']).mean()),
                        class_logits_mean_absolute_change=float(np.abs(arrays['P0_class_logits']-arrays['P1_class_logits']).mean()),
                        query_index_comparison_is_not_identity_matching=True)
                del output,pc,masks,probability;gc.collect();torch.cuda.empty_cache()
        finally:
            for hook in hooks:hook.remove()
        assert all(audits['paired'][k] for k in ('head_coordinate_identical','head_bitwise_identical','backbone_coordinate_identical','backbone_bitwise_identical'))
        np.savez_compressed(d/'data.npz',**arrays)
        seal(d/'receipt.json',dict(complete=True,GT_read=False,safe_columns_poison_test=True,
            color_features_and_XYZ_position_separate=True,sparse_original_mapping_exact=True,paired=audits,
            outputs={str(d/'data.npz'):sha(d/'data.npz')}));files+=list(d.glob('*'))
        stage('stage2_paired_prior',scene=s,index=si+1,paired=audits['paired'])
        del arrays,previous,captured,embeds,data,raw_xyz,rgb,allowed;gc.collect();torch.cuda.empty_cache()
    verify(c['prior_inputs']);assert sha(c['checkpoint'])==c['checkpoint_sha256']
    stage_seal('priors',files,GT_read=False,original_prior_unchanged=True,checkpoint_unchanged=True)


def associate():
    c=read(OUT/'contract.json');guard();verify(c['code']);verify(read(OUT/'priors_frozen.json')['outputs']);files=[]
    for s in c['scenes']:
        d=OUT/'associations'/s;d.mkdir(parents=True,exist_ok=True);rows=[];scores=[];arrays={}
        with np.load(OUT/'structure'/s/'data.npz') as z,np.load(OUT/'priors'/s/'data.npz') as prior:
            roi=prior['point_ids']
            for r in c['parents'][s]:
                cid=r['candidate_id'];halo=z[cid+'_halo'];parent=z[cid+'_parent'];labels=z[cid+'_labels'];n=z[cid+'_offsets'].size-1
                ix=np.searchsorted(roi,halo);assert np.array_equal(roi[ix],halo)
                for mode in ('P0','P1'):
                    p=prior[mode+'_probability'][ix];volume=prior[mode+'_full_volume']
                    own=np.flatnonzero(np.isin(halo,parent));link=core.association(p,volume,own)
                    means,std=core.region_stats(p,labels); region_links=[];queryset=set(link['top'])
                    for rid in range(n):
                        association=core.association(p,volume,np.flatnonzero(labels==rid));queryset.update(association['top'])
                        region_links.append(dict(region=rid,**association))
                    chosen=link['top'][0]
                    arrays[cid+'_'+mode+'_point']=core.probe(parent,halo,p[:,chosen],z[cid+'_visible'])
                    arrays[cid+'_'+mode+'_region']=core.probe(parent,halo,means[labels,chosen],z[cid+'_visible'])
                    arrays[cid+'_'+mode+'_region_mean']=means.astype(np.float32)
                    arrays[cid+'_'+mode+'_region_std']=std.astype(np.float32)
                    rows.append(dict(scene=s,candidate_id=cid,source=r['source'],prior=mode,parent=link,
                        regions=region_links,query_set=sorted(queryset)))
                    for rid in range(n):
                        ranked=np.lexsort((np.arange(p.shape[1]),-means[rid]));gap=float(means[rid,ranked[0]]-means[rid,ranked[1]])
                        for q in sorted(queryset):scores.append(dict(scene=s,candidate_id=cid,prior=mode,region=rid,query=q,
                            mean=float(means[rid,q]),std=float(std[rid,q]),top_mean_gap=gap,
                            region_association_top1=region_links[rid]['top'][0],region_association_ambiguous=region_links[rid]['ambiguous']))
        np.savez_compressed(d/'data.npz',**arrays);seal(d/'links.json',rows);seal(d/'region_scores.json',scores)
        files+=list(d.glob('*'));reserve();stage('stage3_GT_free_associations',scene=s)
    stage_seal('associations',files,GT_read=False,all_four_probes_frozen=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','structures','export','associate']);args=p.parse_args()
    globals()[args.mode]()
