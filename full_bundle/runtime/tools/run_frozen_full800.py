#!/usr/bin/env python3
"""Portable frozen official100 D-v3 / GD10-matched597 B26 / K800 runner.

No training or threshold search. GPU acquisition and sklearn1.3.0 geometry
run in separate subprocesses from sklearn1.3.2 member scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / 'release_assets/official100_gd10_b26_k800'
VAL = ROOT / 'reproduction/scannetv2_val.txt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    from tools.fi1_frontend_protocol import freeze_json
    freeze_json(path, value)


def verify_release():
    manifest = read(ROOT / 'reproduction/manifest.json')
    for relative, expected in manifest['files'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('release checksum differs: ' + relative)
    return manifest


def selected_scenes(args):
    canonical = VAL.read_text().split()
    requested = args.scenes or canonical
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(canonical):
        raise ValueError('Use unique scenes from the fixed validation312 list; no final test access.')
    return sorted(requested)


def check_external(include_frontend=True):
    paths = {
        'sam_checkpoint': ROOT/'pretrained/checkpoints/sam_vit_b_01ec64.pth',
        'groundingdino': ROOT/'pretrained/groundingdino/groundingdino_swint_ogc.pth',
        'mask3d': ROOT/'pretrained/checkpoints/scannet200_val.ckpt',
        'yoloworld': ROOT/'pretrained/checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth',
        'alpha_clip_base': ROOT/'pretrained/alpha_clip/checkpoints/ViT-L-14.pt',
        'alpha_clip_checkpoint': ROOT/'pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth',
    }
    for name, spec in read(ROOT/'reproduction/external_weights.json').items():
        if name=='groundingdino' and not include_frontend:
            continue
        if not paths[name].is_file() or sha(paths[name]) != spec['sha256']:
            raise ValueError('Missing or different external weight: '+str(paths[name]))
    required=[ROOT/'_external/AlphaCLIP/AlphaCLIP-main/alpha_clip',
              ROOT/'_external/segment-anything/segment-anything-main/segment_anything']
    if include_frontend:
        required += [ROOT/'_external/GroundingDINO/GroundingDINO-main/groundingdino']
        required += [ROOT/'pretrained/bert-base-uncased'/name for name in
                     ['config.json','pytorch_model.bin','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt']]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)
    return paths


def prepare_selection(output):
    from tools.fi1_frontend_protocol import digest
    selection = {'dataset_role':'val312_g10_backend', 'scenes':sorted(VAL.read_text().split()),
                 'active_arms':['D'], 'frontend':'grounded_sam', 'stride':10}
    folder=output/'selection'
    write(folder/'selection.json',selection)
    order=folder/'scenes_execution_order.txt'
    expected='\n'.join(selection['scenes'])+'\n'
    if order.exists() and order.read_text()!=expected:
        raise ValueError('Selection order changed')
    order.write_text(expected)
    sources=[ROOT/'资料/训练597场景清单_20260917.json',VAL]
    write(folder/'selection_receipt.json',{'complete':True,
        'inputs':{str(p):sha(p) for p in sources},
        'outputs':{p.name:sha(p) for p in [folder/'selection.json',order]}})
    return selection


def launch(python, tool, *args):
    command=[str(python),str(ROOT/'tools'/tool),*map(str,args)]
    print('STAGE',tool,*map(str,args),flush=True)
    env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1',
         'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    subprocess.run(command,cwd=ROOT,env=env,check=True)


def native(args):
    import numpy as np
    import torch
    import yaml
    from utils import OpenYolo3D
    from tools.export_scannet200_train_scene_native_cache import _save_native
    scene=args.scenes[0]
    out=args.output
    target=out/'records'/scene
    config=ROOT/'pretrained/config_scannet200.yaml'
    inputs={str(config):sha(config)}
    for p in [args.dataset_root/scene/(scene[5:]+'.npy'),
              args.dataset_root/scene/(scene+'_vh_clean_2.ply')]:
        inputs[str(p)]=sha(p)
    marker=target/'native_complete.json'
    if marker.exists():
        saved=read(marker)
        if saved['inputs']!=inputs:
            raise ValueError('Native inputs changed; use new output')
        for p,h in saved['outputs'].items():
            if sha(p)!=h:
                raise ValueError('Native cache changed')
        return
    if target.exists():
        raise FileExistsError('Incomplete native stage retained: '+str(target))
    for directory in [target/'native_cache',target/'yoloworld_bboxes_2d',out/'common/mask3d_masks']:
        directory.mkdir(parents=True,exist_ok=True)
    model=OpenYolo3D(str(config))
    result=model.predict(path_2_scene_data=str(args.dataset_root/scene),depth_scale=1000.,
        datatype='mesh',processed_scene=str(args.dataset_root/scene/(scene[5:]+'.npy')),
        path_to_3d_masks=None,is_gt=False,path_to_2d_preds=str(target/'yoloworld_bboxes_2d'),
        save_2d_preds=True,reuse_2d_preds=False)
    summary=_save_native(target/'native_cache',scene,result[scene])
    masks,scores=model.preds_3d
    maskpath=out/'common/mask3d_masks'/f'{scene}.pt'
    torch.save((masks.detach().cpu(),scores.detach().cpu()),maskpath)
    cache=target/'yoloworld_bboxes_2d'/f'{scene}.pt'
    signed=torch.load(cache,map_location='cpu')
    if signed.get('metadata',{}).get('text_prompts')!=yaml.safe_load(config.read_text())['network2d']['text_prompts']:
        raise ValueError('YOLO vocabulary metadata differs')
    write(cache.parent/'receipt.json',{'summary':{'complete':True,'scene_name':scene},
                                      'outputs':{str(cache.resolve()):sha(cache)}})
    files=[maskpath,cache,*sorted((target/'native_cache').glob('*.npy'))]
    write(marker,{'complete':True,'inputs':inputs,'outputs':{str(p):sha(p) for p in files},'summary':summary})


def frontend(args):
    from tools import run_fi1_frontend_comparison as core
    from tools.fi1_frontend_protocol import digest
    out=args.output/'frontend'
    selection=prepare_selection(args.output)
    contract={'version':'portable_frozen_GD10_v1','assets':core.assets(),'stride':10,'GT_read':False}
    write(out/'selection.json',selection)
    write(out/'contract.json',contract)
    scene=args.scenes[0]
    mp=out/'manifests'/f'{scene}.json'
    if not mp.exists():
        manifest=core.build_scene_manifest(scene,args.dataset_root,args.dataset_root)
        manifest['arms']['D']=list(range(0,manifest['original_frame_count'],10))
        write(mp,manifest)
    manifest=read(mp)
    if manifest['arms']['D']!=list(range(0,manifest['original_frame_count'],10)):
        raise ValueError('Frontend is not GD10')
    core.verify_scene_inputs(manifest)
    domain=core.check_point_order(manifest)
    base={'contract':digest(contract),'manifest':digest(manifest),'frontend':'grounded_sam'}
    identity=digest({**base,'arm':'D'})
    final=out/'arms/D/stage_receipts'/f'{scene}_merged.json'
    frame_root=out/'frame_cache/grounded_sam'/scene
    if not final.exists():
        model=core.Frontends('grounded_sam',out/'grounded_sam_model_receipt.json')
        chosen=set(map(str,manifest['arms']['D']))
        for frame in manifest['frames']:
            if str(frame['frame_id']) in chosen:
                core.export_frame(model,manifest,frame,frame_root/f"frame{int(frame['frame_id']):06d}",
                                  digest({**base,'frame':frame}),domain)
        del model
        core.assemble_arm(out,'D',manifest,frame_root,identity)
    core.track_arm(out,'D',manifest,frame_root,domain,identity)


def adapters(args):
    scene=args.scenes[0]
    out=args.output
    role='val312_g10_backend'
    record=out/'records'/scene
    prep=out/'preparation'
    link=prep/'records_adapter'/scene
    link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists():
        link.symlink_to(record,target_is_directory=True)
    if link.resolve()!=record.resolve():
        raise ValueError('Records adapter differs')
    hashes={}
    for p in sorted((record/'native_cache').glob('*.npy')):
        destination=out/'common/01_native_cache'/p.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            destination.symlink_to(p)
        if destination.resolve()!=p.resolve():
            raise ValueError('Native alias differs')
        hashes[str(prep/'records_adapter'/scene/'native_cache'/p.name)]=sha(p)
    if len(hashes)!=3:
        raise ValueError('Need native masks/classes/scores')
    write(prep/'native_bindings'/scene/'receipt.json',{'scene':scene,'dataset_role':role,'input_files':hashes})
    write(prep/'inventory/input_sources.json',{'config_path':str(ROOT/'pretrained/config_scannet200.yaml'),
        'sources':{role:{'prepared_root':str(args.dataset_root),'records_root':str(prep/'records_adapter'),
                         'alpha_cache':str(ASSETS/'alpha_text')}}})
    yolo=out/'yolo'/scene
    yolo.parent.mkdir(parents=True,exist_ok=True)
    if not yolo.exists():
        yolo.symlink_to(record/'yoloworld_bboxes_2d',target_is_directory=True)


def alpha(args):
    import numpy as np
    import torch
    from tools.build_dm_sms1_alpha_embedding_ledger import _load_alpha_clip,_load_sam,_process_scene
    from tools.complete_fi1_r0_r1_alpha import manifest_row
    from tools.bind_fi1_r0_r1_alpha_cache import recover_alpha
    from tools.fi1_frontend_protocol import digest
    scene=args.scenes[0]
    own=args.output/'scenes'/scene/'own/D'
    views=own/'views'
    target=own/'alpha_completed'/scene
    identity={'version':'portable_frozen_own_alpha_v1','views_identity_sha256':digest(read(views/'identity.json')),
              'text_features_sha256':sha(ASSETS/'alpha_text/text_features.npy'),
              'external_weights':read(ROOT/'reproduction/external_weights.json'),'scene':scene}
    if (target/'receipt.json').exists():
        old=read(target/'receipt.json')
        if old['identity']!=identity or sha(target/'evidence.jsonl')!=old['outputs']['evidence.jsonl']:
            raise ValueError('Alpha resume identity differs')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Alpha requires CUDA')
    requests=rows(views/scene/'yolo_and_alpha_requests.jsonl')
    prompts=read(ASSETS/'alpha_text/text_prompts.json')['prompts']
    settings=NS(alpha_clip_source=ROOT/'_external/AlphaCLIP/AlphaCLIP-main',
        alpha_clip_base=ROOT/'pretrained/alpha_clip/checkpoints/ViT-L-14.pt',
        alpha_clip_checkpoint=ROOT/'pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth',
        sam_source=ROOT/'_external/segment-anything/segment-anything-main',
        sam_checkpoint=ROOT/'pretrained/checkpoints/sam_vit_b_01ec64.pth',sam_model_type='vit_b',
        sam_batch_size=8,alpha_batch_size=32,sms_threshold=0.,output_root=own/'fresh_alpha_cache')
    state=_load_alpha_clip(settings,prompts)
    predictor=_load_sam(settings)
    text=np.load(ASSETS/'alpha_text/text_features.npy',allow_pickle=False)
    manifests=[manifest_row(r) for r in requests]
    _process_scene(scene,manifests,settings,state,predictor,text,digest(manifests))
    cache=settings.output_root/'scenes'/scene
    scale=np.load(cache/'scale_features.npy',mmap_mode='r')
    aggregate=np.load(cache/'aggregate_features.npy',mmap_mode='r')
    similarities=np.load(cache/'similarities.npy',mmap_mode='r')
    acquired={r['geometry_key']:recover_alpha(r,scale,aggregate,similarities,text) for r in rows(cache/'records.jsonl')}
    target.mkdir(parents=True,exist_ok=False)
    with (target/'evidence.jsonl').open('x') as stream:
        for r in requests:
            r['alpha']=acquired[r['geometry_key']]
            stream.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    write(target/'receipt.json',{'complete':True,'identity':identity,'outputs':{'evidence.jsonl':sha(target/'evidence.jsonl')}})


def score(args):
    import joblib
    import numpy as np
    import sklearn
    from tools.fi1_core_ablation import features_and_members,learned_selection
    from tools.fi1_frontend_protocol import digest
    if sklearn.__version__!='1.3.2':
        raise RuntimeError('B26 must use scikit-learn==1.3.2')
    geometries=read(args.geometry_dir/'geometry.json')
    evidence=rows(args.evidence_file)
    scene=args.scenes[0]
    domain=args.point_count
    if domain is None:
        domain=read(args.geometry_dir/'receipt.json')['summary']['point_count']
    if type(domain)!=int or domain<1:
        raise ValueError('Invalid full point domain')
    points={}
    with np.load(args.geometry_dir/'points.npz',allow_pickle=False) as packed:
        for g in geometries:
            p=np.asarray(packed[g['array_key']],dtype=np.int64)
            if p.ndim!=1 or not np.array_equal(p,np.unique(p)) or (len(p) and (p[0]<0 or p[-1]>=domain)):
                raise ValueError('Invalid point indexes')
            if len(p)!=g['point_count'] or hashlib.sha256(p.tobytes()).hexdigest()!=g['points_sha256']:
                raise ValueError('Candidate pointset digest/count differs')
            if g['geometry_key']!=f"{scene}:geometry:{g['points_sha256']}":
                raise ValueError('Geometry belongs to another scene')
            points[g['geometry_key']]=p
    by_key={r['geometry_key']:r for r in evidence}
    if len(by_key)!=len(evidence) or len(points)!=len(geometries):
        raise ValueError('Duplicate geometry/evidence')
    # Verify the visual recipe independently of array dimensions/pointset hashes.
    # Legacy completed evidence binds immutable cache inputs; the portable
    # acquisition stage binds these same weights in its scene receipt.
    proof=getattr(args,'semantic_provenance',None)
    expected=read(ROOT/'reproduction/external_weights.json')
    alpha_hashes={expected[k]['sha256'] for k in ['alpha_clip_base','alpha_clip_checkpoint','sam_checkpoint']}
    text_hash=sha(ASSETS/'alpha_text/text_features.npy')
    scene_receipt=args.evidence_file.parent/'receipt.json'
    legacy_identity=args.evidence_file.parent.parent/'identity.json'
    if proof is not None:
        provenance=read(proof)
        values={provenance.get(k+'_sha256') for k in ['alpha_clip_base','alpha_clip_checkpoint','sam_checkpoint']}
        if values!=alpha_hashes or provenance.get('text_features_sha256')!=text_hash:
            raise ValueError('Explicit semantic model/text provenance differs')
    elif legacy_identity.exists():
        proof=legacy_identity
        values=set(read(proof).get('cache_inputs',{}).values())
        if not alpha_hashes|{text_hash}<=values:
            raise ValueError('Legacy semantic model/text provenance differs')
    elif scene_receipt.exists() and read(scene_receipt).get('identity',{}).get('version')=='portable_frozen_own_alpha_v1':
        proof=scene_receipt
        prior=read(proof)
        if (prior['identity']['external_weights']!=expected
                or prior['identity']['text_features_sha256']!=text_hash
                or prior['outputs']['evidence.jsonl']!=sha(args.evidence_file)):
            raise ValueError('Portable semantic model/text provenance differs')
    else:
        raise ValueError('Semantic model identity is missing. Supply the original --semantic-provenance model_provenance.json.')
    model_path=ASSETS/'b26/B.joblib'
    inputs={str(p.resolve()):sha(p) for p in [args.geometry_dir/'geometry.json',args.geometry_dir/'points.npz',
                                           args.evidence_file,proof,model_path,ROOT/'reproduction/manifest.json']}
    identity={'scene':scene,'point_count':domain,'inputs':inputs,'configuration':'official100_Dv3_GD10_matched597_B26_K800'}
    dest=args.output
    if (dest/'receipt.json').exists():
        prior=read(dest/'receipt.json')
        if prior['identity']!=identity:
            raise ValueError('Score input identity changed')
        for name,h in prior['outputs'].items():
            if sha(dest/name)!=h:
                raise ValueError('Frozen prediction changed')
        return
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Incomplete score output retained: '+str(dest))
    model=joblib.load(model_path)
    members,x26=features_and_members(geometries,points,by_key,domain)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        selected,scores=learned_selection(members,x26,model)
    dest.mkdir(parents=True,exist_ok=True)
    write(dest/'selected.json',selected)
    np.savez_compressed(dest/'features_scores.npz',X26=x26,scores=scores)
    unique=sorted({m['geometry_key'] for m in selected})
    masks={f'g{i:05d}':points[k] for i,k in enumerate(unique)}
    np.savez_compressed(dest/'selected_points.npz',**masks)
    write(dest/'point_keys.json',dict(zip(unique,masks)))
    outputs={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
    write(dest/'receipt.json',{'complete':True,'identity':identity,'outputs':outputs,
        'members_before_budget':len(members),'selected':len(selected),'budget':800,'GT_read':False})
    print(scene,'Full K800 complete:',len(selected),'members',flush=True)


def run(args):
    check_external(include_frontend=not args.frontend_root)
    selection=prepare_selection(args.output)
    requested=selected_scenes(args)
    write(args.output/'run_contract.json',{'scenes':requested,'dataset_root':str(args.dataset_root),
        'release_manifest_sha256':sha(ROOT/'reproduction/manifest.json'),'config':'official100_Dv3_GD10_B26_K800',
        'visual_python':str(args.visual_python),'geometry_python':str(args.geometry_python),
        'backend_python':str(args.backend_python),'frontend_root':str(args.frontend_root) if args.frontend_root else None,
        'new_training':False,'GT_read':False})
    config=ROOT/'pretrained/config_scannet200.yaml'
    for scene in requested:
        front=args.frontend_root or args.output/'frontend'
        if args.frontend_root:
            validate_cached_frontend(front,scene,args.dataset_root)
        folder=args.output/'scenes'/scene
        folder.mkdir(parents=True,exist_ok=True)
        scene_list=folder/'scene.txt'
        scene_list.write_text(scene+'\n')
        stage=['--dataset-root',args.dataset_root,'--output',args.output,'--scenes',scene]
        launch(args.visual_python,'run_frozen_full800.py','native',*stage)
        sam=['--sam_source',ROOT/'_external/segment-anything/segment-anything-main',
             '--sam_checkpoint',ROOT/'pretrained/checkpoints/sam_vit_b_01ec64.pth']
        launch(args.visual_python,'export_sam_automatic_observations.py','--scene_list',scene_list,
            '--dataset_root',args.dataset_root,'--config_path',config,'--output_root',args.output/'common/02_auto',
            '--frame_selection','uniform','--save-mask-rle','--save-exact-same-frame-relations','--resume',*sam)
        launch(args.visual_python,'build_details_same_frame_hierarchy_preprocessor.py','--scene-list',scene_list,
            '--automatic-root',args.output/'common/02_auto','--output-root',args.output/'common/03_d1_hierarchy_safe',
            '--policy','hierarchy_safe','--resume')
        launch(args.visual_python,'export_dense_frame_instance_observations.py','--dataset_root',args.dataset_root,
            '--processed_scene_root',args.dataset_root,'--config_path',config,'--scene_split',scene_list,
            '--path_to_3d_masks',args.output/'common/mask3d_masks',
            '--path_to_2d_preds',args.output/'records'/scene/'yoloworld_bboxes_2d',
            '--output_root',args.output/'common/11_yoloworld_sam_uniform30','--max_frames','30',
            '--frame_selection','uniform','--detection_score_th','0.08','--sam_multimask_topk','1','--resume',*sam)
        if not args.frontend_root:
            launch(args.visual_python,'run_frozen_full800.py','frontend',*stage)
        adapters(NS(**{**vars(args),'scenes':[scene]}))
        common=['--selection-root',args.output/'selection','--dataset-root',args.dataset_root,
                '--full-preparation',args.output/'preparation','--yolo-cache-root',args.output/'yolo','--scenes',scene,'--arms','D']
        launch(args.geometry_python,'evaluate_fi1_frontend_comparison.py','full-geometry',*common,
            '--run',front,'--output',folder/'backend','--legacy-root',args.output/'common',
            '--backend-config',config,'--dv3-models',ASSETS/'dv3','--champion-models',ASSETS/'champion')
        launch(args.backend_python,'evaluate_fi1_frontend_comparison.py','full-bind',*common,
            '--backend-root',folder/'backend','--output',folder/'own')
        own=folder/'own/D'
        launch(args.visual_python,'export_fi1_r0_r1_yolo_views.py','--binding',own/'geometry_binding',
            '--sources',own/'sources.json','--snapshot',ROOT,'--output',own/'views')
        launch(args.visual_python,'run_frozen_full800.py','alpha',*stage)
        launch(args.backend_python,'run_frozen_full800.py','score','--scenes',scene,
            '--geometry-dir',own/'geometry_binding'/scene,'--evidence-file',own/'alpha_completed'/scene/'evidence.jsonl',
            '--output',args.output/'predictions'/scene)
    receipts={s:sha(args.output/'predictions'/s/'receipt.json') for s in requested}
    write(args.output/'predictions_frozen.json',{'complete':True,'scenes':requested,'receipts':receipts,
        'full_validation312':requested==selection['scenes'],'GT_read':False,'budget':800})


def validate_cached_frontend(front,scene,dataset):
    from tools.fi1_frontend_protocol import digest
    manifest_path=front/'manifests'/f'{scene}.json'
    manifest=read(manifest_path)
    if manifest['scene_name']!=scene or manifest['arms']['D']!=list(range(0,manifest['original_frame_count'],10)):
        raise ValueError('Cached frontend is not this scene with original GD10 frames')
    if Path(manifest['files']['processed']['path']).parent.parent.resolve()!=dataset.resolve():
        raise ValueError('Cached manifest addresses another data root. Inspect/migrate cache provenance before reuse.')
    contract=read(front/'contract.json')
    expected_weights=read(ROOT/'reproduction/external_weights.json')
    values=set(contract.get('assets',{}).values())
    if not {expected_weights[k]['sha256'] for k in ['sam_checkpoint','groundingdino']}<=values:
        raise ValueError('Cached frontend does not certify the frozen GD/SAM weights')
    for relative in ['tools/run_fi1_frontend_comparison.py','tools/fi1_frontend_protocol.py',
                     'tools/build_details_same_frame_hierarchy_preprocessor.py',
                     'tools/build_details_frame_siou_tracks.py',
                     'tools/refine_details_automatic_tracks_consensus.py',
                     'tools/build_details_iterative_proposal_merges.py']:
        recorded=[h for p,h in contract['assets'].items() if p.endswith('/'+relative)]
        if recorded!=[sha(ROOT/relative)]:
            raise ValueError('Cached frontend kernel differs from the frozen release: '+relative)
    identity=digest({'contract':digest(contract),'manifest':digest(manifest),'frontend':'grounded_sam','arm':'D'})
    for stage in ['hierarchy','tracks','consensus','merged']:
        receipt=read(front/'arms/D/stage_receipts'/f'{scene}_{stage}.json')
        if receipt['identity']!=identity:
            raise ValueError('Cached frontend stage identity differs')
        for name,h in receipt['files'].items():
            if sha(front/'arms/D'/stage/scene/name)!=h:
                raise ValueError('Cached frontend bytes differ')


def inspect(args):
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    patterns={'GD10_tracks':'automatic_tracks.json','bound_geometry':'geometry.json',
              'own_semantics':'evidence.jsonl','YOLO_Alpha_requests':'yolo_and_alpha_requests.jsonl',
              'Alpha_image_cache':'similarities.npy','YOLO_detections':'.pt','native_masks':'_pred_masks.npy'}
    found={name:[] for name in patterns}
    counts={name:0 for name in patterns}
    for folder,dirs,files in os.walk(args.root):
        dirs[:]=[d for d in dirs if d not in {'.git','.venvs','__pycache__','ground_truth'}]
        for filename in files:
            for name,pattern in patterns.items():
                matched=filename.endswith(pattern) if pattern.startswith(('.', '_')) else filename==pattern
                if matched and (name!='YOLO_detections' or 'yolo' in folder.lower() or 'bbox' in folder.lower()):
                    counts[name]+=1
                    if len(found[name])<5:
                        found[name].append(str(Path(folder)/filename))
    print(json.dumps({'root':str(args.root),'counts':counts,'example_paths':found,
        'note':'File inventory only; presence does not certify scene/pointset/model identity. No GT or model inference.'},indent=2))


def seal(args):
    requested=selected_scenes(args)
    hashes={}
    for scene in requested:
        folder=args.predictions_root/scene
        receipt=read(folder/'receipt.json')
        if not receipt['complete'] or receipt['budget']!=800 or receipt['identity']['scene']!=scene:
            raise ValueError('Incomplete or mismatched scene prediction')
        for name,h in receipt['outputs'].items():
            if sha(folder/name)!=h:
                raise ValueError('Frozen prediction bytes changed')
        hashes[scene]=sha(folder/'receipt.json')
    write(args.predictions_root.parent/'predictions_frozen.json',{'complete':True,'scenes':requested,'receipts':hashes,
        'full_validation312':requested==sorted(VAL.read_text().split()),'GT_read':False,'budget':800})


def evaluate(args):
    import numpy as np
    from tools.fi1_r3b_delta_diagnostic_core import evaluator,metrics
    if not args.allow_gt_evaluation:
        raise ValueError('Evaluation requires --allow-gt-evaluation; inference never reads GT.')
    sealed=read(args.predictions_root.parent/'predictions_frozen.json')
    selected_scenes(NS(scenes=sealed['scenes']))
    if not sealed['complete']:
        raise ValueError('All requested predictions must be sealed before GT')
    # Verify the complete prediction set before opening any GT file.
    for scene in sealed['scenes']:
        folder=args.predictions_root/scene
        receipt=read(folder/'receipt.json')
        if sha(folder/'receipt.json')!=sealed['receipts'][scene] or not receipt['complete']:
            raise ValueError('Prediction receipt changed/incomplete')
        for name,h in receipt['outputs'].items():
            if sha(folder/name)!=h:
                raise ValueError('Prediction bytes changed')
    ev=evaluator()
    matches={}
    for scene in sealed['scenes']:
        folder=args.predictions_root/scene
        selected=read(folder/'selected.json')
        keys=read(folder/'point_keys.json')
        domain=read(folder/'receipt.json')['identity']['point_count']
        gt=args.gt_root/(scene+'.txt')
        if len(np.loadtxt(gt,dtype=np.int64))!=domain:
            raise ValueError('GT does not match full point domain')
        masks=np.zeros((domain,len(selected)),dtype=bool)
        with np.load(folder/'selected_points.npz') as packed:
            for i,m in enumerate(selected):
                masks[packed[keys[m['geometry_key']]],i]=True
        gm,pm=ev.assign_instances_for_scan({'pred_masks':masks,
            'pred_classes':np.asarray([m['class_index'] for m in selected],dtype=np.int64),
            'pred_scores':np.asarray([m['score'] for m in selected],dtype=np.float64)},str(gt))
        matches[scene]={'gt':gm,'pred':pm}
    array,*_=ev.evaluate_matches(matches)
    args.output.mkdir(parents=True,exist_ok=False)
    np.save(args.output/'AP.npy',array)
    result={'complete':True,'AP_fraction':metrics(ev,array),'scenes':sealed['scenes'],
            'full_validation312':sealed['full_validation312'],'original_evaluator':True,
            'predictions_frozen_sha256':sha(args.predictions_root.parent/'predictions_frozen.json')}
    write(args.output/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    check=sub.add_parser('verify',help='CPU file/weight checks; no models run')
    check.add_argument('--with-external-assets',action='store_true')
    p=sub.add_parser('inspect',help='Inventory existing G10 and semantic caches without GPU/GT')
    p.add_argument('--root',type=Path,required=True)
    for name in ['run','native','frontend','alpha']:
        p=sub.add_parser(name)
        p.add_argument('--dataset-root',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--scenes',nargs='+')
        if name=='run':
            p.add_argument('--visual-python',type=Path,required=True)
            p.add_argument('--geometry-python',type=Path,required=True)
            p.add_argument('--backend-python',type=Path,default=Path(sys.executable))
            p.add_argument('--frontend-root',type=Path,help='Reuse a complete audited GD10 frontend on this data root')
    p=sub.add_parser('score',help='CPU replay using bound geometry and own semantic evidence')
    p.add_argument('--geometry-dir',type=Path,required=True)
    p.add_argument('--evidence-file',type=Path,required=True)
    p.add_argument('--semantic-provenance',type=Path,help='Original Alpha model_provenance.json if cache identity was archived separately')
    p.add_argument('--point-count',type=int)
    p.add_argument('--scenes',nargs=1,required=True)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('seal',help='Seal all requested CPU cache replays before evaluation')
    p.add_argument('--predictions-root',type=Path,required=True)
    p.add_argument('--scenes',nargs='+')
    p=sub.add_parser('evaluate')
    p.add_argument('--predictions-root',type=Path,required=True)
    p.add_argument('--gt-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-gt-evaluation',action='store_true')
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,name,value.resolve())
    manifest=verify_release()
    if args.command=='verify':
        if args.with_external_assets:
            check_external()
        print(json.dumps({'complete':True,'configuration':manifest['configuration'],'checked_files':len(manifest['files'])}))
        return
    if args.command in ['native','frontend','alpha'] and (not args.scenes or len(args.scenes)!=1):
        parser.error('Internal stages require exactly one --scenes value')
    if args.command not in ['evaluate','inspect']:
        selected_scenes(args)
    globals()[args.command](args)


if __name__=='__main__':
    main()
