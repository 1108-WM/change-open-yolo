#!/usr/bin/env python3
"""Inventory val312 inputs and verify frozen inference, without GT or fitting."""
import argparse
from collections import Counter,defaultdict
import json
import hashlib
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from tools.prepare_fi1_r0_r1_inputs import read,write,sha,digest,load_partition,main_scene
from tools.fi1_r0_r1_protocol import PROTOCOL,build_pair
from tools.fi1_r2_core import FEATURES,MODEL_PARAMS,geometry_features,select
from tools.fi1_r3_core import EXTRA_FEATURES,pair_overlap,context_features,independent_context

BASE=ROOT/'.local/fi1_r0_r4_preparation_20260913'
HISTORY=Path('/media/jia/软件1/scannet_validation_stream/fi1_d_v3_dm_sms1_val312_local_20260902')
DATA=Path('/home/jia/Wm/Dataset/scannet200')
SCENES=ROOT/'models/Mask3D/third_party/ScanNet/Tasks/Benchmark/scannetv2_val.txt'
PROTOCOL_FILE=ROOT/'docs/FI1_R3B_VAL312_PROTOCOL_20260914.md'


def require(ok,message):
    if not ok:raise ValueError(message)


def verify(files):
    for path,h in files.items():require(sha(path)==h,'changed bound file '+str(path))


def no_gt_io():
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=str(Path(os.fsdecode(args[0])).resolve())
        if any(s in path for s in ['ground_truth','.labels.ply','.aggregation.json','.segs.json']):
            raise PermissionError('preparation must not open GT payload: '+path)
    sys.addaudithook(hook)


def rows(path):
    with Path(path).open() as f:
        for line in f:yield json.loads(line)


def overlaps(scans,other):
    for s in [*scans,*other]:main_scene(s)
    require(len(set(scans))==len(scans) and len(set(other))==len(other),'duplicate scan identity')
    a={main_scene(s) for s in scans};b={main_scene(s) for s in other}
    return {'scans':sorted(set(scans)&set(other)),'main_scenes':sorted(a&b)}


def alpha_plan_hash(plan):
    """Legacy Alpha keys use SHA1 even when the refined plan uses SHA256."""
    algorithm=plan['geometry_digest_algorithm']
    if algorithm=='sha1_point_indices':return plan['geometry_digest']
    require(algorithm=='sha256_point_indices','unknown plan digest algorithm')
    loc=plan['geometry_locator_read_only'];require(loc['kind']=='point_indices_npz','unexpected SHA256 locator')
    with np.load(loc['points_path'],allow_pickle=False) as a:pts=a[loc.get('array_key','point_indices')]
    require(pts.ndim==1 and pts.dtype.kind in 'iu' and np.array_equal(pts,np.unique(pts)),'invalid refined point order')
    raw=np.asarray(pts,dtype='<i8').tobytes()
    require(hashlib.sha256(raw).hexdigest()==plan['geometry_digest'] and len(pts)==plan['point_count'],'refined digest/count differs')
    return hashlib.sha1(raw).hexdigest()


def inspect_yolo(payload,scene,labels):
    """A bare dictionary is readable evidence, not a certified vocabulary binding."""
    import torch
    require(isinstance(payload,dict) and bool(payload),'empty/non-dictionary YOLO cache')
    metadata=payload.get('metadata');pred=payload.get('predictions') if metadata is not None else payload
    require(isinstance(pred,dict) and bool(pred),'invalid frame dictionary')
    ids=[];boxes=0
    for frame,item in pred.items():
        require(str(frame).isdigit() and isinstance(item,dict),'invalid frame identity')
        ids.append(int(frame));bb=item.get('bbox');ll=item.get('labels');ss=item.get('scores')
        require(all(torch.is_tensor(x) for x in (bb,ll,ss)),'YOLO arrays must be tensors')
        require(bb.ndim==2 and bb.shape[1]==4 and ll.ndim==ss.ndim==1 and len(bb)==len(ll)==len(ss),'YOLO shape mismatch')
        require(ll.dtype in (torch.int32,torch.int64) and bool(((ll>=0)&(ll<len(labels))).all()),'YOLO label range differs')
        require(bool(torch.isfinite(bb).all()) and bool(torch.isfinite(ss).all()),'nonfinite YOLO values')
        require(bool((bb[:,2:]>=bb[:,:2]).all()),'inverted bbox')
        boxes+=len(bb)
    require(len(ids)==len(set(ids)),'duplicate numeric frames')
    if metadata is not None:
        require(metadata.get('scene_name')==scene and metadata.get('text_prompts')==labels,'YOLO metadata scene/vocabulary differs')
    return {'frame_count':len(ids),'bbox_count':boxes,'frame_ids':sorted(ids),
            'format':'wrapped' if metadata is not None else 'legacy_bare_frames',
            'current_exporter_metadata_present':metadata is not None,
            'historical_vocabulary_identity_verified':metadata is not None}


def load_models(base):
    r2=read(base/'r2_v1/fit/summary.json');r3=read(base/'r3_v1/fit_summary.json')
    require(r2['complete'] and r3['complete'],'models not complete')
    p2=base/'r2_v1/fit/models/full200.joblib';p3=base/'r3_v1/models/R3b_full200.joblib'
    t2=next(r for r in r2['rows'] if r['model']=='full200')
    t3=next(r for r in r3['rows'] if r['name']=='R3b_full200')
    part=load_partition(base/'approved_split_v1')
    require(t2['fit_scenes']==t3['fit_scenes']==part['roles']['train'],'full200 train identity differs')
    files={str(p2):t2['model_sha256'],str(p3):t3['model_sha256']}
    for p in [base/'r2_v1/fit/summary.json',base/'r3_v1/fit_summary.json',base/'r3_v1/identity.json']:files[str(p)]=sha(p)
    # The inference kernels still match the code actually used to train R3b.
    old=read(base/'r3_v1/identity.json')['code']
    for name in ['fi1_r2_core.py','fi1_r3_core.py']:
        p=ROOT/'tools'/name;files[str(p)]=old[str(p)]
    verify(files);models={'R2':joblib.load(p2),'R3b':joblib.load(p3)}
    for arm,m in models.items():
        require(m.n_features_in_==(20 if arm=='R2' else 26) and m.n_iter_==100,'wrong fitted model dimensions/iterations')
        require(all(m.get_params()[k]==v for k,v in MODEL_PARAMS.items()),'frozen model hyperparameters differ')
    return models,files


def score_scene(geometry,point_sets,evidence,domain,models):
    """Shared GT-free deployment kernel: original evidence -> R0/R1/R2/R3b."""
    require(set(models)=={'R2','R3b'},'missing/extra deployment model')
    pair=build_pair(geometry,evidence,PROTOCOL['scene_budget']);members=pair['R1']['eligible']
    require(bool(members),'empty R1 member set')
    keys=[g['geometry_key'] for g in geometry]
    require(len(keys)==len(set(keys))==len(point_sets),'geometry coverage differs')
    for g,pts in zip(geometry,point_sets):
        require(g['point_count']==len(pts),'point count differs')
        require(g['points_sha256']==hashlib.sha256(np.asarray(pts,dtype='<i8').tobytes()).hexdigest(),'point hash differs')
    gx={g['geometry_key']:geometry_features(g,evidence[g['geometry_key']]) for g in geometry}
    own=np.asarray([gx[m['geometry_key']][m['class_index']] for m in members],dtype=np.float64)
    iou,coverage=pair_overlap(point_sets,domain);cx=context_features(members,keys,iou,coverage)
    require(np.allclose(cx,independent_context(members,keys,iou,coverage),rtol=0,atol=1e-14),'independent context differs')
    outputs={a:pair[a] for a in ['R0','R1']}
    for arm,x in [('R2',own),('R3b',np.column_stack((own,cx)))]:
        require(x.shape==(len(members),20 if arm=='R2' else 26),'wrong feature matrix')
        require(models[arm].n_features_in_==x.shape[1],'wrong model inputs')
        scores=np.clip(models[arm].predict(x),0,1)
        selected=select(members,scores)
        independent=sorted([{**m,'score':float(v)} for m,v in zip(members,scores)],key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:600]
        require(selected==independent,'selection replay differs')
        outputs[arm]={'scores':scores,'selected':selected}
    return outputs


def smoke(base,out):
    source_sha=sha(__file__)
    require(not out.exists(),'refusing existing smoke output')
    models,files=load_models(base);reports=read(base/'r2_v1/dataset/summary.json')['rows']
    # First and last dev scenes are fixed by scan identity, not AP.
    part=load_partition(base/'approved_split_v1');scans=[part['roles']['development_validation'][i] for i in (0,-1)]
    results=[]
    with threadpool_limits(limits=4):
        for s in scans:
            r=next(r for r in reports if r['scene_name']==s)
            geometry=read(r['geometry_file']);pointpath=Path(r['points_file'])
            evidencepath=base/'alpha_completed_v1/ncs_validation60'/s/'evidence.jsonl'
            er=read(evidencepath.parent/'receipt.json');verify({str(evidencepath):er['outputs']['evidence.jsonl']})
            verify({r['geometry_file']:r['geometry_sha256'],r['points_file']:r['points_sha256'],r['R1_eligible']:r['R1_eligible_sha256']})
            files.update({r['geometry_file']:r['geometry_sha256'],r['points_file']:r['points_sha256'],str(evidencepath):er['outputs']['evidence.jsonl']})
            with np.load(pointpath,allow_pickle=False) as pts:points=[pts[g['array_key']] for g in geometry]
            source_receipt=read(base/'r2_v1/dataset'/s/'receipt.json')
            domainfiles=[Path(p) for p in source_receipt['inputs'] if p.endswith('/'+s[5:]+'.npy')];require(len(domainfiles)==1,'ambiguous domain')
            verify({str(domainfiles[0]):source_receipt['inputs'][str(domainfiles[0])]})
            domain=len(np.load(domainfiles[0],mmap_mode='r',allow_pickle=False))
            ev={x['geometry_key']:x for x in rows(evidencepath)};result=score_scene(geometry,points,ev,domain,models)
            require(result['R1']['eligible']==list(rows(r['R1_eligible'])),'deployed R1 members differ')
            for arm,folder,receipt_name in [('R2',base/'r2_v1/fit/predictions'/s,'receipt.json'),('R3b',base/'r3_v1/predictions'/s,'R3b_receipt.json')]:
                rr=read(folder/receipt_name);verify({str(folder/p):h for p,h in rr['outputs'].items()})
                old=np.load(folder/f'{arm}_eligible_scores.npy',allow_pickle=False)
                require(np.array_equal(result[arm]['scores'],old),'deployed scores differ from frozen '+arm)
                require(result[arm]['selected']==list(rows(folder/f'{arm}_selected.jsonl')),'deployed selected members differ')
            results.append({'scene_name':s,'geometry_count':len(geometry),'R1_members':len(result['R1']['eligible']),'R2_R3b_scores_exact':True,'selected_members_order_aliases_exact':True})
    require(sha(__file__)==source_sha,'preparation source changed during smoke')
    out.mkdir(parents=True);write(out/'summary.json',{'complete':True,'scenes':results,'inputs':files,'new_training':False,'new_val312_predictions':False,'new_AP':False,'GT_read':False,'code':{str(Path(__file__).resolve()):source_sha}})
    print('deployment smoke passed',scans,flush=True)


def inventory(base,out):
    import torch,yaml
    source_sha=sha(__file__)
    require(not out.exists(),'refusing existing inventory')
    out.mkdir(parents=True);files={}
    def bind(path,expected=None):
        path=Path(path).resolve();h=sha(path)
        if expected is not None:require(h==expected,'historical receipt differs '+str(path))
        files[str(path)]=h;return read(path) if path.suffix=='.json' else path
    source=read(base/'input_sources_v1.json');configpath=Path(source['config_path']);config=yaml.safe_load(configpath.read_text());labels=config['network2d']['text_prompts']
    require(len(labels)==198 and len(set(labels))==198 and config['openyolo3d']['frequency']==10 and config['openyolo3d']['topk']==40,'frozen vocabulary/settings differ')
    bind(configpath);bind(PROTOCOL_FILE);bind(SCENES)
    scans=sorted(SCENES.read_text().split());require(len(scans)==len(set(scans))==312,'not official312')
    for s in scans:main_scene(s)
    c=HISTORY/'class_agnostic_repro_20260902';d=HISTORY/'dm_sms1_4090_repro_20260902';alpha=d/'05_alpha_embedding_ledger'
    lp=c/'15_legacy_unique_geometry/unique_geometry_ledger.jsonl';plan=c/'17_fi1_d_v3_frozen_inference/complete_plan/complete_inference_ap_plan.jsonl'
    ins=bind(c/'17_fi1_d_v3_frozen_inference/summary.json');ia=bind(c/'18_fi1_d_v3_frozen_inference_audit/summary.json')
    require(ia['audit_valid'] and ia['error_count']==0 and ia['scene_count']==312,'D-v3 audit invalid')
    bind(c/'17_fi1_d_v3_frozen_inference/summary.json',ia['input_provenance']['inference_summary_sha256']);bind(plan,ia['input_provenance']['plan_sha256'])
    modelsummary=Path(source['dv3_model_root'])/'summary.json';ms=bind(modelsummary,ins['model_package_sha256'])
    for m in ms['models']:bind(modelsummary.parent/m['model_file'],m['model_sha256'])
    ls=bind(c/'15_legacy_unique_geometry/summary.json');bind(lp)
    ds=bind(d/'01_unique_geometry/summary.json');bind(lp,ds['input_provenance']['legacy_unique_geometry_ledger_sha256']);bind(plan,ds['input_provenance']['fi1_d_v3_complete_plan_sha256'])
    asc=bind(alpha/'summary.json');mp=bind(alpha/'model_provenance.json',asc['input_provenance']['model_provenance_sha256'])
    prompts=bind(alpha/'text_prompts.json',mp['text_prompts_sha256']);bind(alpha/'text_features.npy',mp['text_features_sha256'])
    require(prompts['class_names']==labels and asc['class_count']==198 and asc['stage_complete'],'Alpha vocabulary/completion differs')
    for key in ['alpha_clip_base','alpha_clip_checkpoint','sam_checkpoint']:bind(mp[key],mp[key+'_sha256'])
    # Compare deployed Alpha weights/text to the train/dev evidence package.
    training_alpha=read(base/'alpha_completed_v1/ncs_validation60/identity.json')
    cache=Path(training_alpha['cache'])
    trainmp=bind(cache/'model_provenance.json')
    for key in ['alpha_clip_base_sha256','alpha_clip_checkpoint_sha256','sam_checkpoint_sha256','text_features_sha256','text_prompts_sha256']:
        require(mp[key]==trainmp[key],'Alpha differs from training package '+key)
    legacy=defaultdict(list);plans=defaultdict(list);locations={}
    for r in rows(lp):
        require(r['member_count']==len(r['members']),'legacy member counts differ');legacy[r['scene_name']].append(r)
        for loc in [r['canonical_geometry_locator'],*[m['geometry_locator'] for m in r['members']]]:
            for key in ['masks_path','points_path']:
                if key in loc:locations[loc[key]]=Path(loc[key]).is_file()
    for r in rows(plan):plans[r['scene_name']].append(r)
    require(set(legacy)==set(plans)==set(scans),'plan/legacy scan coverage differs')
    require(sum(len(v) for v in legacy.values())==ls['unique_geometry_count'] and sum(len(v) for v in plans.values())==ins['challenger_candidate_count'],'geometry/plan counts differ')
    part=load_partition(base/'approved_split_v1');dependency=bind(base/'upstream_direct_dependency_v1.json')
    direct=sorted({s for row in dependency['dependencies'] for s in row['scenes']}) if 'dependencies' in dependency else sorted({s for row in dependency['direct_input_files'] for s in row['scenes']})
    intersection={k:overlaps(scans,v) for k,v in part['roles'].items()};intersection['D_v3_direct_supervision']=overlaps(scans,direct)
    _,model_files=load_models(base);files.update(model_files)
    records=[];missing=[];expected_arrays={};counts=Counter()
    for scene in scans:
        data=DATA/scene;native=c/'01_native_cache';ar=alpha/'scenes'/scene
        paths={'xyz':data/f'{scene[5:]}.npy','mesh':data/f'{scene}_vh_clean_2.ply','intrinsics':data/'intrinsics.txt',
          'native_masks':native/f'{scene}_pred_masks.npy','native_classes':native/f'{scene}_pred_classes.npy','native_scores':native/f'{scene}_pred_scores.npy',
          'yolo':ROOT/'output/scannet200/bboxes_2d'/f'{scene}.pt','alpha_summary':ar/'summary.json'}
        absent=[str(p) for p in paths.values() if not p.is_file()];missing.extend(absent)
        row={'scene_name':scene,'files':{k:str(p) for k,p in paths.items()},'missing':absent,'legacy_geometries':len(legacy[scene]),'legacy_members':sum(r['member_count'] for r in legacy[scene]),'planned_candidates':len(plans[scene])}
        if absent:records.append(row);continue
        xyz=np.load(paths['xyz'],mmap_mode='r',allow_pickle=False);masks=np.load(paths['native_masks'],mmap_mode='r',allow_pickle=False);classes=np.load(paths['native_classes'],allow_pickle=False);scores=np.load(paths['native_scores'],allow_pickle=False)
        require(xyz.ndim==2 and xyz.shape[1]>=3 and masks.shape==(len(xyz),len(classes)) and classes.shape==scores.shape==(600,),'native point/column domain differs '+scene)
        require(np.isfinite(scores).all(),'nonfinite native scores')
        row['point_count']=len(xyz);row['native_columns']=len(classes)
        yi=inspect_yolo(torch.load(paths['yolo'],map_location='cpu'),scene,labels);bind(paths['yolo']);row['yolo']=yi;counts[yi['format']]+=1
        rgb={p.stem for p in (data/'color').glob('*') if p.is_file()};depth={p.stem for p in (data/'depth').glob('*') if p.is_file()};poses={p.stem for p in (data/'poses').glob('*') if p.is_file()}
        needed={str(i) for i in yi['frame_ids']};row['sampled_frames_missing']={'rgb':sorted(needed-rgb),'depth':sorted(needed-depth),'poses':sorted(needed-poses)}
        require(not any(row['sampled_frames_missing'].values()),'sampled RGB-D/pose incomplete '+scene)
        ar_summary=bind(ar/'summary.json');require(ar_summary['scene_complete'] and ar_summary['scene_name']==scene,'Alpha scene incomplete')
        recpath=ar/'records.jsonl';bind(recpath,ar_summary['records_sha256']);ars=list(rows(recpath))
        require(len(ars)==ar_summary['geometry_count'] and len({r['geometry_hash'] for r in ars})==len(ars),'Alpha geometry duplicates/count mismatch')
        # Compare plan geometry identities without applying old SMS deletion flags.
        planned={alpha_plan_hash(r) for r in plans[scene]}
        for r in plans[scene]:
            if r['geometry_digest_algorithm']=='sha256_point_indices':bind(r['geometry_locator_read_only']['points_path'])
        require(planned=={r['geometry_hash'] for r in ars},'Alpha/plan geometry coverage differs')
        row['alpha_geometries']=len(ars);row['alpha_invalid']=ar_summary['alpha_invalid_count'];row['alpha_views']=ar_summary['selected_view_count']
        for fname,hkey,shape in [('aggregate_features.npy','aggregate_features_sha256',(len(ars),768)),('similarities.npy','similarities_sha256',(len(ars),198)),('scale_features.npy','scale_features_sha256',(ar_summary['scale_feature_count'],768))]:
            p=ar/fname;require(p.is_file(),'missing Alpha array '+str(p));a=np.load(p,mmap_mode='r',allow_pickle=False);require(a.shape==shape,'Alpha array shape differs');expected_arrays[str(p)]={'expected_sha256':ar_summary[hkey],'bytes':p.stat().st_size,'shape':list(a.shape),'hash_rechecked_this_inventory':False}
        row['alpha_reuse_status']='candidate_only_pending_exact_geometry_view_crop_and_aggregation_checks'
        row['GT_path_exists_only']=(DATA/'ground_truth'/f'{scene}.txt').is_file();counts['GT_paths_present']+=row['GT_path_exists_only'];records.append(row)
        if len(records)%40==0:print('inventoried',len(records),'/312',flush=True)
    write(out/'scene_inventory.json',records);write(out/'expected_alpha_arrays.json',expected_arrays);write(out/'intersections.json',intersection)
    (out/'val312.txt').write_text(''.join(s+'\n' for s in scans))
    spec={'config_path':str(configpath),'deployment_code_root':source['deployment_code_root'],'dv3_model_root':source['dv3_model_root'],
      'sources':{'val312':{'prepared_root':str(DATA),'records_root':str(out.parent/'records_adapter'),'legacy_geometry_root':str(c/'15_legacy_unique_geometry'),
        'plan':str(plan),'native_cache_root':str(c/'01_native_cache'),'yolo_root':str(ROOT/'output/scannet200/bboxes_2d'),'alpha_cache':str(alpha)}}}
    write(out/'input_sources.json',spec)
    require(not missing and all(locations.values()),'missing required assets; inventory retained')
    summary={'complete':True,'inventory_complete':True,'ready_for_full_inference':False,'scene_count':312,'main_scene_count':len({main_scene(s) for s in scans}),
      'models_verified':True,'D_v3_package_matches_training':True,'alpha_model_and_vocabulary_match_training':True,
      'legacy_geometry_count':sum(r['legacy_geometries'] for r in records),'legacy_member_count':sum(r['legacy_members'] for r in records),
      'planned_candidate_count':sum(r['planned_candidates'] for r in records),'unique_alpha_geometry_count':sum(r['alpha_geometries'] for r in records),
      'native_columns':sum(r['native_columns'] for r in records),'alpha_invalid_geometries':sum(r['alpha_invalid'] for r in records),
      'counts':dict(counts),'missing_files':missing,'locator_files_found':len(locations),'intersections':intersection,
      'pending':['pointwise geometry and original native alias binding','legacy YOLO identity recovery or frozen-config re-extraction','own YOLO votes and Alpha requests','Alpha exact view/crop checks and aggregation replay; repair incompatible cases','full312 frozen scoring and prediction audit','separate GT-bound AP evaluation'],
      'historical_route':'4090 local input lineage; do not mix A100 assets or metrics','earlier_transitive_supervision':'not fully audited; direct-overlap result is not full independence proof',
      'input_files':files,'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'code':{str(Path(__file__).resolve()):sha(__file__)},
      'GT_content_read':False,'new_training':False,'new_visual_inference':False,'new_val312_predictions':False,'new_AP':False}
    require(sha(__file__)==source_sha,'preparation source changed during inventory')
    write(out/'summary.json',summary);print('inventory complete',summary['counts'],flush=True)


def bind_geometry(inventory_root,out,scenes=None):
    """Use the existing pointwise auditor with a local native-path adapter."""
    from tools.audit_fi1_r0_r1_geometry import audit_scene,grouped
    require(not out.exists(),'refusing existing geometry output')
    inv=read(inventory_root/'summary.json');require(inv['inventory_complete'],'inventory incomplete')
    verify(inv['input_files']);verify({str(inventory_root/k):h for k,h in inv['outputs'].items()})
    configpath=inventory_root/'input_sources.json';config=read(configpath);spec=config['sources']['val312']
    full=(inventory_root/'val312.txt').read_text().split();scans=sorted(scenes or full)
    require(len(scans)==len(set(scans)) and set(scans)<=set(full),'invalid geometry subset')
    legacy=grouped(Path(spec['legacy_geometry_root'])/'unique_geometry_ledger.jsonl');plans=grouped(spec['plan'])
    adapter=Path(spec['records_root']);adapter.mkdir(parents=True,exist_ok=True)
    for scene in scans:
        target=adapter/scene/'native_cache';target.mkdir(parents=True,exist_ok=True)
        for suffix in ['masks','classes','scores']:
            src=Path(spec['native_cache_root'])/f'{scene}_pred_{suffix}.npy';dest=target/src.name
            if dest.is_symlink():require(dest.resolve()==src.resolve(),'adapter target differs')
            else:require(not dest.exists(),'adapter would overwrite existing file');dest.symlink_to(src)
    identity={'version':'fi1_r3b_val312_geometry_binding_v1','partition_id':digest({'evaluation_only_scans':full}),
        'sources_sha256':sha(configpath),'source':'val312','scenes':scans,'subset_smoke':scans!=full,
        'inventory_summary_sha256':sha(inventory_root/'summary.json'),
        'code':{str(Path(__file__).resolve()):sha(__file__),str(ROOT/'tools/audit_fi1_r0_r1_geometry.py'):sha(ROOT/'tools/audit_fi1_r0_r1_geometry.py')}}
    out.mkdir(parents=True);write(out/'identity.json',identity);results=[]
    for scene in scans:
        geometry,points,files,summary=audit_scene(scene,spec,legacy[scene],plans[scene])
        dest=out/scene;dest.mkdir();write(dest/'geometry.json',geometry);np.savez_compressed(dest/'points.npz',**points)
        summary.update(role='official_validation',source='val312')
        write(dest/'receipt.json',{'identity_sha256':digest(identity),'input_files':files,
              'output_files':{p.name:sha(p) for p in dest.iterdir()},'summary':summary})
        results.append(summary);print('pointwise geometry bound',scene,summary['unique_geometry_count'],flush=True)
    verify(identity['code']);write(out/'summary.json',{'complete':True,'identity':identity,'scene_count':len(results),'rows':results,
         'totals':{k:sum(r[k] for r in results) for k in ['legacy_geometry_count','legacy_member_count','native_member_count','dv3_plan_count','refined_plan_count','unique_geometry_count','novel_refined_geometry_count']},
         'ground_truth_read':False,'ap_computed':False,'new_visual_inference':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['inventory','smoke','bind']);p.add_argument('--base',type=Path,default=BASE);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path);p.add_argument('--scenes',nargs='+');a=p.parse_args()
    no_gt_io()
    if a.stage=='inventory':inventory(a.base.resolve(),a.output.resolve())
    elif a.stage=='smoke':smoke(a.base.resolve(),a.output.resolve())
    else:
        require(a.inventory is not None,'--inventory required for bind')
        bind_geometry(a.inventory.resolve(),a.output.resolve(),a.scenes)


if __name__=='__main__':main()
