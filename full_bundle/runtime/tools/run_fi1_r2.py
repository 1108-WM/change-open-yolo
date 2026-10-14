#!/usr/bin/env python3
"""Prepare, train and evaluate the preregistered fixed-member R2 baseline.

Each stage creates a separate directory, verifies its dependencies, and leaves
historical predictions untouched. Training labels exist only for train200.
"""
import argparse
from collections.abc import Mapping
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.prepare_fi1_r0_r1_inputs import read, write, sha, digest, load_partition, main_scene
from tools.fi1_r0_r1_protocol import PROTOCOL
from tools.fi1_r2_core import DETAILS, FEATURES, MODEL_PARAMS, geometry_features, geometry_targets, valid_instances, scene_weights, select
from tools.evaluate_fi1_r0_r1_gt import reconstruct_gt, prediction_geometry_paths, verified
from tools.prepare_scannet200_train_scene import DEFAULT_LABEL_MAP, _ply_arrays, _raw_category_to_id
from evaluate.scannet200 import eval_semantic_instance as evaluator

CLASS_IDS = [int(evaluator.PRED_ID_TO_ID[i]) for i in range(198)]
RAW_ROOT = Path('/media/jia/软件1/scannet_train_stream/scans')
DEV_GT = Path('/media/jia/软件1/scannet_validation_stream/dm_sms1_ncs_validation60_20260820/ground_truth_once_20260822')


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def save_rows(path, records):
    with Path(path).open('x') as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True, allow_nan=False)+'\n')


def code_files():
    return {str(p): sha(p) for p in [Path(__file__).resolve(), ROOT/'tools/fi1_r2_core.py',
        ROOT/'tools/fi1_r0_r1_protocol.py', ROOT/'tools/evaluate_fi1_r0_r1_gt.py',
        ROOT/'tools/prepare_scannet200_train_scene.py', ROOT/'tools/prepare_fi1_r0_r1_inputs.py',
        Path(evaluator.__file__), ROOT/'docs/FI1_R2_PREREGISTRATION_20260913.md']}


def context(base, source, partition):
    version = 'v2' if source == 'official100' else 'v1'
    pred = base/f'r0_r1_predictions_{version}'/source
    s = read(pred/'summary.json')
    audit = base/'code_audit_v1'/('official100_v2' if source == 'official100' else source)/'summary.json'
    a = read(audit)
    if not a['audit_valid'] or a['prediction_summary_sha256'] != sha(pred/'summary.json'):
        raise ValueError('independent prediction audit missing or stale')
    ev = Path(s['evidence_root']); ei = read(ev/'identity.json')
    vi = read(Path(ei['views'])/'identity.json'); binding = Path(vi['binding'])
    bi = read(binding/'identity.json')
    if not s['complete'] or s['smoke'] or s['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('incomplete or changed source predictions')
    if s['evidence_identity_sha256'] != digest(ei) or ei['views_identity_sha256'] != digest(vi) or vi['binding_identity_sha256'] != digest(bi):
        raise ValueError('source identity chain differs')
    if bi['partition_id'] != digest(partition) or bi['sources_sha256'] != sha(base/'input_sources_v1.json'):
        raise ValueError('split/source mismatch')
    scenes = partition['source_scans'][source]
    if scenes != ei['scenes'] or scenes != sorted(r['scene_name'] for r in s['rows']) or scenes != sorted(r['scene_name'] for r in a['rows']):
        raise ValueError('source scan coverage differs')
    names = [evaluator.ID_TO_LABEL[c] for c in CLASS_IDS]
    if names != vi['class_names']:
        raise ValueError('feature classes differ from evaluator')
    return pred, ev, binding, {str(p):sha(p) for p in [pred/'summary.json',audit,ev/'identity.json',
        Path(ei['views'])/'identity.json',binding/'identity.json']}


def materialize_train_gt(scene, processed, output):
    raw = RAW_ROOT/scene
    xyz, _, sem = _ply_arrays(raw/f'{scene}_vh_clean_2.labels.ply', normals=False)
    prepared = np.load(processed, mmap_mode='r', allow_pickle=False)
    if sem is None or not np.array_equal(xyz, prepared[:, :3]):
        raise ValueError('GT/prepared XYZ mismatch')
    seg = np.asarray(read(raw/f'{scene}_vh_clean_2.0.010000.segs.json')['segIndices'])
    if seg.shape != (len(xyz),):
        raise ValueError('GT segment domain mismatch')
    instance = np.full(len(xyz), -1, dtype=np.int64)
    sem = np.asarray(sem, dtype=np.int64).copy()
    categories = _raw_category_to_id(DEFAULT_LABEL_MAP)
    for group in read(raw/f'{scene}.aggregation.json')['segGroups']:
        mask = np.isin(seg, group['segments'])
        instance[mask] = int(group['id'])
        sem[mask] = categories.get(str(group['label']), 0)
    gt = sem*1000+instance+1
    np.savetxt(output, gt, fmt='%d')
    return gt, reconstruct_gt(scene, RAW_ROOT, processed, DEFAULT_LABEL_MAP, output)


def prepare(base, output):
    partition = load_partition(base/'approved_split_v1')
    sources = read(base/'input_sources_v1.json')['sources']
    contexts = {s:context(base,s,partition) for s in partition['source_scans']}
    output.mkdir(parents=True, exist_ok=False)
    (output/'ground_truth_train').mkdir()
    identity = {'protocol': DETAILS, 'partition_id': digest(partition), 'partition': partition,
        'code':code_files(), 'class_index_to_dataset_id':CLASS_IDS,
        'input_files':{str(base/'input_sources_v1.json'):sha(base/'input_sources_v1.json')},
        'source_files':{s:c[3] for s,c in contexts.items()}, 'ground_truth_roles':['train'],
        'numpy_version':np.__version__, 'sklearn_version':sklearn.__version__}
    write(output/'identity.json',identity)
    reports = []
    for source,(pred,ev,binding,_) in contexts.items():
        for scene in partition['source_scans'][source]:
            role = next(k for k,v in partition['roles'].items() if scene in v)
            target = output/scene; target.mkdir()
            receipt = read(pred/scene/'receipt.json')
            er = read(ev/scene/'receipt.json')
            if receipt['evidence_receipt_sha256'] != sha(ev/scene/'receipt.json') or er['identity_sha256'] != digest(read(ev/'identity.json')):
                raise ValueError('scene evidence identity differs')
            paths = prediction_geometry_paths(receipt,binding,scene)
            processed = Path(sources[source]['prepared_root'])/scene/f'{scene[5:]}.npy'
            binding_receipt = read(binding/scene/'receipt.json')
            if str(processed) not in binding_receipt['input_files']:
                raise ValueError('processed coordinates lack geometry provenance')
            inputs = {str(pred/scene/k):v for k,v in receipt['files'].items()}
            inputs[str(processed)] = binding_receipt['input_files'][str(processed)]
            inputs.update({str(ev/scene/k):v for k,v in er['outputs'].items()})
            inputs.update({str(paths['geometry_file']):receipt['geometry_sha256'],str(paths['points_file']):receipt['points_sha256']})
            verified(inputs)
            members = rows(pred/scene/'R1_eligible.jsonl')
            if len({(r['geometry_key'],r['class_index']) for r in members}) != len(members):
                raise ValueError('duplicate R1 member')
            geometries = read(paths['geometry_file']); evidence = {r['geometry_key']:r for r in rows(ev/scene/'evidence.jsonl')}
            gx = {g['geometry_key']:geometry_features(g,evidence[g['geometry_key']]) for g in geometries}
            x = np.asarray([gx[r['geometry_key']][r['class_index']] for r in members], dtype=np.float64)
            if not np.allclose(x[:,0], [r['score'] for r in members], atol=1e-14, rtol=0):
                raise ValueError('R2 direct feature differs from frozen R1 score')
            arrays = {'X':x, 'weights':scene_weights(members)}
            gt_receipt = None
            if role == 'train':
                processed = Path(sources[source]['prepared_root'])/scene/f'{scene[5:]}.npy'
                gt, gt_receipt = materialize_train_gt(scene,processed,output/'ground_truth_train'/f'{scene}.txt')
                instances = valid_instances(gt,CLASS_IDS)
                with np.load(paths['points_file'],allow_pickle=False) as points:
                    gy = {g['geometry_key']:geometry_targets(points[g['array_key']],gt,instances,CLASS_IDS) for g in geometries}
                arrays['y'] = np.asarray([gy[r['geometry_key']][r['class_index']] for r in members])
                inputs.update(gt_receipt['files'])
            np.savez(target/'features.npz',**arrays)
            report = {'scene_name':scene,'source':source,'role':role,'member_count':len(members),
                'eligible_geometry_count':len({r['geometry_key'] for r in members}),
                'R1_eligible':str(pred/scene/'R1_eligible.jsonl'), 'R1_eligible_sha256':sha(pred/scene/'R1_eligible.jsonl'),
                'geometry_file':str(paths['geometry_file']),'points_file':str(paths['points_file']),
                'geometry_sha256':receipt['geometry_sha256'],'points_sha256':receipt['points_sha256'],
                'valid_gt_count':len(instances) if role=='train' else None,
                'positive_target_count':int((arrays['y']>0).sum()) if role=='train' else None,
                'gt_binding':gt_receipt,'complete':True}
            write(target/'receipt.json',{'identity_sha256':digest(identity),'inputs':inputs,
                'outputs':{'features.npz':sha(target/'features.npz')},'summary':report})
            reports.append(report)
            print('prepared',scene,role,len(members),'members',flush=True)
    write(output/'summary.json',{'complete':True,'scene_count':len(reports),'rows':reports,
        'identity_sha256':digest(identity),'train_rows':sum(r['member_count'] for r in reports if r['role']=='train'),
        'train_labels_only':True,'new_training':False,'new_ap':False})


def load_dataset(dataset):
    identity = read(dataset/'identity.json'); summary = read(dataset/'summary.json')
    if not summary['complete'] or summary['identity_sha256'] != digest(identity) or identity['protocol'] != DETAILS:
        raise ValueError('dataset identity/protocol incomplete or changed')
    verified(identity['code']); verified(identity['input_files'])
    for files in identity['source_files'].values(): verified(files)
    result = {}
    for report in summary['rows']:
        scene = report['scene_name']; receipt = read(dataset/scene/'receipt.json')
        if receipt['identity_sha256'] != digest(identity) or receipt['summary'] != report:
            raise ValueError('dataset scene identity differs')
        verified({str(dataset/scene/k):h for k,h in receipt['outputs'].items()})
        verified({report['R1_eligible']:report['R1_eligible_sha256'],report['geometry_file']:report['geometry_sha256'],report['points_file']:report['points_sha256']})
        with np.load(dataset/scene/'features.npz',allow_pickle=False) as a:
            if set(a.files) != ({'X','weights','y'} if report['role']=='train' else {'X','weights'}):
                raise ValueError('label role leak or missing training target')
            result[scene] = {k:a[k] for k in a.files}
        if result[scene]['X'].shape != (report['member_count'],20) or not np.isclose(result[scene]['weights'].sum(),1):
            raise ValueError('feature shape/scene weight differs')
    return identity, summary, result


def infer_scene(model, scene, data, report, output, model_path):
    target = output/scene; target.mkdir()
    members = rows(report['R1_eligible'])
    scores = np.clip(model.predict(data['X']),0,1).astype(np.float64)
    selected = select(members,scores)
    np.save(target/'R2_eligible_scores.npy',scores)
    save_rows(target/'R2_selected.jsonl',selected)
    native_pre = {(r['geometry_key'],k) for r in members for k in r['native_alias_keys']}
    native_post = {(r['geometry_key'],k) for r in selected for k in r['native_alias_keys']}
    write(target/'receipt.json',{'R1_eligible':report['R1_eligible'],'R1_eligible_sha256':report['R1_eligible_sha256'],
        'model':str(model_path),'model_sha256':sha(model_path),'member_count':len(members),
        'eligible_member_order':'same rows as frozen R1 eligible JSONL; scores array replaces score only',
        'selected_member_count':len(selected),'selected_native_alias_count':len(native_post),
        'budget_evicted_native_alias_count':len(native_pre-native_post),
        'selected_expanded_classes':sum(r['expanded_class'] for r in selected),
        'outputs':{p.name:sha(p) for p in [target/'R2_eligible_scores.npy',target/'R2_selected.jsonl']}})
    return scores


def train(dataset, output, threads):
    identity, summary, data = load_dataset(dataset)
    partition = identity['partition']; reports = {r['scene_name']:r for r in summary['rows']}
    output.mkdir(parents=True,exist_ok=False); (output/'models').mkdir(); (output/'predictions').mkdir()
    write(output/'identity.json',{'dataset':str(dataset),'dataset_summary_sha256':sha(dataset/'summary.json'),
        'protocol':DETAILS,'code':code_files(),'threads':threads,'sklearn_version':sklearn.__version__,
        'numpy_version':np.__version__,'partition_id':identity['partition_id']})
    train_scenes = partition['roles']['train']; folds = partition['training_heldout_folds']
    if sorted(s for fold in folds for s in fold) != train_scenes:
        raise ValueError('heldout folds do not partition train200')
    offset = {}; start=0
    for s in train_scenes:
        stop = start+len(data[s]['y']); offset[s]=(start,stop);start=stop
    X = np.concatenate([data[s]['X'] for s in train_scenes])
    y = np.concatenate([data[s]['y'] for s in train_scenes])
    w = np.concatenate([data[s]['weights'] for s in train_scenes])
    fit_reports=[]
    with threadpool_limits(limits=threads):
        for fold in range(6):
            heldout = folds[fold] if fold<5 else partition['roles']['development_validation']+partition['roles']['diagnostic']
            fit = [s for s in train_scenes if s not in heldout]
            if fold<5 and {main_scene(s) for s in fit}&{main_scene(s) for s in heldout}:
                raise ValueError('downstream fit/heldout main-scene overlap')
            mask=np.zeros(len(y),dtype=bool)
            for s in fit: mask[slice(*offset[s])]=True
            name = f'fold{fold}' if fold<5 else 'full200'
            model_path=output/'models'/f'{name}.joblib'
            model=HistGradientBoostingRegressor(**MODEL_PARAMS)
            weight=w[mask].copy();weight*=len(weight)/weight.sum()
            started=time.monotonic()
            print('fitting',name,len(fit),'scans',int(mask.sum()),'rows',flush=True)
            model.fit(X[mask],y[mask],sample_weight=weight)
            joblib.dump(model,model_path)
            metrics=[]
            for s in heldout:
                scores=infer_scene(model,s,data[s],reports[s],output/'predictions',model_path)
                if fold<5:
                    metrics.append({'scene_name':s,'weighted_mse':float(np.dot(data[s]['weights'],(scores-data[s]['y'])**2)),
                        'direct_weighted_mse':float(np.dot(data[s]['weights'],(data[s]['X'][:,0]-data[s]['y'])**2))})
            def coverage(scans):
                return {'scan_count':len(scans),'main_scene_count':len({main_scene(s) for s in scans}),
                    'valid_gt_instances':sum(reports[s]['valid_gt_count'] for s in scans),
                    'positive_target_main_scenes':len({main_scene(s) for s in scans if reports[s]['positive_target_count']>0})}
            record={'model':name,'fit_scenes':fit,'prediction_scenes':heldout,'fit_coverage':coverage(fit),
                'heldout_coverage':coverage(heldout) if fold<5 else None,'heldout_metrics':metrics,
                'model_sha256':sha(model_path),'fit_row_count':int(mask.sum()),'iterations':int(model.n_iter_),
                'sample_weight_sum':float(weight.sum()),'seconds':time.monotonic()-started,
                'whole_system_oof':False}
            fit_reports.append(record);write(output/'progress.json',fit_reports)
            print('completed',name,round(record['seconds'],1),'seconds',flush=True)
    write(output/'summary.json',{'complete':True,'new_training':True,'model_count':6,'prediction_scene_count':len(data),
        'dataset_summary_sha256':sha(dataset/'summary.json'),'identity_sha256':sha(output/'identity.json'),
        'rows':fit_reports,'new_ap':False,'R3_R4_completed':False})


def evaluate(dataset, fitted, output):
    identity,summary,_ = load_dataset(dataset)
    fit=read(fitted/'summary.json');fi=read(fitted/'identity.json')
    if not fit['complete'] or fit['dataset_summary_sha256']!=sha(dataset/'summary.json') or fit['identity_sha256']!=sha(fitted/'identity.json'):
        raise ValueError('fit/dataset identity differs')
    verified(fi['code'])
    output.mkdir(parents=True,exist_ok=False)
    reports={r['scene_name']:r for r in summary['rows']};partition=identity['partition']
    # Audit every scored member (OOF and deployment), including tie order and
    # restored source aliases. Verify persisted models by a fresh prediction pass.
    model_hashes={str(fitted/'models'/f"{r['model']}.joblib"):r['model_sha256'] for r in fit['rows']}
    verified(model_hashes)
    selected={}; audit=[]
    with threadpool_limits(limits=4):
        for model_report in fit['rows']:
            path=fitted/'models'/f"{model_report['model']}.joblib"
            model=joblib.load(path)
            for scene in model_report['prediction_scenes']:
                report=reports[scene];target=fitted/'predictions'/scene;r=read(target/'receipt.json')
                if r['model']!=str(path) or r['model_sha256']!=sha(path) or r['R1_eligible_sha256']!=report['R1_eligible_sha256']:
                    raise ValueError('prediction model/member identity differs')
                verified({str(target/k):v for k,v in r['outputs'].items()})
                members=rows(report['R1_eligible']);scores=np.load(target/'R2_eligible_scores.npy',allow_pickle=False)
                with np.load(dataset/scene/'features.npz',allow_pickle=False) as a:
                    replay=np.clip(model.predict(a['X']),0,1)
                if not np.array_equal(scores,replay):raise ValueError('persisted model prediction replay differs')
                ordered=sorted(range(len(members)),key=lambda i:(-float(scores[i]),not members[i]['has_original_native_alias'],members[i]['geometry_key'],members[i]['class_index']))[:600]
                expected=[{**members[i],'score':float(scores[i])} for i in ordered]
                actual=rows(target/'R2_selected.jsonl')
                if actual!=expected:raise ValueError('R2 selected member/score/tie replay differs')
                selected[scene]=actual
                audit.append({'scene_name':scene,'eligible_member_count':len(scores),'selected_count':len(actual),
                    'same_R1_members':True,'model_replay_exact':True,'receipt_sha256':sha(target/'receipt.json')})
    write(output/'prediction_audit.json',{'complete':True,'scene_count':len(audit),'rows':audit})
    roles={k:partition['roles'][k] for k in ['development_validation','diagnostic']}
    # Prepared source paths were frozen in the preparation identity.
    source_config_path=next(iter(identity['input_files']))
    config=read(source_config_path)
    gt_summary=read(DEV_GT/'summary.json');gt_hashes={r['scene_name']:r['sha256'] for r in gt_summary['scenes']}
    if gt_summary['label_map_sha256']!=sha(DEFAULT_LABEL_MAP):raise ValueError('GT class map changed')
    bound=[];domains={};geometries={}
    for scene in sorted(s for scans in roles.values() for s in scans):
        r=reports[scene];processed=Path(config['sources'][r['source']]['prepared_root'])/scene/f'{scene[5:]}.npy'
        verified({str(DEV_GT/f'{scene}.txt'):gt_hashes[scene]})
        g=reconstruct_gt(scene,RAW_ROOT,processed,DEFAULT_LABEL_MAP,DEV_GT/f'{scene}.txt')
        domains[scene]=g['point_count'];geometries[scene]={x['geometry_key']:x for x in read(r['geometry_file'])}
        bound.append({'scene_name':scene,**g})
    write(output/'gt_binding.json',{'rows':bound,'roles':roles,'gt_summary_sha256':sha(DEV_GT/'summary.json')})
    class Predictions(Mapping):
        def __init__(self,scenes):self.scenes=scenes
        def __iter__(self):return iter(self.scenes)
        def __len__(self):return len(self.scenes)
        def __getitem__(self,scene):
            if scene not in self.scenes:raise KeyError(scene)
            rr=selected[scene];mask=np.zeros((domains[scene],len(rr)),dtype=bool)
            with np.load(reports[scene]['points_file'],allow_pickle=False) as pts:
                for i,r in enumerate(rr):mask[pts[geometries[scene][r['geometry_key']]['array_key']],i]=True
            return {'pred_masks':mask,'pred_classes':np.asarray([r['class_index'] for r in rr],dtype=np.int64),
                'pred_scores':np.asarray([r['score'] for r in rr],dtype=np.float64)}
    results={}
    for role,scenes in roles.items():
        with (output/f'{role}.log').open('x') as log,contextlib.redirect_stdout(log):
            av,*_=evaluator.evaluate(Predictions(scenes),str(DEV_GT),str(output/f'{role}.csv'),dataset='scannet200')
        results[role]={k:float(av[v]) for k,v in {'ap':'all_ap','ap50':'all_ap_50%','ap25':'all_ap_25%'}.items()}
        print('R2 AP',role,results[role],flush=True)
    write(output/'summary.json',{'complete':True,'AP':results,'new_training':True,'new_ap':True,
        'fit_summary_sha256':sha(fitted/'summary.json'),'prediction_audit_sha256':sha(output/'prediction_audit.json'),
        'gt_binding_sha256':sha(output/'gt_binding.json'),'code':code_files(),
        'whole_system_oof':False,'validation_is_blind':False,'R3_R4_completed':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train','evaluate'])
    p.add_argument('--base',type=Path,default=ROOT/'.local/fi1_r0_r4_preparation_20260913')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--fitted',type=Path)
    p.add_argument('--threads',type=int,default=8)
    args=p.parse_args()
    if list(evaluator.opt['min_region_sizes']) != [100]:
        raise ValueError('evaluator minimum size differs from frozen R2 target')
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    if args.stage=='prepare':prepare(args.base,args.output)
    elif args.stage=='train':train(args.dataset,args.output,args.threads)
    else:evaluate(args.dataset,args.fitted,args.output)


if __name__=='__main__':main()
