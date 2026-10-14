#!/usr/bin/env python3
"""Deploy the frozen full200 heads, audit outputs, and separately evaluate val312."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from tools.prepare_fi1_r0_r1_inputs import read, write, sha, digest
from tools.prepare_fi1_r3b_val312 import score_scene, load_models, no_gt_io, require, verify, BASE
from tools.fi1_r0_r1_protocol import PROTOCOL
from tools.fi1_r2_core import geometry_features
from tools.audit_fi1_r0_r1_predictions import audit_members
from tools.audit_fi1_r2_dataset import features as independent_features
from tools.extract_fi1_val312_yolo import atomic_json

ARMS = ('R0', 'R2', 'R3b')


def jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def save_rows(path, records):
    with path.open('x') as f:
        for row in records:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')


def code_files():
    names = ['run_fi1_r3b_val312.py','prepare_fi1_r3b_val312.py','fi1_r0_r1_protocol.py',
             'fi1_r2_core.py','fi1_r3_core.py','prepare_fi1_r0_r1_inputs.py',
             'audit_fi1_r0_r1_predictions.py','audit_fi1_r2_dataset.py','extract_fi1_val312_yolo.py']
    return {str(ROOT/'tools'/n):sha(ROOT/'tools'/n) for n in names}


def validate_selected(members, scores, selected):
    require(scores.shape == (len(members),) and np.isfinite(scores).all(), 'invalid score array')
    require(bool(((scores >= 0) & (scores <= 1)).all()), 'scores outside [0,1]')
    expected = sorted([{**m, 'score':float(v)} for m,v in zip(members,scores)],
                      key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:600]
    require(expected == selected, 'eligible-score alignment/selection differs')


def budget(rows):
    return {'selected_class_members':len(rows), 'selected_geometry_count':len({r['geometry_key'] for r in rows}),
            'selected_native_aliases':sum(len(r['native_alias_keys']) for r in rows),
            'selected_expanded_classes':sum(r['expanded_class'] for r in rows),
            'selected_alias_sources':dict(Counter(a['source'] for r in rows for a in r['original_aliases']))}


def scoring_context(prep, evidence, smoke):
    ps = read(prep/'summary.json'); verify(ps['source']); verify(ps['evidence'])
    require(ps['geometry_binding_complete'] and ps['scene_count'] == 312, 'preparation incomplete')
    ei = read(evidence/'identity.json'); es = read(evidence/'summary.json')
    require(es['complete'] and es['identity_sha256'] == digest(ei), 'evidence incomplete/unbound')
    views = Path(ei['views']); vi = read(views/'identity.json')
    require(ei['views_identity_sha256'] == digest(vi), 'evidence/views link differs')
    binding = Path(vi['binding']); bi = read(binding/'identity.json')
    require(binding.resolve() == (prep/'geometry_binding').resolve(), 'unexpected geometry route')
    require(vi['binding_identity_sha256'] == digest(bi), 'view/geometry link differs')
    require(vi['protocol'] == PROTOCOL and vi['protocol_sha256'] == digest(PROTOCOL), 'protocol differs')
    require(not bi['subset_smoke'], 'full geometry binding required')
    scans = ei['scenes']
    require(len(scans) == len(set(scans)) and set(scans) <= set(bi['scenes']), 'invalid scene coverage')
    subset = scans != bi['scenes'] or vi['subset']
    require(smoke or not subset, 'subset evidence cannot be full312 scoring')
    require({r['scene_name'] for r in es['rows']} == set(scans), 'evidence summary coverage differs')
    files = {str(p):sha(p) for p in [prep/'summary.json',evidence/'identity.json',evidence/'summary.json',views/'identity.json',binding/'identity.json']}
    return scans, ei, vi, bi, binding, files


def score(args):
    no_gt_io()
    prep, evidence, out = args.preparation.resolve(), args.evidence.resolve(), args.output.resolve()
    scans, ei, vi, bi, binding, files = scoring_context(prep,evidence,args.smoke)
    models, model_files = load_models(BASE); files.update(model_files)
    import sklearn
    import scipy
    identity = {'version':'fi1_r3b_val312_frozen_deployment','scenes':scans,'smoke':args.smoke,
        'evidence':str(evidence),'binding':str(binding),'inputs':files,'code':code_files(),
        'arms':list(ARMS),'full200_models':model_files,'class_names':vi['class_names'],
        'protocol':PROTOCOL,'new_training':False,'Qwen':False,'GT_content_read':False,
        'runtime':{'python':sys.version,'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__}}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'identity.json').exists():
        require(read(out/'identity.json') == identity, 'scoring resume identity differs')
    else:
        write(out/'identity.json',identity)
    summaries = []
    with threadpool_limits(limits=4):
        for scene in scans:
            target = out/scene
            if (target/'receipt.json').exists():
                r = read(target/'receipt.json')
                require(r['identity_sha256'] == digest(identity), 'scene identity differs')
                verify(r['inputs']); verify({str(target/n):h for n,h in r['outputs'].items()})
                summaries.append(r['summary']); continue
            started = time.monotonic()
            er = read(evidence/scene/'receipt.json'); br = read(binding/scene/'receipt.json')
            require(er['identity_sha256'] == digest(ei) and er['summary']['complete'], 'incomplete scene evidence')
            require(br['identity_sha256'] == digest(bi), 'unbound geometry receipt')
            inputs = {str(evidence/scene/'receipt.json'):sha(evidence/scene/'receipt.json'),
                      str(binding/scene/'receipt.json'):sha(binding/scene/'receipt.json'),
                      **er['inputs'], **br['input_files'],
                      **{str(evidence/scene/n):h for n,h in er['outputs'].items()},
                      **{str(binding/scene/n):h for n,h in br['output_files'].items()}}
            verify(inputs)
            geometry = read(binding/scene/'geometry.json')
            records = jsonl(evidence/scene/'evidence.jsonl'); ev = {r['geometry_key']:r for r in records}
            require(len(ev) == len(records) == len(geometry) and set(ev) == {g['geometry_key'] for g in geometry}, 'evidence coverage differs')
            with np.load(binding/scene/'points.npz',allow_pickle=False) as packed:
                points = [packed[g['array_key']] for g in geometry]
            domain = br['summary']['point_count']
            for g,p in zip(geometry,points):
                require(p.ndim == 1 and p.dtype.kind in 'iu' and np.array_equal(p,np.unique(p)), 'invalid sorted point set')
                require(not len(p) or (p[0] >= 0 and p[-1] < domain), 'point outside bound domain')
            result = score_scene(geometry,points,ev,domain,models)
            member_audit = audit_members(geometry,ev,{a:result[a] for a in ['R0','R1']})
            error = 0.
            for g in geometry:
                own = geometry_features(g,ev[g['geometry_key']]); independent = independent_features(g,ev[g['geometry_key']])
                require(np.isfinite(own).all() and np.isfinite(independent).all(), 'nonfinite features')
                error = max(error,float(np.max(np.abs(own-independent))))
            require(error <= 1e-12, 'independent own-feature replay differs')
            for arm in ['R2','R3b']:
                validate_selected(result['R1']['eligible'],result[arm]['scores'],result[arm]['selected'])
            if target.exists():
                target.rename(out/(scene+'.incomplete_'+str(time.time_ns())))
            target.mkdir()
            for arm in ['R0','R1']:
                for stage in ['eligible','selected']:
                    save_rows(target/f'{arm}_{stage}.jsonl',result[arm][stage])
            for arm in ['R2','R3b']:
                np.save(target/f'{arm}_scores.npy',result[arm]['scores'])
                save_rows(target/f'{arm}_selected.jsonl',result[arm]['selected'])
            summary = {'scene_name':scene,'geometry_count':len(geometry),'point_count':domain,
                'R1_eligible_count':len(result['R1']['eligible']), 'independent_member_audit':member_audit,
                'independent_own_features_max_abs_error':error,'selection_replay_equal':True,
                'budget':{a:budget(result[a]['selected']) for a in ARMS},'seconds':time.monotonic()-started}
            verify(inputs); verify(files); verify(identity['code'])
            atomic_json(target/'receipt.json',{'identity_sha256':digest(identity),'inputs':inputs,
                'outputs':{p.name:sha(p) for p in target.iterdir() if p.is_file()},'summary':summary})
            summaries.append(summary)
            atomic_json(out/'progress.json',{'complete':False,'scene_count':len(summaries),'total':len(scans)})
            print('frozen R0/R2/R3b scored and audited',scene,len(result['R1']['eligible']),flush=True)
    atomic_json(out/'summary.json',{'complete':True,'smoke':args.smoke,'scene_count':len(scans),
        'rows':summaries,'identity_sha256':digest(identity),'GT_content_read':False,'new_AP':False,
        'receipts':{str(out/s/'receipt.json'):sha(out/s/'receipt.json') for s in scans},
        'new_training':False,'Qwen':False})


def bind_gt_arrays(prepared, gt, xyz):
    """Cross-check cached GT encoding against labeled NPY and the bound PLY order."""
    require(prepared.ndim == 2 and prepared.shape[1] == 12, 'GT requires labeled 12-column NPY')
    require(np.array_equal(prepared[:,:3],xyz), 'GT/prepared/mesh XYZ order differs')
    labels = prepared[:,10:12]
    require(np.isfinite(labels).all() and np.array_equal(labels,labels.astype(np.int64)), 'invalid GT columns')
    expected = labels[:,0].astype(np.int64)*1000+labels[:,1].astype(np.int64)+1
    require(gt.ndim == 1 and np.array_equal(expected,gt), 'GT text does not match pointwise NPY labels')
    return len(gt)


def evaluate(args):
    require(args.allow_gt_evaluation, 'explicit GT evaluation stage required')
    from plyfile import PlyData
    from tools.analyze_fi1_r3 import configure_evaluator, compact_metrics, public_json, validate_ap_array
    from evaluate.scannet200 import eval_semantic_instance as evaluator
    pred, out, prep = args.predictions.resolve(), args.output.resolve(), args.preparation.resolve()
    si = read(pred/'identity.json'); ss = read(pred/'summary.json')
    require(ss['complete'] and not ss['smoke'] and not si['smoke'] and ss['scene_count'] == 312, 'full312 predictions required before GT')
    require(ss['identity_sha256'] == digest(si), 'prediction identity differs')
    verify(si['inputs']); verify(si['code']); verify(ss['receipts'])
    scans = (prep/'inventory/val312.txt').read_text().split()
    require(si['scenes'] == scans and len(set(scans)) == 312, 'official scene coverage differs')
    binding = Path(si['binding']); bi = read(binding/'identity.json')
    root = Path(read(prep/'inventory/input_sources.json')['sources']['val312']['prepared_root'])
    gtroot = root/'ground_truth'
    configure_evaluator()
    require(list(evaluator.CLASS_LABELS) == si['class_names'], 'class mapping differs')
    require(np.allclose(evaluator.opt['overlaps'],[*np.arange(.5,.95,.05),.25]), 'AP IoU thresholds differ')
    ei = {'prediction_identity_sha256':sha(pred/'identity.json'),'prediction_summary_sha256':sha(pred/'summary.json'),
          'code':{**code_files(),str(Path(evaluator.__file__)):sha(evaluator.__file__),
                  str(ROOT/'tools/analyze_fi1_r3.py'):sha(ROOT/'tools/analyze_fi1_r3.py')},
          'scenes':scans,'arms':list(ARMS),'GT_root':str(gtroot), 'validation_is_blind':False,
          'GT_binding':'GT text equals labeled NPY columns; NPY XYZ equals bound PLY. Raw annotation files not available for independent reconstruction.'}
    for name in ['util.py','util_3d.py','scannet_constants.py']:
        path = ROOT/'evaluate/scannet200'/name
        ei['code'][str(path)] = sha(path)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'identity.json').exists():
        require(read(out/'identity.json') == ei, 'evaluation resume identity differs')
    else:
        write(out/'identity.json',ei)
    if (out/'summary.json').exists():
        old = read(out/'summary.json')
        require(old['complete'] and old['identity_sha256'] == digest(ei), 'completed evaluation identity differs')
        verify(old['inputs']); verify({str(out/n):h for n,h in old['outputs'].items()})
        print('Full312 AP already complete; verified read-only',flush=True)
        return
    inputs = {}; matches = {a:{} for a in ARMS}; reports = []
    for scene in scans:
        target = out/scene; rp = target/'receipt.json'
        sr = read(pred/scene/'receipt.json'); br = read(binding/scene/'receipt.json')
        require(sr['identity_sha256'] == digest(si) and br['identity_sha256'] == digest(bi), 'scene link differs')
        source_files = {str(pred/scene/n):h for n,h in sr['outputs'].items()}
        source_files.update({str(binding/scene/n):h for n,h in br['output_files'].items()})
        source_files.update(br['input_files']); verify(source_files)
        gtpath = gtroot/f'{scene}.txt'; preparedpath = root/scene/f'{scene[5:]}.npy'
        mesh = root/scene/f'{scene}_vh_clean_2.ply'
        require(str(preparedpath) in br['input_files'] and str(mesh) in br['input_files'], 'GT coordinates not from bound geometry inputs')
        source_files[str(gtpath)] = sha(gtpath)
        source_files[str(pred/scene/'receipt.json')] = sha(pred/scene/'receipt.json')
        inputs.update(source_files)
        if rp.exists():
            r = read(rp)
            require(r['identity_sha256'] == digest(ei) and r['inputs'] == source_files, 'AP match resume inputs differ')
            verify({str(target/n):h for n,h in r['outputs'].items()})
            for a in ARMS:
                with (target/f'{a}_matches.pkl').open('rb') as f: matches[a][scene] = pickle.load(f)
            reports.append(r['summary']); continue
        prepared = np.load(preparedpath,mmap_mode='r',allow_pickle=False)
        verts = PlyData.read(mesh)['vertex']; xyz = np.column_stack([verts[k] for k in ['x','y','z']])
        gt = np.loadtxt(gtpath,dtype=np.int64)
        count = bind_gt_arrays(prepared,gt,xyz)
        require(count == sr['summary']['point_count'], 'prediction/GT domain differs')
        geometry = {g['geometry_key']:g for g in read(binding/scene/'geometry.json')}
        members = jsonl(pred/scene/'R1_eligible.jsonl')
        target.mkdir(exist_ok=True)
        with np.load(binding/scene/'points.npz',allow_pickle=False) as packed:
            for arm in ARMS:
                selected = jsonl(pred/scene/f'{arm}_selected.jsonl')
                if arm != 'R0':
                    validate_selected(members,np.load(pred/scene/f'{arm}_scores.npy',allow_pickle=False),selected)
                require(len(selected) <= 600 and all(0 <= r['class_index'] < 198 and np.isfinite(r['score']) for r in selected), 'invalid selected member')
                masks = np.zeros((count,len(selected)),dtype=bool)
                for j,r in enumerate(selected): masks[packed[geometry[r['geometry_key']]['array_key']],j] = True
                gg,pp = evaluator.assign_instances_for_scan({'pred_masks':masks,
                    'pred_classes':np.array([r['class_index'] for r in selected]),
                    'pred_scores':np.array([r['score'] for r in selected],dtype=np.float64)},str(gtpath))
                matches[arm][scene] = {'gt':gg,'pred':pp}
                with (target/f'{arm}_matches.pkl').open('wb') as f: pickle.dump(matches[arm][scene],f)
                del masks
        report = {'scene_name':scene,'point_count':count,'pointwise_GT_NPY_equal':True,'mesh_NPY_XYZ_equal':True,
                  'raw_annotation_reconstructed':False}
        verify(source_files)
        atomic_json(rp,{'identity_sha256':digest(ei),'inputs':source_files,
            'outputs':{f'{a}_matches.pkl':sha(target/f'{a}_matches.pkl') for a in ARMS},'summary':report})
        reports.append(report)
        atomic_json(out/'progress.json',{'complete':False,'matched_scenes':len(reports),'total':312})
        print('GT bound and three-arm matches computed',scene,flush=True)
    AP = {}
    for arm in ARMS:
        aps,*_ = evaluator.evaluate_matches(matches[arm]); validate_ap_array(aps)
        AP[arm] = compact_metrics(aps)
        np.save(out/f'{arm}_AP.npy',aps)
        write(out/f'{arm}_classes.json',public_json(evaluator.compute_averages(aps)['classes']))
        print('full312 AP',arm,AP[arm],flush=True)
    verify(inputs); verify(ei['code']); verify(ss['receipts'])
    write(out/'GT_binding.json',{'complete':True,'rows':reports,'raw_annotation_reconstructed':False})
    atomic_json(out/'summary.json',{'complete':True,'scene_count':312,'AP':AP,'GT_content_read':True,
        'identity_sha256':digest(ei),'new_training':False,'Qwen':False,'validation_is_blind':False,
        'inputs':inputs,'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name != 'summary.json'}})


def workflow(args):
    """Smoke the complete prediction path before executing the full frozen chain."""
    import fcntl
    import datetime
    prep, out = args.preparation.resolve(), args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'workflow.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    python = str(ROOT/'.venvs/yoloe_gpu/bin/python')
    yolo_python = '/home/jia/anaconda3/envs/openyolo3d/bin/python'
    snapshot = ROOT/'full_bundle/code_sources/evidence_snapshot'
    alpha = Path(read(prep/'inventory/input_sources.json')['sources']['val312']['alpha_cache'])
    code = code_files()
    for name in ['export_fi1_r0_r1_yolo_views.py','bind_fi1_r0_r1_alpha_cache.py','complete_fi1_r0_r1_alpha.py']:
        code[str(ROOT/'tools'/name)] = sha(ROOT/'tools'/name)
    identity = {'preparation':str(prep),'output':str(out),'code':code,
        'preparation_summary_sha256':sha(prep/'summary.json'),'yolo_plan_sha256':sha(out/'yolo/plan.json'),
        'primary_arms':list(ARMS),'new_training':False,'Qwen':False,
        'authorization':'User approved proceeding from preparation through evidence, frozen scoring and the fixed val312 evaluation.',
        'full_chain_smoke_scene':'scene0011_00'}
    if (out/'workflow_identity.json').exists():
        require(read(out/'workflow_identity.json') == identity, 'workflow identity changed')
    else:
        write(out/'workflow_identity.json',identity)
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    stages = []
    def state(stage, status, **extra):
        value = {'stage':stage,'status':status,'pid':os.getpid(),
            'updated_at':datetime.datetime.now().astimezone().isoformat(),
            'workflow_identity_sha256':sha(out/'workflow_identity.json'), 'completed_stages':list(stages),
            'new_training':False,'Qwen':False,**extra}
        atomic_json(out/'execution.json',value)
        current = read(BASE/'preparation_status.json')
        current['val312_R3b_execution'] = {**value,'directory':str(out)}
        current['next_step'] = stage
        atomic_json(BASE/'preparation_status.json',current)
        print('val312 workflow',stage,status,flush=True)
    def command(stage, executable, script, *arguments):
        verify(code)
        require(sha(out/'yolo/plan.json') == identity['yolo_plan_sha256'], 'YOLO plan changed')
        state(stage,'running')
        cmd = [executable,str(ROOT/'tools'/script),*[str(x) for x in arguments]]
        with (out/(stage+'.log')).open('a') as log:
            result = subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{stage} failed with exit {result.returncode}; see {out/(stage+".log")}')
        stages.append(stage)
    def semantics_and_score(folder, prefix, smoke):
        folder.mkdir(parents=True,exist_ok=True)
        view_args = ['--scenes','scene0011_00'] if smoke else []
        command(prefix+'views',python,'export_fi1_r0_r1_yolo_views.py',
            '--binding',prep/'geometry_binding','--sources',prep/'inventory/input_sources.json',
            '--snapshot',snapshot,'--output',folder/'views',*view_args)
        command(prefix+'alpha_bind',python,'bind_fi1_r0_r1_alpha_cache.py',
            '--views',folder/'views','--cache',alpha,'--output',folder/'alpha_bound')
        bound_summary = read(folder/'alpha_bound/summary.json')
        evidence = folder/'alpha_bound'
        if bound_summary['alpha_pending_count']:
            if not (folder/'alpha_completed/plan.json').exists():
                command(prefix+'alpha_plan',python,'complete_fi1_r0_r1_alpha.py','prepare',
                    '--bound',folder/'alpha_bound','--snapshot',snapshot,'--output',folder/'alpha_completed')
            command(prefix+'alpha_complete',python,'complete_fi1_r0_r1_alpha.py','run',
                '--plan',folder/'alpha_completed/plan.json')
            evidence = folder/'alpha_completed'
        command(prefix+'score',python,'run_fi1_r3b_val312.py','score','--preparation',prep,
            '--evidence',evidence,'--output',folder/'predictions',*(['--smoke'] if smoke else []))
    try:
        command('smoke_yolo',yolo_python,'extract_fi1_val312_yolo.py','run','--plan',out/'yolo/plan.json','--limit-scenes','1')
        semantics_and_score(out/'smoke','smoke_',True)
        write(out/'smoke_complete.json',{'complete':True,'prediction_summary_sha256':sha(out/'smoke/predictions/summary.json'),
                                      'GT_content_read':False,'AP_computed':False})
        if args.smoke_only:
            state('smoke_complete','complete',full312_AP_complete=False)
            return
        command('full_yolo',yolo_python,'extract_fi1_val312_yolo.py','run','--plan',out/'yolo/plan.json')
        require(read(out/'yolo/summary.json')['complete'], 'full YOLO incomplete')
        semantics_and_score(out/'full','full_',False)
        command('full_evaluate',python,'run_fi1_r3b_val312.py','evaluate','--preparation',prep,
            '--predictions',out/'full/predictions','--output',out/'evaluation','--allow-gt-evaluation')
        result = read(out/'evaluation/summary.json'); require(result['complete'], 'AP incomplete')
        state('full312_complete','complete',full312_AP_complete=True,AP=result['AP'])
        report = ROOT/'docs/FI1_R3B_VAL312_RESULTS_20260914.md'
        table = '\n'.join(f"| {a} | {100*result['AP'][a]['ap']:.6f} | {100*result['AP'][a]['ap50']:.6f} | {100*result['AP'][a]['ap25']:.6f} |" for a in ARMS)
        report.write_text('# R0/R2/R3b完整val312评测\n\n冻结full200模型，200训练／55开发／5诊断划分不变，无新训练、无Qwen。\n\n'
            '| 方法 | 实例AP | AP50 | AP25 |\n|---|---:|---:|---:|\n'+table+'\n\n'
            'AP为198类三维实例分割、最小100点，IoU 0.50至0.90九档。312为历史验证集，不称盲测。\n\n'
            'YOLO按冻结配置重新提取并绑定RGB哈希；Alpha只复用几何、视角、裁剪及聚合重放通过的记录，其余补算。'
            '历史Alpha缺原RGB字节身份，复用不等于视觉模型重跑。GT文本逐点等于12列NPY语义／实例编码，NPY坐标顺序等于已绑定PLY；未重新从原始标注文件重建GT。\n\n'
            '完整资产、场景回执及逐类AP见 `.local/fi1_r3b_val312_execution_20260914/`。\n')
        atomic_json(out/'summary.json',{'complete':True,'AP':result['AP'],'report':str(report),'report_sha256':sha(report),
                    'evaluation_summary_sha256':sha(out/'evaluation/summary.json'),'new_training':False,'Qwen':False})
        for filename,prefix in [('新开对话阅读内容.md','docs/'),('CURRENT_EXPERIMENT_STATUS.md','docs/'),
                ('docs/CODEX_HANDOFF.md',''),('docs/FI1_R0_R4_PREPARATION_20260913.md',''),('资料/代码文件索引与当前主线地图.md','../docs/')]:
            path = ROOT/filename
            banner = '<!-- FI1_VAL312_RESULTS_20260914 -->\n**2026-09-14：R0/R2/R3b完整312评测已完成。** '+', '.join(f"{a} AP={100*result['AP'][a]['ap']:.6f}" for a in ARMS)+f'。冻结模型，无新训练／Qwen，不称盲测。见[312结果]({prefix}{report.name})。\n\n'
            if not path.read_text().startswith('<!-- FI1_VAL312_RESULTS_20260914 -->'):
                path.write_text(banner+path.read_text())
    except Exception as exc:
        active = read(out/'execution.json').get('stage','startup') if (out/'execution.json').exists() else 'startup'
        state(active,'failed',error=str(exc),full312_AP_complete=False)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage',required=True)
    s = sub.add_parser('score'); s.add_argument('--evidence',type=Path,required=True); s.add_argument('--smoke',action='store_true')
    e = sub.add_parser('evaluate'); e.add_argument('--predictions',type=Path,required=True); e.add_argument('--allow-gt-evaluation',action='store_true')
    w = sub.add_parser('workflow'); w.add_argument('--smoke-only',action='store_true')
    for parser in [s,e,w]:
        parser.add_argument('--preparation',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    {'score':score,'evaluate':evaluate,'workflow':workflow}[args.stage](args)
