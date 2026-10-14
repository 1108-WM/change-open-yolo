#!/usr/bin/env python3
"""Prepare, infer, and apply a hash-bound R2+Qwen independent control without GT."""
import argparse
from collections import Counter
import datetime
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_fi1_r0_r1_inputs import read,write,sha,digest
from tools.fi1_r4_qwen_core import CONTRACT,prompt,route_pairs,choose_views,apply_decisions


def rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f]


def save_rows(path,records):
    with Path(path).open('x') as f:
        for r in records:f.write(json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')


def no_gt_guard():
    violations=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.fsdecode(args[0])
        # Check the spelling AND its target, including relative paths and links.
        paths={os.path.abspath(path),str(Path(path).resolve())}
        forbidden=('ground_truth','/evaluation/','/evaluations/','/r3_v1/','/r3_analysis_', '.labels.ply','.aggregation.json','.segs.json')
        if any(any(x in candidate for x in forbidden) or
               ('/r2_v1/dataset/' in candidate and not candidate.endswith('/identity.json')) for candidate in paths):
            violations.append(path)
            raise PermissionError('R4 inference must not access GT/R3/AP assets: '+path)
    sys.addaudithook(hook)
    return violations


def code_hashes():
    paths=[Path(__file__),ROOT/'tools/fi1_r4_qwen_core.py',ROOT/'tools/prepare_fi1_r0_r1_inputs.py',
           ROOT/'docs/FI1_R4_QWEN_PROTOCOL_20260914.md']
    return {str(p):sha(p) for p in paths}


def verified(files):
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('file identity differs: '+str(p))


def prepare(base,out,model_dir):
    violations=no_gt_guard()
    if out.exists():raise ValueError('refusing to overwrite existing manifest')
    out.mkdir(parents=True)
    inputs={}
    def bind(path,expected=None):
        path=Path(path);h=sha(path)
        if expected is not None and expected!=h:raise ValueError('changed asset '+str(path))
        inputs[str(path)]=h
        return path
    di=read(bind(base/'r2_v1/dataset/identity.json'))
    partition=di['partition']; roles={k:partition['roles'][k] for k in ('development_validation','diagnostic')}
    if list(map(len,roles.values()))!=[55,5]:raise ValueError('incorrect approved roles')
    view_identity=read(bind(base/'yolo_views_v1/ncs_validation60/identity.json'))
    names=view_identity['class_names']
    if len(names)!=198:raise ValueError('class vocabulary differs')
    alpha_root=base/'alpha_completed_v1/ncs_validation60'
    alpha_identity=read(bind(alpha_root/'identity.json'))
    bind(alpha_root/'summary.json')
    tasks=[];scene_receipts=[];baseline=out/'baseline';baseline.mkdir()
    for role,scans in roles.items():
        for scene in scans:
            pd=base/'r2_v1/fit/predictions'/scene;pr=read(bind(pd/'receipt.json'))
            if Path(pr['model']).stem!='full200':raise ValueError('not full200 prediction')
            bind(pr['model'],pr['model_sha256'])
            selected=rows(bind(pd/'R2_selected.jsonl',pr['outputs']['R2_selected.jsonl']))
            save_rows(baseline/f'{scene}.jsonl',selected)
            ep=alpha_root/scene;er=read(bind(ep/'receipt.json'))
            if er['identity_sha256']!=digest(alpha_identity):raise ValueError('Alpha receipt identity differs')
            evidence_rows=rows(bind(ep/'evidence.jsonl',er['outputs']['evidence.jsonl']))
            evidence={r['geometry_key']:r for r in evidence_rows}
            vp=base/'yolo_views_v1/ncs_validation60'/scene;vr=read(bind(vp/'receipt.json'))
            if vr['identity_sha256']!=digest(view_identity):raise ValueError('view receipt identity differs')
            request_rows=rows(bind(vp/'yolo_and_alpha_requests.jsonl',vr['output_files']['yolo_and_alpha_requests.jsonl']))
            requests={r['geometry_key']:r for r in request_rows}
            if len(evidence)!=len(evidence_rows) or len(requests)!=len(request_rows):raise ValueError('duplicate geometry evidence')
            gp=Path(view_identity['binding'])/scene;gr=read(bind(gp/'receipt.json'))
            geometries=read(bind(gp/'geometry.json',gr['output_files']['geometry.json']))
            bound_points={g['geometry_key']:g['points_sha256'] for g in geometries}
            pairs,reasons=route_pairs(selected,evidence);available=[]
            for pair in pairs:
                req=requests[pair['geometry_key']];ev=evidence[pair['geometry_key']]
                if req['points_sha256']!=ev['points_sha256'] or req['points_sha256']!=bound_points[pair['geometry_key']]:raise ValueError('view/evidence point identity differs')
                views=choose_views(req)
                if not views:
                    reasons['insufficient_views']=reasons.get('insufficient_views',0)+1
                    continue
                available.append({**pair,'views':views,'points_sha256':req['points_sha256']})
            routed=available[:CONTRACT['max_geometries_per_scan']]
            for pair in routed:
                for view in pair['views']:bind(view['image_path'],view['image_sha256'])
                row={**pair,'scene_name':scene,'role':role,'current_class_name':names[pair['current_class_index']],
                    'proposed_class_name':names[pair['proposed_class_index']], 'baseline_sha256':sha(baseline/f'{scene}.jsonl')}
                row['task_id']=digest(row)
                tasks.append(row)
            scene_receipts.append({'scene_name':scene,'role':role,'selected':len(selected),'conflicts_before_views':len(pairs),
                'available_conflicts':len(available),'routed':len(routed),'budget_skipped':len(available)-len(routed),
                'skip_reasons':reasons,'baseline_sha256':sha(baseline/f'{scene}.jsonl'),
                'available_pairs':[{k:v for k,v in p.items() if k!='views'} for p in available]})
    model_files={str(p):sha(p) for p in sorted(model_dir.iterdir()) if p.is_file() and p.suffix in ('.json','.safetensors','.txt','.model')}
    if len([p for p in model_files if p.endswith('.safetensors')])!=5:raise ValueError('unexpected model shard set')
    identity={'contract':CONTRACT,'code':code_hashes(),'input_files':inputs,'model_dir':str(model_dir),'model_files':model_files,
              'roles':roles,'partition_id':di['partition_id'],'class_names':names,'ground_truth_read':False}
    write(out/'identity.json',identity);save_rows(out/'tasks.jsonl',tasks)
    verified(inputs)
    if violations:raise ValueError('forbidden data access attempted')
    write(out/'summary.json',{'complete':True,'identity_sha256':sha(out/'identity.json'),'tasks_sha256':sha(out/'tasks.jsonl'),
        'task_count':len(tasks),'maximum_model_calls':2*len(tasks),'scenes':scene_receipts,'GT_access_violations':violations,
        'new_training':False,'new_visual_inference':False})
    print('manifest complete',len(tasks),'tasks',flush=True)


def load_manifest(manifest,check_code=True):
    identity=read(manifest/'identity.json');summary=read(manifest/'summary.json')
    if not summary['complete'] or summary['identity_sha256']!=sha(manifest/'identity.json') or identity['contract']!=CONTRACT:raise ValueError('manifest identity differs')
    if check_code:verified(identity['code'])
    verified({str(manifest/'tasks.jsonl'):summary['tasks_sha256']})
    tasks=rows(manifest/'tasks.jsonl')
    if len(tasks)!=summary['task_count'] or len({t['task_id'] for t in tasks})!=len(tasks):raise ValueError('task coverage differs')
    for task in tasks:
        if task['task_id']!=digest({k:v for k,v in task.items() if k!='task_id'}):raise ValueError('task hash differs')
    return identity,summary,tasks


def atomic(path,value):
    temp=path.with_suffix('.tmp');write(temp,value);temp.replace(path)


def stage_attempt(out,identity,build,validate):
    """Publish a CPU stage only when complete; retain every failed attempt.

    Legacy partial output remains in place until a replacement is validated,
    then is moved to a sibling archive. Completed output is always read-only.
    """
    if (out/'summary.json').exists() and read(out/'summary.json').get('complete'):
        return validate(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with (out.parent/f'.{out.name}.stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'summary.json').exists() and read(out/'summary.json').get('complete'):
            return validate(out)
        attempt=out.parent/f'.{out.name}.attempt-{uuid.uuid4().hex}'
        atomic(attempt.parent/(attempt.name+'.identity.json'),identity)
        build(attempt)
        result=validate(attempt)
        if not result.get('complete'):raise ValueError('stage attempt is incomplete')
        atomic(attempt/'stage_identity.json',identity)
        if out.exists():out.rename(out.parent/f'.{out.name}.interrupted-{uuid.uuid4().hex}')
        attempt.rename(out)
        return result


def validate_answer(task,record):
    if record.get('task_id')!=task['task_id'] or record.get('task_sha256')!=digest(task):
        raise ValueError('answer task identity differs')
    orders=record.get('orders',{})
    if set(orders)-{'ab','ba'}:raise ValueError('unexpected answer order')
    for order,item in orders.items():
        a,b=task['current_class_name'],task['proposed_class_name']
        if order=='ba':a,b=b,a
        if item.get('prompt_sha256')!=digest(prompt(a,b)):raise ValueError('answer prompt differs')
    if record.get('complete') and not record.get('runtime_error'):
        if set(orders)!={'ab','ba'}:raise ValueError('missing completed answer order')
        for order,item in orders.items():
            if record.get('raw_'+order)!=item.get('raw') or 'raw' not in item or item.get('error'):
                raise ValueError('inconsistent completed answer')


def answer_inventory(out,tasks):
    """Verify all existing answers against PREVIOUS receipts, never re-sign them."""
    expected={t['task_id']:t for t in tasks}
    if len(expected)!=len(tasks):raise ValueError('duplicate tasks')
    summary=read(out/'summary.json') if (out/'summary.json').exists() else None
    recorded=summary['task_outputs'] if summary else {}
    if set(recorded)-set(expected):raise ValueError('foreign recorded answer')
    if summary and summary['identity_sha256']!=sha(out/'identity.json'):raise ValueError('inference summary identity differs')
    actual={p.stem:p for p in out.glob('*.json') if len(p.stem)==64}
    if set(actual)-set(expected) or set(recorded)-set(actual):raise ValueError('missing/foreign answer files')
    result={}
    for key,path in actual.items():
        receipt=out/'task_receipts'/f'{key}.json'
        if key in recorded:verified({str(path):recorded[key]})
        elif not receipt.exists():raise ValueError('uncommitted answer has no receipt; preserve and inspect it')
        if receipt.exists():
            rr=read(receipt)
            if rr['task_sha256']!=digest(expected[key]):raise ValueError('checkpoint belongs to another task')
            verified({str(path):rr['answer_sha256']})
        record=read(path);validate_answer(expected[key],record)
        if key in recorded and not record.get('complete'):raise ValueError('recorded complete answer became partial')
        result[key]=record
    if summary:
        if summary['completed_tasks']!=len(recorded) or summary['total_tasks']!=len(tasks):raise ValueError('summary task counts differ')
        if summary.get('complete') and (set(recorded)!=set(expected) or not all(r.get('complete') for r in result.values())):
            raise ValueError('incomplete answer set marked complete')
    return summary,result


def answer_checkpoint(out,task,record):
    validate_answer(task,record)
    path=out/f"{task['task_id']}.json"
    atomic(path,record)
    dest=out/'task_receipts';dest.mkdir(exist_ok=True)
    atomic(dest/path.name,{'task_sha256':digest(task),'answer_sha256':sha(path)})


def infer(manifest,out,count):
    violations=no_gt_guard();identity,summary,tasks=load_manifest(manifest,check_code=False)
    if count is not None and count<1:raise ValueError('positive smoke task count required')
    run_id={'manifest_identity_sha256':summary['identity_sha256'],'tasks_sha256':summary['tasks_sha256'],'code':identity['code']}
    if out.exists() and any(out.iterdir()):
        if read(out/'identity.json')!=run_id:raise ValueError('run identity changed')
        previous,_=answer_inventory(out,tasks)
        if previous and previous.get('complete'):
            print('already complete; verified read-only',len(tasks),flush=True)
            return previous
    verified(identity['code']);verified(identity['input_files']);verified(identity['model_files'])
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'identity.json').exists():
        if read(out/'identity.json')!=run_id:raise ValueError('run identity changed')
    else:write(out/'identity.json',run_id)
    previous,cached=answer_inventory(out,tasks)
    if previous and previous.get('complete'):return previous
    pending=[]
    requested=tasks if count is None else tasks[:count]
    for t in requested:
        path=out/f"{t['task_id']}.json"
        if not path.exists():pending.append(t)
        else:
            r=cached[t['task_id']]
            if not r.get('complete'):pending.append(t)
    if pending:
        import torch
        from PIL import Image
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.manual_seed(CONTRACT['seed']);torch.cuda.manual_seed_all(CONTRACT['seed'])
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(identity['model_dir'],torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval()
        processor=AutoProcessor.from_pretrained(identity['model_dir'],min_pixels=CONTRACT['min_pixels'],max_pixels=CONTRACT['max_pixels'],local_files_only=True)
        runtime={'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'qwen_vl_utils':importlib.metadata.version('qwen-vl-utils'),
            'device':torch.cuda.get_device_name(),'pid':os.getpid()}
        if (out/'runtime.json').exists():
            old_runtime=read(out/'runtime.json')
            if {k:v for k,v in old_runtime.items() if k!='pid'}!={k:v for k,v in runtime.items() if k!='pid'}:raise ValueError('runtime changed across resume')
        write(out/'runtime.json',runtime)
        for task in pending:
            path=out/f"{task['task_id']}.json";record=read(path) if path.exists() else {'task_id':task['task_id'],'task_sha256':digest(task),'complete':False,'orders':{}}
            images=[]
            try:
                for view in task['views']:
                    verified({view['image_path']:view['image_sha256']})
                    with Image.open(view['image_path']) as img:
                        if list(img.size)!=list(reversed(view['image_shape'])):raise ValueError('image dimensions differ')
                        images.append(img.convert('RGB').crop(tuple(view['bbox_xyxy'])))
                for order in ('ab','ba'):
                    if order in record['orders']:
                        if 'raw' not in record['orders'][order]:
                            record['orders'][order].update(raw=None,error='interrupted_attempt_not_retried')
                            answer_checkpoint(out,task,record)
                        continue
                    a,b=task['current_class_name'],task['proposed_class_name']
                    if order=='ba':a,b=b,a
                    text_prompt=prompt(a,b);start=time.monotonic()
                    result={'prompt_sha256':digest(text_prompt)}
                    record['orders'][order]={**result,'attempt_started':True}
                    answer_checkpoint(out,task,record)
                    inputs=None;generated=None
                    try:
                        content=[{'type':'image','image':image} for image in images]+[{'type':'text','text':text_prompt}]
                        messages=[{'role':'user','content':content}]
                        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                        image_inputs,video_inputs=process_vision_info(messages)
                        inputs=processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors='pt').to('cuda')
                        with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=CONTRACT['max_new_tokens'],do_sample=False)
                        new=generated[:,inputs.input_ids.shape[1]:]
                        result.update(raw=processor.batch_decode(new,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0],
                            input_tokens=int(inputs.input_ids.shape[1]),generated_tokens=int(new.shape[1]))
                    except Exception as error:result.update(raw=None,error=repr(error))
                    finally:
                        result['seconds']=time.monotonic()-start
                        del inputs,generated
                        gc.collect();torch.cuda.empty_cache()
                    record['orders'][order]=result;answer_checkpoint(out,task,record)
                record['raw_ab']=record['orders']['ab'].get('raw');record['raw_ba']=record['orders']['ba'].get('raw')
                errors=[o['error'] for o in record['orders'].values() if 'error' in o]
                if errors:record['runtime_error']=errors
            except Exception as error:record['runtime_error']=[repr(error)]
            finally:
                for image in images:image.close()
            record['complete']=True;answer_checkpoint(out,task,record)
            print('Qwen',task['scene_name'],task['task_id'][:12],record.get('raw_ab'),record.get('raw_ba'),record.get('runtime_error'),flush=True)
    _,cached=answer_inventory(out,tasks)
    completed=[t for t in tasks if cached.get(t['task_id'],{}).get('complete')]
    verified(identity['code']);verified(identity['input_files'])
    if violations:raise ValueError('forbidden data access attempted')
    result={'complete':len(completed)==len(tasks),'completed_tasks':len(completed),'total_tasks':len(tasks),
        'GT_access_violations':violations,'new_visual_inference':bool(pending) or bool(previous and previous.get('new_visual_inference')),'new_training':False,
        'task_outputs':{t['task_id']:sha(out/f"{t['task_id']}.json") for t in completed},'identity_sha256':sha(out/'identity.json')}
    if previous!=result:atomic(out/'summary.json',result)
    print('completed',len(completed),'/',len(tasks),flush=True)
    return result


def validate_applied(manifest,inference,out):
    identity,summary,tasks=load_manifest(manifest,check_code=False)
    completed,cached=answer_inventory(inference,tasks)
    if not completed or not completed.get('complete'):raise ValueError('inference incomplete')
    ii=read(inference/'identity.json')
    if ii!={'manifest_identity_sha256':summary['identity_sha256'],'tasks_sha256':summary['tasks_sha256'],'code':identity['code']}:raise ValueError('foreign inference')
    result=read(out/'summary.json')
    if not result.get('complete') or result['manifest_summary_sha256']!=sha(manifest/'summary.json') or result['inference_summary_sha256']!=sha(inference/'summary.json'):
        raise ValueError('prediction summary dependency differs')
    expected_scenes={s:role for role,scans in identity['roles'].items() for s in scans}
    records={r['scene_name']:r for r in result['records']}
    if len(records)!=len(result['records']) or set(records)!=set(expected_scenes) or result['scene_count']!=len(records):raise ValueError('prediction scene coverage differs')
    counts=Counter()
    for scene,role in expected_scenes.items():
        sr=next(r for r in summary['scenes'] if r['scene_name']==scene)
        bp=manifest/'baseline'/f'{scene}.jsonl';verified({str(bp):sr['baseline_sha256']})
        st=[t for t in tasks if t['scene_name']==scene]
        expected,ledger=apply_decisions(rows(bp),st,[cached[t['task_id']] for t in st])
        actual=out/scene/'R4_selected.jsonl'
        verified({str(actual):records[scene]['selected_sha256']})
        if records[scene]['role']!=role or rows(actual)!=expected or records[scene]['ledger']!=ledger:raise ValueError('prediction replay differs')
        counts.update(r['action'] for r in ledger)
    if dict(counts)!=result['decision_counts']:raise ValueError('decision counts differ')
    return result


def apply(manifest,inference,out):
    if (out/'summary.json').exists() and read(out/'summary.json').get('complete'):
        return validate_applied(manifest,inference,out)
    load_manifest(manifest)
    identity={'manifest_summary_sha256':sha(manifest/'summary.json'),
              'inference_summary_sha256':sha(inference/'summary.json'),'code':code_hashes()}
    return stage_attempt(out,identity,lambda work:_apply(manifest,inference,work),
                         lambda work:validate_applied(manifest,inference,work))


def _apply(manifest,inference,out):
    violations=no_gt_guard();identity,summary,tasks=load_manifest(manifest)
    completed=read(inference/'summary.json');ii=read(inference/'identity.json')
    if not completed['complete'] or ii['tasks_sha256']!=summary['tasks_sha256'] or completed['identity_sha256']!=sha(inference/'identity.json'):raise ValueError('incomplete/foreign inference')
    _,outputs=answer_inventory(inference,tasks)
    for key,h in completed['task_outputs'].items():verified({str(inference/f'{key}.json'):h})
    if set(outputs)!=set(completed['task_outputs']):raise ValueError('answer coverage differs')
    out.mkdir(parents=True,exist_ok=False);records=[]
    for role,scans in identity['roles'].items():
        for scene in scans:
            selected=rows(manifest/'baseline'/f'{scene}.jsonl')
            sr=next(r for r in summary['scenes'] if r['scene_name']==scene)
            verified({str(manifest/'baseline'/f'{scene}.jsonl'):sr['baseline_sha256']})
            st=[t for t in tasks if t['scene_name']==scene];rr=[outputs[t['task_id']] for t in st]
            for t,r in zip(st,rr):
                if not r['complete'] or r['task_sha256']!=digest(t):raise ValueError('answer task identity differs')
                for order,a,b in [('ab',t['current_class_name'],t['proposed_class_name']),('ba',t['proposed_class_name'],t['current_class_name'])]:
                    if not r.get('runtime_error') and r['orders'][order]['prompt_sha256']!=digest(prompt(a,b)):raise ValueError('answer prompt differs')
                    if not r.get('runtime_error') and r.get('raw_'+order)!=r['orders'][order].get('raw'):raise ValueError('flattened raw answer differs')
            result,ledger=apply_decisions(selected,st,rr)
            # Independent reconstruction: no calls to parse/decision/apply functions.
            independent=[dict(r) for r in selected]
            for t,r in zip(st,rr):
                if not r.get('runtime_error') and isinstance(r.get('raw_ab'),str) and isinstance(r.get('raw_ba'),str) and r['raw_ab'].strip()=='B' and r['raw_ba'].strip()=='A':
                    for row in independent:
                        if row['geometry_key']==t['geometry_key']:
                            if row['class_index']==t['current_class_index']:row['score']=t['proposed_score']
                            elif row['class_index']==t['proposed_class_index']:row['score']=t['current_score']
            if result!=independent:raise ValueError('independent score exchange differs')
            for before,after in zip(selected,result):
                if {k:v for k,v in before.items() if k!='score'}!={k:v for k,v in after.items() if k!='score'}:raise ValueError('member metadata changed')
            for geom in {r['geometry_key'] for r in selected}:
                if sorted(r['score'] for r in selected if r['geometry_key']==geom)!=sorted(r['score'] for r in result if r['geometry_key']==geom):raise ValueError('geometry score multiset changed')
            target=out/scene;target.mkdir();save_rows(target/'R4_selected.jsonl',result)
            records.append({'scene_name':scene,'role':role,'ledger':ledger,'independent_replay_exact':True,
                'selected_sha256':sha(target/'R4_selected.jsonl'),'unchanged_members_and_geometry_score_multisets':True})
    if violations:raise ValueError('forbidden data access attempted')
    write(out/'summary.json',{'complete':True,'scene_count':len(records),'records':records,'GT_access_violations':violations,
        'inference_summary_sha256':sha(inference/'summary.json'),'manifest_summary_sha256':sha(manifest/'summary.json'),
        'decision_counts':dict(Counter(r['action'] for scene in records for r in scene['ledger']))})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','infer','apply'])
    p.add_argument('--base',type=Path,default=ROOT/'.local/fi1_r0_r4_preparation_20260913')
    p.add_argument('--model-dir',type=Path,default=ROOT/'pretrained/checkpoints/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--manifest',type=Path);p.add_argument('--inference',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int)
    args=p.parse_args()
    if args.stage=='prepare':prepare(args.base.resolve(),args.output.resolve(),args.model_dir.resolve())
    elif args.stage=='infer':infer(args.manifest.resolve(),args.output.resolve(),args.count)
    else:apply(args.manifest.resolve(),args.inference.resolve(),args.output.resolve())


if __name__=='__main__':main()
