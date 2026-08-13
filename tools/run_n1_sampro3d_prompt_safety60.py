#!/usr/bin/env python3
"""Resumable, no-GT N1a SAMPro3D prompt runner over the frozen seed ledger."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PREFIX='n1_sampro3d_prompt_paper_reference_gvc_safety60_uniform30_d2b_20260806'

def _resolve(p):
    p=Path(p); return p if p.is_absolute() else ROOT/p
def _scenes(p): return [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]
def _eligible(ledger,scene):
    return sum(bool(json.loads(x).get('views')) for x in (ledger/scene/'seed_view_ledger.jsonl').read_text().splitlines() if x)
def _write(path,payload):
    temp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}'); temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n'); os.replace(temp,path)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--scene-list',type=Path,required=True); p.add_argument('--seed-ledger-root',type=Path,required=True); p.add_argument('--sam-source',type=Path,required=True); p.add_argument('--sam-checkpoint',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--python',type=Path,default=Path(sys.executable)); p.add_argument('--sam-model-type',default='vit_b'); p.add_argument('--seeds-per-chunk',type=int,default=64); p.add_argument('--views-per-seed',type=int,default=3); p.add_argument('--max-chunks',type=int); p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    for n in ('scene_list','seed_ledger_root','sam_source','sam_checkpoint','output_root','python'): setattr(a,n,_resolve(getattr(a,n)))
    if not a.sam_source.is_dir() or not a.sam_checkpoint.is_file() or not a.python.is_file(): raise SystemExit('SAM source, checkpoint, or python is missing')
    if min(a.seeds_per_chunk,a.views_per_seed)<=0: raise SystemExit('chunk sizes must be positive')
    a.output_root.mkdir(parents=True,exist_ok=True); plans=[]
    for scene in _scenes(a.scene_list):
        for start in range(0,_eligible(a.seed_ledger_root,scene),a.seeds_per_chunk):
            end=start+a.seeds_per_chunk-1; batch=f'{scene}_seed{start:06d}_{min(end,_eligible(a.seed_ledger_root,scene)-1):06d}'; obs=a.output_root/'batches'/batch/'observations'/scene
            if (obs/'summary.json').is_file(): continue
            plans.append((scene,start,batch))
    if a.max_chunks is not None: plans=plans[:a.max_chunks]
    print(json.dumps({'remaining_chunk_count':len(plans),'dry_run':a.dry_run},ensure_ascii=False),flush=True)
    if a.dry_run: return
    lock=a.output_root/f'.{PREFIX}.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise SystemExit(f'another N1a runner may be active: {lock}')
    os.close(fd); state=a.output_root/f'{PREFIX}_runner_state.json'; done=0
    try:
        for number,(scene,start,batch) in enumerate(plans,1):
            root=a.output_root/'batches'/batch; planroot=root/'plans'; obsroot=root/'observations'; state_data={'status':'running','chunk_index':number,'chunk_count':len(plans),'completed_chunks':done,'scene_name':scene,'seed_offset':start,'proposal_materialization_applied':False,'ap_computed':False,'ground_truth_usage':'none'}; _write(state,state_data)
            single=root/'scene.txt'; single.parent.mkdir(parents=True,exist_ok=True); single.write_text(scene+'\n')
            plan_cmd=[str(a.python),str(ROOT/'tools/build_n1_sampro3d_prompt_plan.py'),'--scene-list',str(single),'--seed-ledger-root',str(a.seed_ledger_root),'--output-root',str(planroot),'--max-seeds-per-scene',str(a.seeds_per_chunk),'--max-views-per-seed',str(a.views_per_seed),'--seed-offset',str(start)]
            sam_cmd=[str(a.python),str(ROOT/'tools/export_details_core_prompt_sam_observations.py'),'--scene-list',str(single),'--plan-root',str(planroot),'--sam-source',str(a.sam_source),'--sam-checkpoint',str(a.sam_checkpoint),'--sam-model-type',a.sam_model_type,'--device','cuda','--output-root',str(obsroot)]
            print(f'[run {number}/{len(plans)}] {batch}',flush=True); subprocess.run(plan_cmd,cwd=ROOT,check=True); subprocess.run(sam_cmd,cwd=ROOT,check=True); done+=1
        _write(state,{'status':'completed','chunk_count':len(plans),'completed_chunks':done,'proposal_materialization_applied':False,'ap_computed':False,'ground_truth_usage':'none'})
    finally: lock.unlink(missing_ok=True)
if __name__=='__main__': main()
