#!/usr/bin/env python3
"""Execution-only continuation: immutable v1 scene algorithm, four CPU workers."""
import sys,time,shutil
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from tools import run_fi1_complementary_fragment_diagnostic as run
from threadpoolctl import threadpool_limits

def worker(c,s):
    sys.addaudithook(run.Guard(c).hook)
    with threadpool_limits(1):
        return s,run.feature_scene(c,s)

def main():
    c=run.read(run.OUT/'contract.json');sys.addaudithook(run.Guard(c).hook)
    run.parent.diag.prior.src.ctl.base.verify_files(c['code'])
    assert not (run.OUT/'features_frozen.json').exists()
    extension=run.OUT/'execution_extension.json'
    if not extension.exists():
        snapshot=run.OUT/'execution_code_snapshot';snapshot.mkdir()
        shutil.copyfile(__file__,snapshot/Path(__file__).name)
        run.freeze_json(extension,dict(execution_only=True,workers=4,source=__file__,sha256=run.sha(__file__),algorithm_changes=False,GT_used=False,contract_sha256=run.sha(run.OUT/'contract.json')))
    done={};todo=[]
    for s in c['scenes']:
        d=run.OUT/'features'/s
        if (d/'receipt.json').exists():
            r=run.read(d/'receipt.json');assert r['complete'] and r['GT_used'] is False
            run.parent.diag.prior.src.ctl.base.verify_files(r['outputs']);done[s]=r
        else:
            if d.exists():
                archive=run.OUT/'interrupted_features';archive.mkdir(exist_ok=True)
                d.rename(archive/(s+'_'+str(time.time_ns())))
            todo.append(s)
    run.stage('GT_free_parallel_continuation',completed=len(done),total=len(c['scenes']))
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,c,s) for s in todo]
        for f in as_completed(futures):
            s,r=f.result();done[s]=r
            run.stage('GT_free_parallel_continuation',completed=len(done),scene=s,total=len(c['scenes']))
    outputs={};stats=Counter();projection=Counter()
    for s in c['scenes']:
        r=done[s];outputs.update(r['outputs']);stats.update(r['counts']);projection.update(r['projection_note'])
    run.freeze_json(run.OUT/'features_frozen.json',dict(complete=True,outputs=outputs,counts=dict(stats),projection_note=dict(projection),GT_used=False,execution_extension_sha256=run.sha(extension)))
    run.stage('features_sealed',counts=dict(stats))

if __name__=='__main__':main()
