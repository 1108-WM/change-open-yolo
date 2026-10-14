#!/usr/bin/env python3
"""Freeze geometry inspection regions before training-scope GT diagnosis."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,sys,time,shutil
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_deletion_blocks as old
from tools import fi1_local_region_proposals as core
from tools.fi1_frontend_protocol import write_json,sha256_file as sha,digest
SOURCE=old.SOURCE;PREVIOUS=old.OUT
OUT=ROOT/'.local/fi1_local_region_supply_development200_20261013_v1'
read=old.read;seal=old.seal;OBS=old.OBS
ARMS=('G_cached','G_float32','geometry_raw','geometry48','geometry96','geometry192')


def stage(name,**kw):write_json(OUT/'execution.json',dict(stage=name,updated=time.time(),**old.plain(kw)));print(name,old.plain(kw),flush=True)


class Guard:
    def __init__(self,c,mode,fold=None):self.c,self.mode,self.fold=c,mode,fold
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,Path)):return
        p=Path(args[0]).resolve();name=str(p)
        if p.suffix in ('.py','.pyc','.so'):return
        if 'independent_analysis' in name or p.name in ('candidate_diagnosis.csv','block_opportunity_diagnosis.json','audit.json','model_probe.json'):raise PermissionError('historical GT diagnostic inputs forbidden')
        scene=next((s for s in self.c['scenes'] if '/'+s+'/' in name or p.name==s+'.txt'),None)
        raw=any(q.startswith('ground_truth') for q in p.parts)
        label='/labels/' in name and any(str(root.resolve()) in name for root in (SOURCE,PREVIOUS))
        cached='/diagnosis/' in name and str(OUT.resolve()) in name
        if raw or label or cached:
            if self.mode not in ('diagnose','verify') or not (OUT/'regions_frozen.json').exists():raise PermissionError('GT/cache before proposal seal')
            if scene is None or self.fold is None or self.c['folds'][scene]==self.fold:raise PermissionError('outer-heldout diagnosis input forbidden')
            if raw and name not in self.c['GT_inputs']:raise PermissionError('GT whitelist')
        if self.mode=='generate' and any(str(root.resolve())+'/predictions/' in name or str(root.resolve())+'/generators/' in name for root in (PREVIOUS,SOURCE)):
            raise PermissionError('geometry generation cannot read learned proposals')
        if self.mode=='generate' and str(OUT/'diagnosis') in name:raise PermissionError('diagnostic output cannot guide generation')


def prepare():
    assert not OUT.exists();assert shutil.disk_usage(ROOT).free>=50*1024**3
    c0=read(PREVIOUS/'contract.json');assert read(PREVIOUS/'completion_verified.json')['complete']
    files=[ROOT/'tools'/n for n in ('fi1_local_region_proposals.py','run_fi1_local_region_supply.py','diagnose_fi1_local_region_supply.py','verify_fi1_local_region_supply.py','run_fi1_local_region_supply.sh')]
    files+=[ROOT/'tests/test_fi1_local_region_proposals.py',ROOT/'docs/FI1_LOCAL_REGION_SUPPLY_PROTOCOL_20261013.md']
    code=dict(c0['code']);code[str(ROOT/'tools/fi1_split_grow_core.py')]=sha(ROOT/'tools/fi1_split_grow_core.py');code.update({str(p):sha(p) for p in files})
    bindings={str(PREVIOUS/n):sha(PREVIOUS/n) for n in ('contract.json','generators_frozen.json','labels_frozen.json','completion_verified.json')}
    for root,stages in ((SOURCE,('plan','features','labels')),(PREVIOUS,('generators','labels'))):
        for name in stages:bindings.update(read(root/(name+'_frozen.json'))['outputs'])
    # All integrity hashing occurs before any stage's annotation access guard.
    old.previous.original.audit.prior.old.verify_files(code);old.previous.original.audit.prior.old.verify_files(bindings)
    c={k:c0[k] for k in ('scenes','folds','groups','fixed10','class_ids','jobs','GT_inputs','inputs','frozen_SL_models')}
    c.update(version=OUT.name,parameters=core.PARAMETERS,supply_gate=core.SUPPLY_GATE,arms=ARMS,code=code,source_bindings=bindings,
             GT_free_regions=True,diagnosis_scope='Original outer train160 only, per fold. Five training sets overlap; union200 is exposed development diagnostic, never formal OOF benefit.',
             proposal_action='Inspect only, no learned deletion. Complete six geometric families and fixed48/96/192 subsets frozen before GT. Main96 unchanged after diagnosis.',
             legacy_probability='G_cached matches sealed old inner proposals exactly. G_float32 rebuilds >=.8 in float32 on same stored probabilities, no new inference.',
             no_training=True,new_frames=False,new_features=False,SL_modified=False,AP_run=False)
    OUT.mkdir();seal(OUT/'contract.json',c)
    for p in files:
        target=OUT/'code_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    stage('prepared',scenes=200,parents=1599,primary_budget=96)


def generate_scene(s):
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'generate').hook);d=OUT/'regions'/s
    if (d/'receipt.json').exists():return read(d/'receipt.json')
    # No original superpoint/semantic/instance values enter proposal formation.
    raw=np.load(c['jobs'][s]['paths']['processed'],mmap_mode='r');arrays={};records=[]
    with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(SOURCE/'plan'/s/'domains.npz') as domain:
        for row in read(SOURCE/'plan'/s/'candidates.json'):
            cid=row['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];points=np.asarray(raw[halo,:9],np.float32)
            blocks,metadata,budget,trace=core.propose(points[:,:3],points[:,3:6],points[:,6:9],parent,halo,z[cid+'_'+OBS+'_groups'],z[cid+'_neighbors'],domain[cid+'_roles'][3])
            flat,offset=old.pack(blocks);arrays[cid+'_block_indices']=flat;arrays[cid+'_block_offsets']=offset
            for b,ids in budget.items():arrays[cid+'_budget'+b]=ids
            records.append(dict(candidate_id=cid,source=row['source'],parent=row['parent'],regions=metadata,trace=trace))
    d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'regions.npz',**arrays);seal(d/'meta.json',records);oo={str(d/n):sha(d/n) for n in ('regions.npz','meta.json')}
    receipt=dict(complete=True,GT_read=False,probability_read=False,scene=s,regions=sum(len(r['regions']) for r in records),primary_regions=sum(r['trace']['budget_counts']['96'] if 'budget_counts' in r['trace'] else 0 for r in records),outputs=oo);seal(d/'receipt.json',receipt);return receipt


def generate():
    c=read(OUT/'contract.json');outputs={};counts=dict(raw_regions=0,primary_regions=0)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(generate_scene,s) for s in c['scenes']])):
            r=f.result();outputs.update(r['outputs']);counts['raw_regions']+=r['regions'];counts['primary_regions']+=r['primary_regions']
            if (i+1)%10==0:stage('GT_free_multiscale_regions',completed=i+1,total=200,**counts)
    seal(OUT/'regions_frozen.json',dict(complete=True,GT_read=False,probability_read=False,outputs=outputs,counts=counts,parameters_sha256=digest(core.PARAMETERS)))
    stage('all_regions_sealed',**counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','generate'));args=parser.parse_args();globals()[args.mode]()
