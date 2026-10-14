"""Retire authorized unbound historical output, preserving summary evidence."""
from pathlib import Path
import csv, hashlib, json, os, shutil, tarfile
from audit import ROOT, OUT, dump

receipt=OUT/'output_deletion_executed.json'
assert not receipt.exists(),'Deletion receipt already exists; do not repeat'
assert json.loads((OUT/'refactor_verified.json').read_text())['complete']
assert json.loads((OUT/'legacy_geometry_verified.json').read_text())['complete']
rows=list(csv.DictReader((OUT/'output_inventory_refined.csv').open(encoding='utf-8-sig')))
targets=[r for r in rows if r['classification']=='RETIRE_CANDIDATE_NO_CURRENT_BINDING']
evidence=json.loads((OUT/'output_reference_evidence_refined.json').read_text())
archive=ROOT/'archive/retired_20261014/output_retired_summary_evidence.tar.gz'
assert not archive.exists()
free_before=shutil.disk_usage(ROOT).free
selected=[]; inventory=[];root_sizes={}
for row in targets:
    rel=row['path'];p=ROOT/rel
    assert not evidence['current_bindings'].get(rel)
    assert not evidence['physical_symlink_references'].get(rel)
    assert not evidence['code_defaults'].get(rel)
    assert p.is_dir() and not p.is_symlink() and ROOT/'output' in p.parents
    count=0;blocks=0
    for folder,dirs,names in os.walk(p,followlinks=False):
        for name in dirs+names:
            q=Path(folder)/name
            if q.is_symlink():
                inventory.append(dict(path=str(q.relative_to(ROOT)),kind='symlink',target=os.readlink(q)))
        dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]
        for name in names:
            q=Path(folder)/name
            if q.is_symlink():continue
            s=q.stat();count+=1;blocks+=s.st_blocks*512
            key=name.lower()
            preserve=(q.suffix in ['.md','.txt','.log'] or any(x in key for x in ['summary','contract','manifest','report','receipt','verified','evaluation','metrics','ap.npy','config','identity']))
            saved=preserve and s.st_size<=4*1024**2
            item=dict(path=str(q.relative_to(ROOT)),kind='file',bytes=s.st_size,allocated_bytes=s.st_blocks*512,summary_archived=saved)
            if saved:
                item['sha256']=hashlib.sha256(q.read_bytes()).hexdigest();selected.append(q)
            inventory.append(item)
    root_sizes[rel]=dict(files=count,allocated_bytes=blocks)
with tarfile.open(archive,'w:gz') as tar:
    for p in selected:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
with tarfile.open(archive) as tar:
    for item in inventory:
        if item.get('summary_archived'):
            assert hashlib.sha256(tar.extractfile(item['path']).read()).hexdigest()==item['sha256']
dump('output_removed_file_inventory.json',inventory)
dump('output_summary_archive_verified.json',dict(complete=True,archive=str(archive.relative_to(ROOT)),files=len(selected),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size))
deleted=[]
for row in targets:
    rel=row['path'];p=ROOT/rel
    shutil.rmtree(p)
    assert not p.exists()
    deleted.append(dict(path=rel,**root_sizes[rel]))
    with (OUT/'output_deletion_journal.jsonl').open('a') as f:f.write(json.dumps(deleted[-1])+'\n')
free_after=shutil.disk_usage(ROOT).free
result=dict(complete=True,deleted=deleted,roots=len(deleted),allocated_GiB=sum(d['allocated_bytes'] for d in deleted)/2**30,
            summary_archive=str(archive.relative_to(ROOT)),summary_files_preserved=len(selected),
            free_before=free_before,free_after=free_after,net_free_increase_GiB=(free_after-free_before)/2**30,
            only_retired_unbound_roots=True,current_contracts_modified=False,
            retained_roots=[dict(path=r['path'],classification=r['classification']) for r in rows if r not in targets])
dump(receipt.name,result)
print(json.dumps({k:v for k,v in result.items() if k not in ['deleted','retained_roots']}))
