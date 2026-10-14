"""Retire the two legacy fusion roots after removing their sole CLI defaults."""
from pathlib import Path
import csv, hashlib, json, os, shutil, tarfile
from audit import ROOT, OUT, dump, codeaudit

codeaudit()
targets=['output/backprojection_candidates_scannet200_mv_m20','output/sam_fused_proposals_scannet200_s5_m30_prefilter']
receipt=OUT/'legacy_fusion_output_deletion_executed.json'
assert not receipt.exists()
bindings=json.loads((OUT/'output_reference_evidence_refined.json').read_text())
for rel in targets:
    assert not bindings['current_bindings'].get(rel)
    assert not bindings['physical_symlink_references'].get(rel)
    for base in ['full_bundle/code','sl_bundle/code','research/geometry/code']:
        assert not any(rel in p.read_text() for p in (ROOT/base).rglob('*.py'))
free_before=shutil.disk_usage(ROOT).free
archive=ROOT/'archive/retired_20261014/legacy_fusion_output_summary_evidence.tar.gz'
assert not archive.exists()
inventory=[];selected=[];sizes={}
for rel in targets:
    root=ROOT/rel;assert root.is_dir() and not root.is_symlink()
    blocks=0;count=0
    for folder,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]
        for name in names:
            p=Path(folder)/name
            if p.is_symlink():inventory.append(dict(path=str(p.relative_to(ROOT)),kind='symlink',target=os.readlink(p)));continue
            s=p.stat();blocks+=s.st_blocks*512;count+=1
            keep=p.suffix in ['.json','.jsonl','.csv','.md','.txt','.log'] and s.st_size<=4*1024**2
            record=dict(path=str(p.relative_to(ROOT)),kind='file',bytes=s.st_size,allocated_bytes=s.st_blocks*512,summary_archived=keep)
            if keep:record['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();selected.append(p)
            inventory.append(record)
    sizes[rel]=dict(files=count,allocated_bytes=blocks)
with tarfile.open(archive,'w:gz') as tar:
    for p in selected:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
with tarfile.open(archive) as tar:
    for r in inventory:
        if r.get('summary_archived'):assert hashlib.sha256(tar.extractfile(r['path']).read()).hexdigest()==r['sha256']
dump('legacy_fusion_removed_file_inventory.json',inventory)
deleted=[]
for rel in targets:
    shutil.rmtree(ROOT/rel);assert not (ROOT/rel).exists();deleted.append(dict(path=rel,**sizes[rel]))
result=dict(complete=True,deleted=deleted,roots=2,allocated_GiB=sum(r['allocated_bytes'] for r in deleted)/2**30,
    net_free_increase_GiB=(shutil.disk_usage(ROOT).free-free_before)/2**30,
    summary_archive=str(archive.relative_to(ROOT)),summary_files_preserved=len(selected),
    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
dump(receipt.name,result);print(json.dumps(result))
