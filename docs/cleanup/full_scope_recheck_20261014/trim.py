"""Correct oversized active source snapshots; archive before removing files."""
from pathlib import Path
import ast, hashlib, json, os, shutil, tarfile

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SNAP=ROOT/'full_bundle/code_sources/evidence_snapshot'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def archive(paths,name):
    dest=ROOT/'archive/retired_20261014'/name
    files={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file() and not p.is_symlink()}
    if dest.exists():
        prior=json.loads(dest.with_suffix('.manifest.json').read_text())
        assert prior['files']==files and prior['archive_sha256']==sha(dest)
    else:
        with tarfile.open(dest,'w:gz') as t:
            for rel in files:t.add(ROOT/rel,arcname=rel,recursive=False)
        with tarfile.open(dest,'r:gz') as t:
            for rel,h in files.items():assert hashlib.sha256(t.extractfile(rel).read()).hexdigest()==h
        write(dest.with_suffix('.manifest.json'),dict(files=files,archive_sha256=sha(dest),verified=True))
    return dest

def main():
    seeds=['export_fi1_refined_semantic_evidence','fi1_unified_evidence_core','fi1_full_alpha_adapter',
           'build_official100_fi1_b2_multisemantic','build_dm_sms1_alpha_embedding_ledger','build_z1_yoloworld_multiview_distribution_ledger']
    queue=[SNAP/'tools'/f'{n}.py' for n in seeds];keep=set()
    while queue:
        p=queue.pop()
        if p in keep or not p.exists():continue
        keep.add(p)
        for n in ast.walk(ast.parse(p.read_text())):
            names=[]
            if isinstance(n,ast.Import):names=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom):names=[n.module or '']+[(n.module or '')+'.'+a.name for a in n.names]
            for mod in names:
                if mod.split('.')[0] not in ['tools','utils','evaluate']:continue
                for q in [SNAP/(mod.replace('.','/')+'.py'),SNAP/mod.replace('.','/')/'__init__.py']:
                    if q.is_file():queue.append(q);break
    for base in ['tools','utils','evaluate']:
        p=SNAP/base/'__init__.py'
        if p.exists():keep.add(p)
    config=SNAP/'pretrained/config_scannet200.yaml'
    keep.add(config)
    files=[];links=[]
    for folder,dirs,names in os.walk(SNAP,followlinks=False):
        for n in dirs+names:
            p=Path(folder)/n
            if p.is_symlink():links.append(p)
        dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
        files.extend(Path(folder)/n for n in names if not (Path(folder)/n).is_symlink())
    archived=archive(files,'evidence_snapshot_before_minimization.tar.gz')
    source_ids={str(p.relative_to(SNAP)):sha(p) for p in keep}
    removed=[]
    for p in files:
        if p not in keep:removed.append(str(p.relative_to(SNAP)));p.unlink()
    # Broad project aliases falsely expose unrelated research. Source helper
    # defaults are preserved in archived bytes; callers supply actual roots.
    alias_records=[]
    for p in links:
        alias_records.append(dict(path=str(p.relative_to(SNAP)),target=os.readlink(p)))
        p.unlink()
    for folder,dirs,names in os.walk(SNAP,topdown=False):
        p=Path(folder)
        if p!=SNAP and not list(p.iterdir()):p.rmdir()
    for rel,h in source_ids.items():assert sha(SNAP/rel)==h
    # These five diagnostic CLIs entered Full solely through the previous
    # broad identifier search; exact imports/dynamic literals have no caller.
    names=['diagnose_d2b_native_component_competition_oracle_gt.py',
           'diagnose_mv3dis_boundary_action_oracle_gt.py',
           'diagnose_mv3dis_global_feasible_action_oracle_gt.py',
           'diagnose_n1_sampro3d_candidate_space_oracle_gt.py',
           'diagnose_n2_medoid_candidate_oracle_gt.py']
    candidates=[ROOT/'full_bundle/code/tools'/n for n in names]
    archive(candidates,'full_false_positive_diagnostic_sources.tar.gz')
    original_manifest=ROOT/'docs/cleanup/full_consolidation_execution_20261014/code_relocation_manifest.json'
    # Original execution manifest is historical. New current inventory records
    # these five archived paths rather than silently editing that old receipt.
    for p in candidates:
        alias=ROOT/'tools'/p.name
        assert alias.is_symlink() and alias.resolve()==p
        alias.unlink();p.unlink()
    write(OUT/'source_trim_executed.json',dict(complete=True,snapshot_archive=str(archived),
        snapshot_kept=source_ids,snapshot_removed_files=removed,removed_broad_aliases=alias_records,
        removed_full_diagnostics=[p.name for p in candidates],
        note='删除活跃副本；源码完整压缩归档逐文件核验，原23个静态闭包源码字节保持'))
    print(json.dumps(dict(snapshot_source_files_kept=len(source_ids),snapshot_files_removed=len(removed),
                         broad_links_removed=len(alias_records),full_diagnostics_archived=len(candidates)),ensure_ascii=False))

if __name__=='__main__':main()
