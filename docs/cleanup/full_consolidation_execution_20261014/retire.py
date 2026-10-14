"""Delete the approved retired families after migration and replay.

Never follows directory symlinks. Retains small historical records and full
source archives; records allocated bytes only once for each inode.
"""
from consolidate import ROOT, OUT, PLAN, rows, sha, write, journal, archive_files
from pathlib import Path
import json, os, shutil, tarfile

removed_inodes=set()
deleted=[]

def files_under(p):
    if p.is_symlink():return []
    if p.is_file():return [p]
    result=[]
    for folder,dirs,files in os.walk(p,followlinks=False):
        dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]
        result.extend(Path(folder)/n for n in files if not (Path(folder)/n).is_symlink())
    return result

def remove(p, reason):
    assert p.is_relative_to(ROOT) and not p.is_symlink(),p
    assert '.git' not in p.relative_to(ROOT).parts,p
    allocated=0;size=0;count=0
    for f in files_under(p):
        s=f.stat();key=(s.st_dev,s.st_ino)
        size+=s.st_size;count+=1
        if key not in removed_inodes:
            removed_inodes.add(key)
            # Hardlinked bytes may still exist outside the retired roots.
            if s.st_nlink==1:allocated+=s.st_blocks*512
    record=dict(path=str(p.relative_to(ROOT)),reason=reason,files=count,
                apparent_bytes=size,unique_single_link_allocated_bytes=allocated)
    journal(dict(operation='delete_retired_path_started',**record))
    if p.is_dir():shutil.rmtree(p)
    else:p.unlink()
    assert not p.exists(),p
    deleted.append(record)
    journal(dict(operation='delete_retired_path_complete',**record))

def history(paths,name):
    kept=[]
    for p in paths:
        for f in files_under(p):
            if '.git' in f.parts:continue
            n=f.name.lower()
            if (f.suffix.lower() in ['.md','.log','.yaml','.yml','.sh','.py','.toml']
                or any(k in n for k in ['receipt','contract','summary','verified','identity','seal','frozen','manifest','report'])):
                if f.stat().st_size<=4*1024**2:kept.append(f)
    archive_files(kept,name)

def main():
    # These independent checks have already passed before any large deletion.
    replay=OUT/'replay'
    for scene in ['scene0011_00','scene0153_00']:
        assert json.loads((replay/'full'/scene/'selected.json').read_text())==json.loads((ROOT/'full_bundle/results/fi1_core_ablation_val312_20261006_v1/scenes'/scene/'full_selected.json').read_text())
    for name in ['semantic_SL.json','class_agnostic_SL.json']:
        assert json.loads((replay/'sl/scene0005_00'/name).read_text())==json.loads((ROOT/'sl_bundle/protected_sources/fi1_sl_single200_dev55_20261010_v1/predictions/scene0005_00'/name).read_text())
    assert json.loads((OUT/'verification_frames_extracted.json').read_text())['count']==972
    free_before=shutil.disk_usage(ROOT).free
    # Retired weights are reproducible downloads; retain identity, not another
    # copy of the same weights under an archive name.
    models=[ROOT/r['path'] for r in rows(PLAN/'model_external_candidates.csv') if r['decision']=='RETIRE_MODEL_FAMILY_CANDIDATE']
    weights=[]
    for p in models:
        for f in files_under(p):
            if f.suffix in ['.pt','.pth','.bin','.safetensors','.joblib','.pkl']:
                weights.append(dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size))
    write(OUT/'retired_weight_identities.json',weights)
    history(models,'retired_model_records.tar.gz')
    sources=[f for p in models if '_external' in p.parts for f in files_under(p)
             if '.git' not in f.parts and f.suffix not in ['.pt','.pth','.bin','.safetensors'] and f.stat().st_size<8*1024**2]
    archive_files(sources,'retired_model_source.tar.gz')
    for p in models:remove(p,'approved YOLOE/MobileCLIP, SigLIP or ProxyCLIP retirement')
    comparisons=[ROOT/r['path'] for r in rows(PLAN/'retired_model_comparison_artifact_candidates.csv') if (ROOT/r['path']).exists()]
    history(comparisons,'retired_comparison_records.tar.gz')
    for p in comparisons:remove(p,'retired visual model comparison; records archived')
    for r in rows(PLAN/'model_external_candidates.csv'):
        if r['decision']=='ARCHIVE_SMALL_LEGACY_MODEL':
            p=ROOT/r['path'];archive_files(files_under(p),p.name+'.tar.gz');remove(p,'legacy small model archived')
    for name in ['OV3D-CG','SegDINO3D_official','ESAM']:
        p=ROOT/'_external'/name
        archive_files([f for f in files_under(p) if '.git' not in f.parts],name+'_source.tar.gz')
        remove(p,'reference implementation archived; ESAM metadata migrated')
    meta=ROOT/'_external/ESAM/ESAM-main/data/scannet200/meta_data'
    meta.parent.mkdir(parents=True,exist_ok=True)
    meta.symlink_to(os.path.relpath(ROOT/'full_bundle/metadata/scannet',meta.parent),target_is_directory=True)
    # Keep the verified original Any3DIS ZIP as its one source archive.
    source=ROOT/'_external/Any3DIS_unofficial.zip'
    dest=ROOT/'archive/retired_20261014/Any3DIS_unofficial_original.zip'
    source.rename(dest);journal(dict(operation='move_reference_zip',target=str(dest),sha256=sha(dest)))
    remove(ROOT/'_external/Any3DIS_unofficial','418/418 source files represented by retained original ZIP')
    # Retain the original locally patched file from GD's download ZIP.
    import zipfile
    p=ROOT/'_external/GroundingDINO.zip'
    with zipfile.ZipFile(p) as z:
        for name in z.namelist():
            if name.endswith('/ms_deform_attn.py'):
                dest=ROOT/'archive/retired_20261014/GroundingDINO_original_ms_deform_attn.py';dest.write_bytes(z.read(name))
    remove(p,'duplicate download; original ms_deform_attn and current patched GD source retained')
    arms=[ROOT/r['path'] for r in rows(PLAN/'val48_branch_data_candidates.csv') if r['arm'] in ['A','B']]
    # Full/SL/research actual scene route and current contract scans were checked
    # before execution. GD5/D and current GD10 are explicitly excluded here.
    history(arms,'retired_val48_AB_records.tar.gz')
    for p in arms:remove(p,'abandoned automatic SAM A/B branch; Full GD10 input retained')
    free_after=shutil.disk_usage(ROOT).free
    write(OUT/'deletion_executed.json',dict(complete=True,deleted=deleted,
          deleted_single_link_allocated_bytes=sum(r['unique_single_link_allocated_bytes'] for r in deleted),
          filesystem_free_before=free_before,filesystem_free_after=free_after,
          net_filesystem_free_delta=free_after-free_before,
          caveat='Net delta includes archives and other concurrent filesystem activity; hardlinked bytes excluded from per-root total.'))

if __name__=='__main__':main()
