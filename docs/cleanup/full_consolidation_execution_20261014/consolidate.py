"""Execute the user-authorized consolidation; journal every mutation.

Historical contracts are immutable. Compatibility links retain their paths.
Large external datasets are indexed in place, never duplicated.
"""
from pathlib import Path
import csv, hashlib, json, os, re, shutil, tarfile, sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
OUT = ROOT / 'docs/cleanup/full_consolidation_execution_20261014'
PLAN = ROOT / 'docs/cleanup/full_consolidation_plan_20261014'
WORK = Path('/home/jia/Wm/wm_open-yolo/worktrees/OpenYOLO3D-geometry-semantic-56b2c25')

def rows(p):
    return list(csv.DictReader(p.open(encoding='utf-8-sig')))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')

def journal(x):
    with (OUT/'journal.jsonl').open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')

def link(p, target):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_symlink():
        assert p.resolve()==target.resolve(), (p,target)
    elif p.exists():
        assert p.resolve()==target.resolve(), (p,target)
    else:p.symlink_to(os.path.relpath(target,p.parent),target_is_directory=target.is_dir())

def relocate(source, dest):
    assert source.exists() and not source.is_symlink(),source
    assert not dest.exists(),dest
    dest.parent.mkdir(parents=True,exist_ok=True)
    source.rename(dest)
    link(source,dest)
    journal(dict(operation='move_with_compatibility_link',source=str(source),target=str(dest)))

def archive_files(paths, name):
    dest=ROOT/'archive/retired_20261014'/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    identities={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file() and not p.is_symlink()}
    if dest.exists():
        prior=json.loads(dest.with_suffix(dest.suffix+'.manifest.json').read_text())
        assert prior['files']==identities and prior['archive_sha256']==sha(dest),dest
        return
    with tarfile.open(dest,'w:gz') as tar:
        for relative in identities:tar.add(ROOT/relative,arcname=relative,recursive=False)
    with tarfile.open(dest,'r:gz') as tar:
        for rel,expected in identities.items():assert hashlib.sha256(tar.extractfile(rel).read()).hexdigest()==expected
    write(dest.with_suffix(dest.suffix+'.manifest.json'),dict(archive_sha256=sha(dest),files=identities,verified=True))
    journal(dict(operation='verified_source_archive',path=str(dest),files=len(identities)))

def prepare():
    partition=rows(PLAN/'code_partition_candidates.csv')
    archive_files([ROOT/r['path'] for r in partition], 'all_code_before_relocation.tar.gz')
    imported=rows(ROOT/'docs/cleanup/full_sl_asset_map_20261014/current_code_import_index.csv')
    for r in imported:
        if r['source_root']==str(WORK):
            src=Path(r['actual_path']);dest=ROOT/r['relative_path']
            if dest.exists():assert sha(dest)==sha(src)
            else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            journal(dict(operation='materialize_worktree_dependency',source=str(src),target=str(dest),sha256=sha(dest)))
    snapshot=ROOT/'.local/fi1_unified_evidence_20260910/workspace'
    dest=ROOT/'full_bundle/code_sources/evidence_snapshot'
    shutil.copytree(snapshot,dest,symlinks=True)
    # Frozen snapshot retains its own ROOT and source identity. Visual/data roots
    # are project aliases; these new aliases do not replace any snapshot file.
    for n in ['models','data','output','outputs','.local','_external']:
        if not (dest/n).exists():link(dest/n,ROOT/n)
    for n in ['checkpoints','alpha_clip','clip-vit-base-patch32','groundingdino','bert-base-uncased','configs']:
        if not (dest/'pretrained'/n).exists():link(dest/'pretrained'/n,ROOT/'pretrained'/n)
    dv=ROOT/'full_bundle/models/dv3_official100'
    shutil.copytree(WORK/'pretrained/fi1_d_v3_official100_full_models_20260822',dv,symlinks=True)
    link(ROOT/'pretrained/fi1_d_v3_official100_full_models_20260822',dv)
    learn=Path('/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921/learning')
    for n in ['B.joblib','B.json','summary.json']:
        dest=ROOT/'full_bundle/models/b26'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(learn/'models'/n,dest)
        assert sha(dest)==sha(learn/'models'/n)
    # Real visual assets move on the same filesystem; old paths become aliases.
    for r in rows(PLAN/'model_external_candidates.csv'):
        if r['decision']=='KEEP_FULL':
            src=ROOT/r['path'];base='vendor' if r['path'].startswith('_external/') else 'models/visual'
            rel=Path(r['path']).relative_to('_external' if base=='vendor' else 'pretrained')
            relocate(src,ROOT/'full_bundle'/base/rel)
        elif r['decision'] in ['KEEP_GEOMETRY_RESEARCH','KEEP_RESEARCH_OPTION']:
            src=ROOT/r['path'];rel=Path(r['path'])
            relocate(src,ROOT/'research/geometry/assets'/rel)
    for n in ['config_scannet200.yaml','config_replica.yaml','configs']:
        relocate(ROOT/'pretrained'/n,ROOT/'full_bundle/models/config'/n)
    champion='train_candidate_champion_pair_union_combined_full_official100_v2_prior_corrected'
    relocate(ROOT/'output'/champion,ROOT/'full_bundle/models/champion')
    for n in ['scannetv2_train.txt','scannetv2-labels.combined.tsv']:
        src=ROOT/'_external/ESAM/ESAM-main/data/scannet200/meta_data'/n
        dest=ROOT/'full_bundle/metadata/scannet'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        assert sha(src)==sha(dest)
    # Runtime is an independently frozen release; keep all its original hashes.
    for n in ['pretrained','_external','data']:
        runtime=ROOT/'full_bundle/runtime'/n;runtime.mkdir(exist_ok=True)
        for child in (ROOT/n).iterdir():
            if not (runtime/child.name).exists():link(runtime/child.name,child)
    routes=rows(ROOT/'docs/cleanup/full_sl_asset_map_20261014/scene_asset_routes.csv')
    write(ROOT/'full_bundle/metadata/scene_routes.json',routes)
    fields={'data':'prepared_root','records_resolved':'native_records','actual_front':'gd10_frontend',
            'actual_backend':'backend','actual_own_root':'own','yolosam':'yolosam',
            'bound_geometry':'geometry.json','bound_points':'points.npz','own_evidence':'evidence.jsonl'}
    for row in routes:
        base=ROOT/'full_bundle/artifacts/scannet'/row['group']/row['scene']
        for source,name in fields.items():link(base/name,Path(row[source]))
    for r in rows(PLAN/'full_migration_source_target_plan.csv'):
        if r['kind']=='training_dataset_or_original_data':link(ROOT/r['proposed_target'],Path(r['source']))
    shutil.copy2(ROOT/'资料/训练597场景清单_20260917.json',ROOT/'full_bundle/metadata/train597.json')
    for name in ['fi1_core_ablation_val312_20261006_v1','replica_official_masks_full_20261011_v1','replica_registered_full_20261011_v1']:
        relocate(ROOT/'.local'/name,ROOT/'full_bundle/results'/name)
    for r in rows(PLAN/'full_migration_source_target_plan.csv'):
        if r['kind']=='effective_SL':relocate(Path(r['source']),ROOT/r['proposed_target'])
    for src in list((ROOT/'.local').iterdir()):
        if any(s in src.name for s in ['20261011','20261012','20261013']) and src.name.startswith('fi1_'):
            if src.is_symlink():link(ROOT/'research/geometry/experiments'/src.name,src)
            elif src.is_dir():relocate(src,ROOT/'research/geometry/experiments'/src.name)
    journal(dict(operation='asset_consolidation_stage_complete',scannet_routes=len(routes)))

def code():
    partition=rows(PLAN/'code_partition_candidates.csv')
    candidates={r['path']:r for r in partition}
    kept={r['path']:r for r in partition if not r['decision'].startswith('ARCHIVE')}
    # Include copied historical helper versions and literal subprocess/runpy
    # script references absent from the original static import analysis.
    for p in (ROOT/'tools').glob('*.py'):
        rel=str(p.relative_to(ROOT))
        if rel not in candidates:kept[rel]=dict(scope='Full',decision='KEEP_MATERIALIZED_HELPER')
    fixed=True
    while fixed:
        fixed=False
        texts='\n'.join((ROOT/p).read_text(errors='replace') for p in kept if (ROOT/p).suffix in ['.py','.sh'])
        for rel,r in candidates.items():
            if rel in kept:continue
            p=Path(rel)
            if p.name in texts or re.search(r'\b'+re.escape(p.stem)+r'\b',texts):
                kept[rel]=dict(r,decision='KEEP_DYNAMIC_REFERENCE');fixed=True
    write(OUT/'code_partition_executed.json',dict(kept=kept,archived=[p for p in candidates if p not in kept]))
    relocated=[]
    for rel,r in kept.items():
        src=ROOT/rel
        if src.is_symlink():continue
        scope=r.get('scope','')
        base='sl_bundle/code' if scope=='SL' else ('research/geometry/code' if not scope else 'full_bundle/code')
        dest=ROOT/base/rel
        before=sha(src)
        text=src.read_text()
        if src.suffix=='.py':
            depth=len(dest.relative_to(ROOT).parts)-1
            text=text.replace('Path(__file__).resolve().parents[1]',f'Path(__file__).resolve().parents[{depth}]')
            text=text.replace(".local/fi1_unified_evidence_20260910/workspace",'full_bundle/code_sources/evidence_snapshot')
            text=text.replace(str(WORK),str(ROOT))
            text=text.replace('_external/ESAM/ESAM-main/data/scannet200/meta_data','full_bundle/metadata/scannet')
            compile(text,str(dest),'exec')
        relocate(src,dest)
        dest.write_text(text)
        relocated.append(dict(original=rel,target=str(dest.relative_to(ROOT)),before_sha256=before,after_sha256=sha(dest)))
    write(OUT/'code_relocation_manifest.json',relocated)
    # Archive verified before unlinking. This removes old entrypoints, not the
    # compatibility aliases of the retained Full/SL/research code.
    retired=[ROOT/p for p in candidates if p not in kept]
    archive_files(retired,'retired_code.tar.gz')
    for p in retired:
        assert not p.is_symlink();p.unlink()
    journal(dict(operation='retire_legacy_active_code',files=len(retired)))

def frames():
    from tools.build_c1_gvc_paper_reference_quality_ledger import _frame_contract
    verified=[]
    for row in rows(PLAN/'autosam_common_frame_extraction_plan.csv'):
        src=Path(row['common_frames_source'])
        indices,by_id=_frame_contract(src)
        dest=ROOT/row['target']
        write(dest,dict(frames=[dict(frame_id=k,frame_index=v) for k,v in by_id.items()],
                        source=str(src),frame_indices=indices,exact_existing_contract=True))
        x=json.loads(dest.read_text())
        assert sorted(f['frame_index'] for f in x['frames'])==indices
        assert {f['frame_id']:f['frame_index'] for f in x['frames']}==by_id
        verified.append(dict(scene=row['scene'],group=row['group'],source=str(src),target=str(dest),sha256=sha(dest)))
    write(OUT/'verification_frames_extracted.json',dict(complete=True,count=len(verified),files=verified))
    journal(dict(operation='extract_exact_verification_frames',scenes=len(verified)))

if __name__=='__main__':globals()[sys.argv[1]]()
