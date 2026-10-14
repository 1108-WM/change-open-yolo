"""Retire reviewed old val48 sampling/Q/smoke and early class coverage artifacts."""
from pathlib import Path
import csv,json,hashlib,os,shutil,tarfile,time
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'local_inventory.csv').open(encoding='utf-8-sig')))
refs=json.loads((OUT/'local_reference_evidence.json').read_text())
# These are explicitly historical alternatives: current Full is stride10,
# official100 D-v3, matching597 B26. No current experiment uses these outputs.
def reason(n):
    if n.startswith(('fi1_val48_grounded_stride15_','fi1_val48_grounded_stride20_')):return 'GD15/GD20 comparison retired; production GD10 retained'
    if n.startswith(('fi1_val48_ad_full597_q_','fi1_val48_bc_full597_q_','fi1_val48_grounded_stride10_full597_q_','fi1_full597_bq_error_diagnosis','fi1_full597_q_bottleneck_')):return 'old A/B/C or Q29 scoring comparison; not current B26'
    if n.startswith(('fi1_val48_grounded_stride10_backend_smoke_','fi1_val48_grounded_stride10_evidence_smoke_','fi1_val48_grounded_stride10_reprojection_replay_')):return 'single-scene smoke superseded by retained formal GD10/backend/evidence'
    if n.startswith(('fi1_v32_','fi1_v33_')):return 'early 22-geometry class-coverage pilot; not production input'
selected=[dict(r,reason=reason(r['name'])) for r in rows if reason(r['name']) and r['classification']=='RETIRE_UNBOUND_HISTORY']
names={r['name'] for r in selected}
for r in selected:
    assert not refs['direct'].get(r['name'])
    assert not refs['symlink_inputs'].get(r['name'])
    assert not refs['source_defaults'].get(r['name'])
    r['historical_inbound']=[k for k,v in refs['edges'].items() if r['name'] in v]
    outside=[k for k in r['historical_inbound'] if k not in names]
    # Outside references here are history-only audits/bootstrap inventories,
    # not inputs to the registered Full/SL/current geometry seed graph.
    assert all(k.startswith(('storage_','fi1_val48_stride_bootstrap_')) for k in outside),(r['name'],outside)
plan=dict(candidates=selected,root_count=len(selected),allocated_bytes=sum(int(r['allocated_bytes']) for r in selected),
          basis='manual role review plus current direct metadata, source default, frame-index and inward alias checks; does not delete all 160 audit candidates')
(OUT/'local_retirement_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print('Reviewed roots',len(selected),'GiB',plan['allocated_bytes']/2**30,flush=True)
if '--execute' not in __import__('sys').argv:raise SystemExit
assert not (OUT/'local_deletion_executed.json').exists(),'Do not repeat deletion'
archive=ROOT/'archive/retired_20261014/deep_local_history_capsule.tar.gz'
assert not archive.exists()
kept=[];inventory=[]
with tarfile.open(archive,'w:gz') as tar:
    for r in selected:
        base=ROOT/r['path'];assert base.is_dir() and not base.is_symlink()
        for folder,dirs,files in os.walk(base,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
            for n in files:
                p=Path(folder)/n
                if p.is_symlink():continue
                s=p.stat();rel=p.relative_to(base)
                # Preserve conclusions, identity, source snapshots, execution
                # logs and final aggregate evaluation; not bulk predictions.
                keep=(len(rel.parts)==1 or 'source_snapshot' in rel.parts or 'code_snapshot' in rel.parts
                      or ('evaluation' in rel.parts and p.suffix in ['.json','.csv','.md','.txt']))
                if keep:
                    tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
                    kept.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
                inventory.append(dict(path=str(p.relative_to(ROOT)),bytes=s.st_size,allocated_bytes=s.st_blocks*512,archived=keep))
with tarfile.open(archive) as tar:
    for r in kept:assert hashlib.sha256(tar.extractfile(r['path']).read()).hexdigest()==r['sha256'],r['path']
(OUT/'local_history_capsule_manifest.json').write_text(json.dumps(kept,ensure_ascii=False,indent=2)+'\n')
(OUT/'local_removed_file_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
free_before=shutil.disk_usage(ROOT).free
for r in selected:
    p=ROOT/r['path'];assert p.parent==ROOT/'.local' and not p.is_symlink()
    shutil.rmtree(p)
    with (OUT/'local_deletion_journal.jsonl').open('a') as f:f.write(json.dumps(dict(path=r['path'],reason=r['reason']),ensure_ascii=False)+'\n')
free_after=shutil.disk_usage(ROOT).free
result=dict(complete=True,deleted_roots=selected,root_count=len(selected),deleted_allocated_bytes=plan['allocated_bytes'],
            archive=str(archive.relative_to(ROOT)),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            archived_files=len(kept),removed_real_files=len(inventory),free_before=free_before,free_after=free_after,free_delta=free_after-free_before)
(OUT/'local_deletion_executed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print({k:v for k,v in result.items() if k!='deleted_roots'})
