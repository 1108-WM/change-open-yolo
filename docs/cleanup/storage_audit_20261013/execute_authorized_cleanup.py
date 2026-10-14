"""Execute ONLY the cache whitelist and three previously hash-verified duplicates.

Explicitly authorized in the follow-up user turn; no .local result tree deletion.
Deduplicated historical files retain both paths and hashes, and become read-only
hardlinks. Target mtime becomes source mtime; original metadata is journaled.
Never rerun historical builders into these frozen paths.
"""
import csv, hashlib, json, os, stat, subprocess, time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RECEIPT = OUT/'cleanup_execution.json'
JOURNAL = OUT/'cleanup_actions.jsonl'
assert not RECEIPT.exists() and not JOURNAL.exists(), 'Already attempted: inspect receipt/journal before any recovery'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def inside(rel):
    p=ROOT/rel
    assert p.resolve()==p and ROOT in p.parents, rel
    assert p.is_file() and not p.is_symlink(), rel
    return p

def identity(p):
    s=p.stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def record(row):
    with JOURNAL.open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
caches=list(csv.DictReader((OUT/'safe_cache_candidates.csv').open(encoding='utf-8-sig')))
duplicates=json.loads((OUT/'duplicate_diagnostic_check.json').read_text())
assert len(duplicates)==3
ready=[];skipped=[];source_hashes={}
for r in caches:
    rel=r['path'];p=ROOT/rel
    assert rel not in tracked
    assert p.parent in [ROOT/q for q in ('__pycache__','tools/__pycache__','tests/__pycache__','utils/__pycache__','evaluate/__pycache__')] or rel.startswith('.pytest_cache/')
    if not p.exists():skipped.append(dict(path=rel,reason='already absent'));continue
    inside(rel)
    if r['source']:
        src=inside(r['source']);source_hashes[r['source']]=sha(src)
    if sha(p)!=r['sha256']:
        skipped.append(dict(path=rel,reason='content changed since whitelist'));continue
    ready.append((r,identity(p)))

pairs=[]
for r in duplicates:
    assert r['left'].startswith('docs/diagnostics/official100_fi1_d_v3_stage_c_v2_member_dataset_20260822/')
    assert r['right'].startswith('docs/diagnostics/official100_fi1_d_v3_stage_c_v2_member_dataset_reproduction_20260822/')
    a,b=inside(r['left']),inside(r['right'])
    assert a.name==b.name and a.name in ('pair_unions.jsonl','member_removal_dataset.jsonl','summary.json')
    assert r['left'] not in tracked and r['right'] not in tracked
    assert sha(a)==sha(b)==r['left_sha256']==r['right_sha256']
    sa,sb=a.stat(),b.stat()
    assert sa.st_dev==sb.st_dev and sa.st_ino!=sb.st_ino and sa.st_nlink==sb.st_nlink==1
    assert stat.S_IMODE(sa.st_mode)==stat.S_IMODE(sb.st_mode)
    pairs.append((r,identity(a),identity(b),stat.S_IMODE(sa.st_mode),sb.st_blocks*512))

orphans=list(csv.DictReader((OUT/'cache_exceptions_keep.csv').open(encoding='utf-8-sig')))
orphan_hashes={r['path']:sha(inside(r['path'])) for r in orphans}
free_before=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
record(dict(action='preflight',time=time.time(),cache_files=len(ready),skipped=skipped,
            duplicate_files=len(pairs),source_files=len(source_hashes),orphan_files=len(orphan_hashes)))
cache_bytes=0;dedup_bytes=0
for r,ident in ready:
    p=inside(r['path']);assert identity(p)==ident
    s=p.stat();p.unlink()
    released=s.st_blocks*512 if s.st_nlink==1 else 0;cache_bytes+=released
    record(dict(action='unlink_regenerable_cache',path=r['path'],sha256=r['sha256'],released_allocated_bytes=released))

for r,ia,ib,old_mode,allocated in pairs:
    a,b=inside(r['left']),inside(r['right'])
    assert identity(a)==ia and identity(b)==ib
    tmp=b.with_name(b.name+'.dedup_link_pending')
    backup=b.with_name(b.name+'.dedup_rollback_pending')
    assert not tmp.exists() and not backup.exists()
    record(dict(action='dedup_begin',source=r['left'],target=r['right'],old_mode=oct(old_mode),
                old_source_identity=ia,old_target_identity=ib,sha256=r['left_sha256']))
    os.link(b,backup)  # keep original blocks until validation completes
    try:
        os.chmod(a,old_mode & ~0o222)  # historical frozen data: prevent accidental shared writes
        os.link(a,tmp)
        os.replace(tmp,b)  # atomic path-preserving replacement
        assert a.stat().st_ino==b.stat().st_ino and sha(b)==r['left_sha256']
        backup.unlink();dedup_bytes+=allocated
    except BaseException:
        if backup.exists():os.replace(backup,b)
        if tmp.exists():tmp.unlink()
        os.chmod(a,old_mode)
        raise
    record(dict(action='dedup_verified',source=r['left'],target=r['right'],sha256=r['left_sha256'],
                new_mode=oct(stat.S_IMODE(a.stat().st_mode)),nlink=a.stat().st_nlink,
                released_allocated_bytes=allocated))

assert all(sha(inside(p))==h for p,h in source_hashes.items())
assert all(sha(inside(p))==h for p,h in orphan_hashes.items())
for r in duplicates:
    a,b=inside(r['left']),inside(r['right'])
    assert sha(a)==sha(b)==r['left_sha256'] and os.path.samefile(a,b)
    assert not (stat.S_IMODE(a.stat().st_mode)&0o222)
for r,_ in ready:assert not (ROOT/r['path']).exists(), 'Cache regenerated during cleanup: inspect'
integrity=[]
for r in json.loads((OUT/'asset_integrity_checks.json').read_text()):
    actual=sha(Path(r['path']))
    integrity.append(dict(path=r['path'],sha256=actual,passed=actual==r['expected_sha256']))
assert all(r['passed'] for r in integrity)
free_after=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
result=dict(complete=True,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    authorization='User authorized low-risk cleanup and retaining one copy of duplicate diagnostic data.',
    cache_files_deleted=len(ready),cache_skipped=skipped,cache_allocated_bytes_removed=cache_bytes,
    duplicate_files_replaced_by_readonly_hardlinks=len(pairs),duplicate_allocated_bytes_removed=dedup_bytes,
    removed_data_allocation_bytes=cache_bytes+dedup_bytes,
    original_sources_verified=len(source_hashes),orphan_caches_preserved_verified=len(orphan_hashes),
    asset_integrity=integrity,free_before=free_before,free_after=free_after,
    free_delta_note='df change includes concurrent filesystem activity; per-file allocation is the cleanup accounting.',
    experiment_tree_deletions=0,geometry_or_prediction_content_changed=False,
    historical_audit_snapshots_unchanged=True,
    dedup_policy='Both original paths retained. Mode 0444 prevents accidental shared writes. Regenerate historical datasets in a new directory.')
RECEIPT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='asset_integrity'},ensure_ascii=False,indent=2),flush=True)
