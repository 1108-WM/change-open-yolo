"""Read-only scoped storage audit. Writes reports here, never modifies inputs."""
import collections,csv,gzip,hashlib,json,os,re,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
REL='.local/fi1_val48_frontend_full_pipeline_20260914'
TARGET=ROOT/REL
rows=[];links=[];errors=[];dirs=collections.Counter();kinds=collections.Counter();arms=collections.Counter()
seen=set()

def kind(p):
    name=p.name;s=str(p)
    if name=='candidate_relations.jsonl':return 'stage_b_candidate_relations'
    if name=='merge_round_relations.jsonl':return 'frontend_merge_relations'
    if name=='same_frame_hierarchy_relations.jsonl':return 'frontend_hierarchy_relations'
    if '/alpha_completed/fresh_cache/' in s:return 'alpha_fresh_cache'
    if '/alpha_completed/' in s and name=='evidence.jsonl':return 'completed_own_evidence'
    if '/alpha_bound/' in s:return 'alpha_bound_intermediate'
    if '/views/' in s:return 'own_view_export'
    if '/geometry_binding/' in s:return 'final_geometry_binding'
    if '/predictions/' in s:return 'final_predictions'
    if 'observations' in name:return 'observations'
    if '/refined_points/' in s or p.suffix=='.npz':return 'pointsets_or_npz'
    return 'other'

for base,sub,files in os.walk(TARGET,followlinks=False):
    for d in sub[:]:
        p=Path(base)/d
        if p.is_symlink():links.append(dict(path=str(p.relative_to(ROOT)),target=os.readlink(p)));sub.remove(d)
    for name in files:
        p=Path(base)/name
        if p.is_symlink():links.append(dict(path=str(p.relative_to(ROOT)),target=os.readlink(p)));continue
        s=p.stat();key=(s.st_dev,s.st_ino);unique=key not in seen;seen.add(key)
        r=dict(path=str(p.relative_to(ROOT)),bytes=s.st_size,allocated=s.st_blocks*512,
               unique_allocated=s.st_blocks*512 if unique else 0,nlink=s.st_nlink,device=s.st_dev,inode=s.st_ino,
               mtime_ns=s.st_mtime_ns,kind=kind(p))
        rows.append(r);kinds[r['kind']]+=r['unique_allocated']
        rp=str(p.relative_to(TARGET));parts=rp.split('/')
        for n in range(1,min(4,len(parts))):dirs['/'.join(parts[:n])]+=r['unique_allocated']
        m=re.search(r'/(?:arms|backend|own_evidence)/([ABCD])/',str(p))
        if m:arms[m.group(1)]+=r['unique_allocated']
with gzip.open(OUT/'files.csv.gz','wt',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
inventory=dict(files=len(rows),allocated_files_bytes=sum(r['unique_allocated'] for r in rows),
               kinds=dict(kinds),arms=dict(arms),directories=dict(dirs),links=links,errors=errors)
(OUT/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
print('inventory',len(rows),{k:round(v/2**30,3) for k,v in kinds.items()},flush=True)

# Scoped incoming references. Avoid nested per-scene JSON (many GiB of arrays).
pattern=re.compile(re.escape(str(TARGET))+r'[^\s"\\<>]*|'+re.escape(REL)+r'[^\s"\\<>]*')
scan=[];skipped=[]
for experiment in (ROOT/'.local').iterdir():
    if not experiment.is_dir() or experiment==TARGET:continue
    for p in experiment.glob('*.json'):
        if p.stat().st_size>16*2**20:skipped.append(str(p.relative_to(ROOT)));continue
        scan.append(p)
for p in (ROOT/'tools').glob('*'):
    if p.suffix in ('.py','.sh'):scan.append(p)
for p in (ROOT/'outputs').glob('*/manifest.json'):scan.append(p)
references=[]
for p in scan:
    try:text=p.read_text()
    except (OSError,UnicodeError) as e:errors.append(dict(path=str(p),error=str(e)));continue
    if REL not in text:continue
    hits=sorted(set(m.group(0).rstrip("'),;") for m in pattern.finditer(text)))
    if hits:references.append(dict(source=str(p.relative_to(ROOT)),references=hits))
(OUT/'incoming_references.json').write_text(json.dumps(dict(files_scanned=len(scan),skipped_over_16MiB=skipped,
    scope='top-level local experiment JSON; tools py/sh; output manifests; excludes nested per-scene data; not an exhaustive dynamic dependency proof',read_errors=errors,hits=references),ensure_ascii=False,indent=2)+'\n')
print('incoming refs',len(scan),len(references),'skipped',len(skipped),flush=True)

# Large exact duplicate groups, stable identities and no symlink following.
def fingerprint(r):
    p=ROOT/r['path'];h=hashlib.sha256()
    with p.open('rb') as f:
        h.update(f.read(65536));f.seek(max(0,r['bytes']-65536));h.update(f.read(65536))
    return h.hexdigest()
def sha(r):
    p=ROOT/r['path'];s=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(r['device'],r['inode'],r['bytes'],r['mtime_ns'])
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    assert p.stat().st_mtime_ns==r['mtime_ns'];return h.hexdigest()
groups=collections.defaultdict(list)
for r in rows:
    if r['bytes']>=2**20:groups[(r['bytes'],Path(r['path']).suffix)].append(r)
duplicates=[];checked=0
for g in groups.values():
    uniq={ (r['device'],r['inode']):r for r in g }
    if len(uniq)<2:continue
    quick=collections.defaultdict(list)
    for r in uniq.values():quick[fingerprint(r)].append(r)
    for subgroup in quick.values():
        if len(subgroup)<2:continue
        hashes=collections.defaultdict(list)
        for r in subgroup:hashes[sha(r)].append(r);checked+=1
        for h,files in hashes.items():
            if len(files)>1:
                # Keep first path as canonical; only single-link redundant inodes count.
                files=sorted(files,key=lambda r:(-r['nlink'],r['path']))
                duplicates.append(dict(sha256=h,files=files,
                    conservative_reclaim_bytes=sum(r['allocated'] for r in files[1:] if r['nlink']==1)))
    if checked and checked%50==0:print('full hashes',checked,flush=True)
(OUT/'exact_duplicates.json').write_text(json.dumps(dict(min_file_bytes=2**20,full_files_hashed=checked,groups=duplicates,
    reclaim_bytes=sum(d['conservative_reclaim_bytes'] for d in duplicates),executed=False),ensure_ascii=False,indent=2)+'\n')
print('duplicates',len(duplicates),'GiB',sum(d['conservative_reclaim_bytes'] for d in duplicates)/2**30,flush=True)

# Evaluate lossless compression on one large table, without retaining an archive.
large=max((r for r in rows if r['kind']=='stage_b_candidate_relations'),key=lambda r:r['bytes'])
import zlib
c=zlib.compressobj(3,zlib.DEFLATED,31);size=0;content=hashlib.sha256()
with (ROOT/large['path']).open('rb') as f:
    for b in iter(lambda:f.read(4*2**20),b''):
        content.update(b);size+=len(c.compress(b))
size+=len(c.flush())
(OUT/'compression_probe.json').write_text(json.dumps(dict(path=large['path'],original_bytes=large['bytes'],
    sha256=content.hexdigest(),gzip_level3_bytes=size,ratio=size/large['bytes'],archive_created=False,
    note='Single-file measurement, not a total archive estimate. Compression/archive is a separate future choice; original pathname must be restored for old stage readers.'),ensure_ascii=False,indent=2)+'\n')
print('compression probe',large['bytes'],size,flush=True)
