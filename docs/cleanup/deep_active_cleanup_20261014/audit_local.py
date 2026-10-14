"""Trace local asset bindings, separating history text from actual inputs."""
from pathlib import Path
import collections,csv,json,os,re
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
roots={p.name:p for p in (ROOT/'.local').iterdir() if p.is_dir() and not p.is_symlink()}
rx=re.compile(r'(?:'+re.escape(str(ROOT))+r'/)?\.local/([A-Za-z0-9_.-]+)(?:/[^\s"\'`<>\)\},\]]*)?')
direct=collections.defaultdict(set);edges=collections.defaultdict(set);file_refs=collections.defaultdict(set)
skip={'before','after','handoff','handoff_snapshot','handoff_before','source_snapshot','code_snapshot','workspace','__pycache__','.git','color','depth','poses','views','crops','frame_cache','snapshots','predictions','matches','scenes','dataset'}
scanned=0
skipped_large=[]

def readmeta(base,origin=None,depth_limit=3):
    global scanned
    if base.is_file():files=[base]
    else:
        files=[]
        pending=[(base,0)]
        while pending:
            folder,depth=pending.pop()
            with os.scandir(folder) as entries:
                for e in entries:
                    if e.is_symlink():continue
                    if e.is_dir(follow_symlinks=False):
                        if depth<depth_limit and e.name not in skip:pending.append((Path(e.path),depth+1))
                    elif Path(e.name).suffix in ['.json','.jsonl','.yaml','.yml']:files.append(Path(e.path))
    for p in files:
        if origin and not any(token in p.name.lower() for token in ['contract','identity','inventory','receipt','summary','manifest','verified','binding','input','config','model','seal']):continue
        scanned+=1
        if p.stat().st_size > 8*1024*1024:
            skipped_large.append(str(p))
            continue
        try:text=p.read_text(errors='replace')
        except OSError:continue
        for match in rx.finditer(text):
            name=match.group(1)
            if name not in roots or name==origin:continue
            ref=match.group().rstrip(':;,\\');file_refs[name].add(str(p))
            if origin:edges[origin].add(name)
            else:direct[name].add(str(p))

seeds=[]
for base in ['full_bundle/protected_sources','full_bundle/results','full_bundle/provenance',
             'sl_bundle/protected_sources','research/geometry/experiments']:
    seeds.extend(p.resolve() for p in (ROOT/base).iterdir() if p.is_dir())
routes=json.loads((ROOT/'full_bundle/metadata/scene_routes.json').read_text())
for r in routes:
    for key in ['native_receipt','bound_geometry','own_evidence']:
        p=Path(r[key]);seeds.append(p.parent.resolve())
seeds += [p.resolve() for p in (ROOT/'full_bundle/training').iterdir()]
for i,base in enumerate(dict.fromkeys(seeds)):
    if i<5:print('reading',base,flush=True)
    readmeta(base)
    if (i+1)%100==0:print('current metadata roots read',i+1,flush=True)
for i,(name,base) in enumerate(roots.items()):
    readmeta(base,origin=name,depth_limit=1)
    if (i+1)%25==0:print('history metadata roots read',i+1,flush=True)

# Existing aliases and exact frame-index keys protect physical dependencies.
symlink_in=collections.defaultdict(set)
for base in ['full_bundle','sl_bundle','research/geometry','outputs','data']:
    for folder,dirs,names in os.walk(ROOT/base,followlinks=False):
        dirs[:]=[n for n in dirs if n not in skip]
        for n in dirs+names:
            p=Path(folder)/n
            if p.is_symlink():
                dest=p.resolve()
                if dest.is_relative_to(ROOT/'.local'):
                    parts=dest.relative_to(ROOT/'.local').parts
                    if parts and parts[0] in roots:symlink_in[parts[0]].add(str(p))
        dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]
for name,refs in symlink_in.items():direct[name].update(refs)
index=json.loads((ROOT/'full_bundle/metadata/verification_frames_index.json').read_text())
for path in index:
    p=Path(path)
    if p.is_relative_to(ROOT/'.local'):
        name=p.relative_to(ROOT/'.local').parts[0]
        if name in roots:direct[name].add('verification_frames_index key')

# Source defaults are review evidence, not proof that an old strategy runs.
defaults=collections.defaultdict(set)
for base in ['full_bundle/code','sl_bundle/code','research/geometry/code']:
    for p in (ROOT/base).rglob('*'):
        if p.is_file() and p.suffix in ['.py','.sh']:
            for match in rx.finditer(p.read_text(errors='replace')):
                if match.group(1) in roots:defaults[match.group(1)].add(str(p.relative_to(ROOT)))
protected={k for k,v in direct.items() if v};queue=list(protected)
while queue:
    name=queue.pop()
    for target in edges[name]-protected:protected.add(target);queue.append(target)
results=[]
for i,(name,p) in enumerate(sorted(roots.items())):
    files=0;allocated=0;logical=0;links=0
    for folder,dirs,names in os.walk(p,followlinks=False):
        for n in dirs+names:
            if (Path(folder)/n).is_symlink():links+=1
        dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]
        for n in names:
            q=Path(folder)/n
            if q.is_symlink():continue
            s=q.stat();files+=1;allocated+=s.st_blocks*512;logical+=s.st_size
    role='KEEP_INPUT_OR_LINEAGE' if name in protected else 'REVIEW_SOURCE_DEFAULT' if defaults[name] else 'RETIRE_UNBOUND_HISTORY'
    # Download receipts / operating environments are administrative retention.
    if name.startswith(('scannet_','old260_','tmux','reboot_','opt','completion_status')):role='KEEP_DATA_PROVENANCE_OR_RUNTIME'
    results.append(dict(name=name,path=str(p.relative_to(ROOT)),classification=role,files=files,allocated_bytes=allocated,
                        allocated_GiB=round(allocated/2**30,6),logical_bytes=logical,symlinks=links,
                        current_direct_refs=len(direct[name]),direct_examples=';'.join(sorted(direct[name])[:6]),
                        source_default_refs=len(defaults[name]),source_examples=';'.join(sorted(defaults[name])[:4])))
    if (i+1)%50==0:print('local roots measured',i+1,flush=True)
with (OUT/'local_inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
(OUT/'local_reference_evidence.json').write_text(json.dumps(dict(direct={k:sorted(v) for k,v in direct.items()},edges={k:sorted(v) for k,v in edges.items()},metadata_refs={k:sorted(v) for k,v in file_refs.items()},symlink_inputs={k:sorted(v) for k,v in symlink_in.items()},source_defaults={k:sorted(v) for k,v in defaults.items()}),ensure_ascii=False,indent=2)+'\n')
summary=dict(metadata_files_read=scanned,real_roots=len(results),counts=dict(collections.Counter(r['classification'] for r in results)),
             GiB_by_role={c:sum(r['allocated_bytes'] for r in results if r['classification']==c)/2**30 for c in {r['classification'] for r in results}},
             bounds='metadata depth3 current / depth2 history; no snapshots/old docs; exact current symlinks; files >8MiB skipped; candidates require manual review; no deletion',
             skipped_large_metadata=skipped_large)
(OUT/'local_audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary))
