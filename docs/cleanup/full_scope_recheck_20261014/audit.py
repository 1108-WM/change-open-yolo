"""Audit physical storage and references; no deletion operations."""
from pathlib import Path
import ast, collections, csv, hashlib, json, os, re, time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def csvwrite(name,records):
    if not records:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def measure(p):
    if p.is_symlink():return dict(files=0,real_bytes=0,allocated_bytes=0,symlinks=1)
    stats=collections.Counter();seen=set()
    for folder,dirs,files in os.walk(p,followlinks=False):
        for n in dirs+files:
            q=Path(folder)/n
            if q.is_symlink():stats['symlinks']+=1
        dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]
        for n in files:
            q=Path(folder)/n
            if q.is_symlink():continue
            s=q.stat();stats['files']+=1;stats['real_bytes']+=s.st_size
            if (s.st_dev,s.st_ino) not in seen:stats['allocated_bytes']+=s.st_blocks*512;seen.add((s.st_dev,s.st_ino))
    return {k:stats[k] for k in ['files','real_bytes','allocated_bytes','symlinks']}

def codeaudit():
    source={}
    for base in ['full_bundle/code','sl_bundle/code','research/geometry/code']:
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and p.suffix in ['.py','.sh']:source[str(p.relative_to(ROOT))]=p
    aliases={}
    for base in ['tools','tests','utils','evaluate']:
        for p in (ROOT/base).rglob('*.py'):
            if p.is_file():
                rel=p.relative_to(ROOT);mod='.'.join(rel.with_suffix('').parts)
                if mod.endswith('.__init__'):mod=mod[:-9]
                aliases[mod]=str(p.resolve().relative_to(ROOT))
                if base=='tools' and p.name!='__init__.py':
                    aliases[p.stem]=str(p.resolve().relative_to(ROOT))
    edges=[];constants={};missing=[]
    for rel,p in source.items():
        text=p.read_text(errors='replace');target=set();strings=[]
        if p.suffix=='.py':
            tree=ast.parse(text)
            for node in ast.walk(tree):
                names=[]
                if isinstance(node,ast.Import):names=[a.name for a in node.names]
                elif isinstance(node,ast.ImportFrom):names=[node.module or '']+[(node.module or '')+'.'+a.name for a in node.names]
                for name in names:
                    if name in aliases:target.add(aliases[name])
                if isinstance(node,ast.Constant) and isinstance(node.value,str):strings.append(node.value)
        else:strings=[text]
        # Literal dynamic tool/module names, not mere identifier mentions in
        # comments, function names or generic prose.
        for string in strings:
            for value in re.findall(r'(?:tools/)?([A-Za-z0-9_]+)\.(?:py|sh)\b',string):
                for base in ['tools','tests']:
                    q=ROOT/base/(value+'.py')
                    if q.is_file():target.add(str(q.resolve().relative_to(ROOT)))
            if string in aliases:target.add(aliases[string])
            if 'tools.'+string in aliases:target.add(aliases['tools.'+string])
        constants[rel]=strings
        edges.extend(dict(source=rel,target=q) for q in target)
    neighbors=collections.defaultdict(set)
    for r in edges:neighbors[r['source']].add(r['target'])
    # Full map explicitly registers the complete stages, including preprocessing
    # and training, in addition to the independent release.
    maptext=(ROOT/'资料/代码文件索引与当前主线地图.md').read_text().split('## 有效 SL：')[0]
    seeds=set(re.findall(r'full_bundle/code/(?:tools|utils|tests)/[A-Za-z0-9_./-]+\.(?:py|sh)',maptext))
    queue=list(seeds);closure=set()
    while queue:
        p=queue.pop()
        if p in closure:continue
        closure.add(p);queue.extend(neighbors[p]-closure)
    incoming=collections.defaultdict(list)
    for r in edges:incoming[r['target']].append(r['source'])
    records=[]
    for rel,p in sorted(source.items()):
        imports=sorted(set(incoming[rel]))
        if rel.startswith('full_bundle/'):
            role='REGISTERED_FULL_ENTRY_OR_DEPENDENCY' if rel in closure else ('SHARED_OTHER_SCOPE_DEPENDENCY' if imports else 'NO_REGISTERED_CURRENT_CALLER_CANDIDATE')
        else:role='SL_OR_GEOMETRY_RETAINED_SEPARATELY'
        records.append(dict(path=rel,kind='test' if '/tests/' in rel else 'shell' if p.suffix=='.sh' else 'implementation',
                            classification=role,full_stage_map_entry=rel in seeds,
                            caller_count=len(imports),callers=';'.join(imports),
                            note='文件级引用；历史脚本被import不证明其训练/失败策略CLI仍被使用'))
    csvwrite('code_roles.csv',records);dump('code_edges.json',edges)
    dump('code_summary.json',dict(counts=dict(collections.Counter(r['classification'] for r in records)),
                                   full_closure_files=len(closure),full_map_seeds=len(seeds)))
    return source,constants

def outputaudit(source,constants):
    roots=[p for p in (ROOT/'output').iterdir() if p.is_dir() and not p.is_symlink()]
    # Audit ScanNet's components individually instead of treating its 14.89GiB
    # as one removable root.
    roots=[p for p in roots if p.name!='scannet200']+list((ROOT/'output/scannet200').iterdir())
    roots=[p for p in roots if p.is_dir() and not p.is_symlink()]
    root_rels={str(p.relative_to(ROOT)):p for p in roots}
    ref_rx=re.compile(r'(?:'+re.escape(str(ROOT))+r'/)?(?:output|\.local|full_bundle|sl_bundle|research/geometry)/[^\s"\'`<>\)\},\]]+')
    allrefs=collections.defaultdict(set);direct=collections.defaultdict(set);edges=collections.defaultdict(set)
    def match_root(s):
        for rel in sorted(root_rels,key=len,reverse=True):
            if s==rel or s.startswith(rel+'/'):return rel
    def references(text):
        values=[]
        for s in ref_rx.findall(text):
            if s.startswith(str(ROOT)+'/'):s=s[len(str(ROOT))+1:]
            values.append(s.rstrip(':;,\\'))
        return values
    # Inspect currently retained source strings. Separate old CLI defaults from
    # frozen asset bindings; those defaults alone do not prove runtime access.
    for rel,strings in constants.items():
        for text in strings:
            for ref in references(text):
                target=match_root(ref)
                if target:allrefs[target].add(rel)
    seed_roots=[]
    for base in ['full_bundle/results','full_bundle/protected_sources','full_bundle/provenance',
                 'sl_bundle/protected_sources','research/geometry/experiments']:
        seed_roots.extend(p.resolve() for p in (ROOT/base).iterdir() if p.is_dir())
    seed_roots.append((ROOT/'outputs/Full点云展示_20261011').resolve())
    # Original Full route receipts on the external disk, including 597 training.
    routes=json.loads((ROOT/'full_bundle/metadata/scene_routes.json').read_text())
    for r in routes:
        for key in ['actual_own_root','actual_backend','native_receipt','records_resolved']:
            q=Path(r[key]);seed_roots.append(q.resolve() if q.is_dir() else q.parent.resolve())
    keywords=['contract','receipt','identity','summary','manifest','inventory','binding','frozen','verified','inputs','jobs','sources','plan','selection']
    files=0;seenfiles=set();skipped=[];queue=list(dict.fromkeys(seed_roots));seenroots=set()
    metadata_refs=collections.defaultdict(set)
    def scan(base,origin,active):
        nonlocal files
        for folder,dirs,names in os.walk(base,followlinks=False):
            # Contracts enumerate deeper scientific assets. Inspect metadata
            # to depth3 rather than traversing millions of per-frame crops.
            depth=len(Path(folder).relative_to(base).parts)
            dirs[:]=[d for d in dirs if depth<3 and d not in ['.git','__pycache__','code_snapshot','postseal_code_snapshot','workspace','frame_cache','reprojected_frame_cache','fresh_alpha','fresh_cache','artifacts','crops','views'] and not (Path(folder)/d).is_symlink()]
            for name in names:
                p=Path(folder)/name
                if p.is_symlink() or p.suffix not in ['.json','.yaml','.yml','.md']:continue
                if not any(k in name.lower() for k in keywords) and p.suffix!='.md':continue
                key=str(p.resolve())
                if key in seenfiles:continue
                seenfiles.add(key)
                if p.stat().st_size>4*1024**2:skipped.append(str(p));continue
                text=p.read_text(errors='replace');files+=1
                for ref in references(text):
                    target=match_root(ref)
                    if target:
                        metadata_refs[target].add(str(p))
                        if origin and origin!=target:edges[origin].add(target)
                        elif active:direct[target].add(str(p))
                    # Follow explicitly referenced local experiment roots; no
                    # generic .local directory exposure from snapshot aliases.
                    if active and ref.startswith('.local/'):
                        parts=Path(ref).parts
                        if len(parts)>1:
                            q=ROOT/parts[0]/parts[1]
                            if q.is_dir() and q.resolve() not in seenroots:queue.append(q.resolve())
    while queue:
        base=queue.pop()
        if base in seenroots:continue
        seenroots.add(base);scan(base,None,True)
        if len(seenroots)%200==0:print('current metadata roots',len(seenroots),'files',files,flush=True)
    # Output-to-output references are historical lineage, not current use.
    for rel,p in root_rels.items():scan(p,rel,False)
    protected=set(direct);pending=list(protected)
    while pending:
        r=pending.pop()
        for target in edges[r]-protected:protected.add(target);pending.append(target)
    incoming=collections.defaultdict(set)
    for origin,targets in edges.items():
        for target in targets:incoming[target].add(origin)
    inventory=[]
    for rel,p in sorted(root_rels.items()):
        s=measure(p)
        role='KEEP_CURRENT_BINDING_OR_LINEAGE' if rel in protected else ('REVIEW_CODE_DEFAULT_ONLY' if allrefs[rel] else 'RETIRE_CANDIDATE_NO_CURRENT_BINDING')
        inventory.append(dict(path=rel,allocated_GiB=round(s['allocated_bytes']/2**30,6),**s,
                              classification=role,current_binding_count=len(direct[rel]),
                              current_binding_examples=';'.join(sorted(direct[rel])[:5]),
                              retained_source_reference_count=len(allrefs[rel]),
                              retained_source_examples=';'.join(sorted(allrefs[rel])[:5]),
                              historical_incoming_count=len(incoming[rel]),
                              historical_incoming_examples=';'.join(sorted(incoming[rel])[:5]),
                              note='候选须归档报告/核对必要子资产后删除；未引用不等于穷尽所有动态访问'))
    csvwrite('output_inventory.csv',inventory)
    dump('output_reference_evidence.json',dict(current_bindings={k:sorted(v) for k,v in direct.items()},
         historical_edges={k:sorted(v) for k,v in edges.items()},source_refs={k:sorted(v) for k,v in allrefs.items()},
         current_metadata_refs={k:sorted(v) for k,v in metadata_refs.items()}))
    summary=dict(metadata_files_read=files,metadata_skipped_large=skipped,active_source_roots_scanned=len(seenroots),
                 counts=dict(collections.Counter(r['classification'] for r in inventory)),
                 allocated_GiB_by_classification={c:sum(r['allocated_bytes'] for r in inventory if r['classification']==c)/2**30 for c in {r['classification'] for r in inventory}},
                 scope='registered Full/SL/current geometry, route receipts and recursively bound .local roots; metadata graph to depth3, skip metadata over4MiB; code defaults listed separately; no deletion')
    dump('output_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    source,constants=codeaudit();outputaudit(source,constants)
