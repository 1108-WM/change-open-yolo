"""Slice shared symbols out of retired experiment entry points (dry run by default)."""
from pathlib import Path
import ast, collections, csv, hashlib, json, os, re, sys, tarfile
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
CODE=ROOT/'full_bundle/code/tools'
PREFIXES=('train_candidate_','build_train_candidate_','build_train_scene_candidate_',
          'diagnose_candidate_','diagnose_train_candidate_','evaluate_candidate_quality_',
          'evaluate_frozen_track_','evaluate_official100_','fit_candidate_',
          'train_ncs_fi1_','build_ncs_fi1_','build_frozen_track_','run_frozen_safety60_')
EXTRA={'build_safety60_structured_action_plan','export_sam_fused_proposals',
       'export_mv3dis_relative_depth_observations','build_ncs1_scene_folds'}
targets={p.stem:p for p in CODE.glob('*.py') if p.stem.startswith(PREFIXES) or p.stem in EXTRA}
regenerate='--regenerate' in sys.argv
archived_text={}
if regenerate:
    prior=json.loads((OUT/'shared_slice_executed.json').read_text())
    with tarfile.open(ROOT/prior['archive']) as tar:
        for row in prior['targets']:
            p=ROOT/row['original'];targets[p.stem]=p
            archived_text[p.stem]=tar.extractfile(row['original']).read().decode()
def original_text(stem):return archived_text[stem] if stem in archived_text else targets[stem].read_text()
mapping={stem:'tools.fi1_shared.'+stem for stem in targets}
source={}
for base in ['full_bundle/code','sl_bundle/code','research/geometry/code']:
    for p in (ROOT/base).rglob('*.py'):
        source[p]=p.read_text()

# Tests of old experiments are archived with the original sources. Current
# Full/SL/geometry integration tests remain the verification authority.
retired_tests={p for p,t in source.items() if '/tests/' in str(p) and
               (p.stem.removeprefix('test_') in targets or p.stem=='test_ncs5_proxyclip_core')}
trees={stem:ast.parse(original_text(stem)) for stem,p in targets.items()}
bindings={};needed=collections.defaultdict(set)
all_bindings=collections.defaultdict(lambda:collections.defaultdict(list))
for stem,tree in trees.items():
    names={}
    for node in tree.body:
        node_names=[]
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):node_names=[node.name]
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            for target in node.targets if isinstance(node,ast.Assign) else [node.target]:
                for n in ast.walk(target):
                    if isinstance(n,ast.Name):node_names.append(n.id)
        elif isinstance(node,(ast.Import,ast.ImportFrom)):
            for a in node.names:node_names.append(a.asname or (a.name.split('.')[0] if isinstance(node,ast.Import) else a.name))
        elif isinstance(node,ast.AugAssign):
            node_names=[n.id for n in ast.walk(node.target) if isinstance(n,ast.Name)]
        for name in node_names:names[name]=node;all_bindings[stem][name].append(node)
    bindings[stem]=names

def imported_dependencies(node):
    deps=[];aliases={}
    for n in ast.walk(node):
        if isinstance(n,ast.ImportFrom):
            stem=(n.module or '').split('.')[-1]
            if n.module=='tools':
                for a in n.names:
                    if a.name in targets:aliases[a.asname or a.name]=a.name
            if stem in targets:
                for a in n.names:
                    if a.name=='*':raise ValueError('Wildcard import must be reviewed')
                    deps.append((stem,a.name))
        elif isinstance(n,ast.Import):
            for a in n.names:
                stem=a.name.split('.')[-1]
                if stem in targets:aliases[a.asname or stem]=stem
    for n in ast.walk(node):
        if isinstance(n,ast.Attribute):
            if isinstance(n.value,ast.Name) and n.value.id in aliases:deps.append((aliases[n.value.id],n.attr))
            if isinstance(n.value,ast.Attribute) and n.value.attr in targets:deps.append((n.value.attr,n.attr))
    # Explicit getattr(module, name) loops in the Replica adapter enumerate
    # helper names as string literals. Only matching top-level bindings count.
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and isinstance(n.value,str):
            for stem in aliases.values():
                if n.value in bindings[stem]:deps.append((stem,n.value))
    return deps

for p,text in source.items():
    if p in retired_tests or p.stem in targets:continue
    for stem,name in imported_dependencies(ast.parse(text)):needed[stem].add(name)

processed=set();pending=[(s,n) for s,ns in needed.items() for n in ns]
while pending:
    stem,name=pending.pop()
    if (stem,name) in processed:continue
    processed.add((stem,name))
    if name not in bindings[stem]:raise ValueError('Unknown required symbol '+stem+'.'+name)
    for node in all_bindings[stem][name]:
        refs={n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        for ref in refs:
            if ref in bindings[stem] and ref not in needed[stem]:needed[stem].add(ref);pending.append((stem,ref))
        for dep,symbol in imported_dependencies(node):
            if symbol not in needed[dep]:needed[dep].add(symbol);pending.append((dep,symbol))

def replace_imports(text):
    for stem,new in mapping.items():
        text=re.sub(r'\bfrom (?:tools\.)?'+re.escape(stem)+r' import\b','from '+new+' import',text)
        text=re.sub(r'\bfrom tools import '+re.escape(stem)+r'\b','from tools.fi1_shared import '+stem,text)
        text=re.sub(r'\bimport tools\.'+re.escape(stem)+r'\b','import '+new,text)
    return text

generated={};report=[]
for stem,p in targets.items():
    text=original_text(stem);nodes=[];seen=set()
    # Preserve the original relative statement order, no CLI guard/init.
    for node in trees[stem].body:
        if id(node) in seen:continue
        if any(node in all_bindings[stem].get(n,[]) for n in needed[stem]) or isinstance(node,ast.ImportFrom) and node.module=='__future__':
            seen.add(id(node));nodes.append(node)
    pieces=[];symbols=[]
    for node in nodes:
        segment=ast.get_source_segment(text,node)
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            # Remove unused imports so a single helper never retains a whole
            # historical training dependency merely via its top-level imports.
            keep=[]
            for a in node.names:
                key=a.asname or (a.name.split('.')[0] if isinstance(node,ast.Import) else a.name)
                if key in needed[stem] or isinstance(node,ast.ImportFrom) and node.module=='__future__':keep.append(a)
            new=ast.Import(names=keep) if isinstance(node,ast.Import) else ast.ImportFrom(module=node.module,names=keep,level=node.level)
            segment=ast.unparse(new)
        elif isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            symbols.append(dict(name=node.name,ast_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()))
        segment=replace_imports(segment)
        # Modules now live one directory deeper; root is the only intentional
        # function environment path change. Formula bodies are unchanged.
        segment=segment.replace('Path(__file__).resolve().parents[3]','Path(__file__).resolve().parents[4]')
        pieces.append(segment)
    destination=CODE/'fi1_shared'/(stem+'.py')
    if needed[stem]:
        generated[destination]='"""Shared frozen Full definitions; retired experiment CLI removed."""\n'+'\n\n'.join(pieces)+'\n'
        compile(generated[destination],str(destination),'exec')
    report.append(dict(original=str(p.relative_to(ROOT)),destination=str(destination.relative_to(ROOT)) if needed[stem] else None,
                       required_symbols=sorted(needed[stem]),functions_classes=symbols,
                       original_lines=len(text.splitlines()),retained_lines=len(generated.get(destination,'').splitlines())))
updates={}
for p,text in source.items():
    if p in retired_tests or p.stem in targets:continue
    new=replace_imports(text)
    if new!=text:updates[p]=new
payload=dict(targets=report,retired_tests=[str(p.relative_to(ROOT)) for p in sorted(retired_tests)],
             modified_callers=[str(p.relative_to(ROOT)) for p in sorted(updates)],
             original_lines=sum(r['original_lines'] for r in report),retained_lines=sum(r['retained_lines'] for r in report),
             old_entrypoints=len(targets),shared_modules=len(generated))
(OUT/'shared_slice_plan.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in payload.items() if k not in ['targets','retired_tests','modified_callers']}))
if regenerate:
    for p,text in generated.items():p.write_text(text)
    prior['regenerated_after_multibinding_check']=True
    prior['targets']=report;prior['retained_lines']=payload['retained_lines'];prior['shared_modules']=len(generated)
    (OUT/'shared_slice_executed.json').write_text(json.dumps(prior,ensure_ascii=False,indent=2)+'\n')
    sys.exit(0)
if '--execute' not in sys.argv:sys.exit(0)
archive=ROOT/'archive/retired_20261014/deep_legacy_sources_before_slice.tar.gz'
assert not archive.exists(),'Do not repeat'
originals=list(targets.values())+list(retired_tests)+list(updates)
with tarfile.open(archive,'w:gz') as tar:
    for p in originals:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
with tarfile.open(archive) as tar:
    for p in originals:assert tar.extractfile(str(p.relative_to(ROOT))).read()==p.read_bytes()
(CODE/'fi1_shared').mkdir()
(CODE/'fi1_shared/__init__.py').write_text('"""Shared frozen features; no historical experiment entrypoints."""\n')
(ROOT/'tools/fi1_shared').symlink_to('../full_bundle/code/tools/fi1_shared',target_is_directory=True)
for p,text in generated.items():p.write_text(text)
for p,text in updates.items():p.write_text(text)
for p in list(targets.values())+list(retired_tests):
    alias=ROOT/('tests' if '/tests/' in str(p) else 'tools')/p.name
    p.unlink()
    if alias.is_symlink():alias.unlink()
payload.update(complete=True,archive=str(archive.relative_to(ROOT)),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(OUT/'shared_slice_executed.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
