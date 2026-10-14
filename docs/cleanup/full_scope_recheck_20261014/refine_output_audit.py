"""Refine saved reference evidence without treating a storage inventory as use."""
from pathlib import Path
import collections, csv, json, os, re
from audit import ROOT, OUT, csvwrite, dump, codeaudit

evidence=json.loads((OUT/'output_reference_evidence.json').read_text())
summary=json.loads((OUT/'output_summary.json').read_text())
inventory=list(csv.DictReader((OUT/'output_inventory.csv').open(encoding='utf-8-sig')))
roots={r['path']:ROOT/r['path'] for r in inventory}
direct=collections.defaultdict(set)
for r,refs in evidence['current_bindings'].items():
    direct[r].update(p for p in refs if '/storage_audit_20260916/' not in p)
excluded={r:[p for p in refs if '/storage_audit_20260916/' in p] for r,refs in evidence['current_bindings'].items()}
edges=evidence['historical_edges']

def target_root(p):
    prefix=str(ROOT)+'/output/'
    value=str(p)
    if not value.startswith(prefix):return None
    parts=value[len(str(ROOT))+1:].split('/')
    rel='/'.join(parts[:3] if parts[1]=='scannet200' else parts[:2])
    return rel if rel in roots else None

# The prior bounded pass skipped large metadata; inspect those files now.
rx=re.compile(r'(?:'+re.escape(str(ROOT))+r'/)?output/[^\s"\'`<>\)\},\]]+')
large=[]
for filename in summary['metadata_skipped_large']:
    p=Path(filename)
    if not p.is_file():continue
    count=0
    for m in rx.finditer(p.read_text(errors='replace')):
        ref=m.group().rstrip(':;,\\')
        q=Path(ref) if ref.startswith('/') else ROOT/ref
        rel=target_root(q)
        if rel:direct[rel].add(str(p));count+=1
    large.append(dict(path=str(p),output_references=count))

# Actual inbound symlinks protect assets regardless of metadata spelling.
symlink_refs=collections.defaultdict(set)
scan_bases=[ROOT/'full_bundle',ROOT/'sl_bundle',ROOT/'research/geometry',ROOT/'.local',ROOT/'data']
routes=json.loads((ROOT/'full_bundle/metadata/scene_routes.json').read_text())
for row in routes:
    for key in ['data','records_resolved','actual_front','actual_backend','actual_own_root','yolosam','bound_geometry','bound_points','own_evidence']:
        p=Path(row[key]);scan_bases.append(p if p.is_dir() else p.parent)
seen=set()
for base in scan_bases:
    if base.is_symlink():
        rel=target_root(base.resolve())
        if rel:symlink_refs[rel].add(str(base))
    base=base.resolve()
    if not base.is_dir() or base in seen:continue
    seen.add(base)
    for folder,dirs,files in os.walk(base,followlinks=False):
        current=Path(folder)
        depth=len(current.relative_to(base).parts)
        for name in dirs+files:
            p=current/name
            if p.is_symlink():
                rel=target_root(p.resolve())
                if rel:symlink_refs[rel].add(str(p))
        dirs[:]=[n for n in dirs if not (current/n).is_symlink() and n not in ['.git','__pycache__','color','depth','poses','frame_cache','crops','views','snapshots']]
        # Current route receipts/links live near each scene root.
        if base!=ROOT/'full_bundle' and base!=ROOT/'sl_bundle' and base!=ROOT/'research/geometry' and depth>=2:dirs[:]=[]
    if len(seen)%300==0:print('symlink roots scanned',len(seen),flush=True)
for rel,refs in symlink_refs.items():direct[rel].update(refs)

source,constants=codeaudit()
defaults=collections.defaultdict(set)
for rel,strings in constants.items():
    for string in strings:
        for m in rx.finditer(string):
            raw=m.group().rstrip(':;,\\');p=Path(raw) if raw.startswith('/') else ROOT/raw
            target=target_root(p)
            if target:defaults[target].add(rel)
protected={r for r,v in direct.items() if v}
pending=list(protected)
while pending:
    r=pending.pop()
    for target in set(edges.get(r,[]))-protected:protected.add(target);pending.append(target)
for row in inventory:
    rel=row['path']
    row['classification']='KEEP_CURRENT_BINDING_OR_LINEAGE' if rel in protected else 'REVIEW_CODE_DEFAULT_ONLY' if defaults[rel] else 'RETIRE_CANDIDATE_NO_CURRENT_BINDING'
    row['current_binding_count']=len(direct[rel]);row['current_binding_examples']=';'.join(sorted(direct[rel])[:5])
    row['retained_source_reference_count']=len(defaults[rel]);row['retained_source_examples']=';'.join(sorted(defaults[rel])[:5])
    row['excluded_storage_inventory_mentions']=len(excluded.get(rel,[]))
csvwrite('output_inventory_refined.csv',inventory)
dump('output_reference_evidence_refined.json',dict(current_bindings={r:sorted(v) for r,v in direct.items()},
     physical_symlink_references={r:sorted(v) for r,v in symlink_refs.items()},historical_edges=edges,
     code_defaults={r:sorted(v) for r,v in defaults.items()},excluded_storage_inventory_mentions=excluded,
     previously_skipped_metadata_read=large))
result=dict(counts=dict(collections.Counter(r['classification'] for r in inventory)),
    allocated_GiB_by_classification={c:sum(int(r['allocated_bytes']) for r in inventory if r['classification']==c)/2**30 for c in {r['classification'] for r in inventory}},
    physical_symlink_scan_bases=len(seen),large_metadata_files_read=len(large),no_deletion=True)
dump('output_summary_refined.json',result)
print(json.dumps(result))
