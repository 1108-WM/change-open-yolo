from pathlib import Path
import ast,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
changes=[]
for p in sorted((ROOT/'research/geometry/code/tests').glob('test_*.py')):
    modules=[]
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n,ast.ImportFrom):
            if n.module=='tools':modules += [a.name for a in n.names]
            elif n.module and n.module.startswith('tools.'):modules.append(n.module.split('.')[1])
        elif isinstance(n,ast.Import):
            modules += [a.name.split('.')[1] for a in n.names if a.name.startswith('tools.')]
    scopes=set()
    for m in modules:
        alias=ROOT/'tools'/(m+'.py')
        if alias.exists():
            target=str(alias.resolve().relative_to(ROOT))
            scopes.add('geometry' if target.startswith('research/geometry/') else 'SL' if target.startswith('sl_bundle/') else 'Full')
    if 'geometry' in scopes or not scopes:continue
    scope='SL' if 'SL' in scopes else 'Full'
    dest=ROOT/('sl_bundle' if scope=='SL' else 'full_bundle')/'code/tests'/p.name
    changes.append(dict(source=str(p.relative_to(ROOT)),destination=str(dest.relative_to(ROOT)),scope=scope))
    assert not dest.exists()
    p.rename(dest)
    alias=ROOT/'tests'/p.name
    assert alias.is_symlink()
    alias.unlink();alias.symlink_to(Path('..')/dest.relative_to(ROOT))
(OUT/'test_partition.json').write_text(json.dumps(dict(complete=True,moves=changes,reason='Tests of retained Full/SL helpers belong with their implementation; no tests discarded by partition'),ensure_ascii=False,indent=2)+'\n')
print('Test partition',len(changes));print({s:sum(c['scope']==s for c in changes) for s in ['Full','SL']})
