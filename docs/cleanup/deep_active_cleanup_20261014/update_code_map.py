from pathlib import Path
import ast,collections,csv,hashlib,json,re,tarfile
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
paths=['资料/代码文件索引与当前主线地图.md','full_bundle/README.md','sl_bundle/README.md','research/geometry/README.md','CURRENT_EXPERIMENT_STATUS.md','新开对话阅读内容.md']
archive=OUT/'handoff_before_deep_update.tar.gz'
if not archive.exists():
    with tarfile.open(archive,'w:gz') as tar:
        for p in paths:tar.add(ROOT/p,arcname=p)
p=ROOT/paths[0];text=p.read_text()
r=json.loads((OUT/'shared_slice_executed.json').read_text())
text=text.replace('`train_ncs_fi1_stage_a_quality_oof.py`、','')
for row in r['targets']:
    if row['destination']:
        old=Path(row['original']).name
        text=text.replace('`'+old+'`','`'+row['destination']+'`')
        text=text.replace('`full_bundle/code/tools/'+old+'`','`'+row['destination']+'`')
text=text.replace('文件名（位于 `tools/` 或旧 worktree 的 `tools/`）','文件名（短文件名位于 `full_bundle/code/tools/`，公共抽取项给出完整路径）')
text=text.replace('../docs/cleanup/full_scope_recheck_20261014/current_code_index.csv','../docs/cleanup/deep_active_cleanup_20261014/current_code_index.csv').replace('../docs/cleanup/full_scope_recheck_20261014/code_edges.json','../docs/cleanup/deep_active_cleanup_20261014/code_edges.json')
text=text.replace('本次已解除10个退役入口的依赖，22个抽取函数/常量AST不变，详情见[源码精简回执](../docs/cleanup/full_scope_recheck_20261014/README.md)。','两轮已解除48个退役入口的依赖。本轮38个完整旧脚本拆成29个精简公共模块，133个函数/类仅调整导入路径、定义保持一致；这些模块不再保留旧实验CLI。详情见[深度精简回执](../docs/cleanup/deep_active_cleanup_20261014/README.md)。')
p.write_text(text)
source={}
for scope,base in [('Full','full_bundle/code'),('SL','sl_bundle/code'),('geometry','research/geometry/code'),('evidence_snapshot','full_bundle/code_sources/evidence_snapshot')]:
    for p in (ROOT/base).rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix in ['.py','.sh']:
            source[str(p.relative_to(ROOT))]=(scope,p)
aliases={}
for rel,(scope,p) in source.items():
    if '/code/tools/' in rel:
        module='tools.'+'.'.join(p.relative_to(ROOT/rel.split('/code/tools/')[0]/'code/tools').with_suffix('').parts)
        if module.endswith('.__init__'):module=module[:-9]
        aliases[module]=rel
        if '/fi1_shared/' not in rel:aliases[p.stem]=rel
for base in ['tools','utils','evaluate']:
    for p in (ROOT/base).rglob('*.py'):
        if p.is_file():
            mod='.'.join(p.relative_to(ROOT).with_suffix('').parts)
            if mod.endswith('.__init__'):mod=mod[:-9]
            aliases[mod]=str(p.resolve().relative_to(ROOT))
edges=[];rows=[]
for rel,(scope,p) in sorted(source.items()):
    raw=p.read_bytes();text=raw.decode();deps=set()
    if p.suffix=='.py':
        for n in ast.walk(ast.parse(text)):
            names=[a.name for a in n.names] if isinstance(n,ast.Import) else [(n.module or '')]+[(n.module or '')+'.'+a.name for a in n.names] if isinstance(n,ast.ImportFrom) else []
            deps.update(aliases[name] for name in names if name in aliases)
    edges.extend(dict(source=rel,target=d) for d in sorted(deps))
    rows.append(dict(scope=scope,path=rel,kind='test' if '/tests/' in rel else 'implementation',lines=len(text.splitlines()),sha256=hashlib.sha256(raw).hexdigest(),imports=';'.join(sorted(deps))))
with (OUT/'current_code_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'code_edges.json').write_text(json.dumps(edges,ensure_ascii=False,indent=2)+'\n')
counts={scope:dict(implementation=sum(r['kind']=='implementation' and r['scope']==scope for r in rows),tests=sum(r['kind']=='test' and r['scope']==scope for r in rows)) for scope in ['Full','SL','geometry','evidence_snapshot']}
(OUT/'code_counts.json').write_text(json.dumps(counts,indent=2)+'\n');print(counts)
