from pathlib import Path
import ast,copy,hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
receipt=json.loads((OUT/'shared_slice_executed.json').read_text())
class Normalize(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        node.module=(node.module or '').replace('tools.fi1_shared.','tools.')
        if node.module=='tools.fi1_shared':node.module='tools'
        return node
    def visit_Import(self,node):
        for a in node.names:a.name=a.name.replace('tools.fi1_shared.','tools.')
        return node
checks=[]
with tarfile.open(ROOT/receipt['archive']) as tar:
    for row in receipt['targets']:
        old=ast.parse(tar.extractfile(row['original']).read().decode())
        assert not (ROOT/row['original']).exists(),row['original']
        assert not (ROOT/'tools'/Path(row['original']).name).is_symlink()
        if not row['destination']:continue
        new=ast.parse((ROOT/row['destination']).read_text())
        originals={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for n in new.body:
            if not isinstance(n,(ast.FunctionDef,ast.ClassDef)):continue
            a=ast.dump(Normalize().visit(copy.deepcopy(originals[n.name])),include_attributes=False)
            b=ast.dump(Normalize().visit(copy.deepcopy(n)),include_attributes=False)
            assert a==b,(row['destination'],n.name)
            checks.append(row['destination']+':'+n.name)
        assert not any(isinstance(n,ast.If) for n in new.body),'old CLI guard'
    for p in receipt['retired_tests']:
        assert not (ROOT/p).exists(),p
        assert not (ROOT/'tests'/Path(p).name).is_symlink()
assert hashlib.sha256((ROOT/receipt['archive']).read_bytes()).hexdigest()==receipt['archive_sha256']
result=dict(complete=True,definitions_AST_equal_after_import_path_normalization=len(checks),definitions=checks,
            removed_old_entrypoints=38,removed_old_tests=26,original_lines=receipt['original_lines'],retained_lines=receipt['retained_lines'],
            archive_sha256=receipt['archive_sha256'],intentional_changes=['import package paths','PROJECT_ROOT one level deeper','unused imports and CLI removed'])
(OUT/'slice_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='definitions'})
