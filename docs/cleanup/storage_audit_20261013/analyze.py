"""Build review-only storage manifests. Contains no deletion operations."""
import ast, collections, csv, hashlib, json, os, re, subprocess, time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
os.chdir(ROOT)
inv = json.loads((OUT/'inventory.json').read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def write_csv(name, rows, fields):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

seeds = [
    'tools/fi1_sl_inference.py', 'tools/run_fi1_sl_single200_dev55.py',
    'tools/run_fi1_mask3d_feature_ab.py', 'tools/export_fi1_mask3d_instance_features.py',
    'tools/evaluate_fi1_mask3d_feature_ab.py', 'tools/verify_fi1_mask3d_feature_ab.py',
    'tools/run_fi1_g10_closure.py', 'tools/run_fi1_g10_closure_assets.py',
    'tools/run_fi1_g10_closure_learning.py', 'tools/run_scannet200_train_scene_track_pipeline.py',
    'run_evaluation.py', 'single_scene_inference.py',
    'outputs/Full点云展示_20261011/render.py', 'outputs/Full点云展示_20261011/verify.py',
]
seen, edges, missing, texts = set(), [], [], {}
queue = list(seeds)
while queue:
    rel = queue.pop()
    if rel in seen: continue
    p = ROOT/rel
    if not p.is_file(): missing.append(rel); continue
    seen.add(rel)
    text = p.read_text(errors='replace'); texts[rel] = text
    try: tree = ast.parse(text)
    except SyntaxError: continue
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import): names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [node.module+'.'+a.name for a in node.names]
        for name in names:
            candidate = name.replace('.', '/')+'.py'
            if (ROOT/candidate).is_file():
                edges.append(dict(source=rel, target=candidate, line=node.lineno))
                if candidate not in seen: queue.append(candidate)

protected = collections.defaultdict(list)
def protect(path, reason):
    path = str(path)
    if path.startswith(str(ROOT)+'/'): path = path[len(str(ROOT))+1:]
    if reason not in protected[path]: protected[path].append(reason)

for path in seen: protect(path, '当前入口的静态导入依赖（保守闭包，不等于逐函数运行轨迹）')
for src, text in texts.items():
    for item in re.findall(r'\.local/[A-Za-z0-9_./-]+', text):
        protect('/'.join(item.split('/')[:2]), '当前代码引用: '+src)

latest = ROOT/'.local/fi1_mask3d_feature_ab_internal_20261013_v1/contract.json'
c = json.loads(latest.read_text())
for field in ('source_bindings','inputs','GT_inputs','frozen_SL_models','code'):
    for p in c[field]: protect(p, '最新 Mask3D A/B contract.'+field)
visual = ROOT/'outputs/Full点云展示_20261011/manifest.json'
v = json.loads(visual.read_text())
for p in v['inputs']: protect(p, '保留 Full 展示的 manifest.inputs')
special = {
 'outputs/Full点云展示_20261011':'用户明确保留，含重绘与核验资产',
 'outputs/Full点云展示_20261011.zip':'对应交付压缩包，随明确保留展示一起保护',
 'outputs/Full算法说明_20261012':'当前算法说明交付',
 'outputs/github_frozen_full800_20261008':'Full 冻结复现包，包含所需源码、自训模型、Alpha 文本向量，非普通图例',
 'release_assets':'冻结复现配置和交付资产',
 'release_summaries':'冻结结果摘要与发布依据',
 '.local/fi1_mask3d_feature_ab_internal_20261013_v1':'最新几何实验，外盘链接不能按本盘释放量计算',
 '.local/fi1_sl_single200_dev55_20261010_v1':'有效 SL 单模型、归一化、预测与核验',
 '.local/fi1_s_fixed_L_rescore_development200_20261010_v1':'已登记有效 SL 候选与合同',
 '.local/fi1_s_semantic_rank_residual_development200_20261010_v2':'SL 五折 L 权重与归一化',
 '.local/fi1_sl_replay_stability_development200_20261010_v1':'SL 稳定性复核与模型',
 '.local/fi1_unified_evidence_20260910':'生产 Full 实际调用其中 workspace 源码',
 '.local/fi1_r0_r4_preparation_20260913':'生产历史输入身份与最新几何 GT 绑定',
 '.local/fi1_r3b_val312_execution_20260914':'Full 历史证据与基线重放',
 '.local/fi1_population_visual_20260910':'含被复用的 Alpha 特征；此前 RGB 清理已完成',
 '.local/fi1_core_ablation_val312_20261006_v1':'保留 Full 展示的预测、几何来源',
 '.local/fi1_val48_frontend_full_pipeline_20260914':'历史对照终态几何和上游引用，不能整删',
 '.local/replica_registered_full_20261011_v1':'Replica 对照及后续复用资产',
 '.local/replica_official_masks_full_20261011_v1':'官方候选重评测终态与核验',
 '.local/official_openyolo3d_assets_20261011/extracted':'旧合同实际指向这里的官方掩码',
 'output/replica':'标准官方掩码及安装身份',
 'output/scannet200':'原候选、原生缓存、点域数据，不能整体清空',
 'pretrained/checkpoints/scannet200_val.ckpt':'当前 Mask3D 与最新实例特征导出权重',
 'pretrained/checkpoints/Qwen2.5-VL-7B-Instruct':'旧清理记录明确保留 Qwen；不列入本轮可删项',
 '.venvs/qwen25vl_py311':'Qwen 专用运行环境，旧清理记录明确保留',
 '.venvs/yoloe_gpu':'Full、SL、局部训练当前运行环境',
 '.venvs/tmux_runtime':'现有任务托管工具依赖',
 'OpenYOLO3D/replica':'data/replica 链接的真实原始数据',
 'docs/state_archive':'旧实验合同与交接哈希的历史依据',
 '.git':'版本历史与对象库，禁止按缓存/临时目录整体删除',
}
for p, reason in special.items(): protect(p, reason)
external_protection = {
 '/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921':'生产 Full B/Q、几何、自身语义证据',
 '/media/jia/软件1/scannet_additional_train_stream/fi1_G10_full597_20260919/frontend':'生产 GD10 前端',
 '/home/jia/Wm/wm_open-yolo/worktrees/OpenYOLO3D-geometry-semantic-56b2c25':'D-v3 七模型及历史源码依赖',
 '/home/jia/Wm/Dataset/scannet200':'data/scannet200 链接目标，另计，不清理',
}
for p,reason in external_protection.items(): protect(p,reason)

rows = [dict(path=p,reason=' | '.join(rs)) for p,rs in sorted(protected.items())]
write_csv('protected_paths.csv',rows,['path','reason'])
(OUT/'dependency_evidence.json').write_text(json.dumps(dict(
    seeds=seeds,static_import_files=sorted(seen),import_edges=edges,missing_seed_files=missing,
    dynamic_import_limitation='未执行动态导入或全部生产推理；无静态引用不构成安全删除证据',
    latest_contract_sha256=sha(latest),visual_manifest_sha256=sha(visual),
    binding_counts={k:len(c[k]) for k in ('source_bindings','inputs','GT_inputs','frozen_SL_models')},
    external_protection=external_protection),ensure_ascii=False,indent=2)+'\n')

# Only fully regenerable, untracked, ordinary caches with existing source.
tracked = set(subprocess.check_output(['git','ls-files','-z']).decode().split('\0'))
caches, cache_exceptions = [], []
for folder in ('__pycache__','tools/__pycache__','tests/__pycache__','utils/__pycache__','evaluate/__pycache__'):
    for p in sorted((ROOT/folder).glob('*.pyc')):
        rel = str(p.relative_to(ROOT)); source = p.parent.parent/(p.name.split('.')[0]+'.py')
        if not source.is_file() or rel in tracked or rel in protected or p.is_symlink():
            cache_exceptions.append(dict(path=rel,reason='缺少对应源码或已跟踪/保护；不列入可删'))
            continue
        s=p.stat()
        caches.append(dict(path=rel,allocated_bytes=s.st_blocks*512,apparent_bytes=s.st_size,
            guaranteed_single_link_bytes=s.st_blocks*512 if s.st_nlink==1 else 0,
            sha256=sha(p),source=str(source.relative_to(ROOT)),reason='对应 .py 源码存在，Python 可重新生成',action='仅待下次授权删除'))
for p in sorted((ROOT/'.pytest_cache').rglob('*')):
    if p.is_file() and not p.is_symlink() and str(p.relative_to(ROOT)) not in tracked:
        s=p.stat();caches.append(dict(path=str(p.relative_to(ROOT)),allocated_bytes=s.st_blocks*512,
            apparent_bytes=s.st_size,guaranteed_single_link_bytes=s.st_blocks*512 if s.st_nlink==1 else 0,
            sha256=sha(p),source='',reason='pytest 可重新生成的运行状态缓存，不是测试源码',action='仅待下次授权删除'))
fields=['path','allocated_bytes','apparent_bytes','guaranteed_single_link_bytes','sha256','source','reason','action']
write_csv('safe_cache_candidates.csv',caches,fields)
write_csv('cache_exceptions_keep.csv',cache_exceptions,['path','reason'])

# Every experiment/output top-level directory gets an explicit non-deletion status.
protected_dirs=collections.defaultdict(list)
for p,rs in protected.items():
    if p.startswith(('.local/','output/','outputs/')):
        key='/'.join(p.split('/')[:2]);protected_dirs[key].extend(rs[:1])
directory_rows=[]
for p,values in sorted(inv['directories'].items()):
    parts=p.split('/')
    include=(len(parts)==1 or (len(parts)==2 and parts[0] in ('.local','output','outputs','.venvs','pretrained'))
             or (len(parts)==3 and '/'.join(parts[:2]) in ('docs/diagnostics','docs/visual_checks')))
    if not include:continue
    if p in protected_dirs or p in special:
        status='保留：已发现当前/复现依赖'
        reason=special.get(p) or ' | '.join(dict.fromkeys(protected_dirs[p]))[:650]
    elif p.startswith('outputs/'):
        status='待决定：历史可视化可退役，非明确无用'
        reason='旧图例有独立内容；未列入安全删除，删除会失去历史展示/核验'
    elif p.startswith(('docs/diagnostics/','docs/visual_checks/','.local/','output/')):
        status='待核实：禁止整删'
        reason='盘点存在，但尚未证明所有传递依赖/历史重放均不再需要；失败/旧日期不是删除依据'
    else:
        status='保留：项目、数据、环境或代码';reason='仅个别明确缓存另见白名单'
    directory_rows.append(dict(path=p,allocated_bytes=values[0],GiB=round(values[0]/2**30,6),files=values[2],status=status,reason=reason))
for link in inv['links']:
    if link['path'].count('/')==1 and link['path'].startswith(('.local/','data/')):
        directory_rows.append(dict(path=link['path'],allocated_bytes=0,GiB=0,files='',status='保留：符号链接，目标另计',reason=link['target']))
write_csv('directory_review.csv',directory_rows,['path','allocated_bytes','GiB','files','status','reason'])

code_rows=[]
for base in ('tools','tests'):
    for p in sorted((ROOT/base).iterdir()):
        if p.is_file() and p.suffix in ('.py','.sh'):
            rel=str(p.relative_to(ROOT));code_rows.append(dict(path=rel,bytes=p.stat().st_size,
                tracked=rel in tracked,static_current_dependency=rel in seen,
                decision='保留',reason='当前静态导入依赖' if rel in seen else '历史入口/测试；未引用不能证明以后无用；无安全删除证据'))
write_csv('code_review.csv',code_rows,['path','bytes','tracked','static_current_dependency','decision','reason'])

# Hash the effective small SL assets and production B/Q; no model inference.
checks=[]
bindings=dict(c['frozen_SL_models'])
sl=ROOT/'.local/fi1_sl_single200_dev55_20261010_v1'
slseal=json.loads((sl/'model_frozen.json').read_text())
bindings[str(sl/'model/model.pt')]=slseal['model_sha256']
bindings[str(sl/'model/normalization.npz')]=slseal['normalization_sha256']
bindings[str(ROOT/'pretrained/checkpoints/scannet200_val.ckpt')]=c['mask3d_checkpoint_sha256']
for p,h in bindings.items():
    actual=sha(p) if Path(p).is_file() else None
    checks.append(dict(path=p,expected_sha256=h,actual_sha256=actual,passed=actual==h))
(OUT/'asset_integrity_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
summary=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),files_deleted=0,files_moved=0,
    project_allocated_bytes=inv['total'][0],project_regular_files=inv['total'][2],
    safe_cache_files=len(caches),safe_cache_allocated_bytes=sum(r['allocated_bytes'] for r in caches),
    safe_cache_single_link_bytes=sum(r['guaranteed_single_link_bytes'] for r in caches),
    cache_exceptions_kept=len(cache_exceptions),static_code_files=len(seen),code_rows=len(code_rows),
    protected_paths=len(protected),directory_rows=len(directory_rows),
    model_integrity_passed=sum(q['passed'] for q in checks),model_integrity_total=len(checks),
    measurement_exclusions=inv['errors'],note='No deletion has been authorized for this turn; all lists are review-only.')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
