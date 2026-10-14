"""Read-only inventory of proposed Full consolidation/retirement; no deletion."""
from pathlib import Path
import ast, collections, csv, hashlib, json, os, re, zipfile
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'docs/cleanup/full_sl_asset_map_20261014'
WORK = Path('/home/jia/Wm/wm_open-yolo/worktrees/OpenYOLO3D-geometry-semantic-56b2c25')
SNAP = ROOT / '.local/fi1_unified_evidence_20260910/workspace'
ROOTS = [ROOT, WORK, SNAP]
def write_csv(name, rows, fields=None):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()
def measure(p):
    p = Path(p); seen = set(); size = allocated = count = 0
    if p.is_file():
        s=p.stat(); return s.st_size,s.st_blocks*512,1
    for folder, dirs, files in os.walk(str(p), followlinks=False):
        dirs[:] = [x for x in dirs if not (Path(folder)/x).is_symlink()]
        for name in files:
            q=Path(folder)/name
            if q.is_symlink(): continue
            s=q.stat(); count+=1; size+=s.st_size
            key=(s.st_dev,s.st_ino)
            if key not in seen: allocated+=s.st_blocks*512;seen.add(key)
    return size,allocated,count
with (OLD/'current_code_import_index.csv').open(encoding='utf-8-sig') as f: full_rows=list(csv.DictReader(f))
protected={Path(x['actual_path']):x['scopes'] for x in full_rows}
research_seeds=set()
contracts=[]
for name in ['fi1_mask3d_feature_ab_internal_20261013_v1','fi1_decoupled_region_internal_development200_20261013_v1','fi1_region_affiliation_internal_development200_20261013_v1','fi1_local_region_supply_development200_20261013_v1','fi1_deletion_blocks_development200_20261013_v1','fi1_local_edit_ablation_development200_20261012_v1','fi1_candidate_local_repair_learning_development200_20261012_v1','fi1_fragment_affiliation_learnability_development200_20261011_v1']:
    p=ROOT/'.local'/name/'contract.json'
    if not p.exists(): continue
    j=json.loads(p.read_text()); contracts.append({'path':str(p),'sha256':sha(p)})
    for s in re.findall(r'"([^"\n]+\.py)"',json.dumps(j,ensure_ascii=False)):
        q=Path(s)
        if not q.is_absolute():q=ROOT/q
        if q.is_file() and any(str(q).startswith(str(b)+'/tools/') for b in ROOTS):research_seeds.add(q)
archive=ROOT/'docs/state_archive/20261014_full_sl_code_map_refocus/before/资料/代码文件索引与当前主线地图.md'
text=archive.read_text().split('## 生产Full主线')[0]
for rel in re.findall(r'tools/[A-Za-z0-9_]+\.py',text):
    p=ROOT/rel
    if p.is_file():research_seeds.add(p)
def resolve(mod):
    if mod.split('.')[0] not in ('tools','utils','evaluate'):return None
    for b in ROOTS:
        for p in (b/(mod.replace('.','/')+'.py'),b/mod.replace('.','/')/'__init__.py'):
            if p.is_file():return p
    return None
research=set();queue=collections.deque(research_seeds)
while queue:
    p=queue.popleft()
    if p in research:continue
    research.add(p)
    for n in ast.walk(ast.parse(p.read_text())):
        names=[]
        if isinstance(n,ast.Import):names=[a.name for a in n.names]
        elif isinstance(n,ast.ImportFrom):
            name=n.module or '';names=[name]+[name+'.'+a.name for a in n.names]
        for name in names:
            q=resolve(name)
            if q:queue.append(q)
all_source={}
for folder in ['tools','tests','utils']:
    for p in (ROOT/folder).glob('*'):
        if p.is_file() and p.suffix in ['.py','.sh']:
            all_source[p]=p.read_text(errors='replace')
source_index=[];autosam=[]
for p,t in sorted(all_source.items()):
    rel=str(p.relative_to(ROOT));scope=protected.get(p,'');is_research=p in research
    corresponding=ROOT/'tools'/p.name.replace('test_','',1) if p.parent.name=='tests' else p
    sibling_shell=p.with_suffix('.py') if p.suffix=='.sh' else p
    if scope:decision='KEEP_CURRENT_IMPORT_DEPENDENCY';reason='当前Full/ReplicaFull/SL入口或传递导入；历史函数拆出前保留'
    elif is_research or corresponding in research or sibling_shell in research:decision='KEEP_GEOMETRY_RESEARCH';reason='最新几何合同/归档地图入口及导入；研究仍待继续'
    elif corresponding in protected or sibling_shell in protected:decision='KEEP_CURRENT_TEST_OR_WRAPPER';reason='当前入口对应测试或任务包装'
    else:decision='ARCHIVE_LEGACY_CODE_CANDIDATE';reason='未落入本次当前/研究静态闭包；先压缩源码快照与必要报告，检查动态入口再移出活跃目录；不自动永久删除'
    row={'path':rel,'decision':decision,'scope':scope,'research_static_dependency':is_research,'bytes':p.stat().st_size,'sha256':sha(p),'reason':reason};source_index.append(row)
    if 'automatic_sam' in p.name or 'sam_automatic' in p.name or p.name in ['build_details_frame_siou_tracks.py','refine_details_automatic_tracks_consensus.py','build_details_same_frame_hierarchy_preprocessor.py','build_details_iterative_proposal_merges.py','run_fi1_frontend_comparison.py']:
        autosam.append(row)
write_csv('code_partition_candidates.csv',source_index);write_csv('autosam_code_candidates.csv',autosam)
model_rules={
'pretrained/alpha_clip':('KEEP_FULL','AlphaCLIP当前自身语义base/微调权重'),
'pretrained/bert-base-uncased':('KEEP_FULL','GroundingDINO文本编码，配置/词表/双格式权重合同'),
'pretrained/clip-vit-base-patch32':('KEEP_FULL','当前YOLO-World配置第33行使用，不能按旧CLIP模型删除'),
'pretrained/groundingdino':('KEEP_FULL','GD10框检测'),
'pretrained/yoloe':('RETIRE_MODEL_FAMILY_CANDIDATE','不属于当前Full/SL或已登记几何路线；停用YOLOE后可删除，失去YOLOE文本/框/掩码重算能力'),
'pretrained/sam2':('KEEP_RESEARCH_OPTION','最新next_route_readiness明确列作可能获取新边界的后续先验；未运行，不属于Full'),
'pretrained/z3_full_semantic_reliability_official100_20260812':('ARCHIVE_SMALL_LEGACY_MODEL','旧语义头，非当前Full/SL；先将模型与元数据放历史压缩包'),
'pretrained/z6f_full_inference_bundle_official100_20260812':('ARCHIVE_SMALL_LEGACY_MODEL','旧选择/门控包，非当前B26/SL；体积很小，保留压缩归档'),
'pretrained/checkpoints/Qwen2.5-VL-7B-Instruct':('KEEP_USER_PREFERENCE','用户此前明确保留Qwen；非Full，未被本轮问题撤销'),
'pretrained/checkpoints/dinov2_vits14_pretrain.pth':('KEEP_GEOMETRY_RESEARCH','局部修复/归属视觉特征仍复用'),
'pretrained/checkpoints/ViT-B-16.pt':('RETIRE_MODEL_FAMILY_CANDIDATE','旧ProxyCLIP图像编码权重；当前几何只复用该helper里的DINO loader，该函数不读CLIP或ProxyCLIP；退出旧ProxyCLIP重算后可删除'),
'pretrained/checkpoints/scannet200_val.ckpt':('KEEP_FULL','Mask3D原生候选与当前实例先验'),
'pretrained/checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth':('KEEP_FULL','冻结YOLO-World；不同于YOLOE'),
'pretrained/checkpoints/sam_vit_b_01ec64.pth':('KEEP_FULL','GD框提示SAM、YOLO-SAM验证掩码、Alpha区域；停用autoSAM不取消这些用途')}
external_rules={
'_external/AlphaCLIP':('KEEP_FULL','当前AlphaCLIP自身语义'),
'_external/GroundingDINO':('KEEP_FULL','当前GD10检测源码/配置'),
'_external/segment-anything':('KEEP_FULL','GD-SAM/YOLO-SAM/Alpha共同使用原SAM'),
'_external/ESAM':('SPLIT_METADATA_THEN_RETIRE_REMAINDER','当前preprocess/native/track脚本默认读取data/scannet200/meta_data的train列表/label_map；抽离元数据与改路径后，其余ESAM实现可归档退出'),
'_external/SigLIP':('RETIRE_MODEL_FAMILY_CANDIDATE','旧NCS4语义对照；当前Full/SL和已登记几何不加载；退出SigLIP重算后可删除'),
'_external/YOLOE':('RETIRE_MODEL_FAMILY_CANDIDATE','旧独立二维框/分割对照；配套mobileclip及YOLOE权重，当前Full是YOLO-World'),
'_external/ProxyCLIP':('RETIRE_MODEL_FAMILY_CANDIDATE','旧NCS5密集语义模型；当前DINO-only helper不导入open_clip；退出ProxyCLIP重算后可删除，helper源码仍保留'),
'_external/Any3DIS_unofficial':('ARCHIVE_REFERENCE_CODE_CANDIDATE','旧SAM2跟踪参考源码，现无当前运行入口；保留论文与方法记录/源码归档，SAM2实际实现单独保留'),
'_external/OV3D-CG':('ARCHIVE_REFERENCE_CODE_CANDIDATE','参考/比较代码，未在当前Full/SL入口或已登记几何实现中使用；归档后移出活跃目录'),
'_external/SegDINO3D_official':('ARCHIVE_REFERENCE_CODE_CANDIDATE','参考代码，当前接口为自有DINO/Mask3D适配；保留论文记录/源码归档后移出'),
'_external/sam2':('KEEP_RESEARCH_OPTION','配套待用SAM2先验实现；与旧autoSAM ViT-B配置不同'),
'_external/Any3DIS_unofficial.zip':('DUPLICATE_ARCHIVE_REVIEW','源码ZIP需逐项核对现存解压内容后可选择只保留一份'),
'_external/GroundingDINO.zip':('DUPLICATE_ARCHIVE_REVIEW','源码ZIP需逐项核对；删除下载ZIP不等于删除GD实现')}
refs=[];model_rows=[]
for rel,(decision,reason) in list(model_rules.items())+list(external_rules.items()):
    p=ROOT/rel
    if not p.exists():continue
    size,allocated,count=measure(p)
    search_tokens=[rel,Path(rel).name]
    hits=[]
    for q,t in all_source.items():
        if any(x in t for x in search_tokens):
            hit={'asset':rel,'source':str(q.relative_to(ROOT)),'current_static_dependency':q in protected,'research_static_dependency':q in research}
            refs.append(hit);hits.append(hit)
    model_rows.append({'path':rel,'decision':decision,'apparent_bytes':size,'allocated_bytes':allocated,'files':count,'scope_of_sizes':'current file stats, no symlink traversal; independent row inode accounting; do not add overlapping rows','current_source_reference_count':sum(x['current_static_dependency'] for x in hits),'research_source_reference_count':sum(x['research_static_dependency'] for x in hits),'all_top_source_reference_count':len(hits),'reason':reason})
write_csv('model_external_candidates.csv',model_rows);write_csv('asset_source_reference_evidence.csv',refs)
weights=[]
for p in sorted((ROOT/'pretrained').rglob('*')):
    if p.is_file() and p.suffix in ['.pt','.pth','.ckpt','.bin','.safetensors','.joblib','.pkl']:
        rel=str(p.relative_to(ROOT));rule=next((v for k,v in sorted(model_rules.items(),key=lambda x:len(x[0]),reverse=True) if rel==k or rel.startswith(k+'/')),('REVIEW','其他'))
        weights.append({'path':rel,'bytes':p.stat().st_size,'decision':rule[0],'reason':rule[1],'hash_recomputed':False})
write_csv('weight_file_candidates.csv',weights)
zipchecks=[]
for rel,dest in [('_external/Any3DIS_unofficial.zip','_external/Any3DIS_unofficial'),('_external/GroundingDINO.zip','_external/GroundingDINO')]:
    z=zipfile.ZipFile(ROOT/rel);missing=[];different=[];same=0
    for n in z.namelist():
        if n.endswith('/'):continue
        q=ROOT/dest/n
        if not q.is_file():missing.append(n)
        elif hashlib.sha256(z.read(n)).hexdigest()!=sha(q):different.append(n)
        else:same+=1
    zipchecks.append({'zip':rel,'extracted_root':dest,'same':same,'missing':missing,'different':different,'all_zip_files_exist_identically':not missing and not different})
(OUT/'download_zip_comparison.json').write_text(json.dumps(zipchecks,ensure_ascii=False,indent=2)+'\n')
# Data candidates are per existing branch, not inferred wildcard deletion.
base=ROOT/'.local/fi1_val48_frontend_full_pipeline_20260914';data=[]
paths=[]
for a in ['A','B','C','D']:
    paths.append((base/'frontend/arms'/a,a,'old48_frontend'))
    for batch in sorted((base/'full_batches').glob('batch*')):
        for k in ['backend','own_evidence']:paths.append((batch/k/a,a,'old48_'+k))
for p,arm,kind in paths:
    if not p.is_dir():continue
    size,allocated,count=measure(p)
    if arm in ['A','B']:decision='RETIRE_AUTOSAM_BRANCH_AFTER_ISOLATION';reason='已放弃的autoSAM候选配置；先保留紧凑报告/合同/源快照和可用图例输入，拆出Full/研究引用，再退役大观测/后端/证据/预测'
    elif arm=='D' and kind=='old48_frontend':decision='EXTRACT_GD10_LINEAGE_THEN_ARCHIVE_GD5';reason='旧D是GD5，不是autoSAM；现GD10的48来源需要D observations/RLE/关系/manifest，不能整臂先删'
    else:decision='RETIRE_OTHER_FRONTEND_COMPARISON_AFTER_ISOLATION';reason='旧GD30/GD5比较配置，非当前GD10；保留报告/合同与任何迁出依赖后可归档退出'
    data.append({'path':str(p.relative_to(ROOT)),'arm':arm,'kind':kind,'decision':decision,'apparent_bytes':size,'allocated_bytes':allocated,'files':count,'reason':reason})
write_csv('val48_branch_data_candidates.csv',data)
common=[]
with (OLD/'scene_asset_routes.csv').open(encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        common.append({'scene':row['scene'],'group':row['group'],'common_frames_source':row['common'],'keep_or_extract':'EXTRACT_EXACT_FRAME_CONTRACT_BEFORE_RETIRING_AUTOSAM_DATA','target':f'full_bundle/artifacts/scannet/{row["group"]}/{row["scene"]}/verification_frames.json','reason':'GVC/关系函数读取summary frames或fallback automatic_observations frame_id/frame_index；这些是帧身份，不是最终autoSAM候选'})
for s in ['office0','office1','office2','office3','office4','room0','room1','room2']:
    common.append({'scene':s,'group':'replica','common_frames_source':str(ROOT/'.local/replica_registered_full_20261011_v1/frontend/arms/A/hierarchy'/s),'keep_or_extract':'EXTRACT_EXACT_FRAME_CONTRACT_BEFORE_RETIRING_AUTOSAM_DATA','target':f'full_bundle/artifacts/replica/{s}/verification_frames.json','reason':'保留现有summary/fallback实际帧集合，不从新uniform采样推断替代'})
write_csv('autosam_common_frame_extraction_plan.csv',common)
summary={'date':'2026-10-14','deletion_executed':False,'migration_executed':False,'contracts_read':contracts,'current_import_files':len(protected),'geometry_import_files':len(research),'code_partition_counts':dict(collections.Counter(x['decision'] for x in source_index)),'autoSAM_code_counts':dict(collections.Counter(x['decision'] for x in autosam)),'model_candidates':len(model_rows),'weights':len(weights),'val48_branch_candidates':len(data),'common_frame_contracts':len(common),'mobileclip':{'path':'pretrained/yoloe/mobileclip_blt.pt','bytes':(ROOT/'pretrained/yoloe/mobileclip_blt.pt').stat().st_size,'Full_SL_used':False,'YOLOE_text_encoder_used':True,'decision':'delete candidate when retiring YOLOE; preserve or redownload to rerun YOLOE'},'static_analysis_limit':'Full/SL closure inherited from prior audit; research closure from current contracts/archived map. No executing GPU imports; unreachable is not proof of no future use. Historical paths/seals must retain original identities or an explicit relocation layer.'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k in ['code_partition_counts','autoSAM_code_counts','current_import_files','geometry_import_files','common_frame_contracts','mobileclip']},ensure_ascii=False))
