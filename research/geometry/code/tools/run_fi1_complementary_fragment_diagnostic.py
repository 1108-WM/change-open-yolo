#!/usr/bin/env python3
"""Separate retrieval recall from association precision, original development200.

All pair features and fixed hypothesis masks are sealed before raw GT labels.
No new semantic scorer, training, AP or default output append is performed here.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,sys,json,shutil,time,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from tools import run_fi1_witnessed_local_completion as parent
from tools import fi1_complementary_fragment_evidence as core
from tools.fi1_r3_core import point_matrix,pair_overlap,gt_iou
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
OUT=ROOT/'.local/fi1_complementary_fragment_retrieval_identity_development200_20261011_v1'
read=parent.read

def stage(name,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=name,updated=time.time(),**kw));print(name,kw,flush=True)

class Guard:
    def __init__(self,c,labels=False):self.c=c;self.labels=labels
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(Path(args[0]).resolve())
        if any(s in p for s in ('val312','reserve100','fi1_sl_single200_dev55')):raise PermissionError('excluded datasets/models')
        if any(x.startswith('ground_truth') for x in Path(p).parts):
            if not self.labels or p not in self.c['GT_inputs'] or not (OUT/'features_frozen.json').exists():raise PermissionError('GT requires full feature seal and200 whitelist')
        if not self.labels and any(p.endswith(s) for s in ('target_census.json','overlap.npz','targets.jsonl','opportunity_action_gap.json','geometry_quality.json')):raise PermissionError('post-GTdiagnostic cannot enter retrieval or identity evidence')
        import re
        if any(s not in self.c['scenes'] for s in re.findall(r'scene\d{4}_\d{2}',p)):raise PermissionError('outside development200')

def prepare():
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>=50*1024**3
    p=read(parent.OUT/'contract.json');inputs={};jobs={}
    for s in p['scenes']:
        j=p['jobs'][s];jobs[s]=j
        for n in ('geometry','points'):inputs[j['binding'][n]]=p['inputs'][j['binding'][n]]
        for n in ('processed','manifest','observations'):inputs[j['paths'][n]]=p['inputs'][j['paths'][n]]
        inputs[j['views']]=p['inputs'][j['views']]
        manifest=read(j['paths']['manifest']);chosen=set(read(j['views'])['raw_frames'])
        for n in ('intrinsics','mesh'):a=manifest['files'][n];inputs[a['path']]=a['sha256']
        for f in manifest['frames']:
            if int(f['frame_id']) in chosen:
                for n in ('pose','depth','color'):a=f[n];inputs[a['path']]=a['sha256']
        # Existing fixed bank identities, no label contents opened.
        e=read(parent.diag.OUT/'scenes'/s/'evidence.json');inputs.update(e['inputs'])
    labels={str(parent.diag.OUT/'targets.jsonl'):sha(parent.diag.OUT/'targets.jsonl'),str(parent.OUT/'opportunity_action_gap.json'):sha(parent.OUT/'opportunity_action_gap.json')}
    files=[Path(__file__).resolve(),ROOT/'tools/fi1_complementary_fragment_evidence.py',ROOT/'tests/test_fi1_complementary_fragment_evidence.py',ROOT/'tools/diagnose_fi1_nt_geometry_gaps.py',ROOT/'tools/fi1_core_ablation.py',ROOT/'tools/fi1_r3_core.py',ROOT/'tools/run_fi1_witnessed_local_completion.py',ROOT/'tools/fi1_frontend_protocol.py',ROOT/'tools/run_fi1_complementary_fragment_diagnostic.sh']
    code={str(q):sha(q) for q in files};parent.diag.prior.src.ctl.base.verify_files(inputs)
    c=dict(version=OUT.name,scenes=p['scenes'],folds=p['folds'],groups=p['groups'],thresholds=p['thresholds'],class_ids=p['class_ids'],GT_inputs=p['GT_inputs'],inputs=inputs,label_inputs=labels,code=code,jobs=jobs,parameters=core.PARAMETERS,signatures=core.SIGNATURES,
        scope='User directs disjoint complementary fragment retrieval and same-instance ownership evidence. All200 NT unordered pairs, not just60GT-picked opportunities. SL fixed, semantic/selection improvement deferred.',
        retrieval='Compare legacy.05<pairIoU<.9, exact3D exclusive-region contacts<=.10m >=3, projection exclusive-region bboxgap<=3depthpixels in>=2 posegroups. NoGT/semantics/scores. Atleast3exclusivepoints both sides, IoU<.9. Broad projection bboxes are retrieval only.',
        evidence='Exclusive region jointmaskcoverage>=.5; jointmaskprecision over exclusiveunion>=.8; independentgroups>=2. Any separatingmaskcounterevidence veto. Unknown never permitsmerging. Boundary normal/color and depth continuity descriptors. Contactdepth samples up to128 deterministic evenly spaced pairs.',
        hypotheses='joint_only: broadmask positive control; compact_no_separation: compactbridge with veto; compact_boundary_depth: adds>=3 contacts<=.05m, normal/colorlimits and>=2compact+depthgroups. No rule scan or choosebestweaksignature afterlabels.',
        geometry='One-hop pair union only for fixedhypotheses; duplicateNT omitted, originalsretained. No directions inheritclassor score. This diagnoses complementary union; cannot removeparentpollution.',
        supervision='PostsealGT: reliableexclusive fragments require>=20targetpoints and>=.75purity, sameidentity vs differentvalididentities; mixedor missingevidence unknown. Allpairs reportadditiontarget/pollution inbothdirections, trueunion coverage overcompleteNT and duplicates. Reliableidentitylabels are diagnostics, not evaluatorTP.',
        advancement='Only strongest fixedsignature can advance. Must produce>1true new>.75target overNT (exceed previous1), >50% directional additions have moreanchortarget thanpollution, reliableassociationprecision>=.9 overall and nofold reliableprecision<.9 where>=10labeledpairs. Otherwise stop beforeownsemantic/AP; do not tune on thisresult.',
        new_training=False,new_AP=False,SL_modified=False,production_modified=False,dev55_read=False)
    OUT.mkdir();freeze_json(OUT/'contract.json',c)
    for q in files:
        d=OUT/'code_snapshot'/q.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,d)
    stage('prepared')

def load_nt(j):
    gs=read(j['binding']['geometry']);nt=parent.source_pool(gs)
    with np.load(j['binding']['points']) as z:pts={g['geometry_key']:z[g['array_key']].copy() for g in nt}
    return nt,pts

def load_frames(c,s,raw,points):
    from PIL import Image
    from plyfile import PlyData
    j=c['jobs'][s];manifest=read(j['paths']['manifest']);fs,bound=parent.diag.load_frames(c,s);lookup={int(f['frame_id']):f for f in manifest['frames']}
    ply=PlyData.read(manifest['files']['mesh']['path'])['vertex'].data;xyz=np.column_stack([ply[k] for k in ('x','y','z')]);assert np.array_equal(xyz,raw[:,:3])
    intrinsic=np.loadtxt(manifest['files']['intrinsics']['path']);first=lookup[fs[0]['frame']]
    width,height=Image.open(first['color']['path']).size;dw,dh=Image.open(first['depth']['path']).size
    adapted=intrinsic.copy()
    if (height,width)!=(dh,dw):
        resize_width=int(np.floor(dw*height/width));adapted[0,0]*=resize_width/height;adapted[1,1]*=dw/width;adapted[0,2]*=(dh-1)/(height-1);adapted[1,2]*=(dw-1)/(width-1)
    matrix=point_matrix(points,len(raw));mismatches=Counter()
    for f in fs:
        entry=lookup[f['frame']];pose=np.loadtxt(entry['pose']['path']);pixels,depth,valid=core.project_points(raw[:,:3],pose,adapted);vis=f['visible'];ix=np.flatnonzero(vis)
        assert valid[ix].all() and ((pixels[ix]>=0).all()) and (pixels[ix,0]<dw).all() and (pixels[ix,1]<dh).all()
        measured=np.asarray(Image.open(entry['depth']['path']))[pixels[ix,1],pixels[ix,0]].astype(float)/1000.
        error=np.abs(measured-depth[ix]);mismatches['cached_visible_points']+=len(ix);mismatches['depth_error_gt_0501']+=int((error>.0501).sum())
        # CPU projection rounding can choose an adjacent depth pixel at an edge.
        # Preserve authentic cachedvisibility; expose mismatch, never reinterpret as evidence.
        f.update(pixels=pixels,depth=depth,mask_sizes=np.array([len(m) for m in f['masks']]),candidate_intersections=(matrix@point_matrix(f['masks'],len(raw)).T).toarray(),
            candidate_visible=np.array([vis[p].sum() for p in points]),candidate_boxes=[(pixels[p[vis[p]]].min(axis=0),pixels[p[vis[p]]].max(axis=0)) if vis[p].any() else None for p in points])
    return fs,bound,dict(mismatches)

def feature_scene(c,s):
    j=c['jobs'][s];nt,pts=load_nt(j);keys=[g['geometry_key'] for g in nt];points=[pts[k] for k in keys]
    rawfile=np.load(j['paths']['processed'],mmap_mode='r');raw=np.asarray(rawfile[:,:9]).copy();del rawfile;assert np.isfinite(raw).all()
    fs,bound,projection_note=load_frames(c,s,raw,points);io,_=pair_overlap(points,len(raw));boxes=[(raw[p,:3].min(axis=0),raw[p,:3].max(axis=0)) if len(p) else None for p in points]
    total=Counter();records=[];proposal={sig:{} for sig in core.SIGNATURES};proposal_rows={sig:[] for sig in core.SIGNATURES}
    for a in range(len(keys)):
        for b in range(a+1,len(keys)):
            total['all_pairs']+=1
            common=np.intersect1d(points[a],points[b],assume_unique=True);aa=np.setdiff1d(points[a],common,assume_unique=True);bb=np.setdiff1d(points[b],common,assume_unique=True)
            if len(aa)<3 or len(bb)<3 or io[a,b]>=c['parameters']['maximum_pair_iou']:total['noncomplementary_domain']+=1;continue
            legacy=bool(.05<io[a,b]<.9);coarse=core.bbox_gap(boxes[a],boxes[b])<=c['parameters']['spatial_radius'];projectgroups=set()
            for f in fs:
                if not len(common):
                    na,nb=map(int,f['candidate_visible'][[a,b]]);ba,bc=f['candidate_boxes'][a],f['candidate_boxes'][b]
                else:
                    av=aa[f['visible'][aa]];bv=bb[f['visible'][bb]];na,nb=len(av),len(bv);ba=(f['pixels'][av].min(axis=0),f['pixels'][av].max(axis=0)) if na else None;bc=(f['pixels'][bv].min(axis=0),f['pixels'][bv].max(axis=0)) if nb else None
                if na<3 or nb<3 or na/len(aa)<.1 or nb/len(bb)<.1:continue
                if core.bbox_gap(ba,bc)<=c['parameters']['projection_bbox_gap_pixels']:projectgroups.add(f['group'])
            projection=len(projectgroups)>=c['parameters']['projection_min_groups']
            bd=core.boundary(aa,bb,raw,c['parameters']) if coarse else dict(contacts10=0,contacts05=0,contact_pairs=[],color_difference=None,normal_difference=None,normal_valid_fraction=0.,distance_min=None)
            spatial=bd['contacts10']>=c['parameters']['minimum_contacts'];retrieved=legacy or spatial or projection
            row=dict(left=keys[a],right=keys[b],pair_iou=float(io[a,b]),shared_points=len(common),left_exclusive_points=len(aa),right_exclusive_points=len(bb),retrieval=dict(legacy_overlap=legacy,spatial=spatial,projection=projection,combined=spatial or projection),boundary={k:v for k,v in bd.items() if k!='contact_pairs'})
            if retrieved:
                views=core.view_evidence(aa,bb,fs,bd['contact_pairs'],c['parameters'],endpoints=(a,b),shared=common);sig=core.signatures(bd,views,c['parameters']);row.update(views=views,signatures=sig)
                pp=None
                for name,accepted in sig.items():
                    if not accepted or not (spatial or projection):continue
                    if pp is None:pp=np.union1d(points[a],points[b])
                    digest=hashlib.sha256(pp.astype('<i8').tobytes()).hexdigest();key=s+':geometry:'+digest
                    if key in pts:total[name+'_identicalNT']+=1;continue
                    if key in proposal[name]:continue
                    proposal[name][key]=pp;proposal_rows[name].append(dict(geometry_key=key,points_sha256=digest,point_count=len(pp),left=keys[a],right=keys[b],array_key='g'+str(len(proposal_rows[name])).zfill(5)))
            else:row.update(views=None,signatures={name:False for name in core.SIGNATURES})
            records.append(row)
    d=OUT/'features'/s;d.mkdir(parents=True,exist_ok=False);freeze_json(d/'pairs.json',records);outputs={str(d/'pairs.json'):sha(d/'pairs.json')}
    for name in core.SIGNATURES:
        freeze_json(d/(name+'_geometry.json'),proposal_rows[name]);np.savez_compressed(d/(name+'_points.npz'),**{r['array_key']:proposal[name][r['geometry_key']] for r in proposal_rows[name]})
        for n in (name+'_geometry.json',name+'_points.npz'):outputs[str(d/n)]=sha(d/n)
        total[name+'_proposals']=len(proposal_rows[name])
    total['domain_pairs']=len(records)
    freeze_json(d/'receipt.json',dict(complete=True,counts=dict(total),projection_note=projection_note,inputs=bound,outputs=outputs,GT_used=False))
    return read(d/'receipt.json')

def features(c):
    sys.addaudithook(Guard(c).hook);parent.diag.prior.src.ctl.base.verify_files(c['code']);outputs={};stats=Counter();proj=Counter()
    for i,s in enumerate(c['scenes']):
        stage('GT_free_retrieval_identity',completed=i,scene=s,total=200);r=feature_scene(c,s);outputs.update(r['outputs']);stats.update(r['counts']);proj.update(r['projection_note'])
    freeze_json(OUT/'features_frozen.json',dict(complete=True,outputs=outputs,counts=dict(stats),projection_note=dict(proj),GT_used=False));stage('features_sealed',counts=dict(stats))

def labels(c):
    sys.addaudithook(Guard(c,True).hook);assert read(OUT/'features_frozen.json')['complete'];parent.diag.prior.src.ctl.base.verify_files(c['GT_inputs'])
    prior=[json.loads(x) for x in Path(next(k for k in c['label_inputs'] if k.endswith('targets.jsonl'))).read_text().splitlines()];outputs={}
    for i,s in enumerate(c['scenes']):
        stage('post_seal_labels',completed=i,scene=s,total=200);nt,pts=load_nt(c['jobs'][s]);keys=list(pts);gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);io,ids=gt_iou([pts[k] for k in keys],gt,c['class_ids']);baseline=io.max(axis=0,initial=0);sizes=np.array([np.sum(gt==gid) for gid in ids]);valid=set(ids.tolist());pairlabels=[]
        pairs=read(OUT/'features'/s/'pairs.json');by={tuple(sorted((r['left'],r['right']))):r for r in pairs};opportunity=[]
        for r in prior:
            if r['scene']!=s or not r['missing075'] or not r['NT_top12_pair_union_oracle'] or r['NT_top12_pair_union_oracle']['iou']<=c['thresholds'][1]:continue
            pp=r['NT_top12_pair_union_oracle'];edge=tuple(sorted((pp['left'],pp['right'])));x=by.get(edge);opportunity.append(dict(gt_id=r['gt_id'],left=edge[0],right=edge[1],oracle_iou=pp['iou'],pair_in_domain=x is not None,retrieval=x['retrieval'] if x else None,pair_iou=x['pair_iou'] if x else None,signatures=x['signatures'] if x else None,views=x['views'] if x else None,boundary=x['boundary'] if x else None))
        for r in pairs:
            a=pts[r['left']];b=pts[r['right']];common=np.intersect1d(a,b,assume_unique=True);aa=np.setdiff1d(a,common);bb=np.setdiff1d(b,common);union=np.union1d(a,b)
            def identity(p):
                gg,nn=np.unique(gt[p],return_counts=True);best=int(np.argmax(nn));gid=int(gg[best]);hit=int(nn[best]);purity=hit/len(p)
                return dict(gt_id=gid,purity=purity,target_points=hit,reliable=gid in valid and hit>=20 and purity>=.75)
            left,right=identity(aa),identity(bb);label='same_instance' if left['reliable'] and right['reliable'] and left['gt_id']==right['gt_id'] else ('different_instances' if left['reliable'] and right['reliable'] else 'mixed_or_unreliable')
            ug,un=np.unique(gt[union],return_counts=True);hist=dict(zip(ug,un));inter=np.array([hist.get(gid,0) for gid in ids]);ui=inter/(len(union)+sizes-inter);ui=ui if len(union)>=100 else np.zeros_like(ui);best=int(ui.argmax()) if len(ui) else None
            directional=[]
            for par,extra in ((a,np.setdiff1d(b,a)),(b,np.setdiff1d(a,b))):
                anchor=identity(par);hit=int(np.sum(gt[extra]==anchor['gt_id'])) if anchor['gt_id'] in valid else 0
                directional.append(dict(anchor=anchor,added_target_points=hit,added_pollution_points=len(extra)-hit,target_majority=hit>len(extra)-hit))
            pairlabels.append(dict(left=r['left'],right=r['right'],identity_label=label,left_identity=left,right_identity=right,directions=directional,best_union_iou=float(ui.max(initial=0)),best_union_gt=int(ids[best]) if best is not None else None,new_hits={f'{t:.2f}':ids[(baseline<=t)&(ui>t)].tolist() for t in c['thresholds']}))
        quality={}
        for name in core.SIGNATURES:
            rows=read(OUT/'features'/s/(name+'_geometry.json'))
            with np.load(OUT/'features'/s/(name+'_points.npz')) as z:pp=[z[r['array_key']] for r in rows]
            ni,_=gt_iou(pp,gt,c['class_ids']);newbest=ni.max(axis=0,initial=0);quality[name]={f'{t:.2f}':dict(new_over_complete_NT=ids[(baseline<=t)&(newbest>t)].tolist(),already_NT_covered=ids[(baseline>t)&(newbest>t)].tolist(),qualified_geometry=int(np.sum(ni.max(axis=1,initial=0)>t)),nonqualified_geometry=int(np.sum(ni.max(axis=1,initial=0)<=t))) for t in c['thresholds']}
        d=OUT/'labels'/s;d.mkdir(parents=True,exist_ok=False)
        for n,value in (('pairs.json',pairlabels),('opportunities.json',opportunity),('geometry_quality.json',quality)):
            q=d/n;freeze_json(q,value);outputs[str(q)]=sha(q)
    freeze_json(OUT/'labels_frozen.json',dict(complete=True,outputs=outputs,GT_only_post_features=True));stage('labels_completed')

def report(c):
    assert read(OUT/'labels_frozen.json')['complete'];totals={};op=[];groupresults={}
    for name,scenes in c['groups'].items():
        retrieval={mode:Counter() for mode in ('legacy_overlap','spatial','projection','combined')};signals={sig:Counter() for sig in core.SIGNATURES};quality={sig:{f'{t:.2f}':Counter() for t in c['thresholds']} for sig in core.SIGNATURES};opps=[]
        for s in scenes:
            pairs=read(OUT/'features'/s/'pairs.json');lab=read(OUT/'labels'/s/'pairs.json');assert len(pairs)==len(lab)
            for x,y in zip(pairs,lab):
                assert (x['left'],x['right'])==(y['left'],y['right'])
                for mode,selected in x['retrieval'].items():
                    if selected:retrieval[mode].update(pairs=1,**{y['identity_label']:1})
                for sig,selected in x['signatures'].items():
                    if selected and x['retrieval']['combined']:
                        cc=signals[sig];cc.update(pairs=1,**{y['identity_label']:1});cc['disjoint_pairs']+=x['shared_points']==0;cc['new75_pairs']+=bool(y['new_hits']['0.75']);cc['anchor_reliable_directions']+=sum(z['anchor']['reliable'] for z in y['directions']);cc['target_majority_directions']+=sum(z['anchor']['reliable'] and z['target_majority'] for z in y['directions']);cc['unreliable_anchor_directions']+=sum(not z['anchor']['reliable'] for z in y['directions'])
            qs=read(OUT/'labels'/s/'geometry_quality.json')
            for sig,q in qs.items():
                for t,v in q.items():quality[sig][t].update({k:len(a) if isinstance(a,list) else a for k,a in v.items()})
            for r in read(OUT/'labels'/s/'opportunities.json'):opps.append(dict(scene=s,**r))
        for mode,cc in retrieval.items():cc['recovered_oracle60']=sum(r['retrieval'] is not None and r['retrieval'][mode] for r in opps);cc['recovered_disjoint26']=sum(r['pair_iou']==0 and r['retrieval'][mode] for r in opps if r['retrieval'] is not None)
        for sig,cc in signals.items():
            den=cc['same_instance']+cc['different_instances'];cc['reliable_precision']=cc['same_instance']/den if den else None;cc['reliable_pairs']=den
        groupresults[name]=dict(retrieval={m:dict(cc) for m,cc in retrieval.items()},signatures={m:dict(cc) for m,cc in signals.items()},geometry_quality={sig:{t:dict(v) for t,v in q.items()} for sig,q in quality.items()},oracle_opportunities=len(opps))
        if name=='development200':op=opps
    full=groupresults['development200'];strong=full['signatures']['compact_boundary_depth'];geo=full['geometry_quality']['compact_boundary_depth'];new75=geo['0.75']['new_over_complete_NT'];precision=strong['reliable_precision'];addition=strong['target_majority_directions']/strong['anchor_reliable_directions'] if strong['anchor_reliable_directions'] else 0.
    foldpass=all(v['signatures']['compact_boundary_depth']['reliable_pairs']<10 or (v['signatures']['compact_boundary_depth']['reliable_precision'] or 0)>=.9 for n,v in groupresults.items() if n.startswith('fold'))
    advance=new75>1 and (precision or 0)>=.9 and addition>.5 and foldpass
    freeze_json(OUT/'summary.json',dict(complete=True,groups=groupresults,opportunity_rows=op,advance_to_own_semantics=advance,addition_target_majority_rate=addition,fold_gate=foldpass))
    lines=['# 互补碎片：检索与同实例归属分开检查','',f'已完成原开发200的无GT检索／证据封存及后置GT诊断。强固定签名是否支持进入自身语义＋冻结SL双AP阶段：{advance}。SL模型、评分、类别选择与生产均未修改，本轮无训练或新AP。','',
        '候选全集为所有NT无序配对，双方至少3独有点、pairIoU<.9；不只评估60个已知oracle。旧重叠检索作参考；空间检索使用独有区域≥3个点在.10m内，投影检索使用可见独有区域bbox间隔≤3深度像素且≥2姿态组。投影bbox用于宽检索，不是同实例判断。','',
        '|检索入口|返回配对|可靠同实例|可靠异实例|混合／身份不足|找回60机会|找回26不相交机会|','|---|---:|---:|---:|---:|---:|---:|']
    for n,v in full['retrieval'].items():lines.append('|'+n+'|'+'|'.join(str(v.get(k,0)) for k in ('pairs','same_instance','different_instances','mixed_or_unreliable','recovered_oracle60','recovered_disjoint26'))+'|')
    lines+=['','可靠身份诊断要求独有区域至少20个目标点且纯度≥.75，双方相同／不同有效GT身份；其他为混合或不可靠。这不是实际推理输入或AP TP标签。共同支持缺失仍为未知，异实例负例与未知均保留，不用只在可靠子集中得到的precision宣称输出安全率。','',
        '|固定归属假设|接受配对|同实例|异实例|混合／身份不足|可靠部分precision|','|---|---:|---:|---:|---:|---:|']
    for n,v in full['signatures'].items():lines.append(f"|{n}|{v.get('pairs',0)}|{v.get('same_instance',0)}|{v.get('different_instances',0)}|{v.get('mixed_or_unreliable',0)}|{v['reliable_precision']}|")
    lines+=['','joint_only为共同大掩码正控制；compact_no_separation要求≥2姿态组紧致共同掩码（掩码中≥.8点属于双方独有区域），且任何分离证据否决；compact_boundary_depth再要求.05m接触≥3点、颜色／法向连续及≥2共同紧致＋深度连续姿态组。深度来自物理可见3D点的相机深度，空间近邻的深度连续并不独立证明同实例。最多128接触点为确定性采样，不声称覆盖所有边界。','',
        '|假设|>.50真正新增NT覆盖|>.75真正新增NT覆盖|>.90真正新增NT覆盖|','|---|---:|---:|---:|']
    for sig,q in full['geometry_quality'].items():lines.append(f"|{sig}|{q['0.50']['new_over_complete_NT']}|{q['0.75']['new_over_complete_NT']}|{q['0.90']['new_over_complete_NT']}|")
    lines+=['',f'强签名可靠锚定方向中，新增目标点超过新增污染点的比例={addition:.6f}；五折可靠precision检查={foldpass}。准入条件在标签前固定：>.75新增目标必须超过旧方案的1个，新增区域多数有效，可靠归属precision≥.9且有足够可靠样本的折均≥.9。该门只决定是否值得进行自身语义/AP，不保证AP提高。', '',
        '原NT保留，纯追加无法删除父候选已有污染。汇总与逐场景同时报告已覆盖目标重复、合格／不合格几何、邻物负例与未知。30帧无证据不能证明全部视角漏检；源视角没有排除，只称姿态组多视图佐证，不称新独立采集证据。','',
        '所有200检索、归属证据与固定假设点集先封存，随后读取开发200白名单GT。没有使用GT类别、旧语义分数或SL结果构建检索／归属输入。既有processed文件只提取XYZ/RGB/normal前9列；标注列不进入方法。未读开发55／val312／保留集。规则失败不扫描阈值、不挑较弱签名替代强签名。','',
        '入口contract.json、features_frozen.json、labels_frozen.json、summary.json；逐场景features/*/pairs.json记录检索与真实观测见证，labels/*/pairs.json保存后置身份和新增污染组成。五折只是分组稳定性诊断，本轮没有拟合模型，开发200已暴露，不称整系统盲测。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');freeze_json(OUT/'decision.json',dict(complete=True,advance_to_own_semantics=advance,geometry_action_ready=advance,new75=new75,reliable_identity_precision=precision,added_region_target_majority_rate=addition,fold_gate=foldpass,new_training=False,new_AP=False,SL_modified=False,dev55_read=False,production_modified=False,no_threshold_scan=True))
    stage('diagnostic_completed',advance=advance)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','features','labels','report'));a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):
        if a.mode=='prepare':prepare()
        else:globals()[a.mode](read(OUT/'contract.json'))
