#!/usr/bin/env python3
"""Freeze a GT-free repair scope, then diagnose capacity, visibility and units."""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from tools import fi1_local_repair_capacity as core
from tools import run_fi1_anchor_local_decontamination as old
from tools.fi1_frontend_protocol import freeze_json, write_json, sha256_file as sha

OUT = ROOT/'.local/fi1_candidate_local_repair_capacity_development200_20261012_v1'
read = old.read
DOMAINS = ('old_boundary', 'whole_parent', 'parent_halo15')
THRESHOLDS = (.50, .75, .80, .85, .90, .95)


def status(stage, **kwargs):
    write_json(OUT/'execution.json', dict(stage=stage, updated=time.time(), **kwargs))
    print(stage, kwargs, flush=True)


def stat_binding(path):
    q = Path(path); s = q.stat()
    return dict(path=str(q), size=s.st_size, mtime_ns=s.st_mtime_ns)


def assert_stat(binding):
    assert stat_binding(binding['path']) == binding, binding['path']


class LabelGuard:
    def __init__(self, c, labels=False):
        self.c, self.labels = c, labels

    def hook(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        path = str(Path(args[0]).resolve())
        if Path(path).suffix in ('.py', '.pyc'):
            return
        if any(n in path for n in ('/val312/', '/reserve100/', 'fi1_sl_single200_dev55')):
            raise PermissionError('outside development repair capacity scope')
        if any(n.startswith('ground_truth') for n in Path(path).parts):
            if not self.labels or path not in self.c['GT_inputs'] or not (OUT/'plan_frozen.json').exists():
                raise PermissionError('GT only after input-domain seal, selected-scene whitelist')


def prepare():
    assert not OUT.exists()
    c = read(old.OUT/'contract.json')
    old.verify_files(read(old.OUT/'geometry_frozen.json')['outputs'])
    old.verify_files(c['frozen_SL_models'])
    # Selection is reused exactly, including all original anchor abstentions.
    OUT.mkdir(); sys.addaudithook(LabelGuard(c).hook)
    outputs = {}; inventory = {}; count = Counter()
    for s in c['scenes']:
        j = c['jobs'][s]; raw = np.load(j['paths']['processed'], mmap_mode='r')
        plans = read(old.OUT/'plan'/s/'candidates.json')
        base = Path(j['paths']['processed']).parent
        files = {n:{int(p.stem):p for p in (base/n).iterdir() if p.stem.isdigit()}
                 for n in ('color', 'depth', 'poses')}
        complete = sorted(set.intersection(*(set(v) for v in files.values())))
        poses = {}; records = []; invalid = []
        for frame in complete:
            pose = np.loadtxt(files['poses'][frame])
            if pose.shape != (4,4) or not np.isfinite(pose).all() or abs(np.linalg.det(pose)) < 1e-12:
                invalid.append(frame); continue
            poses[frame] = pose
            r = dict(frame=frame, pose=pose.tolist(), pose_sha256=sha(files['poses'][frame]))
            r.update({n:stat_binding(files[k][frame]) for n,k in
                      (('color','color'),('depth','depth'),('pose_file','poses'))})
            records.append(r)
        groups = core.physical_groups(poses)
        for r in records:r['group'] = groups[r['frame']]
        arrays = {}; rows = []
        with np.load(old.OUT/'plan'/s/'arrays.npz') as z:
            for p in plans:
                cid = p['candidate_id']; parent = z[cid+'_parent']
                assert np.array_equal(parent, np.unique(parent))
                arrays[cid+'_parent'] = parent
                arrays[cid+'_old_boundary'] = parent[z[cid+'_local'] & ~np.isin(parent,z[cid+'_core'])]
                arrays[cid+'_whole_parent'] = parent
                arrays[cid+'_parent_halo15'] = core.halo(parent, raw[:,:3])
                rows.append({k:p[k] for k in ('candidate_id','parent','source','accepted')})
                count['parents'] += 1; count[p['source']] += 1
        d = OUT/'plan'/s; d.mkdir(parents=True)
        np.savez_compressed(d/'domains.npz', **arrays)
        freeze_json(d/'candidates.json', rows)
        freeze_json(d/'frames.json', dict(records=records, invalid_poses=invalid,
                    available={k:len(v) for k,v in files.items()}, complete=len(complete),
                    valid=len(records), physical_groups=len(set(groups.values()))))
        outputs.update({str(q):sha(q) for q in d.iterdir()})
        inventory[s] = dict(complete=len(complete), valid=len(records), invalid_poses=len(invalid),
                            physical_groups=len(set(groups.values())), parents=len(rows))
        count['complete_frames'] += len(complete); count['valid_frames'] += len(records)
        print('planned',s,inventory[s],flush=True)
    scripts = [Path(__file__).resolve(),Path(core.__file__).resolve(),
               ROOT/'tests/test_fi1_local_repair_capacity.py',
               ROOT/'docs/FI1_LOCAL_REPAIR_CAPACITY_PROTOCOL_20261012.md']
    models = c['frozen_SL_models']; old.verify_files(models)
    contract = dict(version=OUT.name, scenes=c['scenes'], jobs=c['jobs'], folds=c['folds'],
                    groups=c['groups'], class_ids=c['class_ids'], GT_inputs=c['GT_inputs'],
                    inputs=c['inputs'], frozen_SL_models=models, inventory=inventory,
                    code={str(p):sha(p) for p in scripts},
                    prior_contract=str(old.OUT/'contract.json'), prior_contract_sha256=sha(old.OUT/'contract.json'),
                    prior_seal_sha256=sha(old.OUT/'geometry_frozen.json'),
                    selection='Exact prior hash-ranked 10-scene 80-parent selection, including anchor abstentions; no replacements or GT stratification.',
                    domains=list(DOMAINS), radius=.15, voxel_size=.02, thresholds=list(THRESHOLDS),
                    decision='Compare unrestricted local point ceiling, all-frame observable ceiling, superpoint/2cm unit ceiling against complete NT; no parameter sweep, learned predictions, training or AP.',
                    visibility='All on-disk color/depth/pose intersection with finite invertible poses; original depth intrinsic convention, measured depth>0, camera z>0, depth difference<=.02m. Count distinct .1m/15degree pose groups. No mask quality or positive veto.',
                    visibility_GT_use='Only post-seal error points need projection; all valid frames are visited in frame order, without GT frame ranking. This is an oracle diagnostic, not an inference view selector.',
                    raw_frame_binding='File size and nanosecond mtime before GT; pose content hash frozen. Depth content hashes collected during full-frame scan and checked against unchanged metadata. RGB header read only, no 2D model.',
                    identity='Postseal best valid parent-target IoU, ties by smallest instance ID; unknown parent separate. Not an inference identity rule.',
                    unit_ceiling='Optimistic whole-unit decisions clipped to fixed domain; exact unconstrained IoU ratio ceiling, <100-point optima explicitly marked. No claim of a feasible full-superpoint action.',
                    original_NT_retained=True, new_training=False, new_instance_AP=False,
                    SL_modified=False, production_modified=False)
    freeze_json(OUT/'contract.json',contract)
    freeze_json(OUT/'plan_frozen.json',dict(complete=True,GT_read=False,outputs=outputs,counts=dict(count)))
    status('GT_free_scope_sealed',counts=dict(count))


def valid_targets(gt, classes):
    ids,sizes = np.unique(gt,return_counts=True)
    valid = (ids >= 1000) & (sizes >= 100) & np.isin(ids//1000,classes)
    return ids[valid],sizes[valid]


def ious(points, gt, ids, sizes):
    if len(points)<100:return np.zeros(len(ids))
    counts=Counter(gt[points].tolist())
    hits=np.array([counts[g] for g in ids])
    return hits/(len(points)+sizes-hits)


def funnel_scene(c,s,gt,ids,sizes,baseline):
    quality=read(old.OUT/'geometry_quality.json')
    known={r['candidate_id']:r['anchor']['gt_id'] for r in quality['actions']
           if r['scene']==s and r['anchor']['reliable']}
    views=read(old.OUT/'geometry'/s/'views.json'); by=defaultdict(list)
    for v in views:by[v['candidate_id']].append(v)
    counts={name:Counter() for name in ('all_non_target','valid_other_instance')}; rows=[]
    with np.load(old.OUT/'plan'/s/'arrays.npz') as a,np.load(old.OUT/'geometry'/s/'evidence.npz') as e:
        for cid,gid in known.items():
            parent=a[cid+'_parent']; local=a[cid+'_local'] & ~np.isin(parent,a[cid+'_core'])
            visible={}; valid={arm:{} for arm in old.ARMS}; excluded={arm:{} for arm in old.ARMS}
            positive={arm:np.zeros(len(parent),bool) for arm in old.ARMS}
            for v in by[cid]:
                g=v['group']; vi=v['view_index']; seen=e[f'{cid}_v{vi}_old_reliable']
                visible[g]=visible.get(g,np.zeros(len(parent),bool)) | seen
                for arm in old.ARMS:
                    if v[arm+'_valid']:
                        valid[arm][g]=valid[arm].get(g,np.zeros(len(parent),bool)) | seen
                        excluded[arm][g]=excluded[arm].get(g,np.zeros(len(parent),bool)) | (seen&e[f'{cid}_v{vi}_{arm}_excluded'])
                        positive[arm] |= seen & e[f'{cid}_v{vi}_{arm}_inside']
            votes=lambda d:np.sum(list(d.values()),axis=0) if d else np.zeros(len(parent),int)
            seen2=votes(visible)>=2
            for name in counts:
                pollution=(gt[parent]!=gid)
                if name=='valid_other_instance':pollution &= np.isin(gt[parent],ids)
                d=counts[name];d['reliable_anchors']+=1;d['parent_pollution']+=int(pollution.sum())
                d['editable_outside_core']+=int((pollution&local).sum())
                d['selected_visible_2groups']+=int((pollution&local&seen2).sum())
                for arm in old.ARMS:
                    d[arm+'_valid_2groups']+=int((pollution&local&(votes(valid[arm])>=2)).sum())
                    d[arm+'_excluded_2groups']+=int((pollution&local&(votes(excluded[arm])>=2)).sum())
                    d[arm+'_after_positive_veto']+=int((pollution&local&(votes(excluded[arm])>=2)&~positive[arm]).sum())
            target=np.flatnonzero(gt==gid); index=int(np.flatnonzero(ids==gid)[0])
            for scope,editable in [('selected_visible_parent',parent[seen2]),
                                   ('selected_visible_old_boundary',parent[seen2&local])]:
                repaired=core.ideal_repair(parent,editable,target)
                score=float(ious(repaired,gt,ids,sizes)[index]);base=float(baseline[index])
                rows.append(dict(scene=s,candidate_id=cid,scope=scope,target=int(gid),
                                 NT_best=base,oracle_iou=score,pool_gain=max(0.,score-base),
                                 value=bool(score>=base+.02 or any(base<=t<score for t in (.75,.90,.95)))))
    return dict(counts={n:dict(v) for n,v in counts.items()},oracles=rows)


def scan_frames(c,s,raw,roi):
    inventory=read(OUT/'plan'/s/'frames.json'); records=inventory['records']
    m=read(c['jobs'][s]['paths']['manifest']); intrinsic=np.loadtxt(m['files']['intrinsics']['path'])
    cached=set(read(c['jobs'][s]['views'])['raw_frames'])
    all_count=np.zeros(len(roi),np.uint8); all_first=np.full(len(roi),-1,np.int32)
    cached_count=all_count.copy(); cached_first=all_first.copy(); hashes={}; compatibility=Counter()
    # One sorted ROI consists only of postseal annotation error points. Every
    # available valid frame is still read; no GT-selective frame subsampling.
    for i,r in enumerate(records):
        for key in ('depth','color','pose_file'):assert_stat(r[key])
        assert sha(r['pose_file']['path'])==r['pose_sha256']
        data=Path(r['depth']['path']).read_bytes(); hashes[r['depth']['path']]=hashlib.sha256(data).hexdigest()
        sensor=np.asarray(Image.open(BytesIO(data)),dtype=np.float64)/1000.
        with Image.open(r['color']['path']) as image:w,h=image.size
        k=core.adapt_intrinsic(intrinsic,(h,w),sensor.shape)
        seen=core.reliable_visibility(raw[roi,:3],np.asarray(r['pose']),k,sensor)
        core.add_group_observation(all_first,all_count,seen,r['group'])
        if r['frame'] in cached:
            core.add_group_observation(cached_first,cached_count,seen,r['group'])
        saved=old.OUT/'geometry'/s/'observations'/f"frame{r['frame']:06d}.npz"
        if saved.exists():
            with np.load(saved) as z:
                before=z['reliable'][roi]
            compatibility['frames']+=1; compatibility['points']+=len(roi)
            compatibility['different_points']+=int((seen!=before).sum())
        if i%200==0:print('visibility',s,i,len(records),'roi',len(roi),flush=True)
    return all_count,cached_count,hashes,dict(compatibility)


def diagnose_scene(s):
    c=read(OUT/'contract.json');sys.addaudithook(LabelGuard(c,labels=True).hook)
    dest=OUT/'diagnosis'/s
    if (dest/'receipt.json').exists():
        receipt=read(dest/'receipt.json');old.verify_files(receipt['outputs']);return receipt
    j=c['jobs'][s];assert sha(j['GT'])==c['GT_inputs'][j['GT']]
    gt=np.loadtxt(j['GT'],dtype=np.int64);raw=np.load(j['paths']['processed'],mmap_mode='r')
    ids,sizes=valid_targets(gt,c['class_ids']);geometry,nt=old.load_nt(j)
    baseline=np.zeros(len(ids))
    for p in nt.values():baseline=np.maximum(baseline,ious(p,gt,ids,sizes))
    plans=read(OUT/'plan'/s/'candidates.json');rows=[];oracle_sets={};roi=[]
    sp=raw[:,9].astype(np.int64);voxel=np.floor(raw[:,:3]/c['voxel_size']).astype(np.int64)
    units={'superpoint':sp,'voxel2cm':voxel}
    with np.load(OUT/'plan'/s/'domains.npz') as domains:
        for p in plans:
            cid=p['candidate_id'];parent=domains[cid+'_parent'];before=ious(parent,gt,ids,sizes)
            index=int(np.argmax(before)) if len(ids) and before.max()>0 else None
            row=dict(scene=s,candidate_id=cid,source=p['source'],original_anchor=p['accepted'],
                     target=None if index is None else int(ids[index]),
                     parent_iou=float(before.max(initial=0)),domains={})
            if index is not None:
                gid=int(ids[index]);target=np.flatnonzero(gt==gid)
                row.update(NT_best=float(baseline[index]),parent_points=len(parent),
                           pollution_points=int(np.sum(gt[parent]!=gid)),
                           other_valid_instance_points=int(np.sum(np.isin(gt[parent],ids)&(gt[parent]!=gid))),
                           missing_target_points=len(target)-int(np.sum(gt[parent]==gid)))
                for name in DOMAINS:
                    editable=domains[cid+'_'+name];ideal=core.ideal_repair(parent,editable,target)
                    oracle_sets[(cid,name)]=ideal
                    errors=np.setxor1d(parent,ideal,assume_unique=True);roi.append(errors)
                    score=float(ious(ideal,gt,ids,sizes)[index])
                    u={n:core.unit_ceiling(parent,editable,gt,gid,values) for n,values in units.items()}
                    # Distinguish valid adjacent instances from ignored/background.
                    _,inv=np.unique(sp[editable],return_inverse=True)
                    tu=np.unique(inv[gt[editable]==gid]);other=np.unique(inv[np.isin(gt[editable],ids)&(gt[editable]!=gid)])
                    mixed=np.intersect1d(tu,other)
                    row['domains'][name]=dict(points=len(editable),oracle_iou=score,
                        removed_pollution=int(np.sum(~np.isin(parent,ideal))),
                        added_target=int(np.sum(~np.isin(ideal,parent))),
                        units=u,mixed_target_other_instance_superpoints=len(mixed),
                        target_points_in_instance_mixed_superpoints=int(np.sum((gt[editable]==gid)&np.isin(inv,mixed))))
            rows.append(row)
        union=np.unique(np.concatenate(roi)) if roi else np.empty(0,np.int64)
        seen,cached,hashes,compat=scan_frames(c,s,raw,union)
        fullseen=np.zeros(len(gt),np.uint8);fullseen[union]=seen
        fullcached=np.zeros(len(gt),np.uint8);fullcached[union]=cached
        for row in rows:
            if row['target'] is None:continue
            cid=row['candidate_id'];parent=domains[cid+'_parent'];gid=row['target'];target=np.flatnonzero(gt==gid)
            index=int(np.flatnonzero(ids==gid)[0])
            for name in DOMAINS:
                d=row['domains'][name];editable=domains[cid+'_'+name];ideal=oracle_sets[(cid,name)]
                errors=np.setxor1d(parent,ideal,assume_unique=True)
                d['repair_error_points']=len(errors)
                for bank,counts in [('cached30',fullcached),('all_frames',fullseen)]:
                    d[bank]={}
                    for minimum in (1,2):
                        repaired=core.ideal_repair(parent,editable,target,counts>=minimum)
                        score=float(ious(repaired,gt,ids,sizes)[index])
                        d[bank][str(minimum)]=dict(iou=score,observed_error_points=int(np.sum(counts[errors]>=minimum)),
                                                 removed_pollution=len(np.setdiff1d(parent,repaired)),
                                                 added_target=len(np.setdiff1d(repaired,parent)))
    funnel=funnel_scene(c,s,gt,ids,sizes,baseline)
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'error_visibility.npz',point_ids=union,all_frames_groups=seen,cached30_groups=cached)
    freeze_json(dest/'rows.json',rows);freeze_json(dest/'funnel.json',funnel)
    freeze_json(dest/'frame_hashes.json',hashes)
    result=dict(complete=True,scene=s,parents=len(rows),targets=len(ids),
                valid_frames=c['inventory'][s]['valid'],projected_error_points=len(union),
                cached_projection_comparison=compat,diagnostic_oracles_only=True,
                outputs={str(q):sha(q) for q in dest.iterdir() if q.is_file()})
    freeze_json(dest/'receipt.json',result)
    print('scene_complete',s,flush=True);return result


def summary(c,receipts):
    rows=[];funnel={n:Counter() for n in ('all_non_target','valid_other_instance')};selected=[]
    for s in c['scenes']:
        rows.extend(read(OUT/'diagnosis'/s/'rows.json'))
        f=read(OUT/'diagnosis'/s/'funnel.json');selected.extend(f['oracles'])
        for n in funnel:funnel[n].update(f['counts'][n])
    scenarios={}
    for domain in DOMAINS:
        for evidence in ('unrestricted','cached30_1','cached30_2','all_frames_1','all_frames_2','superpoint','voxel2cm'):
            name=domain+'/'+evidence; targets={};stats=Counter();examples=[]
            for row in rows:
                if row['target'] is None:continue
                d=row['domains'][domain];base=row['NT_best'];before=row['parent_iou']
                if evidence=='unrestricted':score=d['oracle_iou']
                elif evidence in ('superpoint','voxel2cm'):
                    score=d['units'][evidence]['iou']
                    stats['unit_optimum_below100']+=not d['units'][evidence]['evaluator_eligible']
                else:
                    bank,minimum=evidence.rsplit('_',1);score=d[bank][minimum]['iou']
                    stats['observed_error_points']+=d[bank][minimum]['observed_error_points']
                    stats['repair_error_points']+=d['repair_error_points']
                stats['known_parents']+=1;stats['parent_gain_ge002']+=score>=before+.02
                stats['pool_gain_ge002_parents']+=score>=base+.02
                stats['mixed_target_other_instance_superpoints']+=d['mixed_target_other_instance_superpoints']
                key=(row['scene'],row['target']);existing=targets.get(key)
                if existing is None or score>existing['score']:
                    targets[key]=dict(scene=row['scene'],target=row['target'],base=base,
                                      score=max(base,score),candidate_id=row['candidate_id'])
                if score>=base+.02:
                    examples.append(dict(scene=row['scene'],candidate_id=row['candidate_id'],source=row['source'],
                                         target=row['target'],parent_iou=before,NT_best=base,oracle_iou=score))
            valued=[r for r in targets.values() if r['score']>=r['base']+.02 or any(r['base']<=t<r['score'] for t in (.75,.90,.95))]
            stats['unique_value_targets']=len(valued);stats['value_scenes']=len({r['scene'] for r in valued})
            stats['unique_pool_gain_ge002_targets']=sum(r['score']>=r['base']+.02 for r in targets.values())
            for t in THRESHOLDS:stats[f'crossing_{t:.2f}']=sum(r['base']<=t<r['score'] for r in targets.values())
            scenarios[name]=dict(counts=dict(stats),targets=list(targets.values()),examples=sorted(examples,key=lambda r:-(r['oracle_iou']-r['NT_best'])))
    comparison=Counter()
    for r in receipts:comparison.update(r['cached_projection_comparison'])
    report=dict(complete=True,parents=len(rows),unknown_parents=sum(r['target'] is None for r in rows),
                scenes=len(c['scenes']),valid_frames=sum(r['valid_frames'] for r in receipts),
                funnel={n:dict(v) for n,v in funnel.items()},selected_view_oracles=selected,
                scenarios=scenarios,cached_projection_comparison=dict(comparison),
                diagnostic_oracles_only=True,new_training=False,new_instance_AP=False,
                SL_modified=False,production_modified=False)
    freeze_json(OUT/'summary.json',report)
    lines=['# 候选条件局部修复能力检查','',
           f"固定开发200内上一轮非GT选择的{len(c['scenes'])}场景、{len(rows)}父候选，未知父候选{report['unknown_parents']}；不按GT补选。原五折×旧90/新110，N/T均覆盖，锚失败父候选保留。全部局部域和有效帧清单先封存。",'',
           f"检查磁盘上{report['valid_frames']}个有效RGB-D-pose帧，不运行SAM或学习模型。全部输出为后置GT理想修复诊断，不能当预测、折外收益或实例AP。",'',
           '|范围/证据|父提升≥.02|超过完整NT≥.02的目标|有价值目标/场景|跨>.75|跨>.90|跨>.95|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for n,r in scenarios.items():
        q=r['counts'];lines.append(f"|{n}|{q['parent_gain_ge002']}|{q['unique_pool_gain_ge002_targets']}|{q['unique_value_targets']}/{q['value_scenes']}|{q['crossing_0.75']}|{q['crossing_0.90']}|{q['crossing_0.95']}|")
    lines+=['','旧漏斗同时保留“所有非目标点”和“有效其他实例点”两个口径，按候选累计：','','```json',json.dumps(report['funnel'],ensure_ascii=False,indent=2),'```','',
            'old_boundary：父候选内旧.15m边界，排除旧core；whole_parent：整个父候选，只能删；parent_halo15：整个父候选及其.15m原始点邻域，可增删。后两者不保护N/T交集。全帧和30缓存帧使用相同的全帧物理分组。所有遮挡、缺深度及未观察错误点保持原父状态。',
            '', 'superpoint和voxel2cm为固定范围内整单元选择的理想上界；单元在范围边缘裁断，是比真正不可拆整超点更宽松的上界。<100点最优解单独计数，不能认定其通过评测。没有按GT选实际动作，未输出可用于SL的候选点集。',
            '',f"与旧保存可靠可见性的交叉检查：{json.dumps(dict(comparison),ensure_ascii=False)}。CPU double投影采用原深度内参、整数截断；少量像素边界差异须保留并说明，不能称逐位复现。",'',
            '当前检查只覆盖固定10场景，不是开发200全量结论。几何学习仍需原物理场景五折、冻结骨干与SL；正负/混合/好候选保持监督，报告误删、漂移、重复及池级收益。几何成立后才获取新点集自身语义并重算完整扩池上下文，保留NT-only与允许新候选对照。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    old.verify_files(c['frozen_SL_models']);old.verify_files(c['code']);old.verify_files(read(OUT/'plan_frozen.json')['outputs'])
    outputs={str(OUT/n):sha(OUT/n) for n in ('summary.json','report.md')}
    for r in receipts:outputs.update(r['outputs'])
    freeze_json(OUT/'verified.json',dict(complete=True,outputs=outputs,
                plan_sealed_before_GT=True,SL_weights_unchanged=True,new_training=False,new_instance_AP=False))
    status('complete_capacity_diagnosis',scenes=len(receipts),parents=len(rows),frames=report['valid_frames'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','diagnose']);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.mode=='prepare':prepare();return
    c=read(OUT/'contract.json');old.verify_files(c['code']);old.verify_files(c['frozen_SL_models']);old.verify_files(read(OUT/'plan_frozen.json')['outputs'])
    old.verify_files(c['inputs']);assert sha(c['prior_contract'])==c['prior_contract_sha256']
    status('postseal_capacity_diagnosis',workers=args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(diagnose_scene,s):s for s in c['scenes']};receipts=[]
        for future in as_completed(jobs):receipts.append(future.result())
    summary(c,receipts)


if __name__=='__main__':main()
