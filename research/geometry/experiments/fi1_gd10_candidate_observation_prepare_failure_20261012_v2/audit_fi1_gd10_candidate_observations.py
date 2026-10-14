#!/usr/bin/env python3
"""Supplement existing repair capacity with GD10 and frozen candidate views."""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from io import BytesIO
import argparse,json,sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from tools import diagnose_fi1_local_repair_capacity as prior
from tools import fi1_local_repair_capacity as capacity
from tools import fi1_candidate_observation_selection as selector
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha,validate_observation
from tools.build_details_same_frame_hierarchy_preprocessor import decode_binary_mask_rle
from tools.fi1_r0_r1_protocol import original_aliases

OUT=ROOT/'.local/fi1_gd10_candidate_observation_audit_development200_20261012_v1'
read=prior.read
BANKS=('cached30','gd10_all','gd10_nonempty','gd10_covered','old30_selected6','gd10_selected6','gd10_selected6_covered','all_raw')


class Guard:
    def __init__(self,c,labels=False):self.c,self.labels=c,labels
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(Path(args[0]).resolve())
        if Path(p).suffix in ('.py','.pyc'):return
        if any(q in p for q in ('/val312/','/reserve100/','fi1_sl_single200_dev55')):raise PermissionError('outside frozen development scope')
        if any(q.startswith('ground_truth') for q in Path(p).parts):
            if not self.labels or p not in self.c['GT_inputs'] or not (OUT/'observations_frozen.json').exists():raise PermissionError('GT only after all GT-free view plans sealed')
        if not self.labels and ('/diagnosis/' in p and not p.endswith('/frame_hashes.json') or
                Path(p).name in ('summary.json','geometry_quality.json','target_census.json','overlap.npz')):
            raise PermissionError('GT-derived capacity/identity/error points excluded from view planning')


def prepare():
    assert not OUT.exists()
    c=read(prior.OUT/'contract.json');sys.addaudithook(Guard(c).hook)
    prior.old.verify_files(c['code']);prior.old.verify_files(c['frozen_SL_models'])
    prior.old.verify_files(read(prior.OUT/'plan_frozen.json')['outputs'])
    # GT-free scopes and original data inventory are reused, not recreated.
    inputs={str(prior.OUT/'contract.json'):sha(prior.OUT/'contract.json'),
            str(prior.OUT/'plan_frozen.json'):sha(prior.OUT/'plan_frozen.json')}
    inputs.update(read(prior.OUT/'plan_frozen.json')['outputs']);inputs.update(c['inputs'])
    inventory={};frame_inputs={}
    for s in c['scenes']:
        j=c['jobs'][s];m=read(j['paths']['manifest']);records=read(prior.OUT/'plan'/s/'frames.json')['records']
        valid={r['frame'] for r in records};gd=sorted(valid&set(m['arms']['D']))
        assert gd==sorted(f for f in valid if f%10==0)
        obs=[json.loads(l) for l in Path(j['paths']['observations']).read_text().splitlines()]
        root=Path(obs[0]['point_indices_path']).parent.parent;by=defaultdict(list)
        for o in obs:by[int(o['frame_id'])].append(o)
        stat=Counter(valid_gd10=len(gd),invalid_gd10=len(set(m['arms']['D'])-valid))
        for f in gd:
            bank=root/f'frame{f:06d}';receipt=read(bank/'receipt.json')
            frame_inputs[str(bank/'receipt.json')]=sha(bank/'receipt.json')
            for name,h in receipt['files'].items():frame_inputs[str(bank/name)]=h
            assert not receipt['inference_skipped_invalid_pose']
            bankrows=read(bank/'observations.json');assert receipt['kept_observation_count']==len(bankrows)
            lookup={r['observation_id']:r for r in bankrows}
            assert all(o==lookup[o['observation_id']] for o in by[f])
            stat['nonempty_gd10']+=bool(bankrows);stat['empty_gd10']+=not bankrows
            stat['observations']+=len(bankrows);stat['downstream_ledger_observations']+=len(by[f])
        inventory[s]=dict(stat)
    paths=[Path(__file__).resolve(),Path(selector.__file__).resolve(),Path(capacity.__file__).resolve(),
           ROOT/'tools/verify_fi1_gd10_candidate_observations.py',
           ROOT/'tests/test_fi1_candidate_observation_selection.py',
           ROOT/'docs/FI1_GD10_CANDIDATE_OBSERVATION_AUDIT_20261012.md',
           ROOT/'tools/fi1_frontend_protocol.py',ROOT/'tools/build_details_same_frame_hierarchy_preprocessor.py']
    c.update(version=OUT.name,prior_contract=str(prior.OUT/'contract.json'),budget=selector.BUDGET,
             role_weights=list(selector.ROLE_WEIGHTS),inventory_gd10=inventory,
             inputs=inputs,frame_inputs=frame_inputs,code={str(p):sha(p) for p in paths},
             view_selector='6 views, one per existing full-inventory physical group; first body coverage, then two-group weighted novelty with .25 existing-mask bonus. Opposite N/T top4 IoU>.05 symmetric difference; no GT or semantic score.',
             segmentation_scope='Existing GD10 retained masks only; zero-mask frames remain in geometric ceiling. No new segmentation, tracks, training or predictions.')
    OUT.mkdir();freeze_json(OUT/'contract.json',c)
    for p in paths:
        q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    print('prepared',inventory,flush=True)


def unpack(z,name):return np.unpackbits(z[name],axis=1,count=len(z['roi']),bitorder='little').astype(bool)


def group_counts(visible,groups):
    count=np.zeros(visible.shape[1],np.uint8);first=np.full(visible.shape[1],-1,np.int32)
    for seen,g in zip(visible,groups):capacity.add_group_observation(first,count,seen,int(g))
    return count


def plan_scene(s):
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c).hook);j=c['jobs'][s]
    dest=OUT/'plan'/s
    if (dest/'receipt.json').exists():
        r=read(dest/'receipt.json');prior.old.verify_files(r['outputs']);return r
    raw=np.load(j['paths']['processed'],mmap_mode='r');n=len(raw)
    candidates=read(prior.OUT/'plan'/s/'candidates.json')
    with np.load(prior.OUT/'plan'/s/'domains.npz') as domains:
        halo={p['candidate_id']:domains[p['candidate_id']+'_parent_halo15'] for p in candidates}
        parent={p['candidate_id']:domains[p['candidate_id']+'_parent'] for p in candidates}
    roi=np.unique(np.concatenate(list(halo.values())))
    m=read(j['paths']['manifest']);fm={int(r['frame_id']):r for r in m['frames']}
    records=[r for r in read(prior.OUT/'plan'/s/'frames.json')['records'] if r['frame']%10==0]
    frames=np.array([r['frame'] for r in records]);groups=np.array([r['group'] for r in records])
    rawhash=read(prior.OUT/'diagnosis'/s/'frame_hashes.json')
    obs=[json.loads(l) for l in Path(j['paths']['observations']).read_text().splitlines()];by=defaultdict(list)
    for o in obs:by[int(o['frame_id'])].append(o)
    bankroot=Path(obs[0]['point_indices_path']).parent.parent
    visible=np.zeros((len(records),len(roi)),bool);covered=visible.copy();frame_rows=[];checked={}
    intrinsic=np.loadtxt(m['files']['intrinsics']['path'])
    for i,r in enumerate(records):
        f=r['frame'];bank=bankroot/f'frame{f:06d}';receipt=read(bank/'receipt.json')
        checked[str(bank/'receipt.json')]=sha(bank/'receipt.json')
        assert checked[str(bank/'receipt.json')]==c['frame_inputs'][str(bank/'receipt.json')]
        for name,h in receipt['files'].items():
            assert sha(bank/name)==h==c['frame_inputs'][str(bank/name)];checked[str(bank/name)]=h
        for key in ('color','depth','pose_file'):prior.assert_stat(r[key])
        assert sha(r['pose_file']['path'])==r['pose_sha256']
        data=Path(r['depth']['path']).read_bytes()
        import hashlib
        assert hashlib.sha256(data).hexdigest()==rawhash[r['depth']['path']]
        sensor=np.asarray(Image.open(BytesIO(data)),dtype=float)/1000.
        with Image.open(r['color']['path']) as image:w,h=image.size
        k=capacity.adapt_intrinsic(intrinsic,(h,w),sensor.shape)
        visible[i]=capacity.reliable_visibility(raw[roi,:3],np.array(r['pose']),k,sensor)
        cache=np.load(bank/'visibility.npy');assert cache.dtype==bool and cache.shape==(n,)
        # Cache uses .05m, this capacity audit .02m. Preserve any projection
        # boundary differences rather than quietly clipping the geometric arm.
        cache_difference=int(np.sum(visible[i]&~cache[roi]))
        bankrows=read(bank/'observations.json');assert receipt['kept_observation_count']==len(bankrows)
        lookup={o['observation_id']:o for o in bankrows};assert all(o==lookup[o['observation_id']] for o in by[f])
        allpoints=[];pointsets={}
        for o in bankrows:
            validate_observation(o,fm[f],n)
            mask=decode_binary_mask_rle(o['mask_rle']);assert mask.shape==(h,w) and int(mask.sum())==o['area']
            assert o['predicted_iou']>.88 and o['stability_score']>=.95
            q=Path(o['point_indices_path']);assert q.parent==bank
            with np.load(q) as z:p=z['point_indices']
            assert np.array_equal(p,np.unique(p)) and cache[p].all();allpoints.append(p);pointsets[o['observation_id']]=p
        if allpoints:covered[i]=visible[i]&np.isin(roi,np.unique(np.concatenate(allpoints)))
        tokens={p.astype('<i8').tobytes() for p in allpoints}
        ledger_tokens={pointsets[o['observation_id']].astype('<i8').tobytes() for o in by[f]}
        frame_rows.append(dict(frame=f,group=r['group'],existing_masks=len(bankrows),
                               downstream_ledger_masks=len(by[f]),unique_raw_pointsets=len(tokens),
                               unique_ledger_pointsets=len(ledger_tokens),ledger_missing_unique_pointsets=len(tokens-ledger_tokens),
                               reliable_roi_points=int(visible[i].sum()),mask_covered_roi_points=int(covered[i].sum()),
                               empty_cache=len(bankrows)==0,receipt=str(bank/'receipt.json'),
                               reliable_outside_cached_visibility=cache_difference))
        if i%100==0:print('GD10_evidence',s,i,len(records),'roi',len(roi),flush=True)
    geometry,points=prior.old.load_nt(j);original_source={g['geometry_key']:{a['source'] for a in original_aliases(g)} for g in geometry}
    cached=set(read(j['views'])['raw_frames']);old_indices=np.flatnonzero(np.isin(frames,list(cached)))
    assert len(old_indices)==len(cached)
    has_masks=np.array([r['existing_masks']>0 for r in frame_rows]);plans=[];role_arrays={}
    with np.load(prior.old.OUT/'plan'/s/'arrays.npz') as oldarrays:
        for p in candidates:
            cid=p['candidate_id'];pp=parent[cid];hh=halo[cid];ix=np.searchsorted(roi,hh)
            boundary=pp[oldarrays[cid+'_local']]
            other='track' if p['source']=='native' else 'native';ranked=[]
            for key,otherpoints in points.items():
                if other not in original_source[key] or key==p['parent']:continue
                hit=len(np.intersect1d(pp,otherpoints,assume_unique=True));iou=hit/(len(pp)+len(otherpoints)-hit)
                if iou>.05:ranked.append((-iou,key))
            pairs=sorted(ranked)[:4]
            differences=np.unique(np.concatenate([np.setxor1d(pp,points[key],assume_unique=True) for _,key in pairs])) if pairs else np.empty(0,int)
            roles,weights=selector.region_weights(pp,hh,boundary,differences)
            role_arrays[cid+'_roles']=np.array(roles);role_arrays[cid+'_weights']=weights
            record=dict(candidate_id=cid,source=p['source'],opposite_sources=[dict(key=key,iou=-value) for value,key in pairs],
                        role_points=[int(r.sum()) for r in roles],views={})
            for name,available in [('old30_selected6',old_indices),('gd10_selected6',np.arange(len(records)))]:
                selected,trace=selector.choose_views(frames[available],groups[available],visible[available][:,ix],
                                    covered[available][:,ix],has_masks[available],roles,weights)
                record['views'][name]=dict(frames=frames[available[selected]].tolist(),trace=trace)
            plans.append(record)
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'evidence.npz',roi=roi,frames=frames,groups=groups,has_masks=has_masks,
              reliable=np.packbits(visible,axis=1,bitorder='little'),covered=np.packbits(covered,axis=1,bitorder='little'))
    np.savez_compressed(dest/'roles.npz',**role_arrays)
    freeze_json(dest/'selected.json',plans);freeze_json(dest/'frames.json',frame_rows)
    result=dict(complete=True,GT_read=False,scene=s,valid_frames=len(records),parents=len(plans),
                input_banks=checked,outputs={str(q):sha(q) for q in dest.iterdir() if q.is_file()})
    freeze_json(dest/'receipt.json',result);print('GT_free_views_complete',s,flush=True);return result


def plan(workers):
    c=read(OUT/'contract.json');prior.old.verify_files(c['code']);prior.old.verify_files(c['inputs'])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        receipts=[f.result() for f in as_completed([pool.submit(plan_scene,s) for s in c['scenes']])]
    outputs={};inputs={}
    for r in receipts:outputs.update(r['outputs']);inputs.update(r['input_banks'])
    freeze_json(OUT/'observations_frozen.json',dict(complete=True,GT_read=False,outputs=outputs,input_banks=inputs,
                candidate_budget=selector.BUDGET,parents=sum(r['parents'] for r in receipts),GD10_frames=sum(r['valid_frames'] for r in receipts)))
    print('all_views_sealed_before_GT',flush=True)


def diagnose_scene(s):
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,labels=True).hook)
    j=c['jobs'][s];assert sha(j['GT'])==c['GT_inputs'][j['GT']]
    gt=np.loadtxt(j['GT'],dtype=np.int64);raw=np.load(j['paths']['processed'],mmap_mode='r')
    previous=read(prior.OUT/'diagnosis'/s/'rows.json');plans=read(OUT/'plan'/s/'selected.json');by={r['candidate_id']:r for r in plans}
    _,nt=prior.old.load_nt(j);ids,sizes=prior.valid_targets(gt,c['class_ids']);baseline=np.zeros(len(ids))
    for pp in nt.values():baseline=np.maximum(baseline,prior.ious(pp,gt,ids,sizes))
    rows=[];all_witnesses=[];non_gd_requested=[];gd_new_requested=[];witness_points={}
    with np.load(OUT/'plan'/s/'evidence.npz') as z,np.load(prior.OUT/'plan'/s/'domains.npz') as domains,np.load(prior.OUT/'diagnosis'/s/'error_visibility.npz') as oldev:
        roi=z['roi'];frames=z['frames'];groups=z['groups'];locator=np.full(len(gt),-1,np.int64);locator[roi]=np.arange(len(roi))
        allcount=np.zeros(len(gt),np.uint8);allcount[oldev['point_ids']]=oldev['all_frames_groups']
        vis=unpack(z,'reliable');covered=unpack(z,'covered');oldset=set(read(j['views'])['raw_frames'])
        common={}
        for name,mask,v in [('cached30',np.isin(frames,list(oldset)),vis),('gd10_all',np.ones(len(frames),bool),vis),
                             ('gd10_nonempty',z['has_masks'],vis),('gd10_covered',np.ones(len(frames),bool),covered)]:
            common[name]=group_counts(v[mask],groups[mask])
        for previous_row in previous:
            cid=previous_row['candidate_id'];plan=by[cid];row={k:previous_row[k] for k in ('scene','candidate_id','source','target','parent_iou')}
            row['views']=plan['views'];row['domains']={}
            if row['target'] is None:rows.append(row);continue
            gid=row['target'];index=int(np.flatnonzero(ids==gid)[0]);assert abs(baseline[index]-previous_row['NT_best'])<1e-12
            row['NT_best']=float(baseline[index]);parent=domains[cid+'_parent'];target=np.flatnonzero(gt==gid)
            counts=dict(common)
            for name in ('old30_selected6','gd10_selected6'):
                mask=np.isin(frames,plan['views'][name]['frames']);counts[name]=group_counts(vis[mask],groups[mask])
                if name=='gd10_selected6':counts[name+'_covered']=group_counts(covered[mask],groups[mask])
            full={n:np.zeros(len(gt),np.uint8) for n in counts}
            for n,count in counts.items():full[n][roi]=count
            full['all_raw']=allcount
            for domain in prior.DOMAINS:
                editable=domains[cid+'_'+domain];ideal=capacity.ideal_repair(parent,editable,target)
                errors=np.setxor1d(parent,ideal,assume_unique=True);q={};row['domains'][domain]=q
                for name,count in full.items():
                    result=capacity.ideal_repair(parent,editable,target,count>=2)
                    q[name]=dict(iou=float(prior.ious(result,gt,ids,sizes)[index]),
                                 observed_error_points=int(np.sum(count[errors]>=2)))
                for bank,oldbank in [('cached30','cached30'),('all_raw','all_frames')]:
                    assert abs(q[bank]['iou']-previous_row['domains'][domain][oldbank]['2']['iou'])<1e-12,(s,cid,domain,bank)
                added=errors[(full['gd10_all'][errors]>=2)&(full['cached30'][errors]<2)]
                residual=errors[(allcount[errors]>=2)&(full['gd10_all'][errors]<2)]
                non_gd_requested.append(residual);gd_new_requested.append(added)
                witness_points[cid+'_'+domain+'_new']=added;witness_points[cid+'_'+domain+'_residual']=residual
                if len(added) or len(residual):
                    contributions=[]
                    requested=np.unique(np.r_[added,residual]);positions=locator[requested]
                    # All frame contributions retained; GT only explains the
                    # already-sealed choice, never changes its frame ranking.
                    for i,f in enumerate(frames):
                        seen=requested[vis[i,positions]]
                        if not len(seen):continue
                        contributions.append(dict(frame=int(f),group=int(groups[i]),existing_masks=bool(z['has_masks'][i]),
                                                  in_old30=int(f) in oldset,
                                                  in_selected6=int(f) in plan['views']['gd10_selected6']['frames'],
                                                  new_vs30_points=int(np.isin(seen,added).sum()),
                                                  residual_points=int(np.isin(seen,residual).sum()),
                                                  mask_covered_error_points=int(covered[i,positions].sum())))
                    all_witnesses.append(dict(scene=s,candidate_id=cid,target=gid,domain=domain,
                                              new_vs30_points=len(added),residual_points=len(residual),
                                              GD10_frame_contributions=contributions))
            rows.append(row)
        wanted=np.unique(np.concatenate(non_gd_requested)) if non_gd_requested else np.empty(0,int)
        gd_seen=vis[:,locator[wanted]]
        first=np.full(len(wanted),-1,np.int32);count=np.zeros(len(wanted),np.uint8)
        witness=np.full((len(wanted),2),-1,np.int64);wg=np.full((len(wanted),2),-1,np.int32)
        def add(seen,frame,group):
            for stage in (0,1):
                ok=seen&(count==stage)
                if stage:ok&=first!=group
                # A point observed first in this frame must not count twice.
                if stage:ok&=witness[:,0]!=frame
                witness[ok,stage]=frame;wg[ok,stage]=group
                if stage==0:first[ok]=group
                count[ok]=stage+1
        for f,g,v in zip(frames,groups,gd_seen):add(v,int(f),int(g))
        reads=[];m=read(j['paths']['manifest']);intrinsic=np.loadtxt(m['files']['intrinsics']['path'])
        rawhash=read(prior.OUT/'diagnosis'/s/'frame_hashes.json')
        records=read(prior.OUT/'plan'/s/'frames.json')['records']
        for r in records:
            if r['frame']%10==0 or not np.any(count<2):continue
            data=Path(r['depth']['path']).read_bytes()
            import hashlib
            assert hashlib.sha256(data).hexdigest()==rawhash[r['depth']['path']]
            sensor=np.asarray(Image.open(BytesIO(data)),float)/1000.
            with Image.open(r['color']['path']) as image:w,h=image.size
            k=capacity.adapt_intrinsic(intrinsic,(h,w),sensor.shape)
            seen=capacity.reliable_visibility(raw[wanted,:3],np.array(r['pose']),k,sensor)
            add(seen,r['frame'],r['group']);reads.append(r['frame'])
        assert np.all(count==2),(s,'missing residual witnesses')
        newpoints=np.unique(np.concatenate(gd_new_requested)) if gd_new_requested else np.empty(0,int)
        new_frames=np.full((len(newpoints),2),-1,np.int64);new_groups=np.full((len(newpoints),2),-1,np.int32)
        newcount=np.zeros(len(newpoints),np.uint8)
        for f,g,seen in zip(frames,groups,vis[:,locator[newpoints]]):
            previous_count=newcount.copy()
            for stage in (0,1):
                ok=seen&(previous_count==stage)
                if stage:ok&=new_groups[:,0]!=g
                new_frames[ok,stage]=f;new_groups[ok,stage]=g;newcount[ok]=stage+1
        assert np.all(newcount==2)
        dest=OUT/'diagnosis'/s;dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'nonGD10_witnesses.npz',point_ids=wanted,frames=witness,groups=wg)
        np.savez_compressed(dest/'GD10_new_witnesses.npz',point_ids=newpoints,frames=new_frames,groups=new_groups)
        np.savez_compressed(dest/'candidate_witness_points.npz',**witness_points)
        freeze_json(dest/'rows.json',rows);freeze_json(dest/'GD10_frame_contributions.json',all_witnesses)
        freeze_json(dest/'witness_scan.json',dict(error_points=len(wanted),nonGD10_frames_read=reads,
                    postseal_GT_only=True,new_masks_generated=False))
        result=dict(complete=True,scene=s,outputs={str(q):sha(q) for q in dest.iterdir() if q.is_file()})
        freeze_json(dest/'receipt.json',result);return result


def summarize(c,receipts):
    rows=[]
    for s in c['scenes']:rows+=read(OUT/'diagnosis'/s/'rows.json')
    scenarios={};attribution={};budget=Counter()
    for r in rows:
        for name,views in r['views'].items():budget[name+'_parents']+=1;budget[name+'_frames']+=len(views['frames']);budget[name+'_under6']+=len(views['frames'])<6
    for domain in prior.DOMAINS:
        target_tables={}
        for bank in BANKS:
            targets={};stats=Counter()
            for r in rows:
                if r['target'] is None:continue
                value=r['domains'][domain][bank]['iou'];base=r['NT_best'];key=(r['scene'],r['target'])
                if key not in targets or value>targets[key]['oracle_iou']:
                    targets[key]=dict(scene=r['scene'],target=r['target'],candidate_id=r['candidate_id'],
                                      NT_best=base,oracle_iou=max(value,base))
                stats['parent_gain002']+=value>=r['parent_iou']+.02
            for t in prior.THRESHOLDS:stats[f'crossing_{t:.2f}']=sum(r['NT_best']<=t<r['oracle_iou'] for r in targets.values())
            useful={k for k,r in targets.items() if r['oracle_iou']>=r['NT_best']+.02}
            valued={k for k,r in targets.items() if r['oracle_iou']>=r['NT_best']+.02 or any(r['NT_best']<=t<r['oracle_iou'] for t in (.75,.90,.95))}
            stats['unique_pool_gain002']=len(useful);stats['scenes_pool_gain002']=len({k[0] for k in useful})
            stats['unique_value_targets']=len(valued)
            scenarios[domain+'/'+bank]=dict(counts=dict(stats),targets=list(targets.values()))
            target_tables[bank]=useful
        def keys(values):return [dict(scene=s,target=g) for s,g in sorted(values)]
        gd=target_tables['gd10_all'];allraw=target_tables['all_raw'];old=target_tables['cached30'];chosen=target_tables['gd10_selected6']
        attribution[domain]=dict(GD10_new_vs30=keys(gd-old),only_nonGD10_value=keys(allraw-gd),
                                  GD10_budget_retained=keys(chosen&gd),GD10_budget_lost=keys(gd-chosen),
                                  all_raw_no_gain_targets=keys(set((r['scene'],r['target']) for r in rows if r['target'] is not None)-allraw))
    result=dict(complete=True,parents=len(rows),budget=selector.BUDGET,bank_inventory=c['inventory_gd10'],
                budget_counts=dict(budget),scenarios=scenarios,attribution=attribution,
                diagnostic_oracles_only=True,new_training=False,new_segmentation_inference=False,new_instance_AP=False)
    freeze_json(OUT/'summary.json',result)
    lines=['# 完整GD10观测补核及候选级固定六帧选择','',
           '固定原10场景/80N/T候选、三修改域及全帧物理组。可见性沿原.02m深度规则；全部选帧计划先封存后读GT。下表均为超过完整NT池最佳IoU≥.02的唯一目标数，不是实际修复、折外收益或AP。','',
           '|可改范围|旧30帧|完整有效GD10|全部有效原始帧|旧30中候选选6帧|GD10中候选选6帧|',
           '|---|---:|---:|---:|---:|---:|']
    for domain in prior.DOMAINS:
        ns=[scenarios[domain+'/'+b]['counts']['unique_pool_gain002'] for b in ('cached30','gd10_all','all_raw','old30_selected6','gd10_selected6')]
        lines.append('|'+domain+'|'+'|'.join(map(str,ns))+'|')
    total=Counter()
    for v in c['inventory_gd10'].values():total.update(v)
    lines+=['',f'GD10库存：{dict(total)}。空mask帧仍进入几何可见性上界，但不进入本次六帧图像观测银行。','',
            '|可改范围|GD10非空mask帧可见|GD10任一mask覆盖的区域|选6帧任一mask覆盖的区域|',
            '|---|---:|---:|---:|']
    for domain in prior.DOMAINS:
        ns=[scenarios[domain+'/'+b]['counts']['unique_pool_gain002'] for b in ('gd10_nonempty','gd10_covered','gd10_selected6_covered')]
        lines.append('|'+domain+'|'+'|'.join(map(str,ns))+'|')
    lines+=['','几何可见、所在帧有mask、该点有任一mask覆盖不是同一回事；任一mask覆盖也不证明目标身份，不把mask外点当负例。两组可见是后置理想修复的证据域限制；旧/新六帧选择完全GT-free，第一帧主体、后续边界/父外接触/N/T分歧互补，不保护共有核心。固定预算6，未扫预算或改规则以保留GT机会。','',
            '每个新增机会的场景/目标/最佳候选在summary的targets及attribution；GD10_frame_contributions.json列父候选的帧号/组号、新增误差点、mask覆盖及是否在旧30/所选6帧。candidate_witness_points.npz关联每个候选/范围的新增与残余点；GD10_new_witnesses.npz和nonGD10_witnesses.npz逐点记录首两独立组见证。全部仅诊断，不用于实际选帧/补帧，没有非GD10新mask。','',
            '全原始帧上界复用上轮封存可见性并逐候选IoU核对，不重新运行旧实验。只有GD10仍不足且上轮全帧有证据的点才读非GD10帧找诊断见证。无法观察的区域继续保持父候选。','',
            '此次不训练局部模块、不改轨迹、不运行新分割、不接SL。之后应在原开发200物理五折，用同一局部增删模块比较原NT、旧30选帧和GD10候选选帧；10场景只核实现，几何成立后才自身语义、完整扩池上下文与固定SL双AP。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    outputs={str(OUT/n):sha(OUT/n) for n in ('summary.json','report.md')}
    for r in receipts:outputs.update(r['outputs'])
    freeze_json(OUT/'diagnosis_verified.json',dict(complete=True,outputs=outputs,GT_after_view_seal=True,
                new_training=False,new_segmentation_inference=False,new_instance_AP=False))
    print('complete_GD10_budget_diagnosis',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','plan','diagnose']);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.mode=='prepare':prepare();return
    c=read(OUT/'contract.json');prior.old.verify_files(c['code']);prior.old.verify_files(c['frozen_SL_models'])
    if a.mode=='plan':plan(a.workers);return
    prior.old.verify_files(read(OUT/'observations_frozen.json')['outputs'])
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        receipts=[f.result() for f in as_completed([pool.submit(diagnose_scene,s) for s in c['scenes']])]
    summarize(c,receipts)


if __name__=='__main__':main()
