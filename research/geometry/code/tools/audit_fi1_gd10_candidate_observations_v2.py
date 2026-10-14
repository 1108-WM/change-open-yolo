#!/usr/bin/env python3
"""Post-seal supplement: cover all repair errors, including minimum-size fillers.

The original GT-free plans and code snapshot remain unchanged. This diagnostic
function reuses v1 and extends the previously sparse error visibility domain.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from tools.audit_fi1_gd10_candidate_observations import *


def expand_all_raw(c,s,raw,gt,previous,domains,z,oldev,allcount):
    errors=[]
    for row in previous:
        if row['target'] is None:continue
        cid=row['candidate_id'];parent=domains[cid+'_parent'];hh=domains[cid+'_parent_halo15']
        errors.extend((parent[gt[parent]!=row['target']],hh[(gt[hh]==row['target'])&~np.isin(hh,parent)]))
    full_errors=np.unique(np.concatenate(errors)) if errors else np.empty(0,int)
    extra=np.setdiff1d(full_errors,oldev['point_ids'],assume_unique=True)
    positions=np.searchsorted(z['roi'],extra);assert np.array_equal(z['roi'][positions],extra)
    vis=unpack(z,'reliable');count=np.zeros(len(extra),np.uint8);first=np.full(len(extra),-1,np.int32)
    cached=np.isin(z['frames'],read(c['jobs'][s]['views'])['raw_frames'])
    # Prove previous visibility is exact on the domain it actually measured.
    oldpos=np.searchsorted(z['roi'],oldev['point_ids'])
    assert np.array_equal(group_counts(vis[cached][:,oldpos],z['groups'][cached]),oldev['cached30_groups'])
    for seen,g in zip(vis[:,positions],z['groups']):capacity.add_group_observation(first,count,seen,int(g))
    j=c['jobs'][s];m=read(j['paths']['manifest']);intrinsic=np.loadtxt(m['files']['intrinsics']['path'])
    rawhash=read(prior.OUT/'diagnosis'/s/'frame_hashes.json');reads=[]
    for r in read(prior.OUT/'plan'/s/'frames.json')['records']:
        if r['frame']%10==0 or not np.any(count<2):continue
        data=Path(r['depth']['path']).read_bytes()
        import hashlib
        assert hashlib.sha256(data).hexdigest()==rawhash[r['depth']['path']]
        sensor=np.asarray(Image.open(BytesIO(data)),float)/1000.
        with Image.open(r['color']['path']) as image:w,h=image.size
        k=capacity.adapt_intrinsic(intrinsic,(h,w),sensor.shape)
        seen=capacity.reliable_visibility(raw[extra,:3],np.array(r['pose']),k,sensor)
        capacity.add_group_observation(first,count,seen,r['group']);reads.append(r['frame'])
    allcount[extra]=count
    return allcount,dict(complete=True,prior_error_points=len(oldev['point_ids']),
                        extra_minimum_size_error_points=len(extra),nonGD10_frames_read=reads,
                        original_measured_cached_visibility_exact=True,view_plans_unchanged=True)

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
        allcount, extra_audit = expand_all_raw(c,s,raw,gt,previous,domains,z,oldev,allcount)
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
                    q[bank]['previous_iou']=previous_row['domains'][domain][oldbank]['2']['iou']
                    q[bank]['diagnostic_correction']=q[bank]['iou']-q[bank]['previous_iou']
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
        np.savez_compressed(dest/'all_raw_visibility.npz',point_ids=np.flatnonzero(allcount),groups=allcount[allcount>0])
        freeze_json(dest/'minimum_size_visibility_correction.json',extra_audit)
        result=dict(complete=True,scene=s,outputs={str(q):sha(q) for q in dest.iterdir() if q.is_file()})
        freeze_json(dest/'receipt.json',result);return result


def main():
    c=read(OUT/'contract.json');prior.old.verify_files(c['code']);prior.old.verify_files(c['frozen_SL_models'])
    amendment=read(OUT/'diagnostic_amendment.json');prior.old.verify_files(amendment['added_code'])
    prior.old.verify_files(read(OUT/'observations_frozen.json')['outputs'])
    with ProcessPoolExecutor(max_workers=4) as pool:
        receipts=[f.result() for f in as_completed([pool.submit(diagnose_scene,s) for s in c['scenes']])]
    summarize(c,receipts)
    # Freeze a corrected description separately; original v1 report and source
    # remain tied to their original contract/snapshot, with no silent overwrite.
    text=(OUT/'report.md').read_text().replace(
        '全原始帧上界复用上轮封存可见性并逐候选IoU核对，不重新运行旧实验。只有GD10仍不足且上轮全帧有证据的点才读非GD10帧找诊断见证。',
        '原全帧可见性复用，并补测此前因最低100点填充而未投影的污染点。原误差域内旧30可见性逐位一致；每父/范围的微小IoU修正保留在rows的diagnostic_correction。没有重跑旧实验或修改选帧；非GD10帧只用于遗漏点可见性及残余机会的后置见证。')
    (OUT/'report_v2.md').write_text(text)
    freeze_json(OUT/'diagnosis_v2_verified.json',dict(complete=True,
                original_view_seal_sha256=sha(OUT/'observations_frozen.json'),
                amendment_sha256=sha(OUT/'diagnostic_amendment.json'),
                outputs={str(OUT/n):sha(OUT/n) for n in ('summary.json','report_v2.md','diagnosis_verified.json')},
                new_training=False,new_segmentation_inference=False))


if __name__=='__main__':main()

