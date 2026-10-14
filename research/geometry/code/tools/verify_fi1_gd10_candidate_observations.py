#!/usr/bin/env python3
"""Independent group sets and existing oracle evaluator, no new inference."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from tools import audit_fi1_gd10_candidate_observations as run
from tools import verify_fi1_local_repair_capacity as existing
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def independent_groups(visible,groups):
    total=np.zeros(visible.shape[1],np.int32)
    for g in sorted(set(groups.tolist())):total+=np.any(visible[groups==g],axis=0)
    return np.minimum(total,2).astype(np.uint8)


def main():
    c=run.read(run.OUT/'contract.json');sys.addaudithook(run.Guard(c,labels=True).hook)
    run.prior.old.verify_files(c['code']);run.prior.old.verify_files(c['frozen_SL_models'])
    run.prior.old.verify_files(run.read(run.OUT/'observations_frozen.json')['outputs'])
    run.prior.old.verify_files(run.read(run.OUT/'observations_frozen.json')['input_banks'])
    run.prior.old.verify_files(run.read(run.OUT/'diagnosis_verified.json')['outputs'])
    checks=Counter();point_witnesses=0
    for s in c['scenes']:
        j=c['jobs'][s];assert sha(j['GT'])==c['GT_inputs'][j['GT']]
        gt=np.loadtxt(j['GT'],dtype=np.int64);raw=np.load(j['paths']['processed'],mmap_mode='r')
        plans=run.read(run.OUT/'plan'/s/'selected.json');by={r['candidate_id']:r for r in plans}
        rr=run.read(run.OUT/'diagnosis'/s/'rows.json');cached=set(run.read(j['views'])['raw_frames'])
        with np.load(run.OUT/'plan'/s/'evidence.npz') as z,np.load(run.OUT/'plan'/s/'roles.npz') as roles,np.load(run.prior.OUT/'plan'/s/'domains.npz') as domains,np.load(run.prior.OUT/'diagnosis'/s/'error_visibility.npz') as oldev:
            roi=z['roi'];frames=z['frames'];groups=z['groups'];visible=run.unpack(z,'reliable');covered=run.unpack(z,'covered')
            assert np.all(~covered|visible);assert np.all(frames%10==0)
            assert len(frames)==c['inventory_gd10'][s]['valid_gd10']
            records=run.read(run.prior.OUT/'plan'/s/'frames.json')['records']
            assert set(frames.tolist())=={r['frame'] for r in records if r['frame']%10==0}
            oldmask=np.isin(frames,list(cached));all_raw=np.zeros(len(gt),np.uint8)
            all_raw[oldev['point_ids']]=oldev['all_frames_groups']
            common={name:independent_groups(v,groups[mask]) for name,v,mask in
                    [('cached30',visible[oldmask],oldmask),('gd10_all',visible,np.ones(len(frames),bool)),
                     ('gd10_nonempty',visible[z['has_masks']],z['has_masks']),('gd10_covered',covered,np.ones(len(frames),bool))]}
            for row in rr:
                cid=row['candidate_id'];p=by[cid];halo=domains[cid+'_parent_halo15'];ix=np.searchsorted(roi,halo)
                for name,available in [('old30_selected6',np.flatnonzero(oldmask)),('gd10_selected6',np.arange(len(frames)))]:
                    selected,trace=run.selector.choose_views(frames[available],groups[available],visible[available][:,ix],
                         covered[available][:,ix],z['has_masks'][available],roles[cid+'_roles'],roles[cid+'_weights'])
                    assert frames[available[selected]].tolist()==p['views'][name]['frames']
                    assert trace==p['views'][name]['trace'];assert len(selected)<=c['budget']==6
                    assert len(set(groups[available[selected]].tolist()))==len(selected)
                    checks['GT_free_selection_replays']+=1
                if row['target'] is None:continue
                counts=dict(common)
                for name in ('old30_selected6','gd10_selected6'):
                    mask=np.isin(frames,p['views'][name]['frames']);counts[name]=independent_groups(visible[mask],groups[mask])
                    if name=='gd10_selected6':counts[name+'_covered']=independent_groups(covered[mask],groups[mask])
                full={n:np.zeros(len(gt),np.uint8) for n in counts}
                for n,v in counts.items():full[n][roi]=v
                full['all_raw']=all_raw
                parent=domains[cid+'_parent'];gid=row['target']
                for domain in run.prior.DOMAINS:
                    editable=domains[cid+'_'+domain]
                    for name,count in full.items():
                        result=existing.repair(parent,editable,gt,gid,count>=2)
                        assert abs(existing.score(result,gt,gid)-row['domains'][domain][name]['iou'])<1e-12
                        assert all(count[p]>=2 for p in set(result)^set(parent))
                        checks['independent_oracle_pointsets']+=1
            checks['GD10_frames_bound']+=len(frames)
            with np.load(run.OUT/'diagnosis'/s/'GD10_new_witnesses.npz') as witnesses:
                points=witnesses['point_ids'];wf=witnesses['frames'];wg=witnesses['groups']
                assert np.all(wg[:,0]!=wg[:,1]);position=np.searchsorted(roi,points)
                assert np.array_equal(roi[position],points)
                for column in (0,1):
                    indices=np.searchsorted(frames,wf[:,column])
                    assert np.array_equal(frames[indices],wf[:,column])
                    assert np.array_equal(groups[indices],wg[:,column])
                    assert visible[indices,position].all()
                checks['GD10_new_points_two_group_witness']+=len(points)
        with np.load(run.OUT/'diagnosis'/s/'nonGD10_witnesses.npz') as z:
            points=z['point_ids'];wf=z['frames'];wg=z['groups'];assert np.all(wf>=0) and np.all(wg[:,0]!=wg[:,1])
            lookup={r['frame']:r for r in records};m=run.read(j['paths']['manifest']);intrinsic=np.loadtxt(m['files']['intrinsics']['path'])
            for f in np.unique(wf):
                wanted=np.flatnonzero(np.any(wf==f,axis=1));r=lookup[int(f)]
                sensor=np.asarray(Image.open(r['depth']['path']),float)/1000.
                with Image.open(r['color']['path']) as im:w,h=im.size
                k=run.capacity.adapt_intrinsic(intrinsic,(h,w),sensor.shape)
                assert run.capacity.reliable_visibility(raw[points[wanted],:3],np.array(r['pose']),k,sensor).all()
                for column in (0,1):assert np.all(wg[wf[:,column]==f,column]==r['group'])
                checks['witness_frames_replayed']+=1
            point_witnesses+=len(points)
        print('verified',s,flush=True)
    result=dict(complete=True,checks=dict(checks),nonGD10_residual_points_with_two_group_witness=point_witnesses,
                new_training=False,new_segmentation_inference=False,new_instance_AP=False)
    freeze_json(run.OUT/'independent_verified.json',result)
    freeze_json(run.OUT/'completion_verified.json',dict(complete=True,code=c['code'],
                outputs={str(run.OUT/n):sha(run.OUT/n) for n in ('contract.json','observations_frozen.json','diagnosis_verified.json','summary.json','report.md','independent_verified.json')},
                GT_free_candidate_selection=True,budget=6,SL_unchanged=True,
                new_training=False,new_segmentation_inference=False,new_instance_AP=False))
    print(dict(checks),flush=True)


if __name__=='__main__':main()
