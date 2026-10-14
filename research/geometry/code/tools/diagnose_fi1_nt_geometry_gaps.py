#!/usr/bin/env python3
"""Frozen SL baseline, geometry extent diagnosis on original development200."""
from pathlib import Path
from collections import Counter,defaultdict
import json,sys,argparse,shutil,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
from tools.fi1_core_ablation import source_pool
from tools.fi1_r3_core import point_matrix
from tools import diagnose_fi1_sl_remaining_development200 as prior
from tools.fi1_r0_r1_protocol import original_aliases
OUT=ROOT/'.local/fi1_nt_geometry_gap_diagnosis_development200_20261010_v1'
OLD=prior.OUT
ADMISSION=ROOT/'.local/fi1_union_admission_development200_20261007_v2'
UNION=ROOT/'.local/fi1_union_generation_objective_development200_20261007_v1'
read=prior.read


def extent_kind(purity,coverage,intersection,threshold=.75):
    if intersection==0:return 'no_overlap'
    if purity>threshold and coverage<=threshold:return 'incomplete_clean'
    if coverage>threshold and purity<=threshold:return 'complete_contaminated'
    if purity<=threshold and coverage<=threshold:return 'incomplete_contaminated'
    return 'boundary_both'


def stage(mode,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=mode,updated=time.time(),**kw));print(mode,kw,flush=True)


class Guard:
    def __init__(self,c):self.c=c;self.labels=False
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(Path(args[0]).resolve())
        if 'val312' in p or 'reserve100' in p or 'fi1_sl_single200_dev55' in p:raise PermissionError('fixed geometry research excludes other evaluation/model data')
        if any(x.startswith('ground_truth') for x in Path(p).parts):
            if not self.labels or p not in self.c['GT_inputs'] or not (OUT/'evidence_frozen.json').exists():raise PermissionError('GT only post-evidence diagnosis, original200 whitelist')
        if not self.labels and (p.endswith('overlap.npz') or p.endswith('target_census.json')):raise PermissionError('labels excluded from observation extraction')
        import re
        if any(s not in self.c['scenes'] for s in re.findall(r'scene\d{4}_\d{2}',p)):raise PermissionError('scene outside development200')


def prepare():
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>=50*1024**3
    p=read(OLD/'contract.json');a=read(ADMISSION/'contract.json');u=read(UNION/'contract.json')
    assert p['scenes']==a['scenes']==u['scenes']
    inputs={};gt={};jobs={}
    for s in p['scenes']:
        job=p['jobs'][s];b=dict(job['binding']);paths=a['jobs'][s]['paths']
        views=ADMISSION/'features'/s/'views.json';vr=read(ADMISSION/'features'/s/'receipt.json')
        assert sha(views)==vr['outputs'][str(views)]
        for n in ('geometry','points','evidence'):inputs[b[n]]=p['inputs'][b[n]]
        for n in ('observations','tracks','manifest','processed'):inputs[paths[n]]=a['inputs'][paths[n]]
        inputs[str(views)]=sha(views)
        g=u['bindings'][s];gt[g['gt_path']]=g['gt_sha256']
        jobs[s]=dict(binding=b,paths=paths,views=str(views),GT=g['gt_path'],overlap=job['overlap'],census=job['census'])
    inputs[str(OLD/'contract.json')]=sha(OLD/'contract.json');inputs[str(OLD/'report_verified.json')]=sha(OLD/'report_verified.json')
    code={str(ROOT/n):sha(ROOT/n) for n in ('tools/diagnose_fi1_nt_geometry_gaps.py','tests/test_fi1_geometry_extent.py','tools/fi1_r3_core.py','tools/fi1_core_ablation.py')}
    c=dict(version=OUT.name,scenes=p['scenes'],folds=p['folds'],groups=p['groups'],class_ids=p['class_ids'],thresholds=p['thresholds'],jobs=jobs,
        inputs=inputs,GT_inputs=gt,code=code,authorization='User directs geometry gap diagnosis, GT for explanations/supervision, then implement one evidence-supported geometry action. Frozen SL, no loss/head tuning.',
        diagnostic='All4689 targets, focus877 NT oracle IoU<=.75. Top5 perN/perT, fullNT best purity/recall, true-region union/completion opportunity. Ratios continuous; .75 descriptor is not an inference rule.',
        observation_protocol='Reuse30 deterministic uniform raw cached GD-SAM frames and existing .1m/15deg pose groups. Seal bank identities before GT. Local masks have physical visibility; no claim that30-view failure is frontend failure in all frames.',
        repair_oracles='GT-picked NT pair unions and observation repair are opportunity diagnostics only. Distinguish unobserved, visible-but-unmasked and support dropped by tracks. No oracle geometry exported as predictions.',
        frozen_baseline='Original development200 fivefold SL, NT/S eligibility/K800. No single-model/55 reads. No new method selection based on55.',
        new_geometry=False,new_training=False,new_AP=False,production_modified=False)
    # GT hashes are bound here from prior receipts, not opened.
    prior.src.ctl.base.verify_files(inputs);prior.src.ctl.base.verify_files(code)
    OUT.mkdir();freeze_json(OUT/'contract.json',c)
    for q in code:
        d=OUT/'code_snapshot'/Path(q).relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,d)
    stage('prepared')


def load_frames(c,s,verify=True):
    job=c['jobs'][s];meta=read(job['views']);obs=prior.src.ctl.base.rows(job['paths']['observations']);byframe=defaultdict(list)
    for o in obs:byframe[int(o['frame_id'])].append(o)
    frames=[];bound={job['views']:sha(job['views']),job['paths']['observations']:sha(job['paths']['observations'])}
    for f in meta['raw_frames']:
        bank=Path(ADMISSION).parent  # bank identity comes from the observation itself
        bank=Path(byframe[f][0]['point_indices_path']).parent
        rc=read(bank/'receipt.json');bound[str(bank/'receipt.json')]=sha(bank/'receipt.json')
        q=bank/'visibility.npy';h=sha(q);assert h==rc['files'][q.name];bound[str(q)]=h
        vis=np.load(q,allow_pickle=False);assert vis.shape==(job['binding']['point_count'],) and vis.dtype==np.bool_
        masks=[];ids=[];seen=set()
        for o in byframe[f]:
            q=Path(o['point_indices_path']);assert q.parent==bank
            h=sha(q);assert h==rc['files'][q.name];bound[str(q)]=h
            with np.load(q,allow_pickle=False) as z:p=z['point_indices'].copy()
            assert p.dtype.kind in 'iu' and np.array_equal(np.unique(p),p) and vis[p].all()
            token=p.astype('<i8').tobytes()
            if token in seen:continue
            seen.add(token);masks.append(p);ids.append(int(o['observation_id']))
        frames.append(dict(frame=int(f),group=int(meta['pose_groups'][str(f)]),visible=vis,masks=masks,observation_ids=ids))
    return frames,bound


def evidence(c,guard):
    seals={}
    for i,s in enumerate(c['scenes']):
        stage('observation_evidence',completed=i,scene=s)
        frames,bound=load_frames(c,s);d=OUT/'scenes'/s;d.mkdir(parents=True,exist_ok=True)
        freeze_json(d/'evidence.json',dict(complete=True,GT_read=False,inputs=bound,frames=[dict(frame=f['frame'],group=f['group'],
            visible_points=int(f['visible'].sum()),unique_masks=len(f['masks']),mask_observation_ids=f['observation_ids']) for f in frames]))
        seals[str(d/'evidence.json')]=sha(d/'evidence.json')
    freeze_json(OUT/'evidence_frozen.json',dict(complete=True,scenes=200,outputs=seals,GT_read=False))


def metrics(p,gt,gid):
    n=int(np.sum(gt==gid));inter=int(np.sum(gt[p]==gid));return dict(points=len(p),intersection=inter,
        purity=inter/max(1,len(p)),coverage=inter/n,iou=inter/(len(p)+n-inter))


def diagnose(c,guard):
    assert read(OUT/'evidence_frozen.json')['complete'];guard.labels=True
    results=[];counts=Counter();input_receipts={}
    for ix,s in enumerate(c['scenes']):
        stage('extent_diagnosis',completed=ix,scene=s)
        job=c['jobs'][s];assert sha(job['GT'])==c['GT_inputs'][job['GT']]
        gt=np.loadtxt(job['GT'],dtype=np.int64);assert len(gt)==job['binding']['point_count']
        gs,pts,_=prior.src.ctl.base.load_geometry(job['binding']);nt=source_pool(gs);keys=[g['geometry_key'] for g in nt];sizes=np.array([len(pts[k]) for k in keys])
        from tools.fi1_r3_core import gt_iou
        iou,gids=gt_iou([pts[k] for k in keys],gt,c['class_ids'])
        gi={int(gid):i for i,gid in enumerate(gids)};truthsizes=np.array([np.sum(gt==gid) for gid in gids])
        inter=np.rint(iou*(sizes[:,None]+truthsizes[None,:])/(1+iou)).astype(int)
        assert np.allclose(inter/(sizes[:,None]+truthsizes-inter),iou,atol=1e-14,rtol=0)
        # Validate independently reconstructed NT oracle against prior target census.
        census=read(job['census']);assert {str(int(g)):int(n) for g,n in zip(gids,truthsizes)}==census['targets']
        for j,k in enumerate(keys):assert np.allclose([census['ious'][k][str(int(g))] for g in gids],iou[j],atol=1e-14,rtol=0)
        frames,bound=load_frames(c,s);assert bound==read(OUT/'scenes'/s/'evidence.json')['inputs']
        frame_tables=[];observed_union=np.zeros(len(gt),bool);visible_union=np.zeros(len(gt),bool)
        for f in frames:
            visible_union|=f['visible']
            maskmat=point_matrix(f['masks'],len(gt));gtmat=sparse.csr_matrix((np.ones(len(gt),np.int64),(np.arange(len(gt)),np.array([gi.get(int(g),len(gids)) for g in gt]))),shape=(len(gt),len(gids)+1))
            mm=(maskmat@gtmat).toarray()[:,:len(gids)];ms=np.array([len(p) for p in f['masks']]);vc=np.bincount(np.array([gi.get(int(g),len(gids)) for g in gt[f['visible']]]),minlength=len(gids)+1)[:len(gids)]
            if f['masks']:observed_union[np.concatenate(f['masks'])]=True
            frame_tables.append(dict(f=f,intersection=mm,sizes=ms,visible_gt=vc))
        sourceidx={a:[i for i,g in enumerate(nt) if any(x['source']==a for x in original_aliases(g))] for a in ('native','track')}
        for j,gid in enumerate(gids):
            order=np.lexsort((np.array(keys),-iou[:,j]));best=int(order[0]);p=inter[best,j]/max(1,sizes[best]);rec=inter[best,j]/truthsizes[j]
            missing=iou[best,j]<=c['thresholds'][1];kind=extent_kind(p,rec,inter[best,j])
            representative={a:[dict(geometry_key=keys[i],iou=float(iou[i,j]),purity=float(inter[i,j]/max(1,sizes[i])),coverage=float(inter[i,j]/truthsizes[j]),points=int(sizes[i]))
                for i in sorted(ii,key=lambda i:(-iou[i,j],keys[i]))[:5]] for a,ii in sourceidx.items()}
            row=dict(scene=s,gt_id=int(gid),class_index=c['class_ids'].index(int(gid)//1000),target_points=int(truthsizes[j]),
                missing075=bool(missing),best_iou=float(iou[best,j]),best_purity=float(p),best_coverage=float(rec),kind=kind,
                best_geometry_key=keys[best],representatives=representative,
                visible_in30_fraction=float(np.sum(visible_union&(gt==gid))/truthsizes[j]),
                any_rawmask30_target_coverage=float(np.sum(observed_union&(gt==gid))/truthsizes[j]))
            if not missing:results.append(row);continue
            counts[kind]+=1;counts['gap075']+=1;counts['gap050']+=iou[best,j]<=c['thresholds'][0]
            # GT-picked unions are diagnostic opportunities, never actual actions.
            useful=[i for i in order if inter[i,j]>0][:12];bestpair=(float(iou[best,j]),None)
            for a in useful:
                for b in useful:
                    if b<=a:continue
                    pp=np.union1d(pts[keys[a]],pts[keys[b]]);q=metrics(pp,gt,gid)
                    if q['iou']>bestpair[0]:bestpair=(q['iou'],dict(left=keys[a],right=keys[b],**q))
            clean=[i for i in order if sizes[i]>0 and inter[i,j]/sizes[i]>.9 and inter[i,j]>0]
            pc=np.unique(np.concatenate([pts[keys[i]] for i in clean])) if clean else np.empty(0,int)
            row['NT_clean_fragment_union_oracle']=metrics(pc,gt,gid);row['NT_top12_pair_union_oracle']=bestpair[1]
            # A rawmask is good only for this GT in physically visible domain.
            # Diagnostic identity chosen usingGT; inference may not reuse this choice.
            bestview=None;good_masks=[];goodgroups=set();support=np.zeros(len(gt),np.int16);visible=np.zeros(len(gt),np.int16)
            for ft in frame_tables:
                f=ft['f'];v=int(ft['visible_gt'][j]);visible+=f['visible']
                if v<3:continue
                jj=ft['intersection'][:,j];ms=ft['sizes'];pur=np.divide(jj,ms,out=np.zeros(len(jj)),where=ms>0);cover=jj/v
                q=np.divide(jj,ms+v-jj,out=np.zeros(len(jj),float),where=(ms+v-jj)>0)
                if len(q):
                    m=int(np.argmax(q));w=dict(frame=f['frame'],group=f['group'],observation_id=f['observation_ids'][m],
                        visible_target_points=v,visible_target_fraction=v/truthsizes[j],purity=float(pur[m]),visible_coverage=float(cover[m]),visible_iou=float(q[m]))
                    if bestview is None or w['visible_iou']>bestview['visible_iou']:bestview=w
                accepted=np.flatnonzero((pur>.75)&(cover>.5))
                if len(accepted):
                    goodgroups.add(f['group']);mask=np.zeros(len(gt),bool)
                    for m in accepted:mask[f['masks'][m]]=True;good_masks.append(f['masks'][m])
                    support+=mask
            good=np.unique(np.concatenate(good_masks)) if good_masks else np.empty(0,int)
            row['raw_observation']=dict(best_view=bestview,good_independent_groups=len(goodgroups),good_masks_union=metrics(good,gt,gid),
                missing_from_best_target_supported=int(np.sum((gt[good]==gid)&~np.isin(good,pts[keys[best]]))))
            # Identity oracle demonstrates whether local support CAN remove contamination.
            pp=pts[keys[best]];keep=(visible[pp]<2)|(2*support[pp]>=visible[pp]);crop=pp[keep]
            row['GT_picked_observation_crop_oracle']=metrics(crop,gt,gid)
            targetmask=gt==gid;bestmask=np.zeros(len(gt),bool);bestmask[pp]=True
            row['best_missing_region']=dict(target_points=int(np.sum(targetmask&~bestmask)),
                present_in_other_NT=int(np.sum((np.bincount(np.concatenate([pts[k] for k in keys]),minlength=len(gt))>0)&targetmask&~bestmask)),
                present_in_rawmask30=int(np.sum(observed_union&targetmask&~bestmask)),
                physically_visible30=int(np.sum(visible_union&targetmask&~bestmask)))
            other_ids,other_count=np.unique(gt[pp][gt[pp]!=gid],return_counts=True)
            row['best_pollution']=dict(total=int(len(pp)-np.sum(gt[pp]==gid)),other_valid=int(sum(n for g,n in zip(other_ids,other_count) if int(g) in gi)),
                ignored_or_unknown=int(sum(n for g,n in zip(other_ids,other_count) if int(g) not in gi)),
                other_instances=[dict(gt_id=int(g),points=int(n)) for g,n in zip(other_ids,other_count) if int(g) in gi])
            counts['NT_pair_union_repairs075']+=bestpair[0]>c['thresholds'][1]
            counts['clean_fragment_union_repairs075']+=row['NT_clean_fragment_union_oracle']['iou']>c['thresholds'][1]
            counts['GT_observation_crop_repairs075']+=row['GT_picked_observation_crop_oracle']['iou']>c['thresholds'][1]
            counts['observation_union_repairs075']+=row['raw_observation']['good_masks_union']['iou']>c['thresholds'][1]
            counts['good_observation_2groups']+=len(goodgroups)>=2
            results.append(row)
    assert counts['gap075']==877 and counts['gap050']==323
    (OUT/'targets.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in results))
    groups={}
    for name,scenes in c['groups'].items():
        cc=Counter();rows=[r for r in results if r['scene'] in scenes and r['missing075']]
        for r in rows:
            cc[r['kind']]+=1
            for n,field in (('pair_union','NT_top12_pair_union_oracle'),('clean_union','NT_clean_fragment_union_oracle'),('crop','GT_picked_observation_crop_oracle')):
                q=r[field];cc[n+'_repairs075']+=q is not None and q['iou']>c['thresholds'][1]
            cc['good_observation_2groups']+=r['raw_observation']['good_independent_groups']>=2
        groups[name]=dict(targets=len(rows),counts=dict(cc))
    freeze_json(OUT/'summary.json',dict(complete=True,counts=dict(counts),groups=groups,raw_GT_read=True,diagnostic_oracles_only=True,
        incomplete_observation_scope='30 sampled cached frames; cannot declare frontend failure in all frames'))
    freeze_json(OUT/'diagnosis_frozen.json',dict(complete=True,outputs={str(OUT/n):sha(OUT/n) for n in ('targets.jsonl','summary.json')},GT_inputs=c['GT_inputs']))
    stage('diagnosis_completed',targets=len(results),missing075=877)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));args=p.parse_args()
    with threadpool_limits(4):
        if args.mode=='prepare':prepare()
        else:
            c=read(OUT/'contract.json');g=Guard(c);sys.addaudithook(g.hook);evidence(c,g);diagnose(c,g)
