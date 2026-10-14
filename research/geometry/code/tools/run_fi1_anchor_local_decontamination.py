#!/usr/bin/env python3
"""Paired old-mask / anchor-reprompt experiment, geometry first, no training."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.ndimage import binary_dilation
from tools import diagnose_fi1_nt_geometry_gaps as prior
from tools import fi1_anchor_local_decontamination as core
from tools.fi1_core_ablation import source_pool
from tools.fi1_r0_r1_protocol import original_aliases
from tools.fi1_r3_core import gt_iou
from tools.fi1_frontend_protocol import freeze_json, write_json, sha256_file as sha, digest
from tools.fi1_complementary_fragment_evidence import project_points

OUT = ROOT/'.local/fi1_anchor_local_decontamination_development200_20261012_v1'
STABILITY = ROOT/'.local/fi1_sl_replay_stability_development200_20261010_v1'
SAM_SOURCE = ROOT/'_external/segment-anything/segment-anything-main'
SAM_WEIGHT = ROOT/'pretrained/checkpoints/sam_vit_b_01ec64.pth'
ARMS = ('old', 'reprompt')
SEED = 'anchor-local-decontamination-20261012-v1'
read = prior.read


def stage(name, **kw):
    write_json(OUT/'execution.json', dict(stage=name, updated=time.time(), **kw))
    print(name, kw, flush=True)


def verify_files(files):
    for p, expected in files.items():
        if sha(p) != expected:
            raise ValueError('input identity changed: '+p)


class Guard:
    def __init__(self, c, labels=False):
        self.c, self.labels = c, labels

    def hook(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        p = str(Path(args[0]).resolve())
        if Path(p).suffix in ('.py', '.pyc'):
            return  # Source filenames are not evaluation data.
        if any(n in p for n in ('fi1_sl_single200_dev55', '/val/', '/val312/', '/reserve100/')):
            raise PermissionError('outside fixed development geometry experiment')
        if any(q.startswith('ground_truth') for q in Path(p).parts):
            if not self.labels or p not in self.c['GT_inputs'] or not (OUT/'geometry_frozen.json').exists():
                raise PermissionError('GT requires full geometry seal and selected-scene whitelist')
        if not self.labels and any(p.endswith(n) for n in
                ('target_census.json', 'overlap.npz', 'targets.jsonl', 'geometry_quality.json')):
            raise PermissionError('GT-derived diagnosis excluded from planning/inference')
        import re
        if any(s not in self.c['scenes'] for s in re.findall(r'scene\d{4}_\d{2}', p)):
            raise PermissionError('scene outside frozen small-scale selection')


def prepare():
    assert not OUT.exists()
    assert shutil.disk_usage(ROOT).free >= 50*1024**3
    p = read(prior.OUT/'contract.json')
    b = read(STABILITY/'contract.json')
    inventory = {}
    # Availability selection inspects inference paths only, never GT existence/content.
    for s in p['scenes']:
        j = p['jobs'][s]
        paths = [j['binding'][n] for n in ('geometry', 'points')]
        paths += [j['paths'][n] for n in ('processed', 'manifest', 'observations')]+[j['views']]
        available = all(Path(q).is_file() for q in paths)
        if available:
            m = read(j['paths']['manifest']); v = read(j['views'])
            fm = {int(f['frame_id']):f for f in m['frames']}
            available = all(all(Path(fm[f][k]['path']).is_file() for k in ('color','depth','pose'))
                            for f in v['raw_frames'])
        inventory[s] = dict(fold=p['folds'][s],
                            group='historical90' if s in p['groups']['historical90'] else 'new110',
                            available=available)
    scenes = []
    for fold in range(5):
        for group in ('historical90', 'new110'):
            pool = [s for s,v in inventory.items() if v['available'] and v['fold']==fold and v['group']==group]
            ranked = sorted(pool, key=lambda s:(digest([SEED, s]), s))
            assert ranked, (fold, group)
            scenes.append(ranked[0])
    assert len(scenes)==len({s[:9] for s in scenes})==10
    jobs, inputs, gt = {}, {}, {}
    for s in scenes:
        j = p['jobs'][s]
        jobs[s] = dict(binding=j['binding'], paths={n:j['paths'][n] for n in
                       ('processed','manifest','observations')}, views=j['views'], GT=j['GT'])
        for q in [j['binding'][n] for n in ('geometry','points','evidence')]+[
                j['paths'][n] for n in ('processed','manifest','observations')]+[j['views']]:
            inputs[q] = p['inputs'][q]
        gt[j['GT']] = p['GT_inputs'][j['GT']]
    models = {b['B_model']:b['B_model_sha256']}
    models.update({q:h for q,h in b['original_models'].items()
                   if Path(q).name in ('model.pt','normalization.npz')})
    source_contracts = {str(prior.OUT/'contract.json'):sha(prior.OUT/'contract.json'),
                        str(STABILITY/'contract.json'):sha(STABILITY/'contract.json')}
    code_paths = [Path(__file__).resolve(), ROOT/'tools/fi1_anchor_local_decontamination.py',
                  ROOT/'tools/run_fi1_anchor_local_decontamination.sh',
                  ROOT/'tools/verify_fi1_anchor_local_decontamination.py',
                  ROOT/'tests/test_fi1_anchor_local_decontamination.py',
                  ROOT/'docs/FI1_ANCHOR_LOCAL_DECONTAMINATION_PROTOCOL_20261012.md',
                  ROOT/'tools/fi1_complementary_fragment_evidence.py',
                  ROOT/'tools/diagnose_fi1_nt_geometry_gaps.py', ROOT/'tools/fi1_sl_inference.py',
                  ROOT/'tools/run_fi1_frontend_comparison.py', ROOT/'utils/__init__.py',
                  ROOT/'tools/fi1_mask_projection.py']
    code_paths += sorted((SAM_SOURCE/'segment_anything').rglob('*.py'))
    code_paths += [Path(module.__file__).resolve() for name,module in list(sys.modules.items())
                   if name.startswith('tools.') and getattr(module,'__file__',None)
                   and Path(module.__file__).resolve().is_relative_to(ROOT)]
    code_paths = sorted(set(code_paths))
    code = {str(q):sha(q) for q in code_paths}
    # The original frontend contract explicitly binds the same SAM checkpoint.
    front_contracts = {}
    for s in scenes:
        fc = Path(jobs[s]['paths']['manifest']).parents[1]/'contract.json'
        d = read(fc)
        assert d['assets'][str(SAM_WEIGHT)] == sha(SAM_WEIGHT)
        front_contracts[str(fc)] = sha(fc)
    verify_files(inputs); verify_files(models)
    OUT.mkdir()
    c = dict(version=OUT.name, seed=SEED, scenes=scenes,
             folds={s:p['folds'][s] for s in scenes},
             groups={n:[s for s in scenes if s in ss] for n,ss in p['groups'].items()},
             jobs=jobs, inputs=inputs, GT_inputs=gt, class_ids=p['class_ids'],
             source_contracts=source_contracts, frontend_contracts=front_contracts,
             code=code, frozen_SL_models=models, sam_checkpoint=str(SAM_WEIGHT),
             sam_sha256=sha(SAM_WEIGHT), sam_source=str(SAM_SOURCE), sam_model_type='vit_b',
             parameters=core.PARAMETERS, geometry_gate=core.GATE, thresholds=list(core.THRESHOLDS),
             treatment='Same original GD box, SAM ViT-B, full RGB and single-mask decoding; add three positive points from one frozen 3D superpoint core. No new frames, tracking, model, negative prompts, or crop-operator change.',
             planning='One hash-ranked available scene per original fold x old90/new110; at most4 native and4 track parents hash-ranked without GT/class/score, shared pointsets chosen once. No replacement for failed anchors.',
             editable_domain='Parent points within .15m of an original scene point outside parent, determined before either evidence arm. Core protected. Two independent .02m sensor-visible exclusions, two RGB-pixel margin, any positive view veto. >20% deletion abstains.',
             identity='One interior raw superpoint,>=20 points and>=95% contained in parent, existing-mask support>=95% in>=2 distinct pose groups; this is a geometric anchor, not proven object identity. Invalid core/mask/quality views abstain.',
             semantic_gate='Only after preregistered geometry gate passes: fresh own YOLO/Alpha, frozen original fold L/B26/S800, extended full context; controls original SL, extended-context NT only, and extended-pool selection. All prediction seals precede dual AP and original coverage/424 protection checks.',
             scope='Exposed development subset, no training or checkpoint/threshold selection. GT is post-seal diagnosis only. No production replacement.',
             new_training=False, SL_modified=False, production_modified=False)
    freeze_json(OUT/'contract.json', c)
    freeze_json(OUT/'selection_inventory.json', inventory)
    for q in code_paths:
        target = OUT/'code_snapshot'/q.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(q, target)
    stage('prepared', scenes=scenes, max_parents=80)


def load_nt(job):
    geometry = source_pool(read(job['binding']['geometry']))
    with np.load(job['binding']['points'], allow_pickle=False) as z:
        points = {g['geometry_key']:z[g['array_key']].copy() for g in geometry}
    for g in geometry:
        p = points[g['geometry_key']]
        assert len(p)==g['point_count'] and np.array_equal(p, np.unique(p))
        assert hashlib.sha256(p.astype('<i8').tobytes()).hexdigest()==g['points_sha256']
    return geometry, points


def anchor_plan(parent, raw, frames, observations, params):
    sp = raw[:,9].astype(np.int64)
    ids, counts = np.unique(sp[parent], return_counts=True)
    totals = Counter(sp.tolist())
    eligible = [(sid,n) for sid,n in zip(ids,counts)
                if n>=params['min_core_points'] and n/totals[sid]>=params['core_parent_fraction']]
    center = raw[parent,:3].mean(axis=0)
    candidates = []
    for sid,n in eligible:
        core_points = parent[sp[parent]==sid]
        distance = float(np.linalg.norm(raw[core_points,:3].mean(axis=0)-center))
        candidates.append((distance,-int(n),int(sid),core_points))
    for _,_,sid,points in sorted(candidates, key=lambda q:q[:3])[:params['max_core_trials']]:
        choices=[]
        for f in frames:
            visible_core=points[f['visible'][points]]; visible_parent=parent[f['visible'][parent]]
            if len(visible_core)<params['min_core_visible_points'] or len(visible_core)/len(points)<params['min_core_visible_fraction']:
                continue
            if len(visible_parent)<params['min_parent_visible_points'] or len(visible_parent)/len(parent)<params['min_parent_visible_fraction']:
                continue
            masks=[]
            for mask,oid in zip(f['masks'],f['observation_ids']):
                intersection=len(np.intersect1d(visible_core, mask, assume_unique=True))
                if intersection/len(visible_core)<params['core_mask_coverage']:
                    continue
                hit=len(np.intersect1d(visible_parent,mask,assume_unique=True))
                iou=hit/(len(visible_parent)+len(mask)-hit)
                masks.append((-iou,oid,hit))
            if not masks:
                continue
            _, oid, hit = min(masks)
            old = observations[(f['frame'],oid)]
            if 'gd_box_xyxy' not in old.get('provenance',{}):
                continue
            choices.append(dict(frame=f['frame'], group=f['group'], observation_id=oid,
                                core_visible_fraction=len(visible_core)/len(points),
                                parent_visible_fraction=len(visible_parent)/len(parent),
                                old_disagreement_fraction=1-hit/len(visible_parent),
                                old_points_path=old['point_indices_path'],
                                old_mask_digest=digest(old['mask_rle']),
                                original_box=old['provenance']['gd_box_xyxy']))
        chosen=[]; groups=set()
        for q in sorted(choices,key=lambda v:(-min(v['core_visible_fraction'],v['parent_visible_fraction']),
                                              -v['old_disagreement_fraction'],v['frame'])):
            if q['group'] not in groups:
                chosen.append(q);groups.add(q['group'])
            if len(chosen)==params['views']:
                break
        if len(chosen)>=params['min_pose_groups']:
            return points, dict(accepted=True, core_superpoint=sid, core_point_count=len(points), views=chosen)
    return np.empty(0,np.int64), dict(accepted=False, reason='no_supported_interior_core',
                                    interior_superpoints=len(eligible), tried=min(len(eligible),params['max_core_trials']),views=[])


def plan(c):
    sys.addaudithook(Guard(c).hook)
    verify_files(c['code']);verify_files(c['inputs']);verify_files(c['frozen_SL_models'])
    outputs={}; counts=Counter()
    for s in c['scenes']:
        d=OUT/'plan'/s
        if (d/'receipt.json').exists():
            r=read(d/'receipt.json');verify_files(r['inputs']);verify_files(r['outputs'])
            outputs.update(r['outputs']);counts.update(r['counts']);continue
        stage('GT_free_planning', scene=s)
        j=c['jobs'][s];gs,pts=load_nt(j)
        raw=np.load(j['paths']['processed'],mmap_mode='r',allow_pickle=False)
        assert len(raw)==j['binding']['point_count'] and raw.shape[1]>=10
        frames,bound=prior.load_frames(c,s)
        obs={(int(o['frame_id']),int(o['observation_id'])):o
             for o in prior.prior.src.ctl.base.rows(j['paths']['observations'])}
        chosen=[]; used=set()
        for source in ('native','track'):
            eligible=[g for g in gs if core.PARAMETERS['min_parent_points']<=g['point_count']<=core.PARAMETERS['max_parent_points']
                      and any(a['source']==source for a in original_aliases(g)) and g['geometry_key'] not in used]
            for g in sorted(eligible,key=lambda g:(digest([SEED,s,source,g['geometry_key']]),g['geometry_key']))[:core.PARAMETERS['candidates_per_source']]:
                used.add(g['geometry_key']);chosen.append((source,g))
        arrays={};rows=[];stat=Counter(parents=len(chosen))
        for i,(source,g) in enumerate(chosen):
            parent=pts[g['geometry_key']];cid=f'c{i:03d}'
            anchor,record=anchor_plan(parent,raw,frames,obs,c['parameters'])
            local=core.local_domain(parent,np.asarray(raw[:,:3]),c['parameters']['local_boundary_radius'])
            arrays[cid+'_parent']=parent;arrays[cid+'_core']=anchor;arrays[cid+'_local']=local
            record.update(candidate_id=cid,parent=g['geometry_key'],source=source,
                          source_aliases=sorted({a['source'] for a in original_aliases(g)}),
                          parent_points=len(parent),local_points=int(local.sum()))
            rows.append(record);stat[source]+=1;stat['anchored' if record['accepted'] else 'anchor_abstain']+=1
        m=read(j['paths']['manifest']);fm={int(f['frame_id']):f for f in m['frames']}
        for q in [v for r in rows for v in r['views']]:
            f=fm[q['frame']]
            for name in ('color','depth','pose'):
                path=f[name]['path'];assert sha(path)==f[name]['sha256'];bound[path]=f[name]['sha256']
        for item in m['files'].values():
            assert sha(item['path'])==item['sha256'];bound[item['path']]=item['sha256']
        d.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(d/'arrays.npz',**arrays);freeze_json(d/'candidates.json',rows)
        out={str(d/n):sha(d/n) for n in ('arrays.npz','candidates.json')}
        freeze_json(d/'receipt.json',dict(complete=True,inputs=bound,outputs=out,counts=dict(stat),GT_read=False))
        outputs.update(out);counts.update(stat)
    freeze_json(OUT/'plan_frozen.json',dict(complete=True,outputs=outputs,counts=dict(counts),GT_read=False))
    stage('plan_sealed',counts=dict(counts))


def generate(c):
    assert read(OUT/'plan_frozen.json')['complete']
    sys.addaudithook(Guard(c).hook)
    verify_files(c['code']);verify_files(c['inputs']);verify_files(c['frozen_SL_models'])
    assert sha(c['sam_checkpoint'])==c['sam_sha256']
    import torch
    from PIL import Image
    from tools.run_fi1_frontend_comparison import project_frame
    from tools.fi1_mask_projection import decode_binary_mask_rle, mask_to_visible_points
    assert torch.cuda.is_available(), 'CUDA required; no CPU SAM fallback'
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    sys.path.insert(0,c['sam_source'])
    from segment_anything import SamPredictor,sam_model_registry
    from segment_anything.utils.amg import calculate_stability_score
    model=sam_model_registry['vit_b'](checkpoint=c['sam_checkpoint']).eval().cuda()
    predictor=SamPredictor(model)
    freeze_json(OUT/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
                                      gpu=torch.cuda.get_device_name(0),model='SAM ViT-B',new_training=False))
    outputs={};allcounts=Counter()
    for s in c['scenes']:
        dest=OUT/'geometry'/s
        if (dest/'receipt.json').exists():
            r=read(dest/'receipt.json');verify_files(r['outputs']);outputs.update(r['outputs']);allcounts.update(r['counts']);continue
        stage('paired_SAM_inference',scene=s)
        pd=OUT/'plan'/s;verify_files(read(pd/'receipt.json')['inputs']);verify_files(read(pd/'receipt.json')['outputs'])
        plans=read(pd/'candidates.json');j=c['jobs'][s]
        raw=np.load(j['paths']['processed'],mmap_mode='r',allow_pickle=False)
        manifest=read(j['paths']['manifest']);fm={int(f['frame_id']):f for f in manifest['frames']}
        obs={(int(o['frame_id']),int(o['observation_id'])):o
             for o in prior.prior.src.ctl.base.rows(j['paths']['observations'])}
        views={r['candidate_id']:[] for r in plans};payload={};details=[]
        dest.mkdir(parents=True,exist_ok=False);ed=dest/'observations';ed.mkdir()
        with np.load(pd/'arrays.npz',allow_pickle=False) as arrays:
            requests=defaultdict(list)
            for r in plans:
                for vi,q in enumerate(r['views']):requests[q['frame']].append((r,vi,q))
            for frame,rr in sorted(requests.items()):
                entry=fm[frame];image=np.asarray(Image.open(entry['color']['path']).convert('RGB'))
                sensor=np.asarray(Image.open(entry['depth']['path']),dtype=np.float64)/1000.
                projected,visible,scaling=project_frame(manifest,entry)
                projected=projected[0];visible=visible[0]
                # The exact unchanged GPU projection is used for both old/new masks.
                pose=np.loadtxt(entry['pose']['path'])
                camera=np.column_stack((raw[:,:3],np.ones(len(raw))))@np.linalg.inv(pose).T
                pix=projected.astype(np.int64);in_domain=(pix[:,0]>=0)&(pix[:,0]<sensor.shape[1])&(pix[:,1]>=0)&(pix[:,1]<sensor.shape[0])
                depth=np.zeros(len(raw));ix=np.flatnonzero(in_domain);depth[ix]=sensor[pix[ix,1],pix[ix,0]]
                reliable=visible&in_domain&(depth>0)&(camera[:,2]>0)&(np.abs(depth-camera[:,2])<=c['parameters']['measured_depth_tolerance'])
                rgbpix=np.rint(projected.astype(np.float32)/np.array([scaling[1],scaling[0]],dtype=np.float32)).astype(np.int64)
                in_rgb=(rgbpix[:,0]>=0)&(rgbpix[:,0]<image.shape[1])&(rgbpix[:,1]>=0)&(rgbpix[:,1]<image.shape[0])
                reliable &= in_rgb
                np.savez_compressed(ed/f'frame{frame:06d}.npz',pixels=rgbpix,reliable=reliable)
                predictor.set_image(np.ascontiguousarray(image))
                for r,vi,q in rr:
                    cid=r['candidate_id'];parent=arrays[cid+'_parent'];anchor=arrays[cid+'_core']
                    old=obs[(frame,q['observation_id'])];assert digest(old['mask_rle'])==q['old_mask_digest']
                    oldmask=decode_binary_mask_rle(old['mask_rle']);assert oldmask.shape==image.shape[:2]
                    with np.load(old['point_indices_path'],allow_pickle=False) as z:oldpoints=z['point_indices']
                    replaypoints=mask_to_visible_points(oldmask,0,projected[None],visible[None],scaling)
                    assert np.array_equal(replaypoints,oldpoints), 'cached old mask projection differs'
                    visible_anchor=anchor[reliable[anchor]]
                    prompt_ids=core.positive_prompt_ids(visible_anchor,np.asarray(raw[:,:3]))
                    box=np.asarray(q['original_box'],dtype=np.float32)
                    with torch.inference_mode():
                        baseline_logits,bq,_=predictor.predict(box=box,multimask_output=False,return_logits=True)
                        oldreplay=baseline_logits[0]>model.mask_threshold
                        replay_iou=float((oldreplay&oldmask).sum()/max(1,(oldreplay|oldmask).sum()))
                        assert replay_iou>=c['parameters']['old_replay_iou_min'], ('old SAM replay mismatch',s,cid,frame,replay_iou)
                        if len(prompt_ids)==c['parameters']['prompt_points']:
                            logits,qualities,_=predictor.predict(point_coords=rgbpix[prompt_ids].astype(np.float32),
                                point_labels=np.ones(len(prompt_ids),np.int64),box=box,multimask_output=False,return_logits=True)
                            newmask=logits[0]>model.mask_threshold;quality=float(qualities[0])
                            stability=float(calculate_stability_score(torch.from_numpy(logits),model.mask_threshold,1.)[0])
                        else:
                            newmask=oldmask.copy();quality=stability=0.
                    base_visible=anchor[visible[anchor]]
                    core_visibility_ok=len(visible_anchor)>=c['parameters']['min_core_visible_points'] and len(visible_anchor)/len(anchor)>=c['parameters']['min_core_visible_fraction']
                    record=dict(candidate_id=cid,view_index=vi,frame=frame,group=q['group'],prompt_ids=prompt_ids.tolist(),
                                prompt_xy=rgbpix[prompt_ids].tolist(),old_SAM_replay_iou=replay_iou,
                                old_quality=old['predicted_iou'],old_stability=old['stability_score'],
                                new_quality=quality,new_stability=stability,core_reliable_points=len(visible_anchor),
                                changed_pixel_fraction=float(np.mean(oldmask!=newmask)),
                                mask_iou=float((oldmask&newmask).sum()/max(1,(oldmask|newmask).sum())))
                    vp={}
                    for arm,mask,qscore,stable in (('old',oldmask,old['predicted_iou'],old['stability_score']),
                                                  ('reprompt',newmask,quality,stability)):
                        inside=np.zeros(len(raw),bool);validids=np.flatnonzero(in_rgb)
                        inside[validids]=mask[rgbpix[validids,1],rgbpix[validids,0]]
                        margin=binary_dilation(mask,iterations=c['parameters']['pixel_exclusion_margin'])
                        excluded=np.zeros(len(raw),bool)
                        excluded[validids]=~margin[rgbpix[validids,1],rgbpix[validids,0]]
                        support=float(inside[visible_anchor].mean()) if len(visible_anchor) else 0.
                        valid=bool(core_visibility_ok and support>=c['parameters']['core_mask_coverage'] and
                                   qscore>c['parameters']['sam_quality_min'] and stable>=c['parameters']['sam_stability_min'])
                        vp[arm]=dict(valid=valid,group=q['group'],reliable=reliable[parent],
                                     inside=inside[parent],excluded=excluded[parent])
                        record[arm+'_valid']=valid;record[arm+'_core_support']=support
                        record[arm+'_mask_points']=int((inside&reliable).sum())
                        payload[f'{cid}_v{vi}_{arm}_reliable']=reliable[parent]
                        payload[f'{cid}_v{vi}_{arm}_inside']=inside[parent]
                        payload[f'{cid}_v{vi}_{arm}_excluded']=excluded[parent]
                        # Physical visible regions for post-seal 2D-boundary / identity diagnosis.
                        payload[f'{cid}_v{vi}_{arm}_region']=np.flatnonzero(reliable&inside)
                    np.savez_compressed(ed/f'{cid}_v{vi}.npz',old=oldmask,reprompt=newmask)
                    views[cid].append(vp);details.append(record)
                predictor.reset_image()
            geometry={};decision=[];counts=Counter(parents=len(plans),views=len(details))
            for r in plans:
                cid=r['candidate_id'];parent=arrays[cid+'_parent'];anchor=arrays[cid+'_core'];local=arrays[cid+'_local']
                item=dict(candidate_id=cid,parent=r['parent'],source=r['source'],arms={})
                for arm in ARMS:
                    pp,rec=core.crop(parent,anchor,local,[v[arm] for v in views[cid]],c['parameters'])
                    token=hashlib.sha256(pp.astype('<i8').tobytes()).hexdigest()
                    geometry[cid+'_'+arm]=pp;item['arms'][arm]={**rec, 'points_sha256':token,
                                                             'geometry_key':s+':geometry:'+token}
                    counts[arm+'_'+rec['reason']]+=1
                decision.append(item)
            np.savez_compressed(dest/'points.npz',**geometry)
            np.savez_compressed(dest/'evidence.npz',**payload)
        freeze_json(dest/'decisions.json',decision);freeze_json(dest/'views.json',details)
        files={str(q):sha(q) for q in dest.rglob('*') if q.is_file()}
        freeze_json(dest/'receipt.json',dict(complete=True,GT_read=False,counts=dict(counts),outputs=files))
        outputs.update(files);allcounts.update(counts)
    verify_files(c['inputs']);verify_files(c['frozen_SL_models'])
    freeze_json(OUT/'geometry_frozen.json',dict(complete=True,GT_read=False,outputs=outputs,counts=dict(allcounts)))
    stage('geometry_sealed',counts=dict(allcounts))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','plan','generate','evaluate'));a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):
        if a.mode=='prepare':prepare()
        elif a.mode=='evaluate':
            from tools.verify_fi1_anchor_local_decontamination import evaluate
            evaluate(read(OUT/'contract.json'))
        else:globals()[a.mode](read(OUT/'contract.json'))


if __name__=='__main__':main()
