#!/usr/bin/env python3
"""Nested physical-scene learning curve with frozen C and unchanged evaluation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import audit_fi1_t_source_chain as a
import run_fi1_cross_track_dual as candidate
import run_fi1_pair_relation as visual
import run_fi1_relation_controls as baseline
import fi1_pair_relation_core as relation
import fi1_relation_controls_core as metrics

ROOT, HERE = a.ROOT, Path(__file__).resolve().parent
SOURCE, BASE, GEOMETRY = baseline.OUT, baseline.BASE, baseline.GEOMETRY
AUDIT = a.OUT
OUT = a.EXP/'fi1_relation_training_scale_exposed52_20261015_v1'
DEV = ROOT/'.local/fi1_union_members_development200_20261007_v1/contract.json'
PROTECTED = ROOT/'.local/fi1_mask3d_feature_ab_internal_20261013_v1/contract.json'
ARMS = ('half', 'original', 'double')


def physical(s):
    return s.rsplit('_', 1)[0]


def ordered(scenes, tag):
    return sorted(scenes, key=lambda s: hashlib.sha256(('relation_scale_20261015/'+tag+'/'+physical(s)).encode()).hexdigest())


def guard(allowed_labels=(), gt=False):
    allowed = {str(Path(p).resolve()) for p in allowed_labels}
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if writing and not p.is_relative_to(OUT):
            raise PermissionError('Writes restricted to new scale experiment: '+str(p))
        if not gt and any(c in str(p) for c in ('ground_truth', '/labels/', '/evaluation/',
                'opportunity_diagnostics', '_evidence.csv', '/results.json', '/diagnosis/')) and str(p) not in allowed:
            raise PermissionError('GT or outcome forbidden in this stage: '+str(p))
    sys.addaudithook(hook)


def verify(mapping):
    baseline.prior.verify(mapping)


def contract():
    c = a.read(OUT/'contract.json')
    verify(c['code'])
    verify(c['source_files'])
    return c


def prepare():
    guard()
    a.reserve()
    assert not OUT.exists()
    bc, dc, pc, sc = map(a.read, (SOURCE/'contract.json', DEV, PROTECTED, AUDIT/'contract.json'))
    protected = {physical(s) for s in pc['splits']['holdout']+pc['groups']['fold0']+dc['reserve100']}
    excluded = protected | {physical(s) for s in bc['scenes']}
    frontend = Path(sc['frontend'])
    inventories = []
    for s in sorted(dc['scenes']):
        if physical(s) in excluded:
            inventories.append(dict(scene=s, eligible=False, reason='evaluation_or_protected_physical_site'))
            continue
        b = dc['bindings'][s]
        required = [frontend/'manifests'/f'{s}.json', Path(b['geometry']), Path(b['points']),
                    Path(b['gt_path']), frontend/'arms/D/observations'/s/'automatic_observations.jsonl']
        required += [frontend/'arms/D/stage_receipts'/f'{s}_{stage}.json' for stage in ('merged', 'tracks', 'hierarchy')]
        missing = [str(p) for p in required if not p.is_file()]
        if not missing:
            mf = a.read(required[0])
            required2 = [Path(v['path']) for v in mf['files'].values()]
            for f in mf['frames']:
                if str(f['frame_id']) in set(map(str, mf['arms']['D'])):
                    required2 += [Path(f[k]['path']) for k in ('color', 'depth', 'pose')]
                    cache = frontend/'frame_cache/grounded_sam'/s/('frame%06d' % int(f['frame_id']))
                    required2 += [cache/name for name in ('receipt.json', 'observations.json', 'visibility.npy')]
            missing += [str(p) for p in required2 if not p.is_file()]
        inventories.append(dict(scene=s, eligible=not missing, reason='available_same_frontend' if not missing else 'missing_assets', missing=missing))
    available = [q['scene'] for q in inventories if q['eligible']]
    assert len({physical(s) for s in available}) == len(available)
    nmax = max(sum(bc['folds'][s] != f for s in bc['scenes']) for f in range(5))
    extra = ordered(available, 'extra')[:nmax]
    assert len(extra) == nmax
    splits = {}
    for f in range(5):
        full = [s for s in bc['scenes'] if bc['folds'][s] != f]
        small = ordered(full, 'half')[:len(full)//2]
        large = full+extra[:len(full)]
        heldout = [s for s in bc['scenes'] if bc['folds'][s] == f]
        assert set(small) < set(full) < set(large)
        assert {physical(s) for s in large}.isdisjoint({physical(s) for s in heldout})
        splits[str(f)] = dict(half=small, original=full, double=large, heldout=heldout)
    jobs = {}
    for s in extra:
        b = dc['bindings'][s]
        mf = a.read(frontend/'manifests'/f'{s}.json')
        jobs[s] = dict(job=dict(GT=b['gt_path'], binding={k:b[k] for k in ('geometry', 'points', 'point_count')},
            paths=dict(manifest=str(frontend/'manifests'/f'{s}.json'), processed=mf['files']['processed']['path'])), GT_sha256=b['gt_sha256'])
    delivery = a.read(SOURCE/'delivery_verified.json')
    refs = {str(p):a.sha(p) for p in (DEV, PROTECTED, AUDIT/'contract.json', SOURCE/'contract.json', SOURCE/'delivery_verified.json',
            BASE/'features.npz', BASE/'feature_rows.json', BASE/'labels_frozen.json', GEOMETRY/'predictions_frozen.json')}
    refs.update(bc['source_files'])
    for p, h in delivery['outputs'].items():
        path = Path(p)
        if path.name in ('C_scores.npz', 'C_decisions.json', 'evaluation_frozen.json') or '/predictions/C/' in p or '/models/' in p and path.name in ('C.joblib', 'training.npz', 'receipt.json'):
            refs[p] = h
    code_paths = [HERE/n for n in ('run_fi1_relation_training_scale.py', 'evaluate_fi1_relation_training_scale.py', 'verify_fi1_relation_training_scale.py',
        'test_fi1_relation_training_scale.py', 'run_fi1_cross_track_dual.py', 'fi1_cross_track_dual_core.py',
        'run_fi1_pair_relation.py', 'fi1_pair_relation_core.py', 'run_fi1_relation_controls.py',
        'fi1_relation_controls_core.py', 'audit_fi1_t_source_chain.py', 'run_fi1_nt_region_identity.py', 'evaluate_fi1_nt_region_identity.py')]
    verify(refs)
    a.seal(OUT/'inventory.json', dict(scenes=inventories, available=len(available), selected=extra,
        selection='SHA256 physical scene name only, before new labels/features/model fits; no pair counts, target IDs, opportunities or outcomes.',
        protected_physical=sorted(protected), all597_frontend_manifests=len(list((frontend/'manifests').glob('*.json')))))
    a.seal(OUT/'contract.json', dict(version=OUT.name, time_unix=time.time(), frontend=str(frontend),
        source_files=refs, code={str(p):a.sha(p) for p in code_paths}, scenes=bc['scenes'], folds=bc['folds'],
        extra_scenes=extra, jobs=jobs, splits=splits, model=bc['models']['C'], label_files=bc['label_files'],
        feature_dim=154, evaluation_pairs=4865, weights='Exact C recipe: equal total weight per nonempty clear-fitting scene, natural within-scene class mass; normalize sum weights to fitting row count. No physical-relation balancing.',
        construction='Same frozen whole unions, endpoints and original NT on evaluation52; new training scenes use exact original proposal, 23 geometry+131 RGBD features and full-endpoint labels. No new frames or visual models.',
        isolation='All added physical sites outside every evaluation52 site, original holdout24, outer fold0 forty, reserve100. Previously development200 scenes, not whole-project unseen; source/B26 may have seen them.',
        nested='Each original fold heldout identical. half=floor(original_scene_count/2) SHA256 prefix; original=C exact reuse; double=original plus equally many new scenes from one global independent SHA256 prefix. Zero-clear scenes remain listed and receive no invented training samples; report effective counts.',
        candidate_parity='Before generating new scenes, replay one original scene from same upstream and require exact node geometries, proposed pair endpoints and union hashes. Recompute its154 features and labels against frozen originals. No new fitting until parity passes.',
        diagnostics='Difficult=zero-overlap clear same/different. Train actual rows and common half rows, plus identical heldout candidates each fold. Main metrics on original evaluation distribution; all mixed/unknown included in risk.',
        relationship_trend='Both half->original and original->double must strictly lower pooled heldout difficult balanced logloss, lower it in at least3/5 folds, not increase different high90, not decrease same high90; different high90 must decrease from half to double. No threshold search.',
        geometry_gates=bc['gates'], opportunities='18 only postseal; never used to select scenes, model, thresholds or training labels.',
        limits=dict(capacity_search=False, weight_change=False, threshold_change=False, competition_change=False,
            feature_change=False, risk_network=False, point_edits=False, AP=False, production_modified=False, old_seals_modified=False)))
    for p in code_paths:
        shutil.copyfile(p, OUT/('code_'+p.name))
    (OUT/'tmp').mkdir()
    print('PREPARED available',len(available),'added',len(extra),'sizes',[(f, [len(splits[str(f)][arm]) for arm in ARMS]) for f in range(5)],flush=True)


class Scene(candidate.Scene):
    """Same reconstruction, with stage receipts for new scenes instead of old52 audit."""
    def __init__(self, scene, spec, c):
        self.scene, self.spec = scene, spec
        self.frontend, self.used = Path(c['frontend']), {}
        self.arm = self.frontend/'arms/D'
        self.expected = {}
        mfpath = self.frontend/'manifests'/f'{scene}.json'
        self.expected[str(mfpath)] = a.sha(mfpath)
        mf = a.read(mfpath)
        assert mf['scene_name'] == scene
        for v in mf['files'].values():
            self.expected[v['path']] = v['sha256']
        for stage in ('merged', 'tracks', 'hierarchy'):
            rp = self.arm/'stage_receipts'/f'{scene}_{stage}.json'
            self.expected[str(rp)] = a.sha(rp)
            for name, h in a.read(rp)['files'].items():
                self.expected[str(self.arm/stage/scene/name)] = h
        obs = self.arm/'observations'/scene/'automatic_observations.jsonl'
        self.expected[str(obs)] = self.expected[str(self.arm/'hierarchy'/scene/'original_automatic_observations.jsonl')]
        for k in ('geometry', 'points'):
            p = spec['job']['binding'][k]
            self.expected[p] = a.sha(p)
        selected = set(map(str, mf['arms']['D']))
        for f in mf['frames']:
            if str(f['frame_id']) not in selected:
                continue
            cache = self.frontend/'frame_cache/grounded_sam'/scene/('frame%06d' % int(f['frame_id']))
            rp = cache/'receipt.json'
            self.expected[str(rp)] = a.sha(rp)
            for name, h in a.read(rp)['files'].items():
                self.expected[str(cache/name)] = h


def geometry_scene(s, spec, c, dest):
    start = time.time()
    obj = Scene(s, spec, c).load()
    pairs, census = candidate.propose(obj)
    arrays = dict(frame_node_visible=obj.V, frame_indices=np.array(obj.fis), frame_ids=np.array(obj.frame_ids))
    arrays.update({'n%04d' % n:pp for n, pp in enumerate(obj.points)})
    original = {a.digest(p) for p in obj.nt.values()}
    for p in pairs:
        left, right = obj.points[p['left']], obj.points[p['right']]
        union, inter = np.union1d(left, right), np.intersect1d(left, right)
        arrays[p['pair_id']+'_union'] = union
        arrays[p['pair_id']+'_intersection_visible'] = obj.vis[:,inter].sum(axis=1).astype(np.int64)
        p.update(left_track_ids=obj.nodes[p['left']]['final_track_ids'], right_track_ids=obj.nodes[p['right']]['final_track_ids'],
            union_sha256=a.digest(union), duplicate_original_NT=a.digest(union) in original)
    dest.mkdir(parents=True)
    np.savez_compressed(dest/'data.npz', **arrays)
    for name, value in (('nodes',obj.nodes), ('pairs',pairs), ('visibility',[
        dict(frame_index=fi, frame_id=obj.frame_ids[j], path=obj.visibility_paths[j]) for j,fi in enumerate(obj.fis)])):
        a.seal(dest/(name+'.json'), value)
    a.seal(dest/'receipt.json',dict(inputs=obj.used,GT_used=False,pairs=len(pairs),seconds=time.time()-start,proposal_census=census))
    print('GEOMETRY',s,len(pairs),'pairs',round(time.time()-start,2),'seconds',flush=True)
    return pairs


def generate(pilot=False):
    guard()
    c = contract()
    if pilot:
        s = c['scenes'][0]
        spec = a.read(AUDIT/'contract.json')['jobs'][s]
        dest = OUT/'pilot_geometry/predictions'/s
        pairs = geometry_scene(s,spec,c,dest)
        olddir = GEOMETRY/'predictions'/s
        oldpairs, oldnodes = a.read(olddir/'pairs.json'), a.read(olddir/'nodes.json')
        assert a.read(dest/'nodes.json') == oldnodes
        assert [{k:p[k] for k in ('pair_id','left','right','union_sha256','duplicate_original_NT')} for p in pairs] == [
            {k:p[k] for k in ('pair_id','left','right','union_sha256','duplicate_original_NT')} for p in oldpairs]
        with np.load(dest/'data.npz') as z, np.load(olddir/'data.npz') as old:
            for key in z.files:
                np.testing.assert_array_equal(z[key],old[key])
        a.seal(OUT/'pilot_geometry_frozen.json',dict(complete=True,scene=s,GT_used=False,
            outputs={str(p):a.sha(p) for p in dest.iterdir()}))
        return
    assert a.read(OUT/'pilot_parity.json')['complete']
    outputs={}
    for s in c['extra_scenes']:
        a.reserve()
        dest = OUT/'training_geometry/predictions'/s
        if not (dest/'receipt.json').exists():
            geometry_scene(s,c['jobs'][s],c,dest)
        else:
            verify(a.read(dest/'receipt.json')['inputs'])
        outputs.update({str(p):a.sha(p) for p in dest.iterdir()})
    a.seal(OUT/'training_geometry_frozen.json',dict(complete=True,GT_used=False,outputs=outputs))


def features(pilot=False):
    guard()
    c = contract()
    if pilot:
        seal = a.read(OUT/'pilot_geometry_frozen.json')
        scenes=[seal['scene']]
        root=OUT/'pilot_data'
        jobs={s:a.read(AUDIT/'contract.json')['jobs'][s] for s in scenes}
        geometry=OUT/'pilot_geometry'
    else:
        seal=a.read(OUT/'training_geometry_frozen.json')
        scenes=c['extra_scenes'];root=OUT/'training_data';jobs=c['jobs'];geometry=OUT/'training_geometry'
    verify(seal['outputs'])
    a.seal(geometry/'predictions_frozen.json',seal)
    a.seal(root/'contract.json',dict(jobs=jobs))
    paircount=sum(len(a.read(geometry/'predictions'/s/'pairs.json')) for s in scenes)
    visual.OUT, visual.SOURCE, a.OUT = root, geometry, root
    visual.guard=lambda *args,**kwargs:None
    visual.contract=lambda:dict(scenes=scenes,candidate_count=paircount,reused_GT_free_feature_files={})
    visual.features()


def labels(pilot=False):
    c=contract()
    root=OUT/('pilot_data' if pilot else 'training_data')
    geometry=OUT/('pilot_geometry' if pilot else 'training_geometry')
    scenes=[a.read(OUT/'pilot_geometry_frozen.json')['scene']] if pilot else c['extra_scenes']
    specs={s:a.read(AUDIT/'contract.json')['jobs'][s] for s in scenes} if pilot else c['jobs']
    allowed=[specs[s]['job']['GT'] for s in scenes]
    guard(allowed_labels=allowed+([BASE/'labels'/f'{s}.json' for s in scenes] if pilot else []))
    verify(a.read(root/'features_frozen.json')['outputs'])
    outputs,counts={},Counter()
    dev=a.read(DEV)
    for s in scenes:
        gp=specs[s]['job']['GT']
        expected=a.read(BASE/'labels'/f'{s}.json')['GT_sha256'] if pilot else dev['bindings'][s]['gt_sha256']
        assert a.sha(gp)==expected
        gt=np.loadtxt(gp,dtype=np.int64)
        rows=a.read(root/'features'/s/'rows.json')
        with np.load(geometry/'predictions'/s/'data.npz') as z:
            identities={n:relation.endpoint_label(*np.unique(gt[z['n%04d'%n]],return_counts=True)) for n in {n for q in rows for n in (q['left'],q['right'])}}
        pairs=[dict(pair_id=q['pair_id'],relation=relation.labels(identities[q['left']],identities[q['right']])) for q in rows]
        value=dict(scene=s,GT_path=gp,GT_sha256=expected,nodes=identities,pairs=pairs)
        path=root/'labels'/f'{s}.json';a.seal(path,value);outputs[str(path)]=a.sha(path)
        counts.update(q['relation'] for q in pairs)
    a.seal(root/'labels_frozen.json',dict(complete=True,counts=dict(counts),outputs=outputs))
    if pilot:
        s=scenes[0]
        with np.load(root/'features'/s/'features.npz') as actual, np.load(BASE/'features'/s/'features.npz') as old:
            for key in actual.files:
                np.testing.assert_array_equal(actual[key],old[key])
        assert a.read(root/'features'/s/'rows.json')==a.read(BASE/'features'/s/'rows.json')
        assert a.read(root/'labels'/f'{s}.json')==a.read(BASE/'labels'/f'{s}.json')
        a.seal(OUT/'pilot_parity.json',dict(complete=True,scene=s,geometry_exact=True,features154_exact=True,labels_exact=True,
            feature_inputs_GT_free=True,no_model_fits=True))
    print('LABELS',dict(counts),'pilot',pilot,flush=True)


def combined():
    old=baseline.rows();new=a.read(OUT/'training_data/feature_rows.json')
    xold,_=baseline.features()
    with np.load(OUT/'training_data/features.npz') as z:
        xnew=np.c_[z['geometry'],z['rgbd']]
    return old+new,np.r_[xold,xnew],len(old)


def fit(fold):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    c=contract();split=c['splits'][str(fold)]
    sources={s:(BASE if s in c['scenes'] else OUT/'training_data')/'labels'/f'{s}.json' for s in split['double']}
    guard(allowed_labels=sources.values())
    verify(a.read(OUT/'training_data/features_frozen.json')['outputs'])
    labelseal=a.read(OUT/'training_data/labels_frozen.json')['outputs']
    lookup={};inputs={}
    for s,p in sources.items():
        expected=c['label_files'].get(str(p),labelseal.get(str(p)))
        assert a.sha(p)==expected
        inputs[str(p)]=expected
        lookup.update({(s,q['pair_id']):q['relation'] for q in a.read(p)['pairs']})
    rr,x,nold=combined()
    assert x.shape==(len(rr),154)
    y=np.array([1 if lookup.get((q['scene'],q['pair_id']))=='same' else 0 if lookup.get((q['scene'],q['pair_id']))=='different' else -1 for q in rr])
    d=OUT/'models'/f'fold{fold}';d.mkdir(parents=True)
    for arm in ARMS:
        start=time.time()
        idx=np.array([i for i,q in enumerate(rr) if q['scene'] in split[arm] and y[i]>=0],dtype=np.int64)
        ss=[rr[i]['scene'] for i in idx]
        w=metrics.weights(ss,x[idx,4]>0,y[idx])
        if arm=='original':
            with np.load(SOURCE/'models'/f'fold{fold}'/'training.npz') as z:
                np.testing.assert_array_equal(idx,z['row_indices'])
                np.testing.assert_array_equal(w,z['weights_C'])
                np.testing.assert_array_equal(y[idx],z['y'])
            model=joblib.load(SOURCE/'models'/f'fold{fold}'/'C.joblib')
        else:
            model=HistGradientBoostingClassifier(**c['model'])
            with threadpool_limits(4):
                model.fit(x[idx],y[idx],sample_weight=w)
            joblib.dump(model,d/(arm+'.joblib'))
        np.savez_compressed(d/(arm+'_training.npz'),row_indices=idx,y=y[idx],weights=w)
        with threadpool_limits(4):
            score=model.predict_proba(x)[:,1]
        if arm=='original':
            np.testing.assert_array_equal(score[:nold],np.load(SOURCE/'C_scores.npz')['fold_all'][fold])
        np.save(d/(arm+'_scores.npy'),score)
        a.seal(d/(arm+'_receipt.json'),dict(fold=fold,arm=arm,fit=arm!='original',seconds=time.time()-start,
            scenes=split[arm],effective_clear_scenes=sorted(set(ss)),pairs=len(idx),labels=dict(Counter(y[idx].tolist())),
            label_inputs={str(sources[s]):inputs[str(sources[s])] for s in split[arm]},
            outputs={str(p):a.sha(p) for p in d.glob(arm+'_*') if p.is_file()} | ({str(d/(arm+'.joblib')):a.sha(d/(arm+'.joblib'))} if arm!='original' else {})))
        print('FIT',fold,arm,'scenes',len(split[arm]),'clear scenes',len(set(ss)),'pairs',len(idx),flush=True)


def predict():
    guard();c=contract();rr=baseline.rows();folds=np.array([c['folds'][q['scene']] for q in rr]);outputs={}
    for arm in ARMS:
        score=[]
        for f in range(5):
            d=OUT/'models'/f'fold{f}'
            receipt=a.read(d/(arm+'_receipt.json'));verify(receipt['outputs']);outputs[str(d/(arm+'_receipt.json'))]=a.sha(d/(arm+'_receipt.json'))
            outputs.update(receipt['outputs'])
            score.append(np.load(d/(arm+'_scores.npy'))[:len(rr)])
        score=np.asarray(score);oof=score[folds,np.arange(len(rr))]
        np.savez_compressed(OUT/(arm+'_scores.npz'),fold_all=score,oof=oof)
        allpred=[]
        for s in c['scenes']:
            dest=OUT/'predictions'/arm/s;dest.mkdir(parents=True)
            if arm=='original':
                for name in ('pairs.json','unions.npz'):
                    shutil.copyfile(SOURCE/'predictions/C'/s/name,dest/name)
                pred=a.read(dest/'pairs.json')
            else:
                ix=[i for i,q in enumerate(rr) if q['scene']==s]
                pred=relation.decide([rr[i] for i in ix],oof[ix],[rr[i]['eligible'] for i in ix])
                emitted,arrays=set(),{}
                with np.load(GEOMETRY/'predictions'/s/'data.npz') as z:
                    for q in pred:
                        if q['state']!='accept':continue
                        pp=z[q['pair_id']+'_union'];assert a.digest(pp)==q['union_sha256']
                        if q['duplicate_original_NT'] or q['union_sha256'] in emitted:
                            q['reason']+='_duplicate_geometry_not_emitted';continue
                        emitted.add(q['union_sha256']);arrays[q['pair_id']+'_union']=pp;q['emit']=True
                np.savez_compressed(dest/'unions.npz',**arrays);a.seal(dest/'pairs.json',pred)
            outputs.update({str(p):a.sha(p) for p in dest.iterdir()});allpred.extend(pred)
        a.seal(OUT/(arm+'_decisions.json'),allpred)
        for name in (arm+'_scores.npz',arm+'_decisions.json'):outputs[str(OUT/name)]=a.sha(OUT/name)
        if arm=='original':assert a.sha(OUT/(arm+'_decisions.json'))==a.sha(SOURCE/'C_decisions.json')
        print('PREDICT',arm,dict(Counter(q['state'] for q in allpred)),'emitted',sum(q['emit'] for q in allpred),flush=True)
    a.seal(OUT/'predictions_frozen.json',dict(complete=True,time_unix=time.time(),GT_used=False,outputs=outputs,
        fitted_models=10,original_C_exact_reused=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','generate','features','labels','fit','predict'))
    p.add_argument('--pilot',action='store_true');p.add_argument('--fold',type=int,choices=range(5));args=p.parse_args()
    if args.mode in ('generate','features','labels'):globals()[args.mode](args.pilot)
    elif args.mode=='fit':assert args.fold is not None;fit(args.fold)
    else:globals()[args.mode]()
