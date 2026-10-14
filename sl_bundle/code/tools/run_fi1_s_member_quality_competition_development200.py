#!/usr/bin/env python3
"""Isolated, fixed five-fold S-protocol scoring experiment on exposed dev200."""
from pathlib import Path
from collections import Counter
import argparse
import json
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from tools import run_fi1_native_symmetric_eligibility_development200 as previous
from tools import fi1_s_member_quality_competition as core
from tools.fi1_r2_core import MODEL_PARAMS, scene_weights
from tools.fi1_r3_core import gt_iou, pair_overlap, assign_competition, independent_assignment
from tools.fi1_frontend_protocol import sha256_file as sha, freeze_json, write_json

S = previous.OUT
C = previous.RELATION
OUT = ROOT / '.local/fi1_s_member_quality_competition_development200_20261010_v1'
ARMS = ('R', 'Q', 'F')
VIEWS = ('semantic', 'class_agnostic')
ctl = previous.ctl
read = previous.read
PAIR_PARAMS = dict(MODEL_PARAMS, loss='log_loss', max_iter=80, min_samples_leaf=40, l2_regularization=5.)


def write(p, value):
    freeze_json(p, value)


def stage(name, **kw):
    write_json(OUT/'execution.json', dict(status='running', stage=name, updated=time.time(), **kw))
    print(name, kw, flush=True)


class ReadGuard:
    """Audit file access; cached training targets forbidden in inference.

    Training/evaluation GT whitelist is exactly the approved 200 bindings.
    Fold-specific label access is separately enforced by load_train().
    """
    def __init__(self, c, mode):
        self.c, self.mode, self.opened_gt = c, mode, set()
        self.gt = {str(Path(b['gt_path']).resolve()) for b in c['bindings'].values()}

    def hook(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        p = str(Path(args[0]).resolve())
        if any(x in p for x in ('/scannet200_test/', '/scannetv2_test/', '/scannet_test/', '/Dataset/scannet200/', '/paper_real_figure_assets_', '/paper_furniture_focused_')):
            raise PermissionError('out-of-scope data forbidden')
        if '/.local/' in p and 'val312' in p and '/code_snapshot/' not in p:
            raise PermissionError('validation artifacts forbidden')
        if any(x.startswith('ground_truth') for x in Path(p).parts):
            if self.mode not in ('dataset', 'evaluate') or p not in self.gt:
                raise PermissionError('raw GT forbidden in this stage')
            self.opened_gt.add(p)
        if self.mode in ('predict', 'verify') and ('/labels/' in p or '/evaluation/' in p or Path(p).name in ('labels.npz', 'target_census.json')):
            raise PermissionError('training/evaluation targets forbidden during inference')
        if '/learning/dataset/' in p and Path(p).name in ('features.npz', 'members.jsonl'):
            raise PermissionError('historical training rows forbidden')


def prepare():
    assert not OUT.exists(), 'new isolated experiment required'
    assert sklearn.__version__ == '1.3.2'
    assert shutil.disk_usage(ROOT).free >= 50*1024**3
    p, relation = read(S/'contract.json'), read(C/'contract.json')
    assert read(S/'completion_verified.json')['complete']
    assert len(p['scenes']) == len({s[:9] for s in p['scenes']}) == 200
    assert not {s[:9] for s in p['scenes']} & {s[:9] for s in p['reserve100']}
    inputs = {str(S/n): sha(S/n) for n in ('contract.json', 'completion_verified.json', 'predictions_frozen.json')}
    inputs[str(C/'contract.json')] = sha(C/'contract.json')
    seal = read(S/'predictions_frozen.json')
    jobs = {}
    for s in p['scenes']:
        jobs[s] = dict(p['jobs'][s])
        for n in ('pool.npz', 'members.jsonl', 'semantic_S.json', 'class_agnostic_S.json', 'semantic_C.json', 'class_agnostic_C.json'):
            q = S/'predictions'/s/n
            inputs[str(q)] = seal['outputs'][str(q)]
        for n in ('geometry', 'points', 'evidence'):
            q = p['bindings'][s][n]
            inputs[q] = p['inputs'][q]
        q = jobs[s]['nt_geometry.json']; inputs[q] = p['inputs'][q]
    files = [Path(__file__), ROOT/'tools/fi1_s_member_quality_competition.py', ROOT/'tests/test_fi1_s_member_quality_competition.py']
    code = dict(p['code'])
    code.update({str(q): sha(q) for q in files})
    c = dict(version=OUT.name, scenes=p['scenes'], reporting_groups=p['reporting_groups'],
             bindings=p['bindings'], jobs=jobs, folds=relation['folds'], inputs=inputs, code=code,
             reserve100=p['reserve100'], budget=800, arms=ARMS, views=VIEWS,
             member_params=MODEL_PARAMS, pair_params=PAIR_PARAMS, class_ids=list(map(int, ctl.base.evaluator().VALID_CLASS_IDS)),
             B_model=p['B_model'], B_model_sha256=p['B_model_sha256'],
             hypothesis='Separate own same-class range quality from class-specific duplicate representation, on S eligibility and fixed NT geometry.',
             R='S candidates; original X26 and one-to-one IoU supervision; exactly original HGB capacity/parameters and row weights; outer-fold retrain160 scenes.',
             Q='Independent maximum same-class IoU head on own20; 0.5 frozen B26 + 0.5 own-quality. Exclude evaluatively ignored negatives, retain independent good duplicates.',
             F='Q scores plus learned directed same-class competition; any source; safe/harm/unknown classification, protected direct witnesses, no chains, K800 witness restoration.',
             own_ignore='If own same-class IoU<=.25 and >.5 prediction fraction belongs to void or same-class ignored (<100-point) instances, zero training weight in Q only. R preserves original weights.',
             pair_labels='Reliable unique same-class identity strictly IoU>.5; safe if same target, winner continuously no worse and no supported threshold loss at .50,.55,...,.90. Ties prefer frozen direct member order. Reliable different target or worse direction unsafe; others abstain.',
             pair_sampling='Top3 positive point-IoU same-class neighbors per >=100-point member; symmetric directed closure, N/N N/T T/T; no label or AP sampling.',
             pair_features='Both own20, signed differences, frozen B26 scores and signed gap, point IoU and directional coverages; no class ID, GT, own fitted scores or assignment feature.',
             pair_weight='Equal scene then directed candidate edge; no class balancing or threshold sweep.',
             score_constraint='Q=0.5 B26+0.5 predicted own IoU, allows up/down. Safe posterior>.9 permits multiplicative attenuation 1-0.5 p; always keep eligible members and fixed tie rule.',
             fitting='Reuse exact C five physical-scene folds; new models fit other160, predict heldout40. No inner hyperparameter/model/threshold/AP search. Frozen B26/upstream have seen these scenes.',
             GT_policy='Training labels separated from GT-free cached features; heldout labels never opened by fold fit; inference forbids all targets; all predictions sealed and independently replayed before evaluator.',
             projection='Actual selected member score, exact pointset max selected score, no refill; original semantic and class-agnostic evaluators and ignore rules.',
             success='F must exceed R,Q,S,C on both primary APs and retain all S recovered424 targets at IoU>.75; old90/new110 reported. Development only; no deployment.',
             failure='No AP-driven revision/tuning. Diagnose failure; do not append U or change Uprime unless step1 passes.',
             U_Uprime_output=False, new_geometry=False, new_visual_inference=False, production_modified=False,
             scope='Exposed development200 OOF for newly trained heads only; training size/protocol differ from historical B26 full597, not whole-system blind acceptance.')
    ctl.base.verify_files(inputs)
    ctl.base.verify_files(code)
    OUT.mkdir(parents=True)
    write(OUT/'contract.json', c)
    for q in code:
        dest = OUT/'code_snapshot'/Path(q).relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(q, dest)
    run = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_fi1_s_member_quality_competition.py',
                          'tests/test_fi1_class_agnostic_ablation.py', '-q'], cwd=ROOT, capture_output=True, text=True)
    (OUT/'tests.log').write_text(run.stdout+run.stderr)
    assert run.returncode == 0, run.stdout+run.stderr
    write(OUT/'tests_verified.json', dict(passed=True, log_sha256=sha(OUT/'tests.log')))
    stage('prepared', next='dataset')


def load_features(s):
    with np.load(OUT/'dataset'/s/'features.npz', allow_pickle=False) as z:
        return dict(z)


def dataset(c):
    guard = ReadGuard(c, 'dataset'); sys.addaudithook(guard.hook)
    ctl.base.verify_files(c['inputs']); ctl.base.verify_files(c['code'])
    receipts = {}
    counts = Counter()
    for j, s in enumerate(c['scenes']):
        stage('dataset', scene=s, completed=j, total=200)
        d = OUT/'dataset'/s; d.mkdir(parents=True, exist_ok=True)
        if (d/'receipt.json').exists():
            r = read(d/'receipt.json'); ctl.base.verify_files(r['outputs'])
        else:
            members = ctl.base.rows(S/'predictions'/s/'members.jsonl')
            pool = read(c['jobs'][s]['nt_geometry.json']); keys = [g['geometry_key'] for g in pool]
            _, allpoints = ctl.shared.load_points(c['bindings'][s])
            points = [allpoints[k] for k in keys]; sizes = np.asarray([len(a) for a in points])
            with np.load(S/'predictions'/s/'pool.npz', allow_pickle=False) as z:
                x, frozen = z['x26'], z['scores']
            assert x.shape == (len(members), 26)
            index = {k: i for i, k in enumerate(keys)}
            gi = np.asarray([index[m['geometry_key']] for m in members])
            iou, coverage = pair_overlap(points, c['bindings'][s]['point_count'])
            edges = core.candidate_edges(members, keys, iou, sizes)
            ex = core.edge_features(x, frozen, edges, gi, iou, coverage)
            # These arrays contain no training labels and are the only inference cache.
            np.savez_compressed(d/'features.npz', x26=x, frozen=frozen, edges=edges, edge_x=ex)
            b = c['bindings'][s]; assert sha(b['gt_path']) == b['gt_sha256']
            gt = np.loadtxt(b['gt_path'], dtype=np.int64); assert len(gt) == b['point_count']
            overlap, ids = gt_iou(points, gt, c['class_ids'])
            census = read(c['jobs'][s]['target_census'])
            assert set(map(int, census['targets'])) == set(map(int, ids))
            for a, k in enumerate(keys):
                assert np.array_equal(overlap[a], [census['ious'][k][str(g)] if sizes[a]>=100 else 0 for g in ids])
            original, assignment = assign_competition(members, keys, overlap, ids, c['class_ids'])
            independent, independent_ids = independent_assignment(members, keys, overlap, ids, c['class_ids'])
            assert np.array_equal(original, independent) and np.array_equal(assignment, independent_ids)
            own, targets = core.own_labels(members, keys, overlap, ids, np.asarray(c['class_ids']))
            labels = core.competition_labels(own, targets, edges)
            weights = scene_weights(members)
            ignore = np.zeros(len(members), dtype=bool)
            # evaluator void or same-class tiny/uninstanced GT fraction; feature-free training mask
            valid_class = np.isin(gt//1000, c['class_ids'])
            unique, cnt = np.unique(gt, return_counts=True)
            small = unique[(unique>=1000)&(cnt<100)]
            void = ~valid_class
            for ci in set(m['class_index'] for m in members):
                cls = c['class_ids'][ci]
                ignored = void | ((gt//1000 == cls)&((gt<1000)|np.isin(gt, small)))
                fractions = np.asarray([ignored[p].mean() if len(p) else 1 for p in points])
                rr = np.asarray([i for i,m in enumerate(members) if m['class_index']==ci])
                ignore[rr] = (own[rr] <= .25) & (fractions[gi[rr]] > .5)
            np.savez_compressed(d/'labels.npz', original=original, own=own, targets=targets,
                                competition=labels, weights=weights, own_weights=weights*(~ignore),
                                edge_weights=np.full(len(edges), 1/max(1,len(edges))))
            r = dict(complete=True, fold=c['folds'][s], member_count=len(members), edge_count=len(edges),
                     ignore_count=int(ignore.sum()), pair_label_counts={str(k): int((labels==k).sum()) for k in range(3)},
                     independent_assignment_exact=True, GT_sha256=b['gt_sha256'],
                     outputs={str(d/n): sha(d/n) for n in ('features.npz', 'labels.npz')})
            write(d/'receipt.json', r)
        receipts[str(d/'receipt.json')] = sha(d/'receipt.json')
        counts.update(members=r['member_count'], edges=r['edge_count'], ignored=r['ignore_count'])
    write(OUT/'dataset_frozen.json', dict(complete=True, receipts=receipts, counts=dict(counts), raw_GT_scans=len(guard.opened_gt)))


def load_train(c, fold):
    scenes = [s for s in c['scenes'] if c['folds'][s] != fold]
    assert len(scenes) == 160
    features, labels = [], []
    for s in scenes:
        assert c['folds'][s] != fold
        r = read(OUT/'dataset'/s/'receipt.json'); ctl.base.verify_files(r['outputs'])
        features.append(load_features(s))
        with np.load(OUT/'dataset'/s/'labels.npz', allow_pickle=False) as z:
            labels.append(dict(z))
    x = {k: np.concatenate([f[k] for f in features]) for k in ('x26', 'edge_x')}
    y = {k: np.concatenate([f[k] for f in labels]) for k in ('original', 'own', 'competition', 'weights', 'own_weights', 'edge_weights')}
    return scenes, x, y


def fit(c):
    assert read(OUT/'dataset_frozen.json')['complete']
    ctl.base.verify_files(c['code'])
    receipts = {}
    for fold in range(5):
        stage('fit', fold=fold, total=5)
        d = OUT/'models'/f'fold{fold}'; d.mkdir(parents=True, exist_ok=True)
        if (d/'receipt.json').exists():
            ctl.base.verify_files(read(d/'receipt.json')['outputs'])
            receipts[str(d/'receipt.json')] = sha(d/'receipt.json'); continue
        start = time.monotonic()
        scenes, x, y = load_train(c, fold)
        test = [s for s in c['scenes'] if c['folds'][s] == fold]
        assert not {s[:9] for s in scenes} & {s[:9] for s in test}
        for head, xx, yy, ww, estimator, params in (
            ('R', x['x26'], y['original'], y['weights'], HistGradientBoostingRegressor, MODEL_PARAMS),
            ('own', x['x26'][:, :20], y['own'], y['own_weights'], HistGradientBoostingRegressor, MODEL_PARAMS),
            ('competition', x['edge_x'], y['competition'], y['edge_weights'], HistGradientBoostingClassifier, PAIR_PARAMS)):
            stage('fit_head', fold=fold, head=head, rows=len(xx), features=xx.shape[1])
            ww = ww*len(ww)/ww.sum()
            model = estimator(**params).fit(xx, yy, sample_weight=ww)
            assert model.n_iter_ == params['max_iter']
            joblib.dump(model, d/f'{head}.joblib')
            saved = joblib.load(d/f'{head}.joblib')
            sample = xx[np.linspace(0,len(xx)-1,min(10000,len(xx)),dtype=int)]
            assert np.array_equal(model.predict(sample), saved.predict(sample))
            del model, saved
        write(d/'receipt.json', dict(complete=True, train_scenes=scenes, test_scenes=test,
                                     heldout_labels_opened=False, member_rows=len(x['x26']), edge_rows=len(x['edge_x']),
                                     dataset_seal_sha256=sha(OUT/'dataset_frozen.json'), seconds=time.monotonic()-start,
                                     outputs={str(d/f'{h}.joblib'): sha(d/f'{h}.joblib') for h in ('R','own','competition')}))
        receipts[str(d/'receipt.json')] = sha(d/'receipt.json')
        del x,y
    write(OUT/'models_frozen.json', dict(complete=True, receipts=receipts, folds=5))


def model_scores(models, f):
    r = np.clip(models['R'].predict(f['x26']), 0, 1)
    own = np.clip(models['own'].predict(f['x26'][:, :20]), 0, 1)
    q = (1-core.ANCHOR)*f['frozen']+core.ANCHOR*own
    p = models['competition'].predict_proba(f['edge_x'])[:,list(models['competition'].classes_).index(1)]
    return r, q, own, p


def predict(c):
    guard = ReadGuard(c, 'predict'); sys.addaudithook(guard.hook)
    ctl.base.verify_files(c['code'])
    assert read(OUT/'models_frozen.json')['complete']
    receipts, outputs, counts = {}, {}, Counter()
    for fold in range(5):
        d = OUT/'models'/f'fold{fold}'; r = read(d/'receipt.json'); ctl.base.verify_files(r['outputs'])
        models = {h:joblib.load(d/f'{h}.joblib') for h in ('R','own','competition')}
        for s in r['test_scenes']:
            stage('predict', fold=fold, scene=s)
            dest = OUT/'predictions'/s; dest.mkdir(parents=True, exist_ok=True)
            if (dest/'receipt.json').exists():
                rr = read(dest/'receipt.json'); ctl.base.verify_files(rr['outputs'])
            else:
                members = ctl.base.rows(S/'predictions'/s/'members.jsonl')
                f = load_features(s)
                rscore, q, own, p = model_scores(models, f)
                full, audit = core.compete(members, q, f['edges'], p, budget=800)
                np.savez_compressed(dest/'scores.npz', R=rscore, Q=q, F=full, own=own, safe_probability=p)
                for arm, scores in zip(ARMS, (rscore,q,full)):
                    rows = previous.rank(members, scores)[:800]
                    write(dest/f'semantic_{arm}.json', rows)
                    write(dest/f'class_agnostic_{arm}.json', ctl.collapse_members(rows))
                write(dest/'competition.json', audit)
                files = list(dest.glob('*.json'))+[dest/'scores.npz']
                rr = dict(complete=True, raw_GT_read=False, fold=fold,
                          actions=len(audit['actions']), restored=audit['restored'],
                          outputs={str(q):sha(q) for q in files})
                write(dest/'receipt.json', rr)
            outputs.update(rr['outputs']); counts.update(actions=rr['actions'],restored=rr['restored'])
            receipts[str(dest/'receipt.json')] = sha(dest/'receipt.json')
    assert not guard.opened_gt
    write(OUT/'predictions_frozen.json', dict(complete=True, outputs=outputs, receipts=receipts, counts=dict(counts),raw_GT_read=False))


def verify(c):
    guard = ReadGuard(c,'verify'); sys.addaudithook(guard.hook)
    ctl.base.verify_files(c['code'])
    seal = read(OUT/'predictions_frozen.json'); ctl.base.verify_files(seal['outputs'])
    receipts = {}
    for fold in range(5):
        md = OUT/'models'/f'fold{fold}'; models = {h:joblib.load(md/f'{h}.joblib') for h in ('R','own','competition')}
        for s in [s for s in c['scenes'] if c['folds'][s]==fold]:
            stage('verify_inference',scene=s)
            f = load_features(s); members = ctl.base.rows(S/'predictions'/s/'members.jsonl')
            r,q,own,p = model_scores(models,f)
            scores, audit = core.compete(members,q,f['edges'],p)
            d = OUT/'predictions'/s
            with np.load(d/'scores.npz',allow_pickle=False) as z:
                for key,a in zip(('R','Q','F','own','safe_probability'),(r,q,scores,own,p)):
                    assert np.array_equal(a,z[key])
            assert audit == read(d/'competition.json')
            for arm,a in zip(ARMS,(r,q,scores)):
                # Independent ranking and max-selected-member collapse.
                ranked = sorted([{**m,'score':float(v)} for m,v in zip(members,a)],
                                key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:800]
                assert ranked == read(d/f'semantic_{arm}.json')
                best = {}
                for m in sorted(ranked,key=lambda m:(-m['score'],m['class_index'])):
                    best.setdefault(m['geometry_key'],dict(geometry_key=m['geometry_key'],score=m['score'],winning_class_index=m['class_index']))
                assert sorted(best.values(),key=lambda m:(-m['score'],m['geometry_key'])) == read(d/f'class_agnostic_{arm}.json')
                assert len(ranked)==800
            selected = {ctl.shared.key(m) for m in read(d/'semantic_F.json')}
            for action in audit['actions']:
                assert ctl.shared.key(members[action['winner']]) in selected
                assert members[action['winner']]['class_index']==members[action['loser']]['class_index']
            receipts[s] = dict(scores_replayed=True,independent_rank_and_collapse=True,direct_witnesses_retained=True)
    write(OUT/'inference_verified.json',dict(complete=True,predictions_seal_sha256=sha(OUT/'predictions_frozen.json'),scenes=receipts,raw_GT_read=False))


def evaluate(c):
    assert read(OUT/'inference_verified.json')['predictions_seal_sha256']==sha(OUT/'predictions_frozen.json')
    ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs']);ctl.base.verify_files(c['code'])
    guard = ReadGuard(c,'evaluate');sys.addaudithook(guard.hook)
    ev = ctl.base.evaluator(); ca,valid,uid = ctl.class_agnostic_evaluator()
    receipts = {}
    for j,s in enumerate(c['scenes']):
        stage('matching',scene=s,completed=j,total=200)
        d = OUT/'evaluation'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():
            ctl.base.verify_files(read(d/'receipt.json')['outputs']);receipts[str(d/'receipt.json')]=sha(d/'receipt.json');continue
        b = c['bindings'][s];assert sha(b['gt_path'])==b['gt_sha256']
        gt=np.loadtxt(b['gt_path'],dtype=np.int64);_,points=ctl.shared.load_points(b);sizes={k:len(p) for k,p in points.items()}
        rows={a:read(OUT/'predictions'/s/f'semantic_{a}.json') for a in ARMS}
        universe=list({ctl.shared.key(m):m for a in ARMS for m in rows[a]}.values())
        template=ctl.shared.assign(ev,universe,points,b['gt_path'],len(gt))
        sem={a:ctl.project_multiclass(template,universe,rows[a],sizes,ev.CLASS_LABELS,ev.VALID_CLASS_IDS) for a in ARMS}
        edges=sum(ctl.shared.check_match(sem[a],rows[a],points,gt,ev) for a in ARMS)
        mapped=ctl.remap_gt(gt,valid,uid)
        def loader(path):
            assert Path(path).resolve()==Path(b['gt_path']).resolve()
            return mapped.copy()
        ca.util_3d.load_ids=loader
        cr={a:read(OUT/'predictions'/s/f'class_agnostic_{a}.json') for a in ARMS}
        cu=list({m['geometry_key']:m for a in ARMS for m in cr[a]}.values())
        ct=ctl.shared.assign(ca,cu,points,b['gt_path'],len(gt),True)
        ck=[m['geometry_key'] for m in cu if sizes[m['geometry_key']]>=100]
        ag={a:ctl.project_matches(ct,ck,cr[a],ca.CLASS_LABELS[0]) for a in ARMS}
        edges+=sum(ctl.shared.check_match(ag[a],cr[a],points,mapped,ca,True) for a in ARMS)
        outputs={}
        for view, match in (('semantic',sem),('class_agnostic',ag)):
            for a in ARMS:
                q=d/f'{view}_{a}_matches.json';write(q,ctl.base.serializable_matches(match[a]));outputs[str(q)]=sha(q)
        write(d/'receipt.json',dict(complete=True,GT_sha256=b['gt_sha256'],independent_intersection_edges=edges,outputs=outputs))
        receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    write(OUT/'matches_frozen.json',dict(complete=True,receipts=receipts,predictions_before_evaluator=True))
    ap,arrays={},{}
    for group,scenes in c['reporting_groups'].items():
        ap[group]={}
        for view in VIEWS:
            evaluator=ev if view=='semantic' else ca;ap[group][view]={}
            for arm in ARMS+('S','C'):
                source=OUT if arm in ARMS else S
                matches={s:read(source/'evaluation'/s/f'{view}_{arm}_matches.json') for s in scenes}
                arr,*_=evaluator.evaluate_matches(matches)
                ap[group][view][arm]=ctl.base.metrics(evaluator,arr)
                q=OUT/'evaluation'/f'{group}_{view}_{arm}_AP.npy';np.save(q,arr,allow_pickle=False);arrays[str(q)]=sha(q)
                # Every arm independently rerun through the original AP computation.
                replay,*_=evaluator.evaluate_matches(dict(reversed(list(matches.items()))))
                assert np.allclose(arr,replay,rtol=0,atol=1e-12,equal_nan=True)
                if arm in ('S','C'):
                    assert np.array_equal(arr,np.load(S/'evaluation'/f'{group}_{view}_{arm}_AP.npy'),equal_nan=True)
                print('AP',group,view,arm,ap[group][view][arm],flush=True)
    write(OUT/'evaluation_summary.json',dict(complete=True,AP=ap,AP_arrays=arrays,original_AP_replayed=True,old_S_C_exact=True))


def report(c):
    results=read(OUT/'evaluation_summary.json');ev=ctl.base.evaluator();cid=c['class_ids']
    summary={}; all_recovered=0
    for j,s in enumerate(c['scenes']):
        stage('diagnostics',scene=s,completed=j)
        census=read(c['jobs'][s]['target_census']);gtids=list(map(int,census['targets']))
        keys=sorted(census['ious']);ov={k:{int(g):v for g,v in census['ious'][k].items()} for k in keys}
        selected={a:read((OUT if a in ARMS else S)/'predictions'/s/f'semantic_{a}.json') for a in ARMS+('S','C','NT')}
        def support(rows,t,semantic=True):
            return {g for m in rows for g,v in ov[m['geometry_key']].items() if v>t and (not semantic or g//1000==cid[m['class_index']])}
        restored=support(selected['S'],.75)-support(selected['NT'],.75)
        # Fixed original eligibility548 and restored424, from diagnostic receipt.
        old=read(ROOT/'.local/fi1_s_c_selection_scope_development200_20261010_v1/scenes'/s/'members.json')
        recovered=old['recovered_targets'];ids=set()
        for r in recovered:
            ids.add(int(r['gt_id']))
        all_recovered+=len(ids)
        sd={}
        for arm in ARMS:
            rows=selected[arm];matches=read(OUT/'evaluation'/s/f'semantic_{arm}_matches.json')
            _,points=ctl.shared.load_points(c['bindings'][s]);sizes={k:len(p) for k,p in points.items()}
            status=previous.member_status(rows,matches,sizes,cid)
            coverage={}
            for t in (.5,.75,.9):
                now=support(rows,t);ref=support(selected['S'],t)
                cn=support(rows,t,False);cf=support(selected['S'],t,False)
                duplicates=sum(max(0,sum(ov[m['geometry_key']].get(g,0)>t for m in ctl.collapse_members(rows))-1) for g in gtids)
                coverage[str(t)]=dict(semantic=len(now),semantic_gained=len(now-ref),semantic_lost=len(ref-now),
                                     ca=len(cn),ca_gained=len(cn-cf),ca_lost=len(cf-cn),ca_extra_matching_predictions=duplicates)
            sd[arm]=dict(status=dict(Counter(status.values())),coverage=coverage,
                         recovered424_retained=sorted(ids&support(rows,.75)),recovered424_lost=sorted(ids-support(rows,.75)))
        summary[s]=sd
    assert all_recovered==424
    lost={a:sum(len(v[a]['recovered424_lost']) for v in summary.values()) for a in ARMS}
    main=results['AP']['development200']
    ap_pass=all(main[v]['F']['ap']>main[v][a]['ap'] for v in VIEWS for a in ('R','Q','S','C'))
    success=ap_pass and lost['F']==0
    decision=dict(complete=True,success=success,AP_gate=ap_pass,recovered424_losses=lost,
                  U_followup_authorized_by_gate=success,production_modified=False,post_AP_tuning=False)
    write(OUT/'diagnostics.json',dict(scenes=summary,recovered424_losses=lost))
    write(OUT/'decision.json',decision)
    lines=['# S 协议下的成员质量与竞争实验','',
           '固定 N/T 点集、S 对称资格、K800；五折复用 C 的物理场景划分。新模型只用另160场景训练。上游与冻结 B26 已见过开发域，属于开发 OOF，不能称整系统盲测。',
           '', 'R：同容量原监督重训；Q：独立自身同类 IoU 质量与冻结 B26 等权锚定；F：Q 加当前类别下的学习竞争。F 竞争为软重评分，保留直接见证并在最终 K800 中核验。',
           '', '|方案|类别相关 AP (%)|实际输出去类 AP (%)|', '|---|---:|---:|']
    for a in ('S','C')+ARMS:
        lines.append(f"|{a}|{100*main['semantic'][a]['ap']:.6f}|{100*main['class_agnostic'][a]['ap']:.6f}|")
    lines+=['',f'424 个既定恢复目标损失：{lost}。验收通过：{success}。',
            '', '所有分组 AP/AP50/AP25 见 evaluation_summary.json；逐场景状态、覆盖及重复见 diagnostics.json；模型、输入、监督与动作规则见 contract.json。',
            '', '重训控制相对 S 同时改变训练域大小及 OOF 使用方式，因此 R−S 不能单独解释为对称资格适配。Q−R 还改变自身输入、忽略权重与锚定；F−Q 才隔离本次竞争动作。这些是组合机制验证，不宣称各子改动的单独因果贡献。',
            '', '若未通过，保留 S/C 双参考并暂停 U/U′追加；不根据本轮 AP 继续扫参数或修改生产 Full。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    outputs={str(OUT/n):sha(OUT/n) for n in ('contract.json','dataset_frozen.json','models_frozen.json','predictions_frozen.json','inference_verified.json','matches_frozen.json','evaluation_summary.json','diagnostics.json','decision.json','report.md')}
    write(OUT/'completion_verified.json',dict(complete=True,success=success,outputs=outputs,production_modified=False))
    write_json(OUT/'execution.json',dict(status='complete',stage='complete',success=success,resume_required=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','dataset','fit','predict','verify','evaluate','report']);a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='prepare':prepare()
        else:
            c=read(OUT/'contract.json');globals()[a.stage](c)


if __name__=='__main__':
    main()
