#!/usr/bin/env python3
"""Frozen native-symmetric R1 eligibility control on development200.

The only policy intervention is R1 class expansion for native-only geometry:
keep legacy classes and add every positive own-fusion class.  Geometry,
evidence, B26, budget, ranking and evaluator are otherwise reused unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import subprocess
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools import run_fi1_selection_control_development200 as ctl
from tools.fi1_core_ablation import source_pool
from tools.fi1_r0_r1_protocol import build_pair, direct_score, original_aliases
from tools.fi1_r2_core import geometry_features
from tools.fi1_r3_core import context_features, pair_overlap
from tools.fi1_r3_core import independent_context
from tools.fi1_frontend_protocol import freeze_json, sha256_file as sha
from tools.fi1_union_development_replay import Guard

OUT = ROOT / ".local/fi1_native_symmetric_eligibility_development200_20261009_v1"
CONTROL = ROOT / ".local/fi1_nt_selection_control_development200_20261009_v1"
RELATION = ROOT / ".local/fi1_nt_relation_selection_development200_20261009_v2"
RISK = ROOT / '.local/fi1_replacement_risk_supervision_development200_20261009_v1'
ARMS = ("NT", "NMS", "C", "S")
VIEWS = ("semantic", "class_agnostic")


class ExperimentGuard(Guard):
    def check(self,path):
        p=Path(path).resolve()
        # Historical filenames containing val312 are scientific source files;
        # only their explicitly bound copies, never validation data, are allowed.
        allowed={OUT/'code_snapshot'/Path(q).relative_to(ROOT) for q in self.snapshot_sources}
        if p in allowed:return
        super().check(path)

    def __init__(self,c,raw_GT_allowed=False):
        super().__init__(c,raw_GT_allowed=raw_GT_allowed)
        self.snapshot_sources=c.get('code',{})


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    freeze_json(path, value)


def stage(name, **extra):
    ctl.write(OUT / "execution.json", dict(status="running", stage=name,
                                        updated=time.time(), **extra))


def symmetric_members(geometry, evidence):
    """Build R1 with expansion for every geometry, including native-only."""
    keys = [g['geometry_key'] for g in geometry]
    if len(set(keys)) != len(keys) or set(keys) != set(evidence):
        raise ValueError('geometry/evidence coverage differs')
    if len({k.split(':geometry:')[0] for k in keys}) > 1:
        raise ValueError('budget must be applied separately per scene')
    result = []
    for g in sorted(geometry, key=lambda x: x["geometry_key"]):
        key = g["geometry_key"]
        if evidence[key]['geometry_key'] != key or evidence[key]['points_sha256'] != g['points_sha256']:
            raise ValueError('own evidence geometry identity differs')
        score = direct_score(evidence[key])
        aliases = original_aliases(g)
        if g['point_count'] == 0:
            continue
        existing = {a["class_index"] for a in aliases if 0 <= a["class_index"] < 198}
        expanded = existing | set(np.flatnonzero(score > 0).tolist())
        for cls in sorted(expanded):
            original = [a for a in aliases if a["class_index"] == cls]
            native = [a["alias_key"] for a in original if a["source"] == "native"]
            result.append({
                "geometry_key": key, "class_index": int(cls),
                "score": float(score[cls]),
                "has_original_native_alias": bool(native),
                "native_alias_keys": native,
                "original_aliases": original,
                "expanded_class": cls not in existing,
            })
    return sorted(result, key=lambda r: (-r["score"], not r["has_original_native_alias"],
                                         r["geometry_key"], r["class_index"]))


def rank(members, scores=None):
    rows = members if scores is None else [{**m, "score": float(s)} for m, s in zip(members, scores)]
    return sorted(rows, key=lambda r: (-r["score"], not r["has_original_native_alias"],
                                       r["geometry_key"], r["class_index"]))


def load_contract():
    c = read(CONTROL / "contract.json")
    assert c["B_model_sha256"] == sha(c["B_model"])
    assert len(c["scenes"]) == 200
    return c


def prepare(c):
    if OUT.exists():
        raise RuntimeError(f"isolated output already exists: {OUT}")
    jobs = {}
    inputs = {}
    assert sklearn.__version__ == '1.3.2'
    assert shutil.disk_usage(ROOT).free >= 50 * 1024**3
    guard = ExperimentGuard(c); sys.addaudithook(guard.hook)
    code = dict(c['code'])
    ctl.base.verify_files(code)
    code.update({
        str(ROOT / "tools/run_fi1_native_symmetric_eligibility_development200.py"):
            sha(ROOT / "tools/run_fi1_native_symmetric_eligibility_development200.py"),
        str(ROOT / "tools/fi1_r0_r1_protocol.py"): sha(ROOT / "tools/fi1_r0_r1_protocol.py"),
        str(ROOT / "tools/fi1_r2_core.py"): sha(ROOT / "tools/fi1_r2_core.py"),
        str(ROOT / "tools/fi1_r3_core.py"): sha(ROOT / "tools/fi1_r3_core.py"),
        str(ROOT / 'tests/test_fi1_native_symmetric_eligibility.py'): sha(ROOT / 'tests/test_fi1_native_symmetric_eligibility.py'),
    })
    guard.snapshot_sources=code
    for root in (CONTROL, RELATION):
        for name in ('contract.json','predictions_frozen.json','completion_verified.json'):
            p = root / name; inputs[str(p)] = sha(p)
        assert read(root / 'completion_verified.json')['complete']
    assert len({s[:9] for s in c['scenes']}) == 200
    assert not {s[:9] for s in c['scenes']} & {s[:9] for s in c['reserve100']}
    labels = {}
    risk_seal=read(RISK/'labels_frozen.json')
    for s in c["scenes"]:
        b = c["bindings"][s]
        jobs[s] = dict(c['jobs'][s])
        jobs[s]['target_census']=str(RISK/'labels'/s/'target_census.json')
        labels[jobs[s]['target_census']]=risk_seal['outputs'][jobs[s]['target_census']]
        for key in ("geometry", "points", "evidence"):
            inputs[b[key]] = c["inputs"][b[key]]
        for name in ('nt_geometry.json','nt_members.jsonl','nt_features.npy','nt_scores.npy'):
            q = jobs[s][name]; inputs[q] = c['inputs'][q]
        for old_root, names in ((CONTROL, ("semantic_A.json", "semantic_B.json",
                                             "class_agnostic_A.json", "class_agnostic_B.json")),
                                (RELATION, ("semantic_C.json", "class_agnostic_C.json"))):
            seal = read(old_root / 'predictions_frozen.json')
            for name in names:
                p = old_root / "predictions" / s / name
                inputs[str(p)] = seal['outputs'][str(p)]
            rc = read(old_root / 'evaluation' / s / 'receipt.json')
            for name, h in rc['outputs'].items():
                if name.endswith('_matches.json'): labels[name] = h
    inputs[c["B_model"]] = c["B_model_sha256"]
    contract = {
        "version": OUT.name, "scenes": c["scenes"], "reserve100": c["reserve100"],
        "reporting_groups": c["reporting_groups"], "bindings": c["bindings"],
        "jobs": jobs, "inputs": inputs, "label_inputs": labels, "code": code,
        "B_model": c["B_model"], "B_model_sha256": c["B_model_sha256"],
        "budget": 800, "arms": ARMS, "views": VIEWS,
        "policy": "R0 legacy classes plus own fusion score > 0 for every geometry, including native-only; no other change",
        "controls": {"NT": "frozen prior NT", "NMS": "frozen same-class point IoU>0.95 NMS",
                     "C": "frozen OOF relation selector", "S": "native-symmetric eligibility with frozen B26"},
        "features": "Recomputed 26-dimensional B26 features and same-class competition context on the changed member pool",
        "GT_policy": "No GT is read before all S predictions are sealed; only existing frozen controls are read",
        "new_training": False, "new_visual_inference": False, "new_geometry": False,
        "B26_retrained": False, "production_modified": False,
        "reserve100_read": False, "heldout_read": False, "final_test_read": False,
        "scope": "Exposed development200; B26 and upstream have seen these scenes; not blind acceptance",
        "authorization": "User requested the native-symmetric eligibility fixed comparison.",
        "cached_census_contract_sha256":sha(RISK/'contract.json'),
        "failure_policy": "Identity/replay mismatch stops before GT. Policy is not tuned after AP. Lack of gain is diagnosed without threshold/top-k/model/budget searches.",
        "success": "Report both views AP/AP50/AP25 and old90/new110 versus NT, NMS and C. Replacing C requires exceeding its two views; no deployment or blind acceptance implied.",
        "diagnostics": "New native eligibility vs changed existing scores vs budget; valid same-class match (including duplicates), unmatched FP, ignore, below100; coverage gains/losses and fixed548 subgroup only after sealing.",
    }
    ctl.base.verify_files(inputs)
    OUT.mkdir(parents=True)
    write(OUT / "contract.json", contract)
    for q in code:
        dest = OUT / 'code_snapshot' / Path(q).relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(q,dest)
    result = subprocess.run([sys.executable,'-m','pytest','tests/test_fi1_native_symmetric_eligibility.py',
        'tests/test_fi1_union_development_replay.py','tests/test_fi1_class_agnostic_ablation.py','-q'],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (OUT/'tests.log').write_text(result.stdout); print(result.stdout,flush=True)
    assert result.returncode == 0
    write(OUT/'tests_verified.json',dict(passed=True,log_sha256=sha(OUT/'tests.log')))
    write(OUT / "execution.json", {"status": "prepared", "stage": "predictions", "resume_required": True})


def predictions(c):
    guard = ExperimentGuard(c); sys.addaudithook(guard.hook)
    ctl.base.verify_files(c['inputs']); ctl.base.verify_files(c['code'])
    assert read(OUT/'tests_verified.json')['passed']
    model = joblib.load(c["B_model"])
    outputs, receipts = {}, {}
    totals = Counter()
    for i, s in enumerate(c["scenes"], 1):
        stage("predictions", scene=s, completed=i - 1, total=200)
        d = OUT / "predictions" / s
        d.mkdir(parents=True, exist_ok=True)
        if (d / "receipt.json").exists():
            r = read(d / "receipt.json")
            ctl.base.verify_files(r['outputs'])
            outputs.update(r["outputs"]); receipts[str(d / "receipt.json")] = sha(d / "receipt.json")
            totals.update(r["counts"])
            continue
        spec = c["bindings"][s]
        gs, all_points, evidence = ctl.base.load_geometry(spec)
        pool = source_pool(gs)
        assert pool == read(c['jobs'][s]['nt_geometry.json'])
        points = {g["geometry_key"]: all_points[g["geometry_key"]] for g in pool}
        evidence = {k:evidence[k] for k in points}
        original = build_pair(pool,evidence,budget=800)['R1']['eligible']
        assert original == ctl.base.rows(c['jobs'][s]['nt_members.jsonl'])
        members = symmetric_members(pool, evidence)
        old_keys = {ctl.shared.key(m) for m in original}
        new_keys = {ctl.shared.key(m) for m in members} - old_keys
        native_only = {g['geometry_key'] for g in pool if all(a['source']=='native' for a in original_aliases(g))}
        assert all(k in native_only for k,cls in new_keys)
        assert {ctl.shared.key(m):m for m in members if ctl.shared.key(m) in old_keys} == {ctl.shared.key(m):m for m in original}
        assert all(not m['has_original_native_alias'] and not m['original_aliases'] and not m['native_alias_keys'] for m in members if ctl.shared.key(m) in new_keys)
        keys = [g["geometry_key"] for g in pool]
        x20 = {g["geometry_key"]: geometry_features(g, evidence[g["geometry_key"]]) for g in pool}
        own = np.asarray([x20[m["geometry_key"]][m["class_index"]] for m in members], dtype=np.float64)
        iou, coverage = pair_overlap([points[k] for k in keys], spec["point_count"])
        xold = np.column_stack((np.asarray([x20[m['geometry_key']][m['class_index']] for m in original]),
            context_features(original,keys,iou,coverage)))
        assert np.array_equal(xold,np.load(c['jobs'][s]['nt_features.npy'],allow_pickle=False))
        sold = np.clip(model.predict(xold),0,1)
        assert np.array_equal(sold,np.load(c['jobs'][s]['nt_scores.npy'],allow_pickle=False))
        assert rank(original,sold)[:800] == read(CONTROL/'predictions'/s/'semantic_A.json')
        x26 = np.column_stack((own, context_features(members, keys, iou, coverage)))
        assert x26.shape == (len(members), 26) and np.isfinite(x26).all()
        scores = np.clip(model.predict(x26), 0, 1)
        assert np.isfinite(scores).all()
        selected = rank(members, scores)[:800]
        # Reuse the exact frozen controls for comparison in the same evidence file.
        control_paths = {
            "NT": CONTROL / "predictions" / s / "semantic_A.json",
            "NMS": CONTROL / "predictions" / s / "semantic_B.json",
            "C": RELATION / "predictions" / s / "semantic_C.json",
        }
        local = {}
        np.savez_compressed(d/'pool.npz',x26=x26,scores=scores,old_scores=sold)
        local[str(d/'pool.npz')]=sha(d/'pool.npz')
        q=d/'members.jsonl'
        with q.open('x') as f:
            for m in members: f.write(json.dumps(m,ensure_ascii=False)+'\n')
        local[str(q)]=sha(q)
        index={ctl.shared.key(m):j for j,m in enumerate(members)}
        # Existing own20 is invariant; its context and score are intentionally not.
        old_ix=[index[ctl.shared.key(m)] for m in original]
        assert np.array_equal(x26[old_ix,:20],xold[:,:20])
        changed_context=int(np.any(x26[old_ix,20:]!=xold[:,20:],axis=1).sum())
        changed_scores=int((scores[old_ix]!=sold).sum())
        for arm, rows in (("NT", read(control_paths["NT"])), ("NMS", read(control_paths["NMS"])),
                          ("C", read(control_paths["C"])), ("S", selected)):
            path = d / ("semantic_" + arm + ".json"); write(path, rows); local[str(path)] = sha(path)
            if arm == "S":
                collapsed = ctl.collapse_members(rows)
                q = d / ("class_agnostic_" + arm + ".json"); write(q, collapsed); local[str(q)] = sha(q)
            else:
                src = (CONTROL if arm in ("NT", "NMS") else RELATION) / "predictions" / s / ("class_agnostic_" + ("A" if arm == "NT" else "B" if arm == "NMS" else "C") + ".json")
                rows2 = read(src); q = d / ("class_agnostic_" + arm + ".json"); write(q, rows2); local[str(q)] = sha(q)
        counts = dict(prebudget_members=len(members), original_members=len(original),added_members=len(new_keys),
                      native_geometries=len(native_only),
                      expanded_members=sum(m["expanded_class"] for m in members), selected_members=len(selected),
                      selected_expanded_classes=sum(m["expanded_class"] for m in selected),
                      selected_new_native_members=sum(ctl.shared.key(m) in new_keys for m in selected),
                      existing_context_changed=changed_context,existing_scores_changed=changed_scores,
                      selected_below100=sum(len(points[m['geometry_key']])<100 for m in selected))
        r = {"complete": True, "outputs": local, "counts": counts, "x26_shape": list(x26.shape),
             "x26_sha256": hashlib.sha256(x26.tobytes()).hexdigest(), "raw_GT_read": False,
             "policy_only_change": True}
        r.update(original_NT_full_replay_exact=True,existing_own20_exact=True,new_source_flags_real=True)
        write(d / "receipt.json", r); outputs.update(local); receipts[str(d / "receipt.json")] = sha(d / "receipt.json"); totals.update(counts)
        if i % 10 == 0: print(f"Predictions {i}/200", flush=True)
    assert not guard.gt_opened
    assert totals['original_members']==1756961 and totals['added_members']==1044739
    write(OUT / "predictions_frozen.json", {"complete": True, "predictions": 1600,
        "outputs": outputs, "scene_receipts": receipts, "counts": dict(totals),
        "raw_GT_read": False, "all_features_recomputed": True})


def verify_predictions(c):
    guard=ExperimentGuard(c);sys.addaudithook(guard.hook)
    seal=read(OUT/'predictions_frozen.json');assert seal['complete'] and seal['predictions']==1600
    ctl.base.verify_files(c['code']);ctl.base.verify_files(c['inputs'])
    ctl.base.verify_files(seal['outputs']);ctl.base.verify_files(seal['scene_receipts'])
    model=joblib.load(c['B_model']);totals=Counter();receipts={}
    for i,s in enumerate(c['scenes'],1):
        stage('independent_inference_verification',scene=s,completed=i-1,total=200)
        d=OUT/'verification'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():
            r=read(d/'receipt.json');totals.update(r['counts']);receipts[str(d/'receipt.json')]=sha(d/'receipt.json');continue
        spec=c['bindings'][s];gs,pp,evidence=ctl.base.load_geometry(spec)
        pool=source_pool(gs);points={g['geometry_key']:pp[g['geometry_key']] for g in pool}
        members=ctl.base.rows(OUT/'predictions'/s/'members.jsonl')
        expected={}; native_only=set()
        for g in pool:
            aliases=original_aliases(g);classes={a['class_index'] for a in aliases if 0<=a['class_index']<198}
            scores=direct_score(evidence[g['geometry_key']])
            classes.update(j for j in range(198) if scores[j]>0)
            if all(a['source']=='native' for a in aliases):native_only.add(g['geometry_key'])
            if not g['point_count']:continue
            for j in classes:
                aa=[a for a in aliases if a['class_index']==j];native=[a['alias_key'] for a in aa if a['source']=='native']
                legacy=j in {a['class_index'] for a in aliases}
                expected[g['geometry_key'],j]=dict(geometry_key=g['geometry_key'],class_index=j,score=float(scores[j]),
                    has_original_native_alias=bool(native),native_alias_keys=native,original_aliases=aa,expanded_class=not legacy)
        assert {ctl.shared.key(m):m for m in members}==expected
        assert members==sorted(expected.values(),key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))
        keys=[g['geometry_key'] for g in pool];iou=np.zeros((len(keys),len(keys)));cover=np.zeros_like(iou)
        # Independent sorted-array intersection rather than the sparse production implementation.
        for a,ka in enumerate(keys):
            for b in range(a,len(keys)):
                kb=keys[b];intersection=int(np.intersect1d(points[ka],points[kb],assume_unique=True).size)
                union=len(points[ka])+len(points[kb])-intersection
                iou[a,b]=iou[b,a]=intersection/union if union else 0.
                cover[a,b]=intersection/len(points[ka]) if len(points[ka]) else 0.
                cover[b,a]=intersection/len(points[kb]) if len(points[kb]) else 0.
        independent=independent_context(members,keys,iou,cover)
        with np.load(OUT/'predictions'/s/'pool.npz',allow_pickle=False) as z:
            x26=z['x26'];scores=z['scores']
        assert np.array_equal(x26[:,20:],independent),(s,'independent context')
        own={g['geometry_key']:geometry_features(g,evidence[g['geometry_key']]) for g in pool}
        assert np.array_equal(x26[:,:20],np.asarray([own[m['geometry_key']][m['class_index']] for m in members]))
        assert np.array_equal(scores,np.clip(model.predict(x26),0,1))
        scored=[{**m,'score':float(v)} for m,v in zip(members,scores)]
        selected=sorted(scored,key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:800]
        assert selected==read(OUT/'predictions'/s/'semantic_S.json')
        collapsed={}
        for m in sorted(selected,key=lambda m:(-m['score'],m['class_index'])):
            collapsed.setdefault(m['geometry_key'],dict(geometry_key=m['geometry_key'],score=m['score'],winning_class_index=m['class_index']))
        assert sorted(collapsed.values(),key=lambda m:(-m['score'],m['geometry_key']))==read(OUT/'predictions'/s/'class_agnostic_S.json')
        r=dict(complete=True,counts=dict(scenes=1,members=len(members)),independent_members=True,
            independent_point_intersections=True,independent_context_exact=True,independent_rank_and_collapse=True,raw_GT_read=False)
        write(d/'receipt.json',r);totals.update(r['counts']);receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
        if i%10==0:print(f'Independent inference {i}/200',flush=True)
    assert not guard.gt_opened
    write(OUT/'inference_verified.json',dict(complete=True,counts=dict(totals),scene_receipts=receipts,
        predictions_seal_sha256=sha(OUT/'predictions_frozen.json'),raw_GT_read=False))


def evaluate(c):
    guard=ExperimentGuard(c,raw_GT_allowed=True);sys.addaudithook(guard.hook)
    seal=read(OUT/'predictions_frozen.json');verify=read(OUT/'inference_verified.json')
    assert verify['complete'] and verify['predictions_seal_sha256']==sha(OUT/'predictions_frozen.json')
    ctl.base.verify_files(c['code']);ctl.base.verify_files(seal['outputs'])
    guard.gt_enabled=True;ctl.base.verify_files(c['label_inputs'])
    stage("matching", completed=0, total=200)
    ev = ctl.base.evaluator(); ca, valid, uid = ctl.class_agnostic_evaluator()
    receipts, total_gt = {}, 0
    for i, s in enumerate(c["scenes"], 1):
        spec = c["bindings"][s]; d = OUT / "evaluation" / s; d.mkdir(parents=True, exist_ok=True)
        if (d / "receipt.json").exists():
            r = read(d / "receipt.json");ctl.base.verify_files(r['outputs']); total_gt += r["GT"]; receipts[str(d / "receipt.json")] = sha(d / "receipt.json"); continue
        gt = np.loadtxt(spec["gt_path"], dtype=np.int64); assert len(gt) == spec["point_count"] and sha(spec["gt_path"]) == spec["gt_sha256"]
        _, points = ctl.shared.load_points(spec); sizes = {k: len(v) for k, v in points.items()}
        selected = {v: {a: read(OUT / "predictions" / s / (v + "_" + a + ".json")) for a in ARMS} for v in VIEWS}
        unique = {ctl.shared.key(m): m for a in ARMS for m in selected["semantic"][a]}; universe = list(unique.values())
        template = ctl.shared.assign(ev, universe, points, spec["gt_path"], len(gt))
        semantic = {a: ctl.project_multiclass(template, universe, selected["semantic"][a], sizes, ev.CLASS_LABELS, ev.VALID_CLASS_IDS) for a in ARMS}
        edges=sum(ctl.shared.check_match(semantic[a],selected['semantic'][a],points,gt,ev) for a in ARMS)
        for a,root,alias in (('NT',CONTROL,'A'),('NMS',CONTROL,'B'),('C',RELATION,'C')):
            assert ctl.shared.canonical_match(semantic[a])==ctl.shared.canonical_match(read(root/'evaluation'/s/f'semantic_{alias}_matches.json'))
        mapped = ctl.remap_gt(gt, valid, uid); ids, cnt = np.unique(gt, return_counts=True); keep = (ids >= 1000) & np.isin(ids // 1000, valid); ids, cnt = ids[keep], cnt[keep]
        assert np.array_equal(mapped==0,~((gt>=1000)&np.isin(gt//1000,valid)))
        def loader(path):
            assert Path(path).resolve()==Path(spec['gt_path']).resolve()
            return mapped.copy()
        ca.util_3d.load_ids = loader
        cg = {m["geometry_key"]: m for a in ARMS for m in selected["class_agnostic"][a]}; cu = list(cg.values())
        ct = ctl.shared.assign(ca, cu, points, spec["gt_path"], len(gt), True); ck = [m["geometry_key"] for m in cu if sizes[m["geometry_key"]] >= 100]
        agnostic = {a: ctl.project_matches(ct, ck, selected["class_agnostic"][a], ca.CLASS_LABELS[0]) for a in ARMS}
        edges+=sum(ctl.shared.check_match(agnostic[a],selected['class_agnostic'][a],points,mapped,ca,True) for a in ARMS)
        for a,root,alias in (('NT',CONTROL,'A'),('NMS',CONTROL,'B'),('C',RELATION,'C')):
            assert ctl.shared.canonical_match(agnostic[a])==ctl.shared.canonical_match(read(root/'evaluation'/s/f'class_agnostic_{alias}_matches.json'))
        local = {}
        for view, matchset in (("semantic", semantic), ("class_agnostic", agnostic)):
            for a in ARMS:
                q = d / f"{view}_{a}_matches.json"; ctl.write(q, ctl.base.serializable_matches(matchset[a])); local[str(q)] = sha(q)
        r = {"complete": True, "GT": int((cnt >= 100).sum()), "GT_sha256": spec["gt_sha256"], "outputs": local,
             "independent_point_census_edges":edges,"six_old_matches_exact":True,"raw_GT_read": True}
        write(d / "receipt.json", r); total_gt += r["GT"]; receipts[str(d / "receipt.json")] = sha(d / "receipt.json")
        if i % 10 == 0: print(f"Matching {i}/200", flush=True)
    assert total_gt == 4689
    write(OUT / "matches_frozen.json", {"complete": True, "GT": total_gt, "scene_receipts": receipts,
        "outputs_sealed_before_GT": True, "raw_GT_read": True})
    ap = {g: {v: {} for v in VIEWS} for g in c["reporting_groups"]}; arrays = {}
    for group, scenes in c["reporting_groups"].items():
        for view in VIEWS:
            evaluator = ev if view == "semantic" else ca
            for arm in ARMS:
                matches = {s: read(OUT / "evaluation" / s / f"{view}_{arm}_matches.json") for s in scenes}
                x, *_ = evaluator.evaluate_matches(matches)
                ap[group][view][arm] = ctl.base.metrics(evaluator, x)
                q = OUT / "evaluation" / f"{group}_{view}_{arm}_AP.npy"; np.save(q, x, allow_pickle=False); arrays[str(q)] = sha(q)
                if arm!='S':
                    alias='A' if arm=='NT' else 'B' if arm=='NMS' else 'C'
                    old=(CONTROL if arm in ('NT','NMS') else RELATION)/'evaluation'/f'{group}_{view}_{alias}_AP.npy'
                    assert np.array_equal(x,np.load(old,allow_pickle=False),equal_nan=True),(group,view,arm)
                print('AP',group,view,arm,ap[group][view][arm],flush=True)
    write(OUT / "evaluation_summary.json", {"complete": True, "AP": ap, "AP_arrays": arrays,
        "arms": ARMS, "groups": list(c["reporting_groups"]), "raw_GT_read": True})
    ctl.write(OUT/'execution.json',dict(status='evaluated',stage='diagnostics_and_final_verification',resume_required=True))


def member_status(members, match, sizes, class_ids, threshold=.5):
    """Original evaluator status, including valid matches that may be duplicates."""
    rows=[m for m in members if sizes[m['geometry_key']]>=100]
    pred={}
    for values in match['pred'].values():
        for p in values:
            m=rows[p['pred_id']]
            assert p['label_id']==class_ids[m['class_index']] and p['confidence']==m['score'] and p['vert_count']==sizes[m['geometry_key']]
            pred[ctl.shared.key(m)]=p
    assert len(pred)==len(rows)
    result={}
    for m in members:
        key=ctl.shared.key(m)
        if key not in pred:result[key]='below_min_points';continue
        p=pred[key]
        good=[g for g in p['matched_gt'] if g['intersection']/(g['vert_count']+p['vert_count']-g['intersection'])>threshold]
        if good:
            result[key]='valid_sameclass_match' if any(g['instance_id']>=1000 and g['vert_count']>=100 for g in good) else 'only_ignored_target_match'
        else:
            ignore=p['void_intersection']+sum(g['intersection'] for g in p['matched_gt'] if g['instance_id']<1000 or g['vert_count']<100)
            result[key]='unmatched_FP' if ignore/p['vert_count']<=threshold else 'ignored_by_fraction'
    return result


def diagnostics(c):
    guard=ExperimentGuard(c);guard.gt_enabled=True;sys.addaudithook(guard.hook)
    assert read(OUT/'matches_frozen.json')['complete']
    ev=ctl.base.evaluator();class_ids=list(map(int,ev.VALID_CLASS_IDS));thresholds=[float(t) for t in ev.opt['overlaps']]
    group_stats={};rows={};gaprows=[];p_receipts={}
    for i,s in enumerate(c['scenes'],1):
        stage('posthoc_diagnostics',scene=s,completed=i-1,total=200)
        job=c['jobs'][s];pool=read(job['nt_geometry.json']);sizes={g['geometry_key']:g['point_count'] for g in pool}
        census=read(job['target_census']);ious=census['ious'];targets=census['targets']
        old_members=ctl.base.rows(job['nt_members.jsonl']);members=ctl.base.rows(OUT/'predictions'/s/'members.jsonl')
        old_keys={ctl.shared.key(m) for m in old_members};new_keys={ctl.shared.key(m) for m in members}-old_keys
        selected={a:read(OUT/'predictions'/s/f'semantic_{a}.json') for a in ARMS}
        indices={ctl.shared.key(m):j for j,m in enumerate(members)}
        with np.load(OUT/'predictions'/s/'pool.npz',allow_pickle=False) as z:scores=z['scores'].copy();old_scores=z['old_scores'].copy()
        old_score_map={ctl.shared.key(m):float(v) for m,v in zip(old_members,old_scores)}
        old_new_scores=[{**m,'score':float(scores[indices[ctl.shared.key(m)]])} for m in old_members]
        context_only=rank(old_new_scores)[:800]
        hypothetical=rank([{**m,'score':old_score_map.get(ctl.shared.key(m),float(scores[j]))} for j,m in enumerate(members)])[:800]
        # These two hypothetical rankings are descriptive, not AP arms or tuned policies.
        status=member_status(selected['S'],read(OUT/'evaluation'/s/'semantic_S_matches.json'),sizes,class_ids)
        oldset={ctl.shared.key(m) for m in selected['NT']};sset={ctl.shared.key(m) for m in selected['S']}
        counts=Counter(read(OUT/'predictions'/s/'receipt.json')['counts'])
        counts.update(entered=len(sset-oldset),exited=len(oldset-sset),
            added_members_entered=sum(ctl.shared.key(m) in new_keys for m in selected['S']),
            old_rescoring_only_selection_changes=len({ctl.shared.key(m) for m in context_only}-oldset),
            retain_old_scores_plus_new_members_selection_changes=len({ctl.shared.key(m) for m in hypothetical}-oldset))
        distributions={tag:dict(Counter(status[ctl.shared.key(m)] for m in selected['S'] if condition(m))) for tag,condition in (
            ('all_S',lambda m:True),('new_native',lambda m:ctl.shared.key(m) in new_keys),
            ('existing',lambda m:ctl.shared.key(m) in old_keys),('entered',lambda m:ctl.shared.key(m) not in oldset))}
        eligibility_old={};eligibility_new={};selected_classes={a:{} for a in ARMS}
        for mm,out in ((old_members,eligibility_old),(members,eligibility_new)):
            for m in mm:out.setdefault(m['class_index'],set()).add(m['geometry_key'])
        for a in ARMS:
            for m in selected[a]:selected_classes[a].setdefault(m['class_index'],set()).add(m['geometry_key'])
        coverage={format(t,'.2f'):Counter() for t in thresholds};changes=[]
        for target,n in targets.items():
            cls=class_ids.index(int(target)//1000)
            best_geometry=max(ious[k][target] for k in sizes)
            best_old=max((ious[k][target] for k in eligibility_old.get(cls,set())),default=0.)
            best_new=max((ious[k][target] for k in eligibility_new.get(cls,set())),default=0.)
            best={a:max((ious[k][target] for k in selected_classes[a].get(cls,set())),default=0.) for a in ARMS}
            for t in thresholds:
                st=coverage[format(t,'.2f')];st['GT']+=1
                st['no_NT_geometry']+=best_geometry<=t
                st['old_correct_class_ineligible']+=best_geometry>t and best_old<=t
                st['S_correct_class_ineligible']+=best_geometry>t and best_new<=t
                for a in ARMS:st[a+'_covered']+=best[a]>t
                for a in ('NT','NMS','C'):
                    st['S_vs_'+a+'_gained']+=best['S']>t and best[a]<=t
                    st['S_vs_'+a+'_lost']+=best['S']<=t and best[a]>t
            if any((best['S']>t)!=(best['NT']>t) for t in thresholds):
                changes.append(dict(gt_id=int(target),class_index=cls,best=best))
            if best_geometry>.75 and best_old<=.75:
                good=[k for k in sizes if ious[k][target]>.75]
                candidates=[m for m in members if m['class_index']==cls and m['geometry_key'] in good]
                assert candidates and all(ctl.shared.key(m) in new_keys for m in candidates)
                best_score=max(scores[indices[ctl.shared.key(m)]] for m in candidates)
                rank_position=min(j+1 for j,m in enumerate(rank(members,scores)) if m['class_index']==cls and m['geometry_key'] in good)
                gaprows.append(dict(scene=s,gt_id=int(target),class_index=cls,best_geometry=best_geometry,
                    original_correct_class_iou=best_old,S_correct_class_eligible_iou=best_new,S_selected_iou=best['S'],
                    recovered_at_075=best['S']>.75,best_good_member_B26_score=float(best_score),best_good_member_rank=rank_position,
                    final_K_cutoff=float(selected['S'][-1]['score']),good_member_direct_scores=[m['score'] for m in candidates]))
        r=dict(counts=dict(counts),status=distributions,coverage={k:dict(v) for k,v in coverage.items()},changed_targets=changes)
        d=OUT/'diagnostics'/s/'summary.json';write(d,r);p_receipts[str(d)]=sha(d);rows[s]=r
        if i%20==0:print(f'Diagnostics {i}/200',flush=True)
    assert len(gaprows)==548
    for group,scenes in c['reporting_groups'].items():
        group_stats[group]=dict(counts=dict(sum((Counter(rows[s]['counts']) for s in scenes),Counter())),
            status={tag:dict(sum((Counter(rows[s]['status'][tag]) for s in scenes),Counter())) for tag in rows[scenes[0]]['status']},
            coverage={t:dict(sum((Counter(rows[s]['coverage'][t]) for s in scenes),Counter())) for t in rows[scenes[0]]['coverage']})
    gap=dict(n=548,recovered_at_075=sum(x['recovered_at_075'] for x in gaprows),all_correct_classes_now_eligible=True,
        B26_score_quantiles=np.quantile([x['best_good_member_B26_score'] for x in gaprows],[0,.25,.5,.75,1]).tolist(),
        best_rank_quantiles=np.quantile([x['best_good_member_rank'] for x in gaprows],[0,.25,.5,.75,1]).tolist(),rows=gaprows)
    write(OUT/'diagnostics/summary.json',dict(complete=True,groups=group_stats,fixed548_subgroup=gap,scene_outputs=p_receipts,
        caveat='Posthoc coverage/status and hypothetical score decompositions do not identify additive AP contributions or justify tuning.'))


def final(c):
    guard=ExperimentGuard(c);guard.gt_enabled=True;sys.addaudithook(guard.hook)
    ctl.base.verify_files(c['code']);ctl.base.verify_files(c['inputs']);ctl.base.verify_files(c['label_inputs'])
    for n in ('predictions_frozen.json','inference_verified.json','matches_frozen.json'):
        z=read(OUT/n);assert z['complete']
        for key in ('outputs','scene_receipts'):
            if key in z:ctl.base.verify_files(z[key])
    for s in c['scenes']:ctl.base.verify_files(read(OUT/'evaluation'/s/'receipt.json')['outputs'])
    summary=read(OUT/'evaluation_summary.json');ctl.base.verify_files(summary['AP_arrays'])
    diag=read(OUT/'diagnostics/summary.json');ctl.base.verify_files(diag['scene_outputs'])
    # Independent evaluator replay of every arm/group/view frozen AP array.
    for view in VIEWS:
        evaluator=ctl.base.evaluator() if view=='semantic' else ctl.class_agnostic_evaluator()[0]
        for a in ARMS:
            matches={s:read(OUT/'evaluation'/s/f'{view}_{a}_matches.json') for s in c['scenes']}
            uuids=[p['uuid'] for m in matches.values() for ps in m['pred'].values() for p in ps]
            assert len(uuids)==len(set(uuids))
            for group,scenes in c['reporting_groups'].items():
                x,*_=evaluator.evaluate_matches({s:matches[s] for s in scenes})
                assert np.array_equal(x,np.load(OUT/'evaluation'/f'{group}_{view}_{a}_AP.npy',allow_pickle=False),equal_nan=True)
                assert ctl.base.metrics(evaluator,x)==summary['AP'][group][view][a]
    ap=summary['AP'];full=ap['development200'];effects={}
    for group in ap:
        effects[group]={v:{a:{k:100*(ap[group][v]['S'][k]-ap[group][v][a][k]) for k in ('ap','ap50','ap25')} for a in ('NT','NMS','C')} for v in VIEWS}
    success=all(full[v]['S']['ap']>full[v]['C']['ap'] for v in VIEWS)
    six=all(full[v]['S'][k]>full[v]['C'][k] for v in VIEWS for k in ('ap','ap50','ap25'))
    decision=dict(complete=True,exceeds_C_both_main_AP=success,exceeds_C_all_six_metrics=six,effects_pp=effects,
        scope=c['scope'],production_modified=False,next='Member-level bidirectional risk supervision requires a separate frozen contract; no threshold/model/budget tuning of this arm.')
    write(OUT/'mechanism_decision.json',decision)
    lines=['# native 对称类别资格固定对照','',
        '仅改变 native 类别准入；NT 点集、自身 YOLO/Alpha、冻结 B26、K800 和评测器不变。新池重算 X26。',
        '开发200已暴露，不是整系统盲验收；生产未改变。','',
        '|方法|类别相关 AP/AP50/AP25 %|实际输出去类 AP/AP50/AP25 %|','|---|---|---|']
    for a in ARMS:
        vals=[' / '.join(f"{100*full[v][a][k]:.6f}" for k in ('ap','ap50','ap25')) for v in VIEWS]
        lines.append(f'|{a}|{vals[0]}|{vals[1]}|')
    stats=diag['groups']['development200'];sub=diag['fixed548_subgroup']
    lines+=['',f"原成员 {stats['counts']['original_members']}，新增 native 成员 {stats['counts']['added_members']}；进入 S800 的新 native 成员 {stats['counts']['selected_new_native_members']}。",
        f"原548个 IoU>.75 类别资格缺口全部获得正确类准入，其中最终 S800 恢复 {sub['recovered_at_075']} 个。覆盖恢复不等于 AP 收益。",
        f"同时超过 C 两口径主 AP：{success}；同时超过六指标：{six}。",
        '', '完整旧90/新110及全部 AP 数组见 evaluation_summary.json；资格、分数、预算、FP/ignore 和覆盖伤害见 diagnostics/summary.json。',
        '原NT全池/X26/分数/K800精确重放；新成员来源/独立交集/竞争上下文/评分/排序/去类封存核验；旧NT/NMS/C匹配与18个AP数组精确复现；24个本轮AP数组独立复算。',
        '预测先封存再读开发GT。未读保留100、开发55、验证312、最终测试或图例；无新训练、视觉推理或点集生成。',
        '', '本项未同时训练风险头或加入U/U′；成员级双向风险选择需另立具体合同。']
    path=OUT/'analysis.md';assert not path.exists();path.write_text('\n'.join(lines)+'\n')
    artifacts={str(OUT/n):sha(OUT/n) for n in ('contract.json','tests_verified.json','predictions_frozen.json',
        'inference_verified.json','matches_frozen.json','evaluation_summary.json','diagnostics/summary.json','mechanism_decision.json','analysis.md')}
    write(OUT/'completion_verified.json',dict(complete=True,resume_required=False,production_modified=False,
        no_new_training=True,no_new_geometry=True,no_new_visual_inference=True,predictions_before_GT=True,
        original_controls_exact=True,AP_arrays_independently_replayed=24,artifacts=artifacts,
        B_model_sha256=sha(c['B_model']),reserve100_read=False,heldout_read=False,final_test_read=False))
    ctl.write(OUT/'execution.json',dict(status='complete',stage='complete',resume_required=False,updated=time.time()))
    print('\n'.join(lines),flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("prepare", "predictions", "verify_predictions", "evaluate", "diagnostics", "final", "all"))
    args = p.parse_args()
    if args.stage=='all':
        # Each phase has its own immutable audit hook and process.  A feature
        # phase's no-GT guard must never be weakened by a later phase.
        for phase in ('prepare','predictions','verify_predictions','evaluate','diagnostics','final'):
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),phase],cwd=ROOT)
            if result.returncode:
                if OUT.exists():ctl.write(OUT/'execution.json',dict(status='failed',stage=phase,
                    exit_code=result.returncode,resume_required=True,updated=time.time()))
                raise SystemExit(result.returncode)
        return
    if args.stage in ("prepare", "all"): prepare(load_contract())
    c=read(OUT/'contract.json')
    if args.stage in ("predictions", "all"): predictions(c)
    if args.stage in ("verify_predictions", "all"): verify_predictions(c)
    if args.stage in ("evaluate", "all"): evaluate(c)
    if args.stage in ('diagnostics','all'):diagnostics(c)
    if args.stage in ('final','all'):final(c)


if __name__ == "__main__":
    with threadpool_limits(limits=4): main()
