#!/usr/bin/env python3
"""Evaluate the sealed screened-U factorial selections in two AP views.

The member audit saved selections, not matching templates for B/C. Match the
union of this experiment's outputs afresh, then reuse the certified subset /
score projection. Old A/D matching and pooled AP must reproduce exactly.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from tools import run_fi1_core_ablation_train90 as base
from tools.fi1_class_agnostic_ablation import collapse_members,remap_gt,class_agnostic_evaluator,project_matches
from tools.fi1_union_development_replay import Guard,project_multiclass
from tools.fi1_union_loss_attribution import key
from tools.fi1_union_budget_sensitivity import conditional_effects
from tools.fi1_frontend_protocol import sha256_file as sha,freeze_json,write_json as write

SOURCE=ROOT/'.local/fi1_union_members_development200_20261007_v1'
CLASSIFIER=ROOT/'.local/fi1_union_classifier_development200_20261007_v1'
OUT=ROOT/'.local/fi1_union_screened_factorial_ap_development200_20261007_v1'
ARMS=('A','B','C','D')
VIEWS=('semantic','class_agnostic')
CONTROLS={'A':'NT','D':'classifier_nested'}
read=base.read


def stage(out,name,**extra):
    write(out/'execution.json',dict(status='running',stage=name,updated=time.time(),**extra))


def canonical_match(value):
    """Ignore random UUID values, retaining both directions and all AP fields."""
    if isinstance(value,list):return [canonical_match(v) for v in value]
    if isinstance(value,dict):return {k:canonical_match(v) for k,v in value.items() if k!='uuid'}
    return value


def load_points(spec):
    geometry=read(spec['geometry'])
    with np.load(spec['points'],allow_pickle=False) as z:
        points={g['geometry_key']:z[g['array_key']].copy() for g in geometry}
    for g in geometry:
        p=points[g['geometry_key']]
        assert p.ndim==1 and p.dtype.kind in 'iu' and np.array_equal(p,np.unique(p))
        assert len(p)==g['point_count'] and (not len(p) or 0<=p[0]<=p[-1]<spec['point_count'])
        assert hashlib.sha256(np.asarray(p,dtype='<i8').tobytes()).hexdigest()==g['points_sha256']
    return geometry,points


def prepare(out):
    assert not out.exists(),'fresh output required'
    p=read(SOURCE/'contract.json');accept=read(SOURCE/'acceptance.json')
    assert accept['complete'] and len(p['scenes'])==len({s[:9] for s in p['scenes']})==200
    assert not {s[:9] for s in p['scenes']}&{s[:9] for s in p['reserve100']}
    base.verify_files({str(SOURCE/n):h for n,h in accept['artifacts'].items()})
    inputs={str(SOURCE/n):sha(SOURCE/n) for n in ('contract.json','acceptance.json','evidence_frozen.json','final_verification.json')}
    seal=read(SOURCE/'evidence_frozen.json');labels={};jobs={}
    for s in p['scenes']:
        for n in ('geometry','points'):
            q=p['bindings'][s][n];inputs[q]=p['inputs'][q]
        jobs[s]={}
        for a in ARMS:
            q=SOURCE/'evidence'/s/(a+'_selected.json');inputs[str(q)]=seal['outputs'][str(q)]
            jobs[s][a]=str(q)
        rec=read(CLASSIFIER/'evaluation'/s/'receipt.json')
        for old in CONTROLS.values():
            q=CLASSIFIER/'evaluation'/s/(old+'_matches.json');labels[str(q)]=rec['outputs'][str(q)]
    oldsummary=read(CLASSIFIER/'evaluation/summary.json')
    labels[str(CLASSIFIER/'evaluation/summary.json')]=sha(CLASSIFIER/'evaluation/summary.json')
    for group in p['reporting_groups']:
        for old in CONTROLS.values():
            q=CLASSIFIER/'evaluation'/(group+'_'+old+'_AP.npy');labels[str(q)]=oldsummary['AP_arrays'][str(q)]
    names=('tools/run_fi1_union_screened_factorial_ap_development200.py',
           'tests/test_fi1_union_screened_factorial_ap_development200.py',
           'tools/verify_fi1_union_screened_factorial_ap_development200.py',
           'tools/fi1_class_agnostic_ablation.py','tools/fi1_union_development_replay.py',
           'tools/fi1_union_loss_attribution.py','tools/fi1_union_budget_sensitivity.py',
           'tools/fi1_frontend_protocol.py','tools/run_fi1_core_ablation_train90.py',
           'tools/run_fi1_union_loss_attribution_train90.py','tools/fi1_r3b_delta_diagnostic_core.py',
           'evaluate/scannet200/eval_semantic_instance.py','evaluate/scannet200/util_3d.py','evaluate/scannet200/scannet_constants.py')
    code={str(ROOT/n):sha(ROOT/n) for n in names}
    c=dict(version=out.name,scenes=p['scenes'],reserve100=p['reserve100'],bindings=p['bindings'],jobs=jobs,
           reporting_groups=p['reporting_groups'],inputs=inputs,label_inputs=labels,code=code,
           arms=list(ARMS),views=list(VIEWS),budget=800,kept_U=839,
           definitions=dict(A='NT original s0',B='NT original s0 + screened U s1',
                            C='NT only with screened NTU s1',D='screened NTU s1'),
           class_agnostic='Collapse already selected 800 members by exact pointset; maximum selected member score; class-index tie then geometry-key order; NO refill. Preserve near-duplicate different pointsets.',
           gt='Only frozen development200 raw GT after all1600 outputs sealed. Original198 valid classes map to one class, preserve instance boundaries and small instances, invalid labels mapped void.',
           controls='A/D semantic original match fields and 6 pooled AP arrays reproduce frozen NT/classifier_nested.',
           aggregation='Original evaluator pooled matches for200; report historical90/new110 separately, never average subgroup AP.',
           scope='Posthoc attribution of exposed development200. B26 saw597; not whole-system blind/generalization evidence. CA uses B26-derived scores, not isolated point-set quality.',
           no_new_model=True,no_new_rule=True,no_new_visual_inference=True,no_threshold_search=True,
           reserve100_read=False,heldout_read=False,final_test_read=False,production_modified=False)
    guard=Guard(c);sys.addaudithook(guard.hook)
    base.verify_files(inputs);base.verify_files(code)
    out.mkdir(parents=True);freeze_json(out/'contract.json',c)
    for q in code:
        dst=out/'code_snapshot'/Path(q).relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dst)
    write(out/'execution.json',dict(status='prepared',stage='tests'))


def predictions(out):
    c=read(out/'contract.json');guard=Guard(c);sys.addaudithook(guard.hook)
    assert read(out/'tests_verified.json')['passed'];base.verify_files(c['inputs']);base.verify_files(c['code'])
    outputs={};counts={v:{a:0 for a in ARMS} for v in VIEWS}
    for s in c['scenes']:
        for a in ARMS:
            rows=read(c['jobs'][s][a])
            assert len(rows)==800 and len({key(m) for m in rows})==800
            assert rows==base.rank(rows)
            collapsed=collapse_members(rows)
            independent={}
            for m in sorted(rows,key=lambda m:(-m['score'],m['class_index'])):
                independent.setdefault(m['geometry_key'],dict(geometry_key=m['geometry_key'],score=m['score'],winning_class_index=m['class_index']))
            assert collapsed==sorted(independent.values(),key=lambda m:(-m['score'],m['geometry_key']))
            for view,selected in zip(VIEWS,(rows,collapsed)):
                q=out/'predictions'/s/(view+'_'+a+'.json');freeze_json(q,selected);outputs[str(q)]=sha(q);counts[view][a]+=len(selected)
    assert not guard.gt_opened
    freeze_json(out/'predictions_frozen.json',dict(complete=True,outputs=outputs,counts=counts,predictions=1600,
        original_selections_exact=True,independent_collapse_exact=True,raw_GT_read=False,contract_sha256=sha(out/'contract.json')))
    print('All1600 outputs sealed; no new selections or models.',flush=True)


def assign(ev,rows,points,gt_path,point_count,agnostic=False):
    masks=np.zeros((point_count,len(rows)),dtype=bool)
    for i,m in enumerate(rows):masks[points[m['geometry_key']],i]=True
    gm,pm=ev.assign_instances_for_scan(dict(pred_masks=masks,
        pred_classes=np.asarray([0 if agnostic else m['class_index'] for m in rows],dtype=np.int64),
        pred_scores=np.asarray([m['score'] for m in rows])),str(gt_path))
    for ps in pm.values():
        for p in ps:
            for g in p['matched_gt']:g.pop('matched_pred',None)
    return dict(gt=gm,pred=pm)


def check_match(match,rows,points,gt,ev,agnostic=False):
    """Independent point census checks both directed match edges and void."""
    census={k:dict(zip(*np.unique(gt[p],return_counts=True))) for k,p in points.items()}
    pid={key(m) if not agnostic else m['geometry_key']:i for i,m in enumerate(m for m in rows if len(points[m['geometry_key']])>=100)}
    edges=0
    for ci,label in enumerate(ev.CLASS_LABELS):
        expected=[m for m in rows if (agnostic or m['class_index']==ci) and len(points[m['geometry_key']])>=100]
        ps=match['pred'][label];gs=match['gt'][label]
        assert len(expected)==len(ps)
        reverse={g['instance_id']:{p['uuid']:p for p in g['matched_pred']} for g in gs}
        for m,p in zip(expected,ps):
            cc=census[m['geometry_key']]
            assert p['pred_id']==pid[m['geometry_key'] if agnostic else key(m)]
            assert p['confidence']==m['score'] and p['vert_count']==len(points[m['geometry_key']])
            assert p['label_id']==int(ev.VALID_CLASS_IDS[ci])
            assert p['void_intersection']==sum(n for gid,n in cc.items() if gid//1000 not in ev.VALID_CLASS_IDS)
            hits={g['instance_id']:int(cc[g['instance_id']]) for g in gs if cc.get(g['instance_id'],0)>0}
            assert {g['instance_id']:g['intersection'] for g in p['matched_gt']}==hits
            for g in gs:
                r=reverse[g['instance_id']].get(p['uuid']);assert (r is not None)==(g['instance_id'] in hits)
                if r:
                    assert r=={**{k:v for k,v in p.items() if k!='matched_gt'},'intersection':hits[g['instance_id']]}
                    edges+=1
    return edges


def target_review(matches,ev,uid_mapping=None):
    """Maximum valid matching confidence by target, not exclusive TP assignment."""
    thresholds=list(map(float,ev.opt['overlaps']));front={}
    for a,match in matches.items():
        table={}
        for label,gs in match['gt'].items():
            for g in gs:
                if g['instance_id']<1000 or g['vert_count']<100:continue
                gid=g['instance_id'] if uid_mapping is None else uid_mapping[g['instance_id']]
                table[gid]=[max((p['confidence'] for p in g['matched_pred']
                    if p['intersection']/(g['vert_count']+p['vert_count']-p['intersection'])>t),default=None) for t in thresholds]
        front[a]=table
    assert all(set(front[a])==set(front['A']) for a in ARMS)
    counts={};cases=[]
    for before,after in (('A','B'),('A','C'),('C','D'),('B','D'),('A','D')):
        pair=after+'_minus_'+before;counts[pair]={}
        for j,t in enumerate(thresholds):
            ct=Counter(gained=0,lost=0,shared_score_up=0,shared_score_down=0)
            for gid in front[before]:
                b,a=front[before][gid][j],front[after][gid][j]
                ct['gained']+=b is None and a is not None;ct['lost']+=b is not None and a is None
                if b is not None and a is not None:ct['shared_score_up']+=a>b;ct['shared_score_down']+=a<b
            counts[pair][f'{t:.2f}']=dict(ct)
    for gid in sorted(front['A']):
        if any(front[a][gid]!=front['A'][gid] for a in ARMS[1:]):
            cases.append(dict(gt_id=gid,frontiers={a:front[a][gid] for a in ARMS}))
    return dict(GT=len(front['A']),thresholds=thresholds,counts=counts,changed_targets=cases)


def matching(out):
    c=read(out/'contract.json');guard=Guard(c,raw_GT_allowed=True);sys.addaudithook(guard.hook)
    seal=read(out/'predictions_frozen.json');assert seal['complete'] and seal['predictions']==1600
    base.verify_files(seal['outputs']);base.verify_files(c['code']);guard.gt_enabled=True;base.verify_files(c['label_inputs'])
    ev=base.evaluator();ca,valid,uid=class_agnostic_evaluator();total=0;receipts={}
    for n,s in enumerate(c['scenes'],1):
        stage(out,'original_matching',scene=s,completed=n-1,total=200);d=out/'evaluation'/s
        if (d/'receipt.json').exists():r=read(d/'receipt.json');base.verify_files(r['outputs'])
        else:
            spec=c['bindings'][s];assert sha(spec['gt_path'])==spec['gt_sha256']
            gt=np.loadtxt(spec['gt_path'],dtype=np.int64);assert len(gt)==spec['point_count']
            _,points=load_points(spec);sizes={k:len(v) for k,v in points.items()}
            selected={v:{a:read(out/'predictions'/s/(v+'_'+a+'.json')) for a in ARMS} for v in VIEWS}
            unique={key(m):m for a in ARMS for m in selected['semantic'][a]};universe=[unique[k] for k in sorted(unique)]
            template=assign(ev,universe,points,spec['gt_path'],len(gt))
            semantic={a:project_multiclass(template,universe,selected['semantic'][a],sizes,ev.CLASS_LABELS,ev.VALID_CLASS_IDS) for a in ARMS}
            edges=sum(check_match(semantic[a],selected['semantic'][a],points,gt,ev) for a in ARMS)
            for a,old in CONTROLS.items():
                assert canonical_match(semantic[a])==canonical_match(read(CLASSIFIER/'evaluation'/s/(old+'_matches.json'))),('old match differs',s,a)
            mapped=remap_gt(gt,valid,uid)
            ids,cnt=np.unique(gt,return_counts=True);keep=(ids>=1000)&np.isin(ids//1000,valid);ids,cnt=ids[keep],cnt[keep]
            assert sorted(np.unique(mapped[mapped>0],return_counts=True)[1].tolist())==sorted(cnt.tolist())
            assert np.array_equal(mapped==0,~((gt>=1000)&np.isin(gt//1000,valid)))
            reverse={uid*1000+i+1:int(g) for i,g in enumerate(ids)}
            def mapped_loader(path):
                assert Path(path).resolve()==Path(spec['gt_path']).resolve()
                return mapped.copy()
            ca.util_3d.load_ids=mapped_loader
            cg={m['geometry_key']:m for a in ARMS for m in selected['class_agnostic'][a]};cu=[cg[k] for k in sorted(cg)]
            ct=assign(ca,cu,points,spec['gt_path'],len(gt),True)
            ck=[m['geometry_key'] for m in cu if sizes[m['geometry_key']]>=100]
            agnostic={a:project_matches(ct,ck,selected['class_agnostic'][a],ca.CLASS_LABELS[0]) for a in ARMS}
            edges+=sum(check_match(agnostic[a],selected['class_agnostic'][a],points,mapped,ca,True) for a in ARMS)
            d.mkdir(parents=True,exist_ok=True)
            for view,matches in zip(VIEWS,(semantic,agnostic)):
                for a in ARMS:write(d/(view+'_'+a+'_matches.json'),base.serializable_matches(matches[a]))
            reviews=dict(semantic=target_review(semantic,ev),class_agnostic=target_review(agnostic,ca,reverse))
            assert reviews['semantic']['GT']==reviews['class_agnostic']['GT']==int((cnt>=100).sum())
            write(d/'target_review.json',reviews)
            r=dict(complete=True,GT=int((cnt>=100).sum()),all_instances=len(ids),GT_sha256=spec['gt_sha256'],
                direct_census_edges=edges,old_semantic_controls_exact=True,
                outputs={str(q):sha(q) for q in d.iterdir() if q.is_file()})
            freeze_json(d/'receipt.json',r)
        total+=r['GT'];receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
        if n%10==0:print(f'Matches {n}/200',flush=True)
    assert total==4689
    freeze_json(out/'matches_frozen.json',dict(complete=True,GT=total,scene_receipts=receipts,raw_GT_read=True,
        outputs_sealed_before_GT=True,reserve100_read=False,heldout_read=False,final_test_read=False))


def evaluate(out):
    c=read(out/'contract.json');guard=Guard(c);guard.gt_enabled=True;sys.addaudithook(guard.hook)
    seal=read(out/'matches_frozen.json');assert seal['complete'];base.verify_files(seal['scene_receipts']);base.verify_files(c['code'])
    AP={g:{v:{} for v in VIEWS} for g in c['reporting_groups']};arrays={};counts={}
    for s in c['scenes']:base.verify_files(read(out/'evaluation'/s/'receipt.json')['outputs'])
    for view in VIEWS:
        ev=base.evaluator() if view=='semantic' else class_agnostic_evaluator()[0]
        for a in ARMS:
            matches={s:read(out/'evaluation'/s/(view+'_'+a+'_matches.json')) for s in c['scenes']}
            uuids=[p['uuid'] for m in matches.values() for ps in m['pred'].values() for p in ps];assert len(set(uuids))==len(uuids)
            counts[view+'_'+a]=len(uuids)
            for group,scenes in c['reporting_groups'].items():
                stage(out,'pooled_AP',view=view,arm=a,group=group)
                x,*_=ev.evaluate_matches({s:matches[s] for s in scenes});AP[group][view][a]=base.metrics(ev,x)
                dest=out/'evaluation'/(group+'_'+view+'_'+a+'_AP.npy');np.save(dest,x,allow_pickle=False);arrays[str(dest)]=sha(dest)
                if view=='semantic' and a in CONTROLS:
                    assert np.array_equal(x,np.load(CLASSIFIER/'evaluation'/(group+'_'+CONTROLS[a]+'_AP.npy'),allow_pickle=False),equal_nan=True)
            print(view,a,AP['development200'][view][a],flush=True)
    effects={g:{v:conditional_effects(AP[g][v]) for v in VIEWS} for g in AP}
    reviews={s:read(out/'evaluation'/s/'target_review.json') for s in c['scenes']}
    agg={}
    for group,scenes in c['reporting_groups'].items():
        agg[group]={}
        for v in VIEWS:
            agg[group][v]={}
            for pair in reviews[scenes[0]][v]['counts']:
                agg[group][v][pair]={t:dict(sum((Counter(reviews[s][v]['counts'][pair][t]) for s in scenes),Counter()))
                    for t in reviews[scenes[0]][v]['counts'][pair]}
    freeze_json(out/'evaluation/summary.json',dict(complete=True,AP=AP,effects_pp=effects,AP_units='fraction',
        AP_arrays=arrays,GT=4689,evaluated_prediction_counts=counts,target_changes=agg,scope=c['scope'],
        old_control_arrays_exact=6,raw_GT_read=True,reserve100_read=False,heldout_read=False,final_test_read=False,production_modified=False))
    write(out/'execution.json',dict(status='complete',stage='evaluated'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('prepare','predictions','matching','evaluate'));ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args();out=args.output.resolve()
    with threadpool_limits(limits=1):globals()[args.phase](out)


if __name__=='__main__':main()
