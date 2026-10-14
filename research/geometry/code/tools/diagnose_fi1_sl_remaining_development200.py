#!/usr/bin/env python3
"""SL heldout error census and isolated existing-U budget probes; no new AP."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, json, shutil, sys, time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import joblib, torch
from threadpoolctl import threadpool_limits
from tools import run_fi1_s_fixed_L_rescore_development200 as old
from tools import run_fi1_native_symmetric_eligibility_development200 as src
from tools import fi1_sl_inference as inf
from tools.fi1_core_ablation import source_pool
from tools.fi1_r0_r1_protocol import original_aliases, direct_score
from tools.fi1_frontend_protocol import freeze_json, write_json, sha256_file as sha
OUT=ROOT/'.local/fi1_sl_remaining_errors_union_opportunities_development200_20261010_v3'
SEALED_PROBES=ROOT/'.local/fi1_sl_remaining_errors_union_opportunities_development200_20261010_v2'
U=ROOT/'.local/fi1_union_generation_objective_development200_20261007_v1'
ABL=ROOT/'.local/fi1_union_refinement_ablation_development200_20261007_v1'
S,L,SL=old.S,old.L,old.OUT
SINGLE=ROOT/'.local/fi1_sl_single200_dev55_20261010_v1'
read=src.read
key=lambda m:(m['geometry_key'],m['class_index'])


def missing_kind(geometry, eligible, selected, threshold):
    if geometry<=threshold:return 'no_NT_geometry'
    if eligible<=threshold:return 'no_correct_class_eligibility'
    if selected<=threshold:return 'outside_S800'
    return 'covered'


def prefix(values, score):
    """Ascending values: strict-above and tie counts, never arbitrary tie order."""
    x=np.sort(np.asarray(values,dtype=float))
    return dict(above=int(len(x)-np.searchsorted(x,score,side='right')),
                tied=int(np.searchsorted(x,score,side='right')-np.searchsorted(x,score,side='left')))


def frontier(rows, lookup, iou, classes):
    result=np.zeros(len(classes))
    for m in rows:
        q=iou[lookup[m['geometry_key']]]
        result=np.maximum(result,np.where(classes==m['class_index'],q,0))
    return result


def stage(name,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=name,updated=time.time(),**kw))
    print(name,kw,flush=True)


class Guard:
    """Exclude raw truth and all other datasets, even in diagnostic phase."""
    def __init__(self,scenes):self.scenes=set(scenes);self.inference=False
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=Path(args[0]).resolve();v=str(p)
        if any(x.startswith('ground_truth') for x in p.parts):raise PermissionError('raw GT excluded; use frozen intersections')
        if self.inference and (p.name in ('overlap.npz','target_census.json') or p.name.endswith('_matches.json')):
            raise PermissionError('cached labels excluded from budget probe inference')
        if 'val312' in v or 'reserve100' in v or '/scannet200_test/' in v:raise PermissionError('other evaluation datasets excluded')
        if SINGLE in p.parents and p.relative_to(SINGLE).parts[0] not in ('model','model_frozen.json'):
            raise PermissionError('development55 outputs excluded; only immutable model identity')
        import re
        found=re.findall(r'scene\d{4}_\d{2}',v)
        if any(s not in self.scenes for s in found):raise PermissionError('outside original development200')


def prepare():
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>=50*1024**3
    c=read(SL/'contract.json');u=read(U/'contract.json');a=read(ABL/'contract.json')
    assert c['scenes']==u['scenes']==a['scenes'] and len({s[:9] for s in c['scenes']})==200
    ev=src.ctl.base.evaluator();ts=[float(t) for t in ev.opt['overlaps'] if any(abs(t-x)<1e-8 for x in (.5,.75,.9))]
    assert len(ts)==3
    inputs={};labels={};jobs={}
    for r,n in ((SL,'contract.json'),(SL,'predictions_frozen.json'),(SL,'diagnostics.json'),(U,'contract.json'),(ABL,'contract.json')):
        inputs[str(r/n)]=sha(r/n)
    seals={r:read(r/'predictions_frozen.json')['outputs'] for r in (S,SL)}
    for s in c['scenes']:
        b=u['bindings'][s];jobs[s]=dict(binding={k:b[k] for k in ('geometry','points','evidence','point_count')},
            overlap=u['jobs'][s]['overlap'],census=c['jobs'][s]['target_census'],gate=a['jobs'][s]['gate'],
            regions=u['jobs'][s]['features'])
        for n in ('geometry','points','evidence'):inputs[b[n]]=u['inputs'][b[n]]
        for n in ('overlap','census'):
            q=jobs[s][n];labels[q]=(u['label_inputs'] if n=='overlap' else c['label_inputs'])[q]
        q=jobs[s]['gate'];inputs[q]=a['inputs'][q]
        q=jobs[s]['regions'];inputs[q]=u['inputs'][q]
        probe_seal=read(SEALED_PROBES/'probes'/s/'sealed.json')
        assert probe_seal['complete'] and not probe_seal['GT_used'] and probe_seal['fold']==c['folds'][s]
        inputs.update(probe_seal['outputs']);inputs[str(SEALED_PROBES/'probes'/s/'sealed.json')]=sha(SEALED_PROBES/'probes'/s/'sealed.json')
        for r,names in ((S,('members.jsonl','pool.npz','semantic_S.json')),(SL,('scores.npz','semantic_SL.json'))):
            for n in names:
                q=str(r/'predictions'/s/n);inputs[q]=seals[r][q]
        q=str(SL/'evaluation'/s/'semantic_SL_matches.json');labels[q]=sha(q)
    model=str(src.ctl.base.MODEL);inputs[model]=src.ctl.base.MODEL_HASH
    for f in range(5):
        rc=L/'models'/f'fold{f}'/'receipt.json';original_models=read(rc)['outputs']
        assert set(read(rc)['test_scenes'])=={s for s in c['scenes'] if c['folds'][s]==f}
        inputs[str(rc)]=sha(rc)
        for n in ('model.pt','normalization.npz'):
            q=str(L/'models'/f'fold{f}'/n);inputs[q]=original_models[q]
    seal=read(SINGLE/'model_frozen.json')
    for n,h in (('model.pt',seal['model_sha256']),('normalization.npz',seal['normalization_sha256'])):
        q=str(SINGLE/'model'/n);inputs[q]=h
    inputs[str(SINGLE/'model_frozen.json')]=sha(SINGLE/'model_frozen.json')
    code={str(ROOT/n):sha(ROOT/n) for n in ('tools/diagnose_fi1_sl_remaining_development200.py',
        'tests/test_fi1_sl_remaining_diagnostic.py','tools/fi1_sl_inference.py','tools/fi1_core_ablation.py',
        'tools/fi1_r0_r1_protocol.py','tools/fi1_r2_core.py','tools/fi1_r3_core.py',
        'tools/fi1_s_semantic_rank_residual.py','tools/run_fi1_native_symmetric_eligibility_development200.py')}
    contract=dict(version=OUT.name,scenes=c['scenes'],folds=c['folds'],groups=c['reporting_groups'],
        class_ids=c['class_ids'],thresholds=ts,jobs=jobs,inputs=inputs,label_inputs=labels,code=code,B_model=model,
        reference='Original fivefold heldout SL, fixed S800. Single200 model and development55 sealed and excluded from analysis.',
        classification='Strict evaluator thresholds; no NT geometry -> no correct class eligibility -> outside S800; covered separately.',
        ranking='Global same-class strict FP/structural duplicate prefixes, ties separate. Representatives chosen independently by each score, not fixed S. Diagnostic counts, not AP contribution.',
        union='All U, frozen screened U, all Uprime, unique masks vs NT-shared aliases; own fusion/YOLO/Alpha evidence. No inherited parent semantics.',
        cache_join='Full NTU intersections plus risk census for20 missing Uprime masks. Restore integer intersections from IoU, mask/GT sizes; strict checks and agreement with shared cached rows. No raw GT.',
        probe_replay='v2 completed and sealed all600 GT-free probes before final JSON serialization failed. v3 reuses these exact identities/scores, fixes numpy scalar serialization only, no reranking or retuning.',
        probe='Three isolated symmetric NTU/screened_NTU/NTUp pools. Rebuild complete x26 context, frozen B26 chooses800, original corresponding heldout L rescores. Record membership changes and coverage only, never publish predictions or compute AP. Distribution changed; original SL validation does not extend here.',
        region='GT intersections used only after GT-free probe identities/scores sealed. Parent subset verified using frozen pointsets; added target/other-valid/ignored-or-unknown partitions diagnostic only.',
        raw_GT_read=False,new_training=False,new_geometry=False,new_AP=False,production_modified=False,dev55_results_read=False)
    sys.addaudithook(Guard(c['scenes']).hook)
    src.ctl.base.verify_files(inputs);src.ctl.base.verify_files(labels);src.ctl.base.verify_files(code)
    OUT.mkdir();freeze_json(OUT/'contract.json',contract)
    for q in code:
        dst=OUT/'code_snapshot'/Path(q).relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dst)
    stage('prepared')


def semantic_witness(e,cls):
    f=direct_score(e);y=e['yolo'];a=e['alpha']
    return dict(fusion=float(f[cls]),fusion_rank=int(1+np.sum(f>f[cls])),fusion_ties=int(np.sum(f==f[cls])),
        yolo_available=y['available'],yolo_correct_votes=int(y['votes'][cls]) if y['available'] else 0,
        alpha_available=a['available'],alpha_rank=int(1+np.sum(np.asarray(a['cosine'])>a['cosine'][cls])) if a['available'] else None)


def probes(c,s,geometry,points,evidence,model,lmodel):
    d=OUT/'probes'/s;d.mkdir(parents=True,exist_ok=True);seal={}
    if SEALED_PROBES.exists():
        prior=read(SEALED_PROBES/'probes'/s/'sealed.json')
        for q,h in prior['outputs'].items():
            assert sha(q)==h==c['inputs'][q]
            dst=d/Path(q).name;freeze_json(dst,read(q));seal[str(dst)]=sha(dst)
        freeze_json(d/'sealed.json',dict(complete=True,fold=c['folds'][s],GT_used=False,outputs=seal))
        return
    keep=set(read(c['jobs'][s]['gate'])['kept'])
    for arm,allowed in (('NTU',('native','track','pair_union')),('screened_NTU',('native','track','pair_union')),('NTUp',('native','track','refined_union'))):
        pool=source_pool(geometry,allowed)
        if arm=='screened_NTU':
            pool=[]
            for g in geometry:pool.extend(source_pool([g],allowed if g['geometry_key'] in keep else ('native','track')))
        ks=[g['geometry_key'] for g in pool];ev={k:evidence[k] for k in ks}
        members=src.symmetric_members(pool,ev)
        own={g['geometry_key']:src.geometry_features(g,ev[g['geometry_key']]) for g in pool}
        oi,co=src.pair_overlap([points[k] for k in ks],c['jobs'][s]['binding']['point_count'])
        x=np.column_stack((np.asarray([own[m['geometry_key']][m['class_index']] for m in members]),src.context_features(members,ks,oi,co)))
        baseline=np.clip(model.predict(x),0,1)
        selected,_,indices,_=inf.rescore_pool(members,x,baseline,*lmodel)
        smap={key(members[i]):float(baseline[i]) for i in indices}
        # Internal probe metadata only. No mask/prediction exporter or AP evaluator.
        freeze_json(d/(arm+'.json'),dict(arm=arm,context_members=len(members),context_geometries=len(pool),
            selected=[dict(geometry_key=m['geometry_key'],class_index=m['class_index'],score=m['score'],
                           S_score=smap[key(m)]) for m in selected]))
        seal[str(d/(arm+'.json'))]=sha(d/(arm+'.json'))
    freeze_json(d/'sealed.json',dict(complete=True,fold=c['folds'][s],GT_used=False,outputs=seal))


def run(c):
    guard=Guard(c['scenes']);sys.addaudithook(guard.hook);torch.set_num_threads(4)
    src.ctl.base.verify_files(c['inputs']);src.ctl.base.verify_files(c['label_inputs']);src.ctl.base.verify_files(c['code'])
    models={f:inf.load_L(L/'models'/f'fold{f}'/'model.pt',L/'models'/f'fold{f}'/'normalization.npz') for f in range(5)}
    b26=joblib.load(c['B_model']);events=[];ranking=[];scene_stats={};catalog=[];targets=[]
    ts=c['thresholds'];class_ids=c['class_ids'];fp=defaultdict(list);dups=defaultdict(list);positive_rows=defaultdict(list)
    for count,s in enumerate(c['scenes']):
        stage('scene',scene=s,completed=count,total=200)
        job=c['jobs'][s];gs,pts,ev=src.ctl.base.load_geometry(job['binding'])
        guard.inference=True
        probes(c,s,gs,pts,ev,b26,models[c['folds'][s]])
        guard.inference=False
        with np.load(job['overlap'],allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        census=read(job['census'])
        # The older region cache predates20 independent refined geometries.
        # Exact cached IoU and both integer sizes uniquely restore intersection.
        missing=sorted({g['geometry_key'] for g in gs}-set(map(str,data['geometry_keys'])))
        for k in missing:
            g=next(g for g in gs if g['geometry_key']==k);n=g['point_count'];assert n>=100
            q=np.array([census['ious'][k][str(int(t))] for t in data['gt_ids']])
            raw=q*(n+data['gt_sizes'])/(1+q);counts=np.rint(raw).astype(np.int64)
            assert np.max(abs(raw-counts),initial=0)<1e-7 and np.all(counts<=np.minimum(n,data['gt_sizes']))
            rebuilt=counts/(n+data['gt_sizes']-counts);assert np.allclose(q,rebuilt,rtol=0,atol=1e-14)
            data['geometry_keys']=np.append(data['geometry_keys'],k);data['point_counts']=np.append(data['point_counts'],n)
            data['intersections']=np.vstack((data['intersections'],counts));data['iou']=np.vstack((data['iou'],rebuilt))
        ks=list(map(str,data['geometry_keys']));lookup={k:i for i,k in enumerate(ks)}
        ids=data['gt_ids'];classes=data['gt_classes'];sizes=data['point_counts'];inter=data['intersections'];truthsize=data['gt_sizes']
        assert set(ks)=={g['geometry_key'] for g in gs} and np.all(truthsize>=100)
        assert np.array_equal(classes,[class_ids.index(int(g)//1000) for g in ids])
        assert all(sizes[lookup[g['geometry_key']]]==g['point_count']==len(pts[g['geometry_key']]) for g in gs)
        io=inter/(sizes[:,None]+truthsize[None,:]-inter);io[sizes<100]=0
        assert np.allclose(io,data['iou'],rtol=0,atol=1e-14)
        assert {str(int(g)):int(n) for g,n in zip(ids,truthsize)}==census['targets']
        for k,v in census['ious'].items():assert np.allclose([v[str(int(g))] for g in ids],io[lookup[k]],rtol=0,atol=1e-14)
        nt=source_pool(gs);ntks={g['geometry_key'] for g in nt};ntidx=[lookup[k] for k in ntks]
        full=src.ctl.base.rows(S/'predictions'/s/'members.jsonl');selected=read(SL/'predictions'/s/'semantic_SL.json');srows=read(S/'predictions'/s/'semantic_S.json')
        with np.load(S/'predictions'/s/'pool.npz') as z:ss=z['scores'].copy()
        with np.load(SL/'predictions'/s/'scores.npz') as z:ls=z['full'].copy()
        assert len(full)==len(ss)==len(ls) and set(ntks)=={m['geometry_key'] for m in full}
        assert {key(m) for m in selected}=={key(m) for m in srows} and len(selected)==800
        ix={key(m):i for i,m in enumerate(full)};chosen={key(m) for m in selected}
        assert all(m['score']==ls[ix[key(m)]] for m in selected)
        ntbest=io[ntidx].max(axis=0,initial=0);eligible=frontier(full,lookup,io,classes);chosenbest=frontier(selected,lookup,io,classes)
        cabest=io[[lookup[m['geometry_key']] for m in selected]].max(axis=0,initial=0)
        keep=set(read(job['gate'])['kept']);sources={g['geometry_key']:{a['source'] for a in original_aliases(g)} for g in gs}
        arms={'U':{k for k in ks if 'pair_union' in sources[k]},'screened_U':keep,
              'Uprime':{k for k in ks if 'refined_union' in sources[k]}}
        assert keep<=arms['U']
        stats={format(t,'.2f'):Counter() for t in ts}
        statuses={format(t,'.2f'):src.member_status(selected,read(SL/'evaluation'/s/'semantic_SL_matches.json'),dict(zip(ks,map(int,sizes))),class_ids,t) for t in ts}
        # Derive reliable positive/duplicate identity independently of S's representative.
        for t in ts:
            tag=format(t,'.2f');positive=defaultdict(list)
            for m in selected:
                hits=np.flatnonzero((classes==m['class_index'])&(io[lookup[m['geometry_key']]]>t))
                assert len(hits)<=1, 'ambiguous target identity must not silently form duplicate group'
                if len(hits):positive[int(ids[hits[0]])].append(m)
                elif statuses[tag][key(m)]=='unmatched_FP':
                    fp[(tag,m['class_index'],'S')].append(float(ss[ix[key(m)]]));fp[(tag,m['class_index'],'SL')].append(m['score'])
            for gid,rows in positive.items():
                for arm,score in (('S',lambda m:float(ss[ix[key(m)]])),('SL',lambda m:m['score'])):
                    ordered=sorted(rows,key=lambda m:(-score(m),m['geometry_key']))
                    for m in ordered[1:]:dups[(tag,m['class_index'],arm)].append(score(m))
                    positive_rows[(tag,rows[0]['class_index'],arm)].append(score(ordered[0]))
            for a in ('S','SL'):
                assert sum(len(v) for v in positive.values())==sum(v=='valid_sameclass_match' for v in statuses[tag].values())
        regions=read(job['regions']);parents=defaultdict(set)
        for r in regions:parents[r['union_key']].add(r['parent_key'])
        for arm,ak in arms.items():
            for k in sorted(ak):
                i=lookup[k];g=next(g for g in gs if g['geometry_key']==k);j=int(io[i].argmax()) if len(ids) else None
                catalog.append(dict(scene=s,source=arm,geometry_key=k,shared_NT=k in ntks,point_count=int(sizes[i]),
                    best_target=int(ids[j]) if j is not None else None,best_iou=float(io[i,j]) if j is not None else 0,
                    own=semantic_witness(ev[k],int(classes[j])) if j is not None else None))
        for j,gid in enumerate(ids):
            ci=int(classes[j]);candidates=[i for i,m in enumerate(full) if m['class_index']==ci and io[lookup[m['geometry_key']],j]>.5]
            targets.append(dict(scene=s,gt_id=int(gid),class_index=ci,gt_points=int(truthsize[j]),NT_best=float(ntbest[j]),
                eligible_best=float(eligible[j]),SL_best=float(chosenbest[j]),SL_CA_best=float(cabest[j]),
                union_best={a:float(io[[lookup[k] for k in ak],j].max(initial=0)) for a,ak in arms.items()}))
            for t in ts:
                tag=format(t,'.2f');kind=missing_kind(ntbest[j],eligible[j],chosenbest[j],t);st=stats[tag];st[kind]+=1;st['GT']+=1
                good=[i for i in candidates if io[lookup[full[i]['geometry_key']],j]>t]
                row=dict(scene=s,fold=c['folds'][s],gt_id=int(gid),class_index=ci,threshold=t,kind=kind)
                if good:
                    bi=max(good,key=lambda i:ls[i]);row.update(best_full_L_score=float(ls[bi]),best_full_S_score=float(ss[max(good,key=lambda i:ss[i])]),
                        best_full_L_rank=int(1+np.sum(ls>ls[bi])),best_full_S_rank=int(1+np.sum(ss>max(ss[good]))))
                if kind=='covered':
                    rr=[m for m in selected if m['class_index']==ci and io[lookup[m['geometry_key']],j]>t]
                    row.update(S_best_score=max(float(ss[ix[key(m)]]) for m in rr),SL_best_score=max(m['score'] for m in rr),
                        correct_selected_members=len(rr))
                    ranking.append(row)
                else:events.append(row)
                for arm,ak in arms.items():
                    ug=[k for k in ak if io[lookup[k],j]>t]
                    if not ug:continue
                    st[arm+'_covered']+=1
                    if ntbest[j]>t:st[arm+'_already_NT_covered']+=1
                    else:
                        st[arm+'_new_over_NT']+=1
                        for n in (1,5,10):st[arm+'_new_fusion_top'+str(n)]+=any(semantic_witness(ev[k],ci)['fusion_rank']<=n for k in ug)
                        st[arm+'_new_own_positive']+=any(direct_score(ev[k])[ci]>0 for k in ug)
                        st[arm+'_new_yolo_vote']+=any(ev[k]['yolo']['available'] and ev[k]['yolo']['votes'][ci]>0 for k in ug)
                    if chosenbest[j]<=t:st[arm+'_SL_missing_opportunity']+=1
                    if ntbest[j]<=t or chosenbest[j]<=t:
                        witnesses=[]
                        for k in sorted(ug,key=lambda k:(-io[lookup[k],j],k)):
                            extra=[]
                            for p in sorted(parents[k]):
                                assert np.array_equal(np.intersect1d(pts[p],pts[k]),pts[p])
                                delta=inter[lookup[k]]-inter[lookup[p]];extra_size=len(pts[k])-len(pts[p])
                                assert np.all(delta>=0) and delta.sum()<=extra_size
                                extra.append(dict(parent=p,added_points=extra_size,target_points=int(delta[j]),
                                    other_valid_points=int(delta.sum()-delta[j]),ignored_or_unassigned_points=int(extra_size-delta.sum())))
                            witnesses.append(dict(geometry_key=k,iou=float(io[lookup[k],j]),shared_NT=k in ntks,own=semantic_witness(ev[k],ci),regions=extra))
                        events.append(dict(scene=s,gt_id=int(gid),class_index=ci,threshold=t,union_source=arm,
                            no_NT_geometry=ntbest[j]<=t,SL_missing=chosenbest[j]<=t,witnesses=witnesses))
        # Budget probes are already sealed before opening label caches above.
        probe_stats={}
        for arm in ('NTU','screened_NTU','NTUp'):
            pr=read(OUT/'probes'/s/(arm+'.json'));rows=pr['selected'];pk={key(m) for m in rows};entered=pk-chosen;exited=chosen-pk
            before={key(m):m for m in selected};after={key(m):m for m in rows};pfront=frontier(rows,lookup,io,classes)
            pc=io[[lookup[m['geometry_key']] for m in rows]].max(axis=0,initial=0)
            pd=dict(context_members=pr['context_members'],entered=len(entered),exited=len(exited),
                new_geometry_members=sum(k[0] not in ntks for k in entered),NT_members_entered=sum(k[0] in ntks for k in entered),
                coverage={},exited_witnesses=[])
            for t in ts:
                tag=format(t,'.2f');lost=np.flatnonzero((chosenbest>t)&(pfront<=t));gained=np.flatnonzero((chosenbest<=t)&(pfront>t))
                pd['coverage'][tag]=dict(semantic_gained=len(gained),semantic_lost=len(lost),
                    CA_gained=int(np.sum((cabest<=t)&(pc>t))),CA_lost=int(np.sum((cabest>t)&(pc<=t))),
                    true_new_NT_geometry_targets=int(np.sum((ntbest<=t)&(pfront>t))),
                    exited_sameclass_valid=sum(any((classes==ci)&(io[lookup[k]]>t)) for k,ci in exited),
                    entered_sameclass_valid=sum(any((classes==ci)&(io[lookup[k]]>t)) for k,ci in entered),
                    new_geometry_correct_members=sum(k not in ntks and any((classes==ci)&(io[lookup[k]]>t)) for k,ci in entered),
                    lost_targets=list(map(int,ids[lost])),gained_targets=list(map(int,ids[gained])))
            for k,ci in sorted(exited):
                hits=np.flatnonzero((classes==ci)&(io[lookup[k]]>ts[0]))
                if len(hits):pd['exited_witnesses'].append(dict(geometry_key=k,class_index=ci,SL_score=before[(k,ci)]['score'],
                    gt_ids=list(map(int,ids[hits])),ious=list(map(float,io[lookup[k],hits]))))
            probe_stats[arm]=pd
        scene_stats[s]=dict(errors={k:dict(v) for k,v in stats.items()},probe=probe_stats,
            identities=dict(NT=len(ntks),U=len(arms['U']),screened_U=len(keep),Uprime=len(arms['Uprime']),
                Uprime_shared_NT=len(arms['Uprime']&ntks),Uprime_unique=len(arms['Uprime']-ntks),full=len(gs)),GT=len(ids))
        freeze_json(OUT/'scenes'/s/'diagnosis.json',scene_stats[s])
    # Across-scene same-class rank pressure, with per-score representative recomputed.
    for r in ranking:
        tag=format(r['threshold'],'.2f');ci=r['class_index']
        for a in ('S','SL'):
            score=r[a+'_best_score'];r[a+'_FP_prefix']=prefix(fp[(tag,ci,a)],score)
            r[a+'_duplicate_prefix']=prefix(dups[(tag,ci,a)],score)
            r[a+'_positive_prefix']=prefix(positive_rows[(tag,ci,a)],score)
    for name,rows in (('targets',targets),('events',events),('ranking',ranking),('union_catalog',catalog)):
        q=OUT/(name+'.jsonl');q.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else None)+'\n' for r in rows))
    groups={}
    for name,scenes in c['groups'].items():
        d={tag:Counter() for tag in scene_stats[scenes[0]]['errors']};p={a:{tag:Counter() for tag in d} for a in ('NTU','screened_NTU','NTUp')};ident=Counter()
        for s in scenes:
            ident.update(scene_stats[s]['identities'])
            for tag,st in scene_stats[s]['errors'].items():d[tag].update(st)
            for a in p:
                for tag,st in scene_stats[s]['probe'][a]['coverage'].items():p[a][tag].update({k:v for k,v in st.items() if isinstance(v,int)})
        groups[name]=dict(errors={k:dict(v) for k,v in d.items()},probe={a:{k:dict(v) for k,v in pp.items()} for a,pp in p.items()},identities=dict(ident))
    pressure={}
    for t in ts:
        tag=format(t,'.2f');rr=[r for r in ranking if r['threshold']==t];summary=Counter(covered=len(rr))
        byclass={}
        for ci in sorted({r['class_index'] for r in rr}):
            cr=[r for r in rr if r['class_index']==ci];v={}
            for a in ('S','SL'):
                v[a]=dict(targets=len(cr),FP_above_mean=float(np.mean([r[a+'_FP_prefix']['above'] for r in cr])),
                    duplicate_above_mean=float(np.mean([r[a+'_duplicate_prefix']['above'] for r in cr])),
                    FP_pressured=sum(r[a+'_FP_prefix']['above']>0 for r in cr),
                    duplicate_pressured=sum(r[a+'_duplicate_prefix']['above']>0 for r in cr))
            byclass[str(ci)]=v
        for r in rr:
            for kind in ('FP','duplicate'):
                a=r['S_'+kind+'_prefix']['above'];b=r['SL_'+kind+'_prefix']['above']
                summary[kind+'_prefix_'+('improved' if b<a else 'worsened' if b>a else 'equal')]+=1
                summary['SL_'+kind+'_pressured']+=b>0
        pressure[tag]=dict(counts=dict(summary),classes=byclass,
            macro_FP_above_mean={a:float(np.mean([v[a]['FP_above_mean'] for v in byclass.values()])) for a in ('S','SL')},
            total_selected_FP=len([v for (tt,ci,a),vv in fp.items() if tt==tag and a=='SL' for v in vv]),
            total_selected_structural_duplicates=len([v for (tt,ci,a),vv in dups.items() if tt==tag and a=='SL' for v in vv]))
    freeze_json(OUT/'summary.json',dict(complete=True,groups=groups,rank_pressure=pressure,targets=len(targets),
        raw_GT_read=False,new_AP=False,model_training=False,probe_distribution_shift=True))
    outputs={str(q):sha(q) for q in OUT.rglob('*') if q.is_file() and 'code_snapshot' not in q.parts and q.name not in ('execution.json','completion_verified.json')}
    src.ctl.base.verify_files(c['inputs']);src.ctl.base.verify_files(c['label_inputs']);src.ctl.base.verify_files(c['code'])
    freeze_json(OUT/'completion_verified.json',dict(complete=True,scenes=200,targets=len(targets),outputs=outputs,inputs_unchanged=True,
        single_model_unchanged=True,dev55_results_read=False,raw_GT_read=False,new_AP=False))
    stage('completed',scenes=200,targets=len(targets))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));a=p.parse_args()
    with threadpool_limits(4):
        if a.mode=='prepare':prepare()
        else:run(read(OUT/'contract.json'))
