#!/usr/bin/env python3
"""Independent cached reconstruction of SL error categories and U opportunities."""
from pathlib import Path
from collections import Counter
import json, sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fi1_frontend_protocol import freeze_json, sha256_file as sha
OUT=ROOT/'.local/fi1_sl_remaining_errors_union_opportunities_development200_20261010_v3'
S=ROOT/'.local/fi1_native_symmetric_eligibility_development200_20261009_v1'
SL=ROOT/'.local/fi1_s_fixed_L_rescore_development200_20261010_v1'
L=ROOT/'.local/fi1_s_semantic_rank_residual_development200_20261010_v2'
def read(q):return json.loads(Path(q).read_text())
def rows(q):return [json.loads(x) for x in Path(q).read_text().splitlines()]
def identity(m):return (m['geometry_key'],int(m['class_index']))


def replay_probes(c):
    import joblib, torch
    from threadpoolctl import threadpool_limits
    from tools import run_fi1_native_symmetric_eligibility_development200 as src
    from tools.fi1_core_ablation import source_pool
    from tools import fi1_sl_inference as inf
    b26=joblib.load(c['B_model']);torch.set_num_threads(4);checked=[]
    # Deterministic first/middle/last scene, fixed before inspecting probe gains.
    for s in [c['scenes'][0],c['scenes'][99],c['scenes'][199]]:
        gs,pts,ev=src.ctl.base.load_geometry(c['jobs'][s]['binding']);keep=set(read(c['jobs'][s]['gate'])['kept'])
        lm=inf.load_L(L/'models'/f"fold{c['folds'][s]}"/'model.pt',L/'models'/f"fold{c['folds'][s]}"/'normalization.npz')
        with threadpool_limits(4):
            for arm,allowed in (('NT',('native','track')),('NTU',('native','track','pair_union')),('screened_NTU',('native','track','pair_union')),('NTUp',('native','track','refined_union'))):
                pool=[]
                for g in gs:pool.extend(source_pool([g],('native','track') if arm=='screened_NTU' and g['geometry_key'] not in keep else allowed))
                ks=[g['geometry_key'] for g in pool];evidence={k:ev[k] for k in ks};members=src.symmetric_members(pool,evidence)
                own={g['geometry_key']:src.geometry_features(g,evidence[g['geometry_key']]) for g in pool}
                ov,cov=src.pair_overlap([pts[k] for k in ks],c['jobs'][s]['binding']['point_count'])
                x26=np.column_stack((np.asarray([own[m['geometry_key']][m['class_index']] for m in members]),src.context_features(members,ks,ov,cov)))
                baseline=np.clip(b26.predict(x26),0,1);selection=src.rank(members,baseline)[:800]
                ind={identity(m):i for i,m in enumerate(members)};indices=np.array([ind[identity(m)] for m in selection])
                x28=np.column_stack((x26,baseline,inf.core.base_logits(baseline))).astype(np.float32)
                scores,_=inf.score_selected(*lm[:1],x28,baseline,indices,*lm[1:])
                slscores={identity(m):float(v) for m,v in zip(selection,scores)}
                if arm=='NT':
                    actual=read(SL/'predictions'/s/'semantic_SL.json');assert {identity(m) for m in actual}==set(slscores)
                    assert all(m['score']==slscores[identity(m)] for m in actual)
                    with np.load(S/'predictions'/s/'pool.npz') as z:assert np.array_equal(baseline,z['scores']) and np.array_equal(x26,z['x26'])
                else:
                    probe=read(OUT/'probes'/s/(arm+'.json'));actual=probe['selected'];assert {identity(m) for m in actual}==set(slscores)
                    assert all(m['score']==slscores[identity(m)] and m['S_score']==baseline[ind[identity(m)]] for m in actual)
                    assert len(members)==probe['context_members']
                checked.append(dict(scene=s,arm=arm,members=len(members),selected=800,exact=True))
    return checked


def main():
    c=read(OUT/'contract.json');done=read(OUT/'completion_verified.json');assert done['complete']
    mutable_log=str(OUT/'run.log');log_receipt_diff=sha(mutable_log)!=done['outputs'][mutable_log]
    # The terminal completion line was emitted after the in-process output seal.
    # Retain that historical seal; audit every scientific file and seal final log separately.
    for part in (done['outputs'],c['inputs'],c['label_inputs'],c['code']):
        for q,h in part.items():
            if part is done['outputs'] and q==mutable_log:continue
            assert sha(q)==h,q
    targetrows=rows(OUT/'targets.jsonl');events=rows(OUT/'events.jsonl');ranking=rows(OUT/'ranking.jsonl')
    targetlookup={(r['scene'],r['gt_id']):r for r in targetrows};ranklookup={(r['scene'],r['gt_id'],r['threshold']):r for r in ranking}
    missinglookup={(r['scene'],r['gt_id'],r['threshold']):r for r in events if 'kind' in r}
    assert len(targetlookup)==len(targetrows)==4689
    oldc=read(SL/'contract.json');baseline=read(SL/'diagnostics.json');summary=read(OUT/'summary.json')
    totals={format(t,'.2f'):Counter() for t in c['thresholds']};inputmembers=0;geom=0;probe_counts=Counter()
    fp={};duplicates={}
    for s in c['scenes']:
        nt=read(oldc['jobs'][s]['nt_geometry.json']);ntkeys={g['geometry_key'] for g in nt};geom+=len(ntkeys)
        full=rows(S/'predictions'/s/'members.jsonl');selected=read(SL/'predictions'/s/'semantic_SL.json');inputmembers+=len(full)
        index={identity(m):i for i,m in enumerate(full)}
        with np.load(S/'predictions'/s/'pool.npz') as z:ss=z['scores'].copy()
        with np.load(SL/'predictions'/s/'scores.npz') as z:ls=z['full'].copy()
        with np.load(L/'predictions'/s/'scores.npz') as z:assert np.array_equal(ls,z['full'])
        assert len(full)==len(ss)==len(ls) and len(selected)==800
        assert set(identity(m) for m in selected)==set(identity(m) for m in read(S/'predictions'/s/'semantic_S.json'))
        assert all(m['score']==ls[index[identity(m)]] and m['score']>0 for m in selected)
        with np.load(c['jobs'][s]['overlap']) as z:
            keys=list(map(str,z['geometry_keys']));ids=list(map(int,z['gt_ids']));cs=list(map(int,z['gt_classes']));iou=z['iou'].copy();size=z['point_counts'].copy();inter=z['intersections'].copy();gsize=z['gt_sizes'].copy()
        gs=read(c['jobs'][s]['binding']['geometry']);risk=read(c['jobs'][s]['census'])
        for g in gs:
            k=g['geometry_key']
            if k in keys:continue
            values=np.array([risk['ious'][k][str(gid)] for gid in ids]);n=g['point_count']
            counts=np.rint(values*(n+gsize)/(1+values)).astype(np.int64)
            exact=counts/(n+gsize-counts);assert np.allclose(values,exact,atol=1e-14,rtol=0)
            keys.append(k);iou=np.vstack((iou,exact));inter=np.vstack((inter,counts));size=np.append(size,n)
        assert np.allclose((inter/(size[:,None]+gsize-inter))[size>=100],iou[size>=100],atol=1e-14,rtol=0)
        lk={k:i for i,k in enumerate(keys)};iou[size<100]=0
        eligible={ci:{m['geometry_key'] for m in full if m['class_index']==ci} for ci in set(cs)}
        selectedclasses={ci:{m['geometry_key'] for m in selected if m['class_index']==ci} for ci in set(cs)}
        def best(keys,j):return max((float(iou[lk[k],j]) for k in keys),default=0.)
        aliases={}
        for g in gs:
            alias={m['candidate_source'] for m in g['legacy_members']}
            if any(p['append_only'] for p in g['dv3_plans']):alias.add('refined_union')
            aliases[g['geometry_key']]=alias
        arms={'U':{k for k,a in aliases.items() if 'pair_union' in a},'Uprime':{k for k,a in aliases.items() if 'refined_union' in a},'screened_U':set(read(c['jobs'][s]['gate'])['kept'])}
        per_scene={format(t,'.2f'):Counter() for t in c['thresholds']}
        for j,gid in enumerate(ids):
            tr=targetlookup[(s,gid)];ci=cs[j]
            q=[best(ntkeys,j),best(eligible.get(ci,set()),j),best(selectedclasses.get(ci,set()),j)]
            assert [tr['NT_best'],tr['eligible_best'],tr['SL_best']]==q
            for arm,ak in arms.items():assert tr['union_best'][arm]==best(ak,j)
            for t in c['thresholds']:
                tag=format(t,'.2f');test=[v>t for v in q];name=('no_NT_geometry' if not test[0] else 'no_correct_class_eligibility' if not test[1] else 'outside_S800' if not test[2] else 'covered')
                per_scene[tag][name]+=1;totals[tag][name]+=1;totals[tag]['GT']+=1
                if name=='covered':assert (s,gid,t) in ranklookup
                else:assert missinglookup[(s,gid,t)]['kind']==name
                for arm,ak in arms.items():
                    if best(ak,j)>t:
                        per_scene[tag][arm+'_covered']+=1
                        per_scene[tag][arm+('_already_NT_covered' if test[0] else '_new_over_NT')]+=1
                        if not test[2]:per_scene[tag][arm+'_SL_missing_opportunity']+=1
        originalmatches=read(SL/'evaluation'/s/'semantic_SL_matches.json')
        filtered=[m for m in selected if size[lk[m['geometry_key']]]>=100]
        preds={int(p['pred_id']):p for vv in originalmatches['pred'].values() for p in vv}
        assert len(filtered)==len(preds)
        for t in c['thresholds']:
            tag=format(t,'.2f');positive={}
            for pi,m in enumerate(filtered):
                p=preds[pi];assert m['score']==p['confidence'] and c['class_ids'][m['class_index']]==p['label_id']
                qualifying=[];anygood=False
                for g in p['matched_gt']:
                    value=g['intersection']/(g['vert_count']+p['vert_count']-g['intersection'])
                    if value>t:
                        anygood=True
                        if g['instance_id']>=1000 and g['vert_count']>=100:qualifying.append(g['instance_id'])
                if qualifying:
                    assert len(qualifying)==1
                    positive.setdefault(qualifying[0],[]).append(m)
                elif not anygood:
                    ignored=p['void_intersection']+sum(g['intersection'] for g in p['matched_gt'] if g['instance_id']<1000 or g['vert_count']<100)
                    if ignored/p['vert_count']<=t:
                        for a,sc in (('S',ss[index[identity(m)]]),('SL',m['score'])):fp.setdefault((tag,m['class_index'],a),[]).append(float(sc))
            for rr in positive.values():
                for a in ('S','SL'):
                    scores=sorted([float(ss[index[identity(m)]]) if a=='S' else m['score'] for m in rr],reverse=True)
                    duplicates.setdefault((tag,rr[0]['class_index'],a),[]).extend(scores[1:])
        sd=read(OUT/'scenes'/s/'diagnosis.json')
        for tag,st in per_scene.items():
            for k,v in st.items():assert sd['errors'][tag][k]==v
        before={identity(m) for m in selected}
        for arm in ('NTU','screened_NTU','NTUp'):
            probe=read(OUT/'probes'/s/(arm+'.json'));seal=read(OUT/'probes'/s/'sealed.json');q=OUT/'probes'/s/(arm+'.json')
            assert seal['outputs'][str(q)]==sha(q) and seal['GT_used'] is False
            after={identity(m) for m in probe['selected']};assert len(after)==800
            probe_counts[arm+'_entered']+=len(after-before);probe_counts[arm+'_new_geometry_entered']+=sum(k not in ntkeys for k,ci in after-before)
            for t in c['thresholds']:
                oldgood=set();newgood=set()
                for members,dest in ((before,oldgood),(after,newgood)):
                    for k,ci in members:
                        for j in np.flatnonzero((np.array(cs)==ci)&(iou[lk[k]]>t)):dest.add(ids[j])
                ps=sd['probe'][arm]['coverage'][format(t,'.2f')]
                assert set(ps['lost_targets'])==oldgood-newgood and set(ps['gained_targets'])==newgood-oldgood
                assert ps['semantic_lost']==len(oldgood-newgood) and ps['semantic_gained']==len(newgood-oldgood)
    assert inputmembers==2801700 and geom==14150
    for tag,st in totals.items():
        assert sum(st[k] for k in ('no_NT_geometry','no_correct_class_eligibility','outside_S800','covered'))==4689
        assert st['covered']==baseline['groups']['development200']['SL']['absolute_coverage'][tag]['semantic']
    for r in ranking:
        tag=format(r['threshold'],'.2f');ci=r['class_index']
        for a in ('S','SL'):
            score=r[a+'_best_score']
            for kind,source in (('FP',fp),('duplicate',duplicates)):
                x=np.asarray(source.get((tag,ci,a),[]));expected=dict(above=int(np.sum(x>score)),tied=int(np.sum(x==score)))
                assert expected==r[a+'_'+kind+'_prefix']
    replay=replay_probes(c)
    freeze_json(OUT/'independent_verified.json',dict(complete=True,scenes=200,targets=4689,NT_geometry=geom,S_pool_members=inputmembers,
        threshold_partitions={k:dict(v) for k,v in totals.items()},probes=dict(probe_counts),ranking_rows=len(ranking),
        original_OOF_scores_exact=True,original_SL_coverage_exact=True,match_based_FP_and_duplicate_prefix_exact=True,
        cached_labels_only=True,single_model_unchanged=True,dev55_results_read=False,new_AP=False,sampled_actual_inference_replay=replay,
        historical_log_seal_diff=log_receipt_diff,final_log_sha256=sha(mutable_log),scientific_outputs_hashes_exact=True,
        source_sha256=sha(Path(__file__))))
    print('Independent reconstruction passed:200 scenes,4689 targets,all3 thresholds,global prefixes and budget coverage.',flush=True)


if __name__=='__main__':main()
