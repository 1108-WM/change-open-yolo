#!/usr/bin/env python3
"""Postseal independent audit and full per-class/physical-scene reporting."""
from pathlib import Path
from collections import Counter
import sys,warnings
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_sl_single200_dev55 as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def main():
    out=run.OUT;c=run.read(out/'contract.json');decision=run.read(out/'decision.json');assert decision['complete']
    guard=run.Guard(c,'audit');sys.addaudithook(guard.hook);torch.set_num_threads(4)
    for hashes in (c['inputs'],c['training_inputs'],c['metadata_inputs'],c['receipt_inputs'],c['code']):run.ctl.base.verify_files(hashes)
    run.ctl.base.verify_files(run.read(out/'model/receipt.json')['outputs'])
    seal=run.read(out/'predictions_frozen.json');run.ctl.base.verify_files(seal['outputs']);run.ctl.base.verify_files(seal['receipts'])
    assert seal['single_model_sha256']==sha(out/'model/model.pt')
    assert run.read(out/'model_frozen.json')['model_sha256']==seal['single_model_sha256']
    assert len(c['train_scenes'])==len(run.mains(c['train_scenes']))==200 and len(run.mains(c['scenes']))==29
    master=run.read(run.MASTER);assert c['scenes']==master['roles']['development_validation']
    assert not run.mains(c['scenes'])&run.mains(master['roles']['train'])
    assert not run.mains(c['scenes'])&run.mains(c['train_scenes'])
    model,mean,std=run.inference.load_L(out/'model/model.pt',out/'model/normalization.npz')
    a=run.parent.load_training_scenes(c,c['train_scenes'])
    assert np.array_equal(a['x'].mean(0,dtype=np.float64).astype(np.float32),mean)
    assert np.array_equal(np.maximum(a['x'].std(0,dtype=np.float64),1e-6).astype(np.float32),std)
    expected=run.core.make_pairs(a['S'],a['classes'],a['scene_ids'],a['identity'],a['states'],run.SEED)
    with np.load(out/'model/training_pairs.npz') as z:
        assert all(np.array_equal(expected[k],z[k]) for k in expected)
    p,n=expected['pairs'].T;assert np.array_equal(a['classes'][p],a['classes'][n])
    for ci in np.unique(a['classes'][p]):assert abs(expected['weights'][a['classes'][p]==ci].sum()-1/len(np.unique(a['classes'][p])))<1e-12
    train_delta=run.parent.original.infer(model,a['x'],mean,std);train_scores=run.core.adjusted(a['S'],train_delta)
    training=run.read(out/'model/training_metrics.json')
    assert training['baseline']==run.core.ranking_metrics(a['S'],expected) and training['residual']==run.core.ranking_metrics(train_scores,expected)
    assert [h['epoch'] for h in training['history']]==list(range(1,21))
    assert run.training_loop_ast(run.fit)==run.training_loop_ast(run.parent.original.fit)
    del a,train_delta,train_scores
    print('training200_normalization_pairs_model_metrics_verified',flush=True)
    counts=Counter();matches={v:{a:{} for a in run.ARMS} for v in run.VIEWS};cover=run.read(out/'coverage_verified.json')['scenes']
    ev=run.ctl.base.evaluator();ca,_,_=run.ctl.class_agnostic_evaluator();ts=np.asarray(ev.opt['overlaps'])
    for s in c['scenes']:
        d=out/'predictions'/s;pool=run.read(d/'members.json');rows={a:run.read(d/f'semantic_{a}.json') for a in run.ARMS}
        ix={run.ctl.shared.key(m):i for i,m in enumerate(pool)}
        with np.load(d/'pool.npz') as z:x26,base=z['x26'],z['scores']
        x28=np.column_stack((x26,base,run.core.base_logits(base))).astype(np.float32)
        full=run.core.adjusted(base,run.parent.original.infer(model,x28,mean,std))
        assert all(m['score']==full[ix[run.ctl.shared.key(m)]] for m in rows['SL'])
        assert all({k:v for k,v in m.items() if k!='score'}=={k:v for k,v in pool[ix[run.ctl.shared.key(m)]].items() if k!='score'} for m in rows['SL'])
        assert {run.ctl.shared.key(m) for m in rows['S']}=={run.ctl.shared.key(m) for m in rows['SL']}
        for arm in run.ARMS:
            best={}
            for m in rows[arm]:
                k=m['geometry_key']
                if k not in best or (m['score'],-m['class_index'])>(best[k]['score'],-best[k]['class_index']):best[k]=m
            ag=sorted([dict(geometry_key=k,score=m['score'],winning_class_index=m['class_index']) for k,m in best.items()],key=lambda m:(-m['score'],m['geometry_key']))
            assert ag==run.read(d/f'class_agnostic_{arm}.json');counts['CA_geometries']+=len(ag)
        sizes={g['geometry_key']:g['point_count'] for g in run.read(c['bindings'][s]['geometry'])}
        run.ctl.base.verify_files(run.read(out/'evaluation'/s/'receipt.json')['outputs'])
        for v in run.VIEWS:
            old=run.read(d/f'{v}_S.json');new=run.read(d/f'{v}_SL.json')
            for arm in run.ARMS:matches[v][arm][s]=run.read(out/'evaluation'/s/f'{v}_{arm}_matches.json')
            template=matches[v]['S'][s]
            assert run.parent.fixed.independent_rescore_match(template,old,new,sizes,v=='class_agnostic')==matches[v]['SL'][s]
            counts['bidirectional_matches']+=1
            for t in ts:
                results={}
                for arm in run.ARMS:
                    presence=set();positive=set()
                    for gg in matches[v][arm][s]['gt'].values():
                        for g in gg:
                            if g['instance_id']<1000 or g['vert_count']<100:continue
                            good=[p for p in g['matched_pred'] if p['intersection']/(g['vert_count']+p['vert_count']-p['intersection'])>t]
                            if good:presence.add(g['instance_id'])
                            if any(p['confidence']>0 for p in good):positive.add(g['instance_id'])
                    results[arm]=(presence,positive)
                assert results['S']==results['SL']
                saved=cover[s][v][f'{t:.2f}'];assert saved['presence']==len(results['SL'][0]) and saved['positive']==len(results['SL'][1]) and saved['lost']==saved['gained']==0
                counts['threshold_coverage']+=1
        counts['members']+=800;print('scene_verified',s,flush=True)
    summary=run.read(out/'evaluation_summary.json');run.ctl.base.verify_files(summary['AP_arrays']);class_changes={};physical={}
    for v,evaluator in (('semantic',ev),('class_agnostic',ca)):
        arrays={a:np.load(out/'evaluation'/f'development55_{v}_{a}_AP.npy') for a in run.ARMS}
        for arm,arr in arrays.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                independent=dict(ap=float(np.nanmean(arr[0,:,~np.isclose(ts,.25)])),ap50=float(np.nanmean(arr[0,:,np.isclose(ts,.5)])),ap25=float(np.nanmean(arr[0,:,np.isclose(ts,.25)])))
            assert all(abs(independent[k]-summary['AP'][v][arm][k])<1e-14 for k in independent);counts['AP_aggregates']+=1
        if v=='semantic':
            for ci,label in enumerate(evaluator.CLASS_LABELS):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    values={a:float(np.nanmean(arr[0,ci,~np.isclose(ts,.25)])) for a,arr in arrays.items()}
                class_changes[label]={k:(val if np.isfinite(val) else None) for k,val in values.items()}
                class_changes[label]['delta_pp']=100*(values['SL']-values['S']) if all(np.isfinite(x) for x in values.values()) else None
        for main in c['physical_scenes']:
            ss=[s for s in c['scenes'] if s.split('_')[0]==main];physical.setdefault(main,dict(scans=ss,AP={}))
            physical[main]['AP'][v]={}
            for arm in run.ARMS:
                result,*_=evaluator.evaluate_matches({s:matches[v][arm][s] for s in ss})
                physical[main]['AP'][v][arm]=run.ctl.base.metrics(evaluator,result)
    both=all(summary['AP'][v]['SL']['ap']>summary['AP'][v]['S']['ap'] for v in run.VIEWS)
    assert both==decision['both_AP_passed'] and decision['coverage_passed']
    ch=Counter('up' if z['delta_pp']>1e-12 else 'down' if z['delta_pp']< -1e-12 else 'unchanged' for z in class_changes.values() if z['delta_pp'] is not None)
    ph={v:dict(Counter('up' if d['AP'][v]['SL']['ap']>d['AP'][v]['S']['ap']+1e-14 else 'down' if d['AP'][v]['SL']['ap']<d['AP'][v]['S']['ap']-1e-14 else 'unchanged' for d in physical.values())) for v in run.VIEWS}
    freeze_json(out/'supplemental.json',dict(complete=True,per_class=class_changes,class_direction_counts=dict(ch),physical_scene=physical,physical_direction_counts=ph))
    lines=['# 开发55逐类别与物理场景补充','',f'有效类别AP方向：{dict(ch)}。29物理场景的单独AP方向：{ph}。',
        '', '物理场景AP是描述性子集结果，不能平均替代全55的逐类别宏AP；每类GT少，亦不能把方向计数当总AP贡献或置信度。','',
        '|物理场景|扫描数|类别S→SL AP (%)|去类S→SL AP (%)|','|---|---:|---:|---:|']
    for main,d in physical.items():
        vals=['→'.join(f"{100*d['AP'][v][a]['ap']:.6f}" for a in run.ARMS) for v in run.VIEWS];lines.append(f'|{main}|{len(d["scans"])}|{vals[0]}|{vals[1]}|')
    lines+=['','所有198类（无GT类记null）及单场景AP/AP50/AP25见supplemental.json。没有据这些诊断筛选子集或调整方法。']
    (out/'supplemental_report.md').write_text('\n'.join(lines)+'\n')
    assert not guard.opened_gt
    freeze_json(out/'independent_verified.json',dict(complete=True,counts=dict(counts),training200_normalization_pairs_metrics_exact=True,
        one_model_all55=True,full_pool_vs_selected_inference_exact=True,all_S_coverage_exact=True,AP_aggregates_independent=True,
        outputs={str(out/n):sha(out/n) for n in ('contract.json','identity_audit.json','model_frozen.json','predictions_frozen.json',
          'evaluation_summary.json','decision.json','report.md','supplemental.json','supplemental_report.md')},verifier_sha256=sha(__file__),raw_GT_read=False))
    print('independent verification complete',dict(counts),decision,flush=True)


if __name__=='__main__':main()
