#!/usr/bin/env python3
"""Independent fold/pair/score/gate and optional coverage/AP checks after sealing."""
from pathlib import Path
from collections import Counter
import sys,bisect,warnings
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools import run_fi1_s_semantic_rank_residual_development200 as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def independent_metrics(scores,data):
    # Explicit odds logarithms, independent of training helpers.
    p,n=data['pairs'].T;s=np.clip(scores,1e-6,1-1e-6)
    margin=np.log(s[p]/(1-s[p]))-np.log(s[n]/(1-s[n]));w=data['weights']
    return dict(pair_loss=float(np.sum(w*np.logaddexp(0,-margin))),
                inversion=float(np.sum(w*np.where(scores[p]<scores[n],1,np.where(scores[p]==scores[n],.5,0)))))


def main():
    run.torch.set_num_threads(4)
    out=run.OUT;c=run.read(out/'contract.json');done=run.read(out/'completion_verified.json');assert done['complete']
    sys.addaudithook(run.Guard(c,'postseal').hook);run.ctl.base.verify_files(done['outputs']);run.ctl.base.verify_files(c['inputs']);run.ctl.base.verify_files(c['label_inputs']);run.ctl.base.verify_files(c['code'])
    seal=run.read(out/'predictions_frozen.json');run.ctl.base.verify_files(seal['outputs']);counts=Counter();gate=run.read(out/'learnability.json');analysis={}
    for s in c['scenes']:
        print('independent_rank_verify',s,flush=True)
        run.ctl.base.verify_files(run.read(out/'dataset'/s/'receipt.json')['outputs'])
        with np.load(out/'dataset'/s/'features.npz',allow_pickle=False) as z:base=z['S'];x=z['x'];classes=z['classes']
        with np.load(run.S/'predictions'/s/'pool.npz',allow_pickle=False) as z:
            assert np.array_equal(base,z['scores']);assert np.array_equal(x[:,:26],z['x26'].astype(np.float32));assert np.array_equal(x[:,26],base.astype(np.float32))
        with np.load(out/'predictions'/s/'scores.npz',allow_pickle=False) as z:scores,delta=z['full'],z['delta']
        assert np.all(abs(delta)<=1) and np.all((scores>=0)&(scores<=1))
        inner=(base>0)&(base<1);logodds=np.log(scores[inner]/(1-scores[inner]))-np.log(base[inner]/(1-base[inner]))
        assert np.allclose(logodds,delta[inner],rtol=0,atol=1e-10);assert np.array_equal(scores[~inner],base[~inner])
        members=run.ctl.base.rows(run.S/'predictions'/s/'members.jsonl');assert np.array_equal(classes,[m['class_index'] for m in members])
        ix={run.ctl.shared.key(m):i for i,m in enumerate(members)};rows=run.read(out/'predictions'/s/'semantic_L.json')
        expected=sorted(range(len(members)),key=lambda i:(-scores[i],not members[i]['has_original_native_alias'],members[i]['geometry_key'],members[i]['class_index']))[:800]
        assert [ix[run.ctl.shared.key(m)] for m in rows]==expected
        for m in rows:
            i=ix[run.ctl.shared.key(m)];assert m['score']==scores[i]
            assert {k:v for k,v in m.items() if k!='score'}=={k:v for k,v in members[i].items() if k!='score'}
        best={}
        for m in rows:
            k=m['geometry_key']
            if k not in best or (m['score'],-m['class_index'])>(best[k]['score'],-best[k]['class_index']):best[k]=m
        geom=sorted([dict(geometry_key=k,score=m['score'],winning_class_index=m['class_index']) for k,m in best.items()],key=lambda m:(-m['score'],m['geometry_key']))
        assert geom==run.read(out/'predictions'/s/'class_agnostic_L.json')
        counts.update(full_scores=len(base),selected_members=800,geometry_projections=len(geom),positive_delta=int((delta>0).sum()),negative_delta=int((delta<0).sum()),S_zero_fixed=int((base==0).sum()))
    train_loss=[];train_inv=[];test_loss=[];test_inv=[]
    for fold in range(5):
        md=out/'models'/f'fold{fold}';receipt=run.read(md/'receipt.json');run.ctl.base.verify_files(receipt['outputs'])
        train=[s for s in c['scenes'] if c['folds'][s]!=fold];test=[s for s in c['scenes'] if c['folds'][s]==fold]
        assert receipt['train_scenes']==train and receipt['test_scenes']==test and receipt['heldout_labels_opened'] is False
        assert len(train)==160 and len(test)==40 and not {s[:9] for s in train}&{s[:9] for s in test}
        for name,scenes in (('train',train),('heldout',test)):
            a=run.load_scenes(c,scenes)
            path=md/'training_pairs.npz' if name=='train' else out/f'heldout_fold{fold}_pairs.npz'
            with np.load(path,allow_pickle=False) as z:data=dict(z)
            p,n=data['pairs'].T;assert np.all(a['classes'][p]==a['classes'][n]);assert np.isclose(data['weights'].sum(),1)
            endpoint_ci=a['classes'][p];unique_ci=np.unique(endpoint_ci)
            for ci in unique_ci:assert np.isclose(data['weights'][endpoint_ci==ci].sum(),1/len(unique_ci))
            cross=data['kinds']==1;assert np.all(a['scene_ids'][p[cross]]!=a['scene_ids'][n[cross]])
            duplicate=data['kinds']==3;assert np.all(a['scene_ids'][p[duplicate]]==a['scene_ids'][n[duplicate]]) and np.all(a['identity'][p[duplicate]]==a['identity'][n[duplicate]])
            # Every group is one class/threshold. Positive and negative state independently checked.
            for g in np.unique(data['groups']):
                ii=data['groups']==g;gp,gn=p[ii],n[ii]
                admissible=[ti for ti in range(9) if np.all(a['states'][gp,ti]==1) and np.all(np.isin(a['states'][gn,ti],(0,2)))]
                assert admissible,'pair label absent from every threshold'
            if name=='train':
                model,mean,std=run.model_load(fold);delta=run.infer(model,a['x'],mean,std);scores=run.core.adjusted(a['S'],delta)
                assert np.array_equal(mean,a['x'].mean(axis=0,dtype=np.float64).astype(np.float32))
                assert np.array_equal(std,np.maximum(a['x'].std(axis=0,dtype=np.float64),1e-6).astype(np.float32))
                before=gate['folds'][str(fold)]['train']['baseline'];after=gate['folds'][str(fold)]['train']['residual']
            else:
                scores=np.concatenate([np.load(out/'predictions'/s/'scores.npz')['full'] for s in scenes])
                before=gate['folds'][str(fold)]['heldout_before'];after=gate['folds'][str(fold)]['heldout_after']
            b=independent_metrics(a['S'],data);r=independent_metrics(scores,data)
            assert all(abs(b[k]-before[k])<1e-12 and abs(r[k]-after[k])<1e-12 for k in b)
            (train_loss if name=='train' else test_loss).append(r['pair_loss']-b['pair_loss']);(train_inv if name=='train' else test_inv).append(r['inversion']-b['inversion'])
            # Hardness and attainable pair order under fixed±1 residual; diagnosis only.
            sb=np.clip(a['S'],1e-6,1-1e-6);gap=np.log(sb[n]/(1-sb[n]))-np.log(sb[p]/(1-sb[p]))
            analysis[f'{fold}_{name}']=dict(before=b,after=r,finite_class_count=len(unique_ci),
                max_inverse_gap=float(gap.max()),weighted_inversions_beyond_fixed_bound=float(np.sum(data['weights']*(gap>2))),
                weighted_original_inversions=float(np.sum(data['weights']*(gap>0))),pairs=len(p))
            counts['independent_pair_checks']+=len(p)
    passed=all(x<0 for x in train_loss) and np.mean(train_inv)<0 and np.mean(test_loss)<0 and np.mean(test_inv)<0 and sum(l<0 and i<0 for l,i in zip(test_loss,test_inv))>=3
    assert passed==gate['passed'];decision=run.read(out/'decision.json');assert decision['AP_computed']==passed
    if passed:
        summary=run.read(out/'evaluation_summary.json');run.ctl.base.verify_files(summary['AP_arrays']);d=run.read(out/'diagnostics.json');ev=run.ctl.base.evaluator();ts=np.asarray(ev.opt['overlaps']);t75=int(np.argmin(abs(ts-.75)))
        for group in c['reporting_groups']:
            for view in run.VIEWS:
                arr=np.load(out/'evaluation'/f'{group}_{view}_L_AP.npy')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    direct=dict(ap=float(np.nanmean(arr[0,:,~np.isclose(ts,.25)])),ap50=float(np.nanmean(arr[0,:,np.isclose(ts,.5)])),ap25=float(np.nanmean(arr[0,:,np.isclose(ts,.25)])))
                assert all(abs(direct[k]-summary['AP'][group][view]['L'][k])<1e-14 for k in direct);counts['AP_aggregates']+=1
        for s in c['scenes']:
            run.ctl.base.verify_files(run.read(out/'evaluation'/s/'receipt.json')['outputs'])
            census=run.read(c['jobs'][s]['target_census']);ids=np.asarray(list(map(int,census['targets'])));rows=run.read(out/'predictions'/s/'semantic_L.json')
            iou=np.asarray([[census['ious'][m['geometry_key']][str(g)] for g in ids] for m in rows]);hits=iou[:,:,None]>ts[None,None,:]
            same=np.asarray([c['class_ids'][m['class_index']] for m in rows])[:,None]==ids[None,:]//1000
            positive=np.asarray([m['score']>0 for m in rows])[:,None,None]
            cover=dict(semantic=(hits & same[:,:,None]).any(axis=0),positive=(hits & same[:,:,None] & positive).any(axis=0),CA=hits.any(axis=0),positive_CA=(hits & positive).any(axis=0))
            refs={}
            for a in ('S','C','R'):
                rr=run.read((run.R if a=='R' else run.S)/'predictions'/s/f'semantic_{a}.json')
                hi=np.asarray([[census['ious'][m['geometry_key']][str(g)] for g in ids] for m in rr])[:,:,None]>ts[None,None,:]
                sc=np.asarray([c['class_ids'][m['class_index']] for m in rr])[:,None]==ids[None,:]//1000
                refs[a]=dict(semantic=(hi & sc[:,:,None]).any(axis=0),CA=hi.any(axis=0))
            for ti,t in enumerate(ts):
                saved=d['scenes'][s]['L']['coverage'][f'{t:.2f}'];assert all(saved[k]==int(v[:,ti].sum()) for k,v in cover.items())
                for a in refs:
                    for view in ('semantic','CA'):
                        x,y=cover[view][:,ti],refs[a][view][:,ti]
                        assert saved['vs'][a][view+'_gained']==sorted(map(int,ids[x & ~y]))
                        assert saved['vs'][a][view+'_lost']==sorted(map(int,ids[y & ~x]))
                counts['coverage_checks']+=1
            for r in d['scenes'][s]['L']['recovered424']:
                gi=int(np.flatnonzero(ids==r['gt_id'])[0]);assert r['member_present']==bool(cover['semantic'][gi,t75]);assert r['positive_present']==bool(cover['positive'][gi,t75]);counts['recovered424_checks']+=1
        mainap=summary['AP']['development200'];totals=d['groups']['development200']['L']
        accepted=all(mainap[v]['L']['ap']>mainap[v][a]['ap'] for v in run.VIEWS for a in ('S','C','R')) and not totals['counts']['recovered_member_lost'] and not totals['counts']['recovered_positive_lost'] and not totals['coverage']['0.75']['S']['semantic_lost'] and not totals['coverage']['0.75']['S']['CA_lost']
        assert accepted==decision['accepted']
    q=out/'independent_analysis.json';freeze_json(q,dict(complete=True,counts=dict(counts),fold_pair_analysis=analysis,learnability_gate_exact=True,
        weighting='Each pair-active class finite equal total; sparse/no-positive classes not learnable via this pair objective. No class-specific inference.',
        first_attempt='v1 dataset halted before training because Full geometry binding was not restricted to NT. v2 explicitly uses registered NT; v1 contract/log preserved.'))
    freeze_json(out/'independent_verified.json',dict(complete=True,code_sha256=sha(__file__),counts=dict(counts),training_folds_normalization_pairs_verified=True,
        outputs={str(q):sha(q),str(out/'completion_verified.json'):sha(out/'completion_verified.json')},prediction_GT_free=True,raw_GT_read=False,original_models_predictions_unchanged=True))
    print('Independent verification complete',dict(counts),flush=True)


if __name__=='__main__':main()
