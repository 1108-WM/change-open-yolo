#!/usr/bin/env python3
"""Single frozen residual ranking experiment; exposed dev200, original five folds."""
from pathlib import Path
from collections import Counter
import argparse,json,shutil,sys,time,subprocess
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tools import fi1_s_semantic_rank_residual as core
from tools import run_fi1_sr_score_factorization_development200 as factor
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
OUT=ROOT/'.local/fi1_s_semantic_rank_residual_development200_20261010_v2'
S,R=factor.S,factor.R;ctl=factor.ctl;read=factor.read;VIEWS=factor.VIEWS


def stage(name,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=name,updated=time.time(),**kw));print(name,kw,flush=True)


class Guard(factor.cross.source.ReadGuard):
    def __init__(self,c,mode,fold=None):super().__init__(c,mode);self.fold=fold
    def hook(self,event,args):
        super().hook(event,args)
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=Path(args[0]).resolve()
        if self.mode=='fit' and '/dataset/' in str(p) and p.name=='labels.npz':
            assert self.fold is not None and self.c['folds'][p.parent.name]!=self.fold,'heldout targets in fitting'
        if self.mode in ('predict','verify') and p.name=='labels.npz':raise PermissionError('labels forbidden in inference')


def prepare():
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>=50*1024**3
    parent=read(factor.OUT/'contract.json');assert read(factor.OUT/'handoff_verified.json')['complete']
    inputs={str(S/n):sha(S/n) for n in ('contract.json','predictions_frozen.json','completion_verified.json')}
    labels={};code=dict(parent['code']);Sseal=read(S/'predictions_frozen.json')
    for s in parent['scenes']:
        for n in ('members.jsonl','pool.npz','semantic_S.json','class_agnostic_S.json','semantic_C.json','class_agnostic_C.json'):
            q=S/'predictions'/s/n;inputs[str(q)]=Sseal['outputs'][str(q)]
        for n in ('geometry','points'):
            q=parent['bindings'][s][n];inputs[q]=parent['inputs'][q]
        q=parent['jobs'][s]['nt_geometry.json'];inputs[q]=parent['inputs'][q]
        for view in VIEWS:
            q=S/'evaluation'/s/f'{view}_S_matches.json';labels[str(q)]=sha(q)
            q=S/'evaluation'/s/f'{view}_C_matches.json';labels[str(q)]=sha(q)
        for q in (Path(parent['jobs'][s]['target_census']),ROOT/'.local/fi1_s_c_selection_scope_development200_20261010_v1/scenes'/s/'members.json'):
            labels[str(q)]=parent['label_inputs'][str(q)]
    for q in (factor.OUT/'supplemental_analysis.json',factor.OUT/'handoff_verified.json'):
        labels[str(q)]=sha(q)
    for n in ('tools/fi1_s_semantic_rank_residual.py','tools/run_fi1_s_semantic_rank_residual_development200.py',
              'tools/run_fi1_s_semantic_rank_residual_development200.sh','tests/test_fi1_s_semantic_rank_residual.py'):
        q=ROOT/n;code[str(q)]=sha(q)
    c=dict(version=OUT.name,scenes=parent['scenes'],folds=parent['folds'],reporting_groups=parent['reporting_groups'],
        bindings=parent['bindings'],jobs=parent['jobs'],class_ids=parent['class_ids'],inputs=inputs,label_inputs=labels,code=code,budget=800,
        model='28(X26,S,logitS) ->32 tanh ->1; zero initialized final layer; one class-shared model, no class/scene/GT ID input.',
        residual='delta=tanh(network), fixed |delta|<=1. score=S*exp(delta)/(1-S+S*exp(delta)); actual S=0/1 stay exact. Numerical1e-6 floor used only loss/input; no floor in output. Per member, both directions allowed.',
        supervision='All original nine mainAP IoU thresholds, exact evaluator ignore per threshold. Per valid same-class GT select highestS member as fixed representative, others duplicates; negative unmatchedFP distinct from ignored and small masks. Training surrogate not exact dynamic AP or immutable TP.',
        pairs='<=128 reps/class/threshold; per rep up to4 hardFP(top64),2 cross-scene hardFP(top64),2 randomFP,2 same-target duplicate; dedup, exclude immutable score endpoints. Sampling only training-fold rows, heldout pairs separate after models sealed.',
        weighting='Equal total each class with pairs, then its available IoU thresholds, then pairs. Finite class contribution, no class-specific head or inverse frequency infinity. Macro ranking surrogate, not an AP guarantee.',
        objective='weighted softplus(-(logitS_pos+delta_pos-logitS_neg-delta_neg)) +0.1 mean pair-endpoint delta^2 +0.1 class-balanced random-full-pool-anchor delta^2.',
        params=dict(epochs=20,batch=2048,optimizer='Adam',learning_rate=.001,bound=1.,anchor=.1,seed=20261010,threads=4),
        normalization='Training160 feature mean/std only; numerical minimumstd1e-6. No heldout normalization or labels used during fitting. Fixed20 epochs, no earlystop/selection/retraining.',
        learnability_gate='Before original AP: macro pairlogloss and inversion improve vsS in pooled five heldout-folds, >=3folds both improve; eachfold training loss strictly improves and pooled training inversion improves. If fails stop before AP, report all training/test class and negative-kind metrics; no parameter repair.',
        acceptance='If learnability passes, original K800 actual CA from selected member max, no refill. Compare S,C,R mainAP, AP50/AP25,old90/new110/folds,all thresholds and424. Accept only both mainAP>S,C,R, no424 member/positive loss and noS>.75 semantic/CA loss; report minimum>=S semantic and>C CA too.',
        GT_policy='Only original approved dev200 bindings/hash whitelist for label construction/evaluation. No GT/labels in prediction/verification; all predictions sealed before evaluation. Foldfit guard prevents heldout labels; diagnostic scoring never changes frozen model.',
        scope='Exposed dev200; new head fivefold physical-scene OOF only, upstream/S saw training-domain scenes; not whole-system blind.',
        exclusions='No R score inputs, geometry multiplier, IoU regression head, Q/F append, fusion/threshold/bound/K/architecture sweep, U/Uprime or production mutation.',
        new_training=True,new_geometry=False,U_Uprime_output=False,production_modified=False)
    ctl.base.verify_files(inputs);ctl.base.verify_files(labels);ctl.base.verify_files(code)
    OUT.mkdir();freeze_json(OUT/'contract.json',c)
    for q in code:
        dest=OUT/'code_snapshot'/Path(q).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dest)
    tests=subprocess.run([sys.executable,'-m','pytest','tests/test_fi1_s_semantic_rank_residual.py','tests/test_fi1_sr_cross_control.py','tests/test_fi1_class_agnostic_ablation.py','-q'],capture_output=True,text=True)
    (OUT/'tests.log').write_text(tests.stdout+tests.stderr);assert tests.returncode==0,tests.stdout+tests.stderr
    freeze_json(OUT/'tests_verified.json',dict(passed=True,sha256=sha(OUT/'tests.log')));stage('prepared')


def dataset(c):
    guard=Guard(c,'dataset');sys.addaudithook(guard.hook);ctl.base.verify_files(c['code']);ctl.base.verify_files(c['inputs']);ctl.base.verify_files(c['label_inputs'])
    ev=ctl.base.evaluator();thresholds=np.asarray(ev.opt['overlaps'][:-1]);assert len(thresholds)==9
    counts=Counter();receipts={}
    for j,s in enumerate(c['scenes']):
        stage('dataset',scene=s,completed=j,total=200);d=OUT/'dataset'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():receipt=read(d/'receipt.json');ctl.base.verify_files(receipt['outputs'])
        else:
            members=ctl.base.rows(S/'predictions'/s/'members.jsonl');_,allpoints=ctl.shared.load_points(c['bindings'][s])
            geoms=read(c['jobs'][s]['nt_geometry.json']);keys=[g['geometry_key'] for g in geoms];points={k:allpoints[k] for k in keys}
            assert {m['geometry_key'] for m in members}==set(keys)
            idx={g:i for i,g in enumerate(keys)};gi=np.asarray([idx[m['geometry_key']] for m in members]);classes=np.asarray([m['class_index'] for m in members],dtype=np.int16)
            sizes=np.asarray([len(points[g]) for g in keys]);b=c['bindings'][s];assert sha(b['gt_path'])==b['gt_sha256']
            gt=np.loadtxt(b['gt_path'],dtype=np.int64);assert len(gt)==b['point_count']
            census=read(c['jobs'][s]['target_census']);targets=np.full((len(keys),198),-1,dtype=np.int64);own=np.zeros((len(keys),198))
            for g,k in enumerate(keys):
                for tid,v in census['ious'][k].items():
                    tid=int(tid);ci=c['class_ids'].index(tid//1000)
                    if v>own[g,ci]:own[g,ci]=v;targets[g,ci]=tid
            ignored=np.zeros((len(keys),198));fraction=np.zeros_like(ignored);ids,nums=np.unique(gt,return_counts=True);ncount=dict(zip(ids,nums));ci_lookup={cid:i for i,cid in enumerate(c['class_ids'])}
            for g,k in enumerate(keys):
                ids,n=np.unique(gt[points[k]],return_counts=True);hits=dict(zip(ids,n));void=sum(n for gid,n in hits.items() if gid//1000 not in ci_lookup)
                fraction[g,:]=void/max(1,sizes[g])
                for gid,n in hits.items():
                    ci=ci_lookup.get(gid//1000)
                    if ci is not None and (gid<1000 or ncount[gid]<100):
                        fraction[g,ci]+=n/max(1,sizes[g]);ignored[g,ci]=max(ignored[g,ci],n/(sizes[g]+ncount[gid]-n))
            with np.load(S/'predictions'/s/'pool.npz',allow_pickle=False) as z:x=z['x26'];scores=z['scores']
            x=np.column_stack((x,scores,core.base_logits(scores))).astype(np.float32)
            # Reorder temporarily to original S deterministic ties before fixed representative selection.
            priority=factor.cross.source.core.order(members,scores);rev=np.argsort(priority)
            states,identity=core.member_states(scores[priority],gi[priority],classes[priority],sizes,own,targets,fraction,ignored,thresholds)
            states=states[rev];identity=identity[rev]
            np.savez_compressed(d/'features.npz',x=x,S=scores,classes=classes)
            np.savez_compressed(d/'labels.npz',states=states,identity=identity)
            # Independently rebuild states from frozen original matching of S800. Representative identity must agree.
            match=read(S/'evaluation'/s/'semantic_S_matches.json');selected=read(S/'predictions'/s/'semantic_S.json');ix={ctl.shared.key(m):i for i,m in enumerate(members)}
            filtered=[m for m in selected if sizes[idx[m['geometry_key']]]>=100]
            for ti,t in enumerate(thresholds):
                winners={}
                for label,gg in match['gt'].items():
                    for g in gg:
                        if g['instance_id']<1000 or g['vert_count']<100:continue
                        eligible=[p for p in g['matched_pred'] if p['intersection']/(p['vert_count']+g['vert_count']-p['intersection'])>t]
                        if eligible:winners[min(eligible,key=lambda p:(-p['confidence'],p['pred_id']))['pred_id']]=g['instance_id']
                for pp in match['pred'].values():
                    for p in pp:
                        mi=ix[ctl.shared.key(filtered[p['pred_id']])]
                        valid=[g for g in p['matched_gt'] if g['instance_id']>=1000 and g['vert_count']>=100 and g['intersection']/(p['vert_count']+g['vert_count']-g['intersection'])>t]
                        if valid:expected=1 if p['pred_id'] in winners else 2
                        else:
                            found=any(g['intersection']/(p['vert_count']+g['vert_count']-g['intersection'])>t for g in p['matched_gt'])
                            ignore=p['void_intersection']+sum(g['intersection'] for g in p['matched_gt'] if g['instance_id']<1000 or g['vert_count']<100)
                            expected=-1 if found or ignore/p['vert_count']>t else 0
                        assert states[mi,ti]==expected,(s,mi,t,states[mi,ti],expected)
            receipt=dict(complete=True,members=len(members),state_counts={str(v):int((states==v).sum()) for v in (-1,0,1,2)},
                S800_all9_thresholds_independent_match_labels=True,GT_sha256=b['gt_sha256'],outputs={str(d/n):sha(d/n) for n in ('features.npz','labels.npz')})
            freeze_json(d/'receipt.json',receipt)
        counts.update(members=receipt['members']);counts.update(receipt['state_counts']);receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    freeze_json(OUT/'dataset_frozen.json',dict(complete=True,counts=dict(counts),receipts=receipts,GT_scans=len(guard.opened_gt)))


def load_scenes(c,scenes,labels=True):
    arrays=[];ys=[];ss=[]
    for i,s in enumerate(scenes):
        ctl.base.verify_files(read(OUT/'dataset'/s/'receipt.json')['outputs'])
        with np.load(OUT/'dataset'/s/'features.npz',allow_pickle=False) as z:arrays.append(dict(z));ss.append(np.full(len(z['S']),i,dtype=np.int16))
        if labels:
            with np.load(OUT/'dataset'/s/'labels.npz',allow_pickle=False) as z:ys.append(dict(z))
    result={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]};result['scene_ids']=np.concatenate(ss)
    if labels:result.update({k:np.concatenate([a[k] for a in ys]) for k in ys[0]})
    return result


def infer(model,x,mean,std):
    values=[]
    with torch.no_grad():
        for start in range(0,len(x),8192):values.append(model(torch.from_numpy(((x[start:start+8192]-mean)/std).astype(np.float32))).numpy().astype(float))
    return np.concatenate(values)


def fit(c,fold):
    sys.addaudithook(Guard(c,'fit',fold).hook);torch.set_num_threads(4);torch.manual_seed(core.SEED+fold);torch.use_deterministic_algorithms(True)
    ctl.base.verify_files(c['code']);assert read(OUT/'dataset_frozen.json')['complete']
    d=OUT/'models'/f'fold{fold}';d.mkdir(parents=True,exist_ok=True);assert not (d/'receipt.json').exists()
    train=[s for s in c['scenes'] if c['folds'][s]!=fold];test=[s for s in c['scenes'] if c['folds'][s]==fold]
    assert len(train)==160 and len(test)==40 and not {s[:9] for s in train}&{s[:9] for s in test}
    stage('fit',fold=fold);a=load_scenes(c,train)
    mean=a['x'].mean(axis=0,dtype=np.float64).astype(np.float32);std=np.maximum(a['x'].std(axis=0,dtype=np.float64),1e-6).astype(np.float32)
    pairs=core.make_pairs(a['S'],a['classes'],a['scene_ids'],a['identity'],a['states'],core.SEED+fold)
    np.savez_compressed(d/'training_pairs.npz',**pairs);np.savez(d/'normalization.npz',mean=mean,std=std)
    model=core.Residual();optimizer=torch.optim.Adam(model.parameters(),lr=.001);x=torch.from_numpy(((a['x']-mean)/std).astype(np.float32));base=torch.from_numpy(core.base_logits(a['S']).astype(np.float32))
    p,n=pairs['pairs'].T;wt=pairs['weights']*len(p);rng=np.random.default_rng(core.SEED+fold)
    anchors=np.concatenate([rng.choice(np.flatnonzero(a['classes']==ci),min(128,int((a['classes']==ci).sum())),replace=False) for ci in np.unique(a['classes'])])
    history=[];start=time.monotonic();stage('fit_pairs',fold=fold,pairs=len(p),parameters=sum(t.numel() for t in model.parameters()))
    for epoch in range(core.EPOCHS):
        order=rng.permutation(len(p));losses=[]
        for offset in range(0,len(p),2048):
            ii=order[offset:offset+2048];pi,ni=p[ii],n[ii];dp=model(x[pi]);dn=model(x[ni]);w=torch.from_numpy(wt[ii].astype(np.float32))
            rank=(F.softplus(-(base[pi]+dp-base[ni]-dn))*w).mean()
            keep=(w*(dp.square()+dn.square())*.5).mean()
            ai=rng.choice(anchors,min(2048,len(anchors)),replace=False);anch=model(x[ai]).square().mean()
            loss=rank+core.ANCHOR*(keep+anch);optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses))));stage('fit_epoch',fold=fold,**history[-1])
    torch.save(model.state_dict(),d/'model.pt');delta=infer(model,a['x'],mean,std);scores=core.adjusted(a['S'],delta)
    baseline=core.ranking_metrics(a['S'],pairs);after=core.ranking_metrics(scores,pairs)
    byclass={};bykind={}
    for ci in np.unique(a['classes'][p]):
        mask=a['classes'][p]==ci;sub={k:v[mask] for k,v in pairs.items()};sub['weights']=sub['weights']/sub['weights'].sum()
        byclass[int(ci)]=dict(before=core.ranking_metrics(a['S'],sub),after=core.ranking_metrics(scores,sub))
    for kind in range(4):
        mask=pairs['kinds']==kind
        if mask.any():
            sub={k:v[mask] for k,v in pairs.items()};sub['weights']=sub['weights']/sub['weights'].sum()
            bykind[kind]=dict(before=core.ranking_metrics(a['S'],sub),after=core.ranking_metrics(scores,sub))
    freeze_json(d/'training_metrics.json',dict(baseline=baseline,residual=after,history=history,seconds=time.monotonic()-start,
        training_byclass=byclass,training_bykind=bykind,
        max_abs_delta=float(abs(delta).max()),train_scenes=train,heldout_labels_read=False))
    freeze_json(d/'receipt.json',dict(complete=True,train_scenes=train,test_scenes=test,heldout_labels_opened=False,
        dataset_seal_sha256=sha(OUT/'dataset_frozen.json'),outputs={str(d/n):sha(d/n) for n in ('model.pt','normalization.npz','training_pairs.npz','training_metrics.json')}))


def model_load(fold):
    d=OUT/'models'/f'fold{fold}';ctl.base.verify_files(read(d/'receipt.json')['outputs']);m=core.Residual();m.load_state_dict(torch.load(d/'model.pt',map_location='cpu',weights_only=True));m.eval()
    with np.load(d/'normalization.npz',allow_pickle=False) as z:mean,std=z['mean'],z['std']
    return m,mean,std


def predict(c,verify=False):
    sys.addaudithook(Guard(c,'verify' if verify else 'predict').hook);torch.set_num_threads(4);ctl.base.verify_files(c['code']);outputs={};receipts={}
    for fold in range(5):
        model,mean,std=model_load(fold)
        for s in [s for s in c['scenes'] if c['folds'][s]==fold]:
            stage('verify' if verify else 'predict',fold=fold,scene=s);d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True)
            with np.load(OUT/'dataset'/s/'features.npz',allow_pickle=False) as z:x,base=z['x'],z['S']
            delta=infer(model,x,mean,std);scores=core.adjusted(base,delta);members=ctl.base.rows(S/'predictions'/s/'members.jsonl')
            selected=factor.cross.source.previous.rank(members,scores)[:800];geom=ctl.collapse_members(selected)
            if verify:
                receipt=read(d/'receipt.json');ctl.base.verify_files(receipt['outputs'])
                with np.load(d/'scores.npz',allow_pickle=False) as z:assert np.array_equal(scores,z['full']) and np.array_equal(delta,z['delta'])
                independent=sorted([{**m,'score':float(v)} for m,v in zip(members,scores)],key=lambda m:(-m['score'],not m['has_original_native_alias'],m['geometry_key'],m['class_index']))[:800]
                assert independent==selected==read(d/'semantic_L.json');assert geom==read(d/'class_agnostic_L.json')
            else:
                np.savez_compressed(d/'scores.npz',full=scores,delta=delta)
                freeze_json(d/'semantic_L.json',selected);freeze_json(d/'class_agnostic_L.json',geom)
                receipt=dict(complete=True,fold=fold,raw_GT_read=False,max_abs_delta=float(abs(delta).max()),
                    outputs={str(d/n):sha(d/n) for n in ('scores.npz','semantic_L.json','class_agnostic_L.json')});freeze_json(d/'receipt.json',receipt)
            outputs.update(receipt['outputs']);receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    if verify:freeze_json(OUT/'inference_verified.json',dict(complete=True,predictions_seal_sha256=sha(OUT/'predictions_frozen.json'),scenes=200,raw_GT_read=False))
    else:freeze_json(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,receipts=receipts,raw_GT_read=False))


def learnability(c):
    sys.addaudithook(Guard(c,'diagnostics').hook);ctl.base.verify_files(c['code']);assert read(OUT/'inference_verified.json')['complete'];folds={};train_losses=[];test_losses=[];train_inv=[];test_inv=[]
    for fold in range(5):
        stage('learnability',fold=fold);scenes=[s for s in c['scenes'] if c['folds'][s]==fold];a=load_scenes(c,scenes)
        pairs=core.make_pairs(a['S'],a['classes'],a['scene_ids'],a['identity'],a['states'],core.SEED+100+fold)
        scores=np.concatenate([np.load(OUT/'predictions'/s/'scores.npz')['full'] for s in scenes]);before=core.ranking_metrics(a['S'],pairs);after=core.ranking_metrics(scores,pairs)
        train=read(OUT/'models'/f'fold{fold}'/'training_metrics.json');train_losses.append(train['residual']['pair_loss']-train['baseline']['pair_loss']);train_inv.append(train['residual']['inversion']-train['baseline']['inversion'])
        test_losses.append(after['pair_loss']-before['pair_loss']);test_inv.append(after['inversion']-before['inversion'])
        byclass={};p,n=pairs['pairs'].T
        for ci in np.unique(a['classes'][p]):
            mask=a['classes'][p]==ci;sub={k:(v[mask] if k!='pairs' else v[mask]) for k,v in pairs.items()};sub['weights']=sub['weights']/sub['weights'].sum()
            byclass[int(ci)]=dict(before=core.ranking_metrics(a['S'],sub),after=core.ranking_metrics(scores,sub),training_contribution='finite equal class weight')
        bykind={}
        for kind in range(4):
            mask=pairs['kinds']==kind
            if mask.any():
                sub={k:v[mask] for k,v in pairs.items()};sub['weights']=sub['weights']/sub['weights'].sum();bykind[kind]=dict(before=core.ranking_metrics(a['S'],sub),after=core.ranking_metrics(scores,sub))
        np.savez_compressed(OUT/f'heldout_fold{fold}_pairs.npz',**pairs)
        folds[fold]=dict(train=train,heldout_before=before,heldout_after=after,heldout_byclass=byclass,heldout_bykind=bykind)
    passed=all(x<0 for x in train_losses) and np.mean(train_inv)<0 and np.mean(test_losses)<0 and np.mean(test_inv)<0 and sum(l<0 and i<0 for l,i in zip(test_losses,test_inv))>=3
    freeze_json(OUT/'learnability.json',dict(complete=True,passed=bool(passed),folds=folds,train_loss_delta_mean=float(np.mean(train_losses)),train_inversion_delta_mean=float(np.mean(train_inv)),heldout_loss_delta_mean=float(np.mean(test_losses)),heldout_inversion_delta_mean=float(np.mean(test_inv)),
        heldout_used_for_fit_or_checkpoint=False,definition='Equal average fold macro pair metrics, not pooled AP; fixed sampling diagnostic. No AP computed at this gate.'))


def evaluate(c):
    gate=read(OUT/'learnability.json');assert gate['complete']
    if not gate['passed']:stage('AP_skipped_no_learnability');return
    sys.addaudithook(Guard(c,'evaluate').hook);assert read(OUT/'inference_verified.json')['predictions_seal_sha256']==sha(OUT/'predictions_frozen.json')
    ctl.base.verify_files(c['code']);ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs'])
    ev=ctl.base.evaluator();ca,valid,uid=ctl.class_agnostic_evaluator();receipts={}
    for j,s in enumerate(c['scenes']):
        stage('matching',scene=s,completed=j,total=200);d=OUT/'evaluation'/s;d.mkdir(parents=True,exist_ok=True)
        b=c['bindings'][s];assert sha(b['gt_path'])==b['gt_sha256'];gt=np.loadtxt(b['gt_path'],dtype=np.int64);_,points=ctl.shared.load_points(b)
        rows=read(OUT/'predictions'/s/'semantic_L.json');geom=read(OUT/'predictions'/s/'class_agnostic_L.json')
        sem=ctl.shared.assign(ev,rows,points,b['gt_path'],len(gt));checks=ctl.shared.check_match(sem,rows,points,gt,ev)
        mapped=ctl.remap_gt(gt,valid,uid)
        def loader(path):assert Path(path).resolve()==Path(b['gt_path']).resolve();return mapped.copy()
        ca.util_3d.load_ids=loader;ag=ctl.shared.assign(ca,geom,points,b['gt_path'],len(gt),True);checks+=ctl.shared.check_match(ag,geom,points,mapped,ca,True)
        for v,value in zip(VIEWS,(sem,ag)):freeze_json(d/f'{v}_L_matches.json',ctl.base.serializable_matches(value))
        receipt=dict(complete=True,GT_sha256=b['gt_sha256'],independent_intersection_edges=checks,outputs={str(d/f'{v}_L_matches.json'):sha(d/f'{v}_L_matches.json') for v in VIEWS});freeze_json(d/'receipt.json',receipt);receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    freeze_json(OUT/'matches_frozen.json',dict(complete=True,receipts=receipts,predictions_before_GT=True))
    refs=read(factor.OUT/'supplemental_analysis.json')['AP_with_C_fold_controls'];ap={};arrays={}
    for group,ss in c['reporting_groups'].items():
        ap[group]={}
        for view,evaluator in (('semantic',ev),('class_agnostic',ca)):
            stage('AP',group=group,view=view);matches={s:read(OUT/'evaluation'/s/f'{view}_L_matches.json') for s in ss};x,*_=evaluator.evaluate_matches(matches)
            ap[group][view]={a:refs[group][view][a] for a in ('S','C','R')};ap[group][view]['L']=ctl.base.metrics(evaluator,x)
            q=OUT/'evaluation'/f'{group}_{view}_L_AP.npy';np.save(q,x,allow_pickle=False);arrays[str(q)]=sha(q)
            import warnings
            ts=np.asarray(evaluator.opt['overlaps'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                direct=dict(ap=float(np.nanmean(x[0,:,~np.isclose(ts,.25)])),ap50=float(np.nanmean(x[0,:,np.isclose(ts,.5)])),ap25=float(np.nanmean(x[0,:,np.isclose(ts,.25)])))
            assert all(abs(direct[k]-ap[group][view]['L'][k])<1e-14 for k in direct)
            if group=='development200':repeat,*_=evaluator.evaluate_matches(dict(reversed(list(matches.items()))));assert np.allclose(x,repeat,atol=1e-12,rtol=0,equal_nan=True)
            print('result',group,view,ap[group][view]['L'],flush=True)
    freeze_json(OUT/'evaluation_summary.json',dict(complete=True,AP=ap,AP_arrays=arrays,independent_aggregates=16))
    # Diagnostics helper only; preserves new experiment namespace, never old output files.
    factor.OUT=OUT;factor.ARMS=('L',);factor.diagnostics(c)


def report(c):
    gate=read(OUT/'learnability.json');accepted=False;lines=['# S成员级语义排序残差实验','',f"排序可学习性门：{gate['passed']}。固定20轮，无留出调参／选checkpoint。",'',
        '固定NT点集、S资格和K800；X26+S+logitS输入，小模型输出每个类别成员独立残差、允许上下调。R最高分不作为输入；不拼接S/R、不新增IoU头或Q/F竞争模块、不加U/U′。',
        '', '|折|训练配对损失变化|训练逆序率变化|留出配对损失变化|留出逆序率变化|','|---|---:|---:|---:|---:|']
    for f,d in gate['folds'].items():
        b,a=d['train']['baseline'],d['train']['residual'];hb,ha=d['heldout_before'],d['heldout_after']
        lines.append(f"|{f}|{a['pair_loss']-b['pair_loss']:+.6f}|{a['inversion']-b['inversion']:+.6f}|{ha['pair_loss']-hb['pair_loss']:+.6f}|{ha['inversion']-hb['inversion']:+.6f}|")
    if gate['passed']:
        ap=read(OUT/'evaluation_summary.json')['AP'];diag=read(OUT/'diagnostics.json')['groups']['development200']['L'];main=ap['development200']
        accepted=all(main[v]['L']['ap']>main[v][a]['ap'] for v in VIEWS for a in ('S','C','R')) and not diag['counts']['recovered_member_lost'] and not diag['counts']['recovered_positive_lost'] and not diag['coverage']['0.75']['S']['semantic_lost'] and not diag['coverage']['0.75']['S']['CA_lost']
        lines+=['', '|分组|臂|类别AP／AP50／AP25 (%)|实际去类AP／AP50／AP25 (%)|','|---|---|---:|---:|']
        for group in ap:
            for a in ('S','C','R','L'):
                v=['／'.join(f'{100*ap[group][view][a][k]:.6f}' for k in ('ap','ap50','ap25')) for view in VIEWS];lines.append(f'|{group}|{a}|{v[0]}|{v[1]}|')
        lines+=['',f"424成员缺失{diag['counts']['recovered_member_lost']}，无正分（含缺失）{diag['counts']['recovered_positive_lost']}；相对S的>.75覆盖：{diag['coverage']['0.75']['S']}。",f'双AP及覆盖验收：{accepted}。']
    else:lines+=['','固定小模型未通过折外排序信号门，按合同停止完整AP，不调参数、换头或追加模块。这仅限本监督／抽样／容量及现有摘要特征，不能证明视觉特征完全无用；应先核查失败语义证据，不继续同类残差加工。']
    lines+=['','监督是固定S代表及多IoU档同类配对的宏平均排序替代目标，不是可加总ΔAP。错误类别和重复分别抽样；ignore按每档，<100点及S=0/1不可调端点不参与配对。S端点输出精确保持，模型无法救回原零分成员；限制事前冻结。',
        '', '类权重是有限等权贡献；没有类ID输入／稀有类特例。训练统计与留出统计分别报告；所有模型封存后才读留出标签做诊断，不据其选择模型。旧90／新110／原五折均属暴露开发，不称整系统盲测。',
        '', '生产Full保持冻结；当前实验无参数扫描、倍率拼接或U/U′。完整逐类／负例类型排序见learnability.json；若完成AP，逐档全部目标及424真实排名见diagnostics.json。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');freeze_json(OUT/'decision.json',dict(complete=True,learnability_passed=gate['passed'],accepted=bool(accepted),AP_computed=gate['passed'],next_training=False,U_Uprime_added=False,production_modified=False))
    files=[p for p in OUT.glob('*.json') if p.name not in ('execution.json','completion_verified.json')]+[OUT/'report.md',OUT/'tests.log']
    freeze_json(OUT/'completion_verified.json',dict(complete=True,outputs={str(p):sha(p) for p in files},accepted=bool(accepted),production_modified=False))
    write_json(OUT/'execution.json',dict(status='complete',stage='complete',resume_required=False,accepted=bool(accepted),AP_computed=gate['passed']))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','dataset','fit','predict','verify','learnability','evaluate','report']);p.add_argument('--fold',type=int);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='fit':fit(read(OUT/'contract.json'),a.fold)
    elif a.stage=='verify':predict(read(OUT/'contract.json'),True)
    else:globals()[a.stage](read(OUT/'contract.json'))


if __name__=='__main__':main()
