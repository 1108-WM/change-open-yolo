#!/usr/bin/env python3
"""Frozen single L trained on200, S-vs-SL confirmation on historical dev55."""
from pathlib import Path
from collections import Counter
import argparse, ast, hashlib, inspect, json, shutil, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from tools import run_fi1_sl_replay_stability_development200 as parent
from tools import fi1_sl_inference as inference
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
OUT=ROOT/'.local/fi1_sl_single200_dev55_20261010_v1'
INVENTORY=ROOT/'.local/fi1_g10_full597_closure_20260921_v1/inventory.json'
MASTER=ROOT/'资料/训练597场景清单_20260917.json'
ASSETS=Path('/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921')
GT_BIND=ROOT/'.local/fi1_r0_r4_preparation_20260913/r2_v1/evaluation/gt_binding.json'
L=parent.L;ctl=parent.ctl;read=parent.read;core=parent.original.core
ARMS=('S','SL');VIEWS=parent.original.VIEWS;SEED=20261010


def mains(scenes):return {s.split('_')[0] for s in scenes}


def stage(name,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=name,updated=time.time(),resume_required=True,**kw));print(name,kw,flush=True)


class Guard:
    def __init__(self,c,mode):
        self.c,self.mode=c,mode;self.opened_gt=set()
        self.gt={str(Path(b['gt_path']).resolve()) for b in c['bindings'].values()}
    def check(self,path):
        p=Path(path).resolve();text=str(p)
        if '/scannet200_test/' in text or '/scannetv2_test/' in text or '/scannet_test/' in text:
            raise PermissionError('hidden test data outside dev55 protocol')
        if '/.local/' in text and 'val312' in text and '/code_snapshot/' not in text:
            if p not in {Path(q).resolve() for q in self.c['code']}:raise PermissionError('val312 artifacts outside protocol')
        if any(v.startswith('ground_truth') for v in p.parts):
            if self.mode!='evaluate' or text not in self.gt or not (OUT/'predictions_frozen.json').exists():
                raise PermissionError('dev55 GT requires every frozen prediction and evaluation stage')
            self.opened_gt.add(text)
        if self.mode in ('fit','predict','verify') and ('/evaluation/' in text or p.name=='target_census.json'):
            raise PermissionError('evaluation labels excluded from fit/inference')
        if p.name=='labels.npz':
            if self.mode!='fit' or p.parent.name not in self.c['train_scenes'] or p.parent.parent!=(L/'dataset').resolve():
                raise PermissionError('only frozen train200 labels may enter fitting')
        if self.mode=='fit' and '/learning/dataset/' in text:raise PermissionError('dev55/historical dataset excluded from fitting')
    def hook(self,event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,Path)):self.check(args[0])


def training_loop_ast(function):
    tree=ast.parse(inspect.getsource(function))
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    return ast.dump(loop,include_attributes=False)


def prepare():
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>=50*1024**3
    old=read(parent.OUT/'contract.json');assert read(parent.OUT/'independent_verified.json')['complete']
    inv=read(INVENTORY);master=read(MASTER);train=old['scenes'];scenes=inv['groups']['dev']
    assert len(train)==len(mains(train))==200 and len(scenes)==55 and len(mains(scenes))==29
    assert scenes==master['roles']['development_validation'] and set(train)<=set(master['roles']['train'])
    backend=read(ASSETS/'learning/models/summary.json');assert backend['complete'] and not backend['pilot']
    assert backend['fit_scenes']==inv['groups']['train']==master['roles']['train']
    physical=mains(scenes);assert not physical&mains(train) and not physical&mains(backend['fit_scenes'])
    official=ROOT/'models/Mask3D/third_party/ScanNet/Tasks/Benchmark'
    split={n:(official/f'scannetv2_{n}.txt').read_text().split() for n in ('train','val','test')}
    historical=read(ROOT/'.local/fi1_dev55_val312_diagnostics_20260914/identity.json')
    assert historical['scenes']['dev55']==scenes
    reserve=read(ROOT/'.local/fi1_union_classifier_reserve100_20261007_v1/reserve/contract.json')['scenes']
    diagnostic=master['roles']['diagnostic']
    groups=dict(train200=train,backend_train597=backend['fit_scenes'],development55=scenes,
        diagnostic5=diagnostic,consumed_reserve100=reserve,validation312=inv['groups']['val'],
        official_train=split['train'],official_test=split['test'])
    overlaps={k:dict(exact_scans=sorted(set(scenes)&set(ss)),physical_scenes=sorted(physical&mains(ss))) for k,ss in groups.items() if k!='development55'}
    metadata=[INVENTORY,MASTER,GT_BIND,ASSETS/'learning/models/summary.json',ASSETS/'learning/identity.json',
        ASSETS/'learning/dev_evaluation_contract.json',ROOT/'.local/fi1_dev55_val312_diagnostics_20260914/identity.json',
        ROOT/'.local/fi1_union_classifier_reserve100_20261007_v1/reserve/contract.json',parent.OUT/'contract.json',parent.OUT/'independent_verified.json']
    metadata += [official/f'scannetv2_{n}.txt' for n in split]
    # The history files above contain identities; historical AP values are not used.
    gt_rows={r['scene_name']:r for r in read(GT_BIND)['rows']}
    bindings={};inputs={old['B_model']:old['B_model_sha256']};label_inputs={};receipts={}
    for s in scenes:
        drpath=ASSETS/'learning/dataset'/s/'receipt.json';dr=read(drpath)
        own=Path(inv['jobs'][s]['folder'])/'own/D'
        b=dict(geometry=str(own/'geometry_binding'/s/'geometry.json'),points=str(own/'geometry_binding'/s/'points.npz'),
               evidence=str(own/'alpha_completed'/s/'evidence.jsonl'),point_count=dr['point_count'])
        assert dr['complete'] and dr['group']=='dev' and dr['geometry_file']==b['geometry'] and dr['points_file']==b['points']
        for n in ('geometry','points','evidence'):inputs[b[n]]=dr['inputs'][b[n]]
        receipts[str(drpath)]=sha(drpath)
        candidates={q:h for q,h in gt_rows[s]['files'].items() if 'ground_truth' in q and q.endswith('/'+s+'.txt')}
        assert len(candidates)==1 and gt_rows[s]['point_count']==b['point_count']
        b.update(gt_path=next(iter(candidates)),gt_sha256=next(iter(candidates.values())))
        assert Path(b['gt_path']).is_file();label_inputs.update(candidates);bindings[s]=b
    training_inputs={}
    for s in train:
        q=L/'dataset'/s/'receipt.json';rc=read(q);receipts[str(q)]=sha(q);training_inputs.update(rc['outputs'])
    code=dict(old['code'])
    for n in ('tools/run_fi1_sl_single200_dev55.py','tools/run_fi1_sl_single200_dev55.sh',
              'tools/verify_fi1_sl_single200_dev55.py','tests/test_fi1_sl_single200_dev55.py'):
        q=ROOT/n;code[str(q)]=sha(q)
    c=dict(version=OUT.name,authorization='User explicitly selected complete development55 for validation after identity review.',
        train_scenes=train,scenes=scenes,physical_scenes=sorted(physical),bindings=bindings,
        reporting_groups={'development55':scenes},arms=list(ARMS),views=list(VIEWS),budget=800,
        B_model=old['B_model'],B_model_sha256=old['B_model_sha256'],class_ids=old['class_ids'],
        inputs=inputs,training_inputs=training_inputs,metadata_inputs={str(q):sha(q) for q in metadata},
        receipt_inputs=receipts,label_inputs=label_inputs,code=code,
        params=dict(epochs=20,batch=2048,optimizer='Adam',learning_rate=.001,hidden=32,bound=1.,anchor=.1,seed=SEED,threads=4),
        original_method='Same original L pair sampling,28 inputs,32hidden residual,loss,20epochs; train all200 once, training200 mean/std only; no fold routing or seed/checkpoint choice.',
        selection='Complete symmetric NT eligibility/context -> frozen B26 S800 -> one L scores only those800. Same member/class/pointset; no drop/refill. CA exact selected member max.',
        evaluation='All55 scans pooled by original198-class evaluator, semantic and actualCA AP/AP50/AP25. Physical29 are disjoint from L200/B26597/D-v3 official100; scans within a physical scene are not independent samples.',
        exposure='Historical dev55 previously evaluated and diagnosed. Also a subset of official ScanNet training identities, so no claim of entire visual-system unseen/independent blind test. This tests the fixed downstream single-head transfer only.',
        GT_policy='No rawGT in audit/preparation/training/prediction/verification. Seal all55x2methodsx2views predictions before opening only bound dev55 GT. No dev55 targets/features in fitting or normalization.',
        acceptance='SL both mainAP strictly>S; all original10thresholds S presence and positive coverage retained. AP50/AP25/perclass/physical-scene effects reported; no posthoc criterion relaxation or tuning.',
        failure='Report result as observed even if eitherAP declines. No retraining/seed switch/threshold/budget/feature/architecture scan, no U/Uprime, no automatic production replacement.',
        prohibited=['reserve100 data','val312 data','official hidden-test payloads','paper figure assets'],
        new_training=True,new_visual_frontend=False,new_geometry=False,production_modified=False)
    ctl.base.verify_files(inputs);ctl.base.verify_files(training_inputs);ctl.base.verify_files(c['metadata_inputs']);ctl.base.verify_files(code)
    assert training_loop_ast(fit)==training_loop_ast(parent.original.fit),'optimization loop must remain exactly original'
    OUT.mkdir();freeze_json(OUT/'contract.json',c)
    remaining=set(split['train'])-set(inv['jobs'])
    audit=dict(complete=True,groups={k:dict(scans=len(v),physical=len(mains(v)),scene_names=v) for k,v in groups.items()},
        dev55_overlaps=overlaps,physical_disjoint_from_L200=True,physical_disjoint_from_B26597=True,
        Dv3_official100_in_B26597=True,historical_dev55_exposure=True,
        official_train_membership=True,entire_visual_system_unseen_not_established=True,
        additional_official_train_scans=len(remaining),additional_new_physical_scenes=sorted(mains(remaining)-mains(inv['jobs'])),
        GT_payload_opened=False,performance_used_for_selection=False,
        symlink_alias_scannet200=Path('/home/jia/Wm/Dataset/Scannet200').resolve()==Path('/home/jia/Wm/Dataset/scannet200').resolve())
    freeze_json(OUT/'identity_audit.json',audit)
    for q in code:
        d=OUT/'code_snapshot'/Path(q).relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,d)
    tests=subprocess.run([sys.executable,'-m','pytest','tests/test_fi1_sl_single200_dev55.py','tests/test_fi1_sl_inference.py',
        'tests/test_fi1_s_semantic_rank_residual.py','tests/test_fi1_union_development_replay.py','tests/test_fi1_class_agnostic_ablation.py','-q'],capture_output=True,text=True)
    (OUT/'tests.log').write_text(tests.stdout+tests.stderr);assert tests.returncode==0,tests.stdout+tests.stderr
    stage('prepared',train_scans=200,validation_scans=55,validation_physical=29)


def fit(c):
    guard=Guard(c,'fit');sys.addaudithook(guard.hook);ctl.base.verify_files(c['code']);ctl.base.verify_files(c['training_inputs'])
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    d=OUT/'model';d.mkdir();train=c['train_scenes'];fold='single200'
    stage('fit');a=parent.load_training_scenes(c,train)
    mean=a['x'].mean(axis=0,dtype=np.float64).astype(np.float32);std=np.maximum(a['x'].std(axis=0,dtype=np.float64),1e-6).astype(np.float32)
    pairs=core.make_pairs(a['S'],a['classes'],a['scene_ids'],a['identity'],a['states'],SEED)
    np.savez_compressed(d/'training_pairs.npz',**pairs);np.savez(d/'normalization.npz',mean=mean,std=std)
    model=core.Residual();optimizer=torch.optim.Adam(model.parameters(),lr=.001);x=torch.from_numpy(((a['x']-mean)/std).astype(np.float32));base=torch.from_numpy(core.base_logits(a['S']).astype(np.float32))
    p,n=pairs['pairs'].T;wt=pairs['weights']*len(p);rng=np.random.default_rng(SEED)
    anchors=np.concatenate([rng.choice(np.flatnonzero(a['classes']==ci),min(128,int((a['classes']==ci).sum())),replace=False) for ci in np.unique(a['classes'])])
    history=[];start=time.monotonic()
    # This loop is AST-checked against the frozen original fit, not re-designed.
    for epoch in range(core.EPOCHS):
        order=rng.permutation(len(p));losses=[]
        for offset in range(0,len(p),2048):
            ii=order[offset:offset+2048];pi,ni=p[ii],n[ii];dp=model(x[pi]);dn=model(x[ni]);w=torch.from_numpy(wt[ii].astype(np.float32))
            rank=(F.softplus(-(base[pi]+dp-base[ni]-dn))*w).mean()
            keep=(w*(dp.square()+dn.square())*.5).mean()
            ai=rng.choice(anchors,min(2048,len(anchors)),replace=False);anch=model(x[ai]).square().mean()
            loss=rank+core.ANCHOR*(keep+anch);optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses))));stage('fit_epoch',fold=fold,**history[-1])
    torch.save(model.state_dict(),d/'model.pt');delta=parent.original.infer(model,a['x'],mean,std);scores=core.adjusted(a['S'],delta)
    freeze_json(d/'training_metrics.json',dict(baseline=core.ranking_metrics(a['S'],pairs),residual=core.ranking_metrics(scores,pairs),
        history=history,train_scenes=train,validation_labels_read=False,seconds=time.monotonic()-start,max_abs_delta=float(abs(delta).max())))
    assert not guard.opened_gt
    freeze_json(d/'receipt.json',dict(complete=True,seed=SEED,train_scenes=train,validation_scenes_used=[],
        normalization_train_only=True,epochs=20,outputs={str(d/n):sha(d/n) for n in ('model.pt','normalization.npz','training_pairs.npz','training_metrics.json')}))
    freeze_json(OUT/'model_frozen.json',dict(complete=True,single_model=True,model_receipt_sha256=sha(d/'receipt.json'),
        model_sha256=sha(d/'model.pt'),normalization_sha256=sha(d/'normalization.npz'),train_scenes=train,seed=SEED,raw_GT_read=False))


def predict(c):
    guard=Guard(c,'predict');sys.addaudithook(guard.hook);torch.set_num_threads(4)
    ctl.base.verify_files(c['code']);ctl.base.verify_files(c['inputs']);ctl.base.verify_files(read(OUT/'model/receipt.json')['outputs'])
    b26=joblib.load(c['B_model']);model,mean,std=inference.load_L(OUT/'model/model.pt',OUT/'model/normalization.npz');outputs={};receipts={}
    for j,s in enumerate(c['scenes']):
        stage('predict',scene=s,completed=j,total=55);d=OUT/'predictions'/s;d.mkdir(parents=True)
        members,x26,scores=inference.build_pool(c['bindings'][s],b26)
        baseline=inference.source.rank(members,scores)[:800]
        rows,ca,indices,delta=inference.rescore_pool(members,x26,scores,model,mean,std)
        assert len(rows)==len(baseline)==800
        freeze_json(d/'members.json',members);np.savez_compressed(d/'pool.npz',x26=x26,scores=scores,indices=indices,delta=delta)
        for arm,sem,ag in (('S',baseline,ctl.collapse_members(baseline)),('SL',rows,ca)):
            freeze_json(d/f'semantic_{arm}.json',sem);freeze_json(d/f'class_agnostic_{arm}.json',ag)
        local={str(q):sha(q) for q in d.iterdir() if q.is_file()}
        freeze_json(d/'receipt.json',dict(complete=True,outputs=local,pool_members=len(members),raw_GT_read=False))
        outputs.update(local);receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    assert not guard.opened_gt
    freeze_json(OUT/'predictions_frozen.json',dict(complete=True,scenes=55,predictions=220,outputs=outputs,receipts=receipts,
        single_model_sha256=sha(OUT/'model/model.pt'),model_frozen_before_predictions=True,raw_GT_read=False))


def verify(c):
    guard=Guard(c,'verify');sys.addaudithook(guard.hook);torch.set_num_threads(4)
    seal=read(OUT/'predictions_frozen.json');ctl.base.verify_files(seal['outputs']);ctl.base.verify_files(seal['receipts'])
    model,mean,std=inference.load_L(OUT/'model/model.pt',OUT/'model/normalization.npz');b26=joblib.load(c['B_model']);count=Counter()
    for j,s in enumerate(c['scenes']):
        stage('verify_predictions',scene=s,completed=j,total=55);d=OUT/'predictions'/s
        members,x26,scores=inference.build_pool(c['bindings'][s],b26)
        assert members==read(d/'members.json')
        with np.load(d/'pool.npz') as z:assert np.array_equal(x26,z['x26']) and np.array_equal(scores,z['scores'])
        rows,ca,indices,delta=inference.rescore_pool(members,x26,scores,model,mean,std)
        assert rows==read(d/'semantic_SL.json') and ca==read(d/'class_agnostic_SL.json')
        with np.load(d/'pool.npz') as z:assert np.array_equal(indices,z['indices']) and np.array_equal(delta,z['delta'])
        old=read(d/'semantic_S.json');assert old==inference.source.rank(members,scores)[:800]
        assert read(d/'class_agnostic_S.json')==ctl.collapse_members(old)
        assert {ctl.shared.key(m) for m in old}=={ctl.shared.key(m) for m in rows}
        assert all((m['score']>0)==(next(x['score'] for x in rows if ctl.shared.key(x)==ctl.shared.key(m))>0) for m in old)
        count.update(pool_members=len(members),members=800,geometries=len(ca),zero_selected=sum(m['score']==0 for m in rows))
    assert not guard.opened_gt
    freeze_json(OUT/'inference_verified.json',dict(complete=True,counts=dict(count),full_frontend_rebuilt_exact=True,
        model_reload_exact=True,all_S_member_identity_positive_exact=True,predictions_seal_sha256=sha(OUT/'predictions_frozen.json'),raw_GT_read=False))


def evaluate(c):
    guard=Guard(c,'evaluate');sys.addaudithook(guard.hook);ctl.base.verify_files(c['code'])
    seal=read(OUT/'predictions_frozen.json');ctl.base.verify_files(seal['outputs'])
    assert read(OUT/'inference_verified.json')['predictions_seal_sha256']==sha(OUT/'predictions_frozen.json')
    ev=ctl.base.evaluator();ca,valid,uid=ctl.class_agnostic_evaluator();receipts={};coverage={}
    for j,s in enumerate(c['scenes']):
        stage('matching',scene=s,completed=j,total=55);d=OUT/'evaluation'/s;d.mkdir(parents=True)
        b=c['bindings'][s];assert sha(b['gt_path'])==b['gt_sha256']
        gt=np.loadtxt(b['gt_path'],dtype=np.int64);assert len(gt)==b['point_count']
        geometry,points=ctl.shared.load_points(b);sizes={k:len(v) for k,v in points.items()}
        sem={a:read(OUT/'predictions'/s/f'semantic_{a}.json') for a in ARMS};ag={a:read(OUT/'predictions'/s/f'class_agnostic_{a}.json') for a in ARMS}
        sm=ctl.shared.assign(ev,sem['S'],points,b['gt_path'],len(gt));checks=ctl.shared.check_match(sm,sem['S'],points,gt,ev)
        mapped=ctl.remap_gt(gt,valid,uid)
        def loader(path):assert Path(path).resolve()==Path(b['gt_path']).resolve();return mapped.copy()
        ca.util_3d.load_ids=loader
        cm=ctl.shared.assign(ca,ag['S'],points,b['gt_path'],len(gt),True);checks+=ctl.shared.check_match(cm,ag['S'],points,mapped,ca,True)
        projected={}
        for view,template,rr in (('semantic',sm,sem),('class_agnostic',cm,ag)):
            template=ctl.base.serializable_matches(template);projected[view]={'S':template}
            if view=='semantic':other=ctl.project_multiclass(template,rr['S'],rr['SL'],sizes,ev.CLASS_LABELS,ev.VALID_CLASS_IDS)
            else:other=ctl.project_matches(template,[m['geometry_key'] for m in rr['S'] if sizes[m['geometry_key']]>=100],rr['SL'],ca.CLASS_LABELS[0])
            independent=parent.fixed.independent_rescore_match(template,rr['S'],rr['SL'],sizes,view=='class_agnostic');assert independent==other
            projected[view]['SL']=other
            for arm,match in projected[view].items():freeze_json(d/f'{view}_{arm}_matches.json',match)
        # Cover every evaluator threshold, including ignored/small target rules.
        scene_cov={}
        for view,evaluator in (('semantic',ev),('class_agnostic',ca)):
            scene_cov[view]={}
            for t in map(float,evaluator.opt['overlaps']):
                hits={}
                for arm,match in projected[view].items():
                    presence=set();positive=set()
                    for gg in match['gt'].values():
                        for g in gg:
                            if g['instance_id']<1000 or g['vert_count']<100:continue
                            pp=[p for p in g['matched_pred'] if p['intersection']/(g['vert_count']+p['vert_count']-p['intersection'])>t]
                            if pp:presence.add(g['instance_id'])
                            if any(p['confidence']>0 for p in pp):positive.add(g['instance_id'])
                    hits[arm]=(presence,positive)
                assert hits['S']==hits['SL'];scene_cov[view][f'{t:.2f}']=dict(presence=len(hits['SL'][0]),positive=len(hits['SL'][1]),lost=0,gained=0)
        coverage[s]=scene_cov
        local={str(q):sha(q) for q in d.iterdir() if q.is_file()}
        freeze_json(d/'receipt.json',dict(complete=True,outputs=local,GT_sha256=b['gt_sha256'],independent_intersection_edges=checks,SL_bidirectional_projection_exact=True))
        receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
    assert len(guard.opened_gt)==55
    freeze_json(OUT/'matches_frozen.json',dict(complete=True,receipts=receipts,GT_scenes_opened=55,predictions_before_GT=True))
    freeze_json(OUT/'coverage_verified.json',dict(complete=True,scenes=coverage,all10_thresholds_S_presence_and_positive_exact=True,
        recovered424='not applicable outside development200; no GT-defined exceptions'))
    ap={};arrays={}
    for view,evaluator in (('semantic',ev),('class_agnostic',ca)):
        ap[view]={}
        for arm in ARMS:
            stage('AP',view=view,arm=arm)
            matches={s:read(OUT/'evaluation'/s/f'{view}_{arm}_matches.json') for s in c['scenes']}
            x,*_=evaluator.evaluate_matches(matches)
            reverse,*_=evaluator.evaluate_matches(dict(reversed(list(matches.items()))));assert np.allclose(x,reverse,rtol=0,atol=1e-12,equal_nan=True)
            ap[view][arm]=ctl.base.metrics(evaluator,x);q=OUT/'evaluation'/f'development55_{view}_{arm}_AP.npy';np.save(q,x);arrays[str(q)]=sha(q)
            print('result',view,arm,ap[view][arm],flush=True)
    freeze_json(OUT/'evaluation_summary.json',dict(complete=True,AP=ap,AP_arrays=arrays,pooled_scans=55,physical_scenes=29,
        four_reverse_order_replays=True,validation_is_blind=False,single_model=True))


def report(c):
    ap=read(OUT/'evaluation_summary.json')['AP'];both=all(ap[v]['SL']['ap']>ap[v]['S']['ap'] for v in VIEWS)
    coverage=read(OUT/'coverage_verified.json')['all10_thresholds_S_presence_and_positive_exact']
    lines=['# 单模型训练200 → 开发55验证：S与SL','',
        '固定原种子20261010，全部开发200训练一个L，20轮无选checkpoint。完整开发55含29个物理场景，与L训练200、B26训练597和D-v3官方100的物理场景无交集。',
        '', '开发55曾参与历史评测／诊断，且位于官方ScanNet训练划分；因此本项是固定下游单模型跨场景验证，不称独立盲测或整个视觉系统未见集。未读取验证312、已消费保留100或官方隐藏测试数据。','',
        '|方法|类别AP／AP50／AP25 (%)|实际去类AP／AP50／AP25 (%)|','|---|---:|---:|']
    for arm in ARMS:
        vals=['／'.join(f'{100*ap[v][arm][k]:.6f}' for k in ('ap','ap50','ap25')) for v in VIEWS];lines.append(f'|{arm}|{vals[0]}|{vals[1]}|')
    delta={v:{k:100*(ap[v]['SL'][k]-ap[v]['S'][k]) for k in ('ap','ap50','ap25')} for v in VIEWS}
    lines+=['',f"SL−S主AP：类别{delta['semantic']['ap']:+.6f}pp，实际去类{delta['class_agnostic']['ap']:+.6f}pp。双AP通过={both}；S全部原IoU档存在及正分覆盖保护={coverage}。",'',
        '两方法使用同一完整NT／S资格池和S800名单，只改变最终分数。训练监督、归一化均仅来自200；没有从55挑场景、调参、改阈值或选择种子。所有220份预测输出先封存并重放核验，随后才读取绑定的55份GT，原评测器重新决定TP/FP并计算AP。',
        '', '55扫描存在同一物理场景的重复扫描，结果按标准扫描协议汇总，不能当55个独立物理样本；不计算虚假的独立扫描置信区间。424子群属于开发200，本集不适用；保护按本集所有S已覆盖目标逐档核验。',
        '', '无论本轮结果如何均保留，不按验证反馈重训或换种子，不追加U/U′，不自动替换生产Full。独立未暴露测试仍需另行建立身份和评测入口。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    freeze_json(OUT/'decision.json',dict(complete=True,both_AP_passed=both,coverage_passed=coverage,accepted_for_downstream_transfer=both and coverage,
        validation_is_blind=False,delta_pp=delta,production_modified=False,next_training=False))
    stage('results_complete')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','predict','verify','evaluate','report']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:globals()[a.stage](read(OUT/'contract.json'))


if __name__=='__main__':main()
