#!/usr/bin/env python3
"""Frozen same-parent query-only pilot; expansion requires every registered gate."""
from pathlib import Path
import argparse, hashlib, json, shutil, sys, time
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from tools.fi1_candidate_local_segmenter import CandidateLocalSegmenter, repair, PARAMETERS
from tools.fi1_deletion_block_core import corrected_loss
from tools.fi1_frontend_protocol import sha256_file as sha, freeze_json
import run_fi1_nt_region_identity as diagnostic
import evaluate_fi1_nt_region_identity as metrics

OUT = ROOT/'research/geometry/experiments/fi1_query_same_parent_pilot_20261014_v1'
DIAG = diagnostic.OUT
SOURCE = diagnostic.SOURCE
LABELS = ROOT/'.local/fi1_deletion_blocks_development200_20261013_v1/labels'
SEED = 20261014
MODES = ('A','B','B_zero','B_shuffle_0','B_shuffle_1','B_shuffle_2')
SHUFFLE_SEEDS = (2026101401,2026101402,2026101403)

def read(p): return json.loads(Path(p).read_text())
def seal(p, v): freeze_json(p, diagnostic.plain(v))
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def reserve():
    for p in (ROOT, diagnostic.AB.resolve()):
        assert shutil.disk_usage(p).free >= 50*2**30, '50GiB disk reserve'

def inputs(e, mode):
    q=e['query'].copy()
    if mode in ('A','B_zero'): q.fill(0)
    elif mode.startswith('B_shuffle_'):
        i=int(mode.rsplit('_',1)[1])
        seed=int.from_bytes(hashlib.sha256(f'{SHUFFLE_SEEDS[i]}:{e["scene"]}:{e["candidate_id"]}'.encode()).digest()[:8],'little')
        q=q[np.random.default_rng(seed).permutation(len(q))]
    return np.concatenate((e['base'], e['head'], q[:,None]),1).astype(np.float32)

def collate(ee,mode,device):
    xx=[]; nn=[]; ww=[]; parts=[]; start=0
    for e in ee:
        xx.append(inputs(e,mode));nn.append(e['neighbors']+start);ww.append(e['weights'])
        parts.append((start,start+len(e['halo'])));start=parts[-1][1]
    return (torch.tensor(np.concatenate(xx),device=device),
            torch.tensor(np.concatenate(nn),device=device,dtype=torch.long),
            torch.tensor(np.concatenate(ww),device=device),parts)

def load(supervised=True):
    c=read(OUT/'contract.json'); norm=np.load(OUT/'normalization.npz');data={}
    for r in c['samples']:
        s,cid=r['scene'],r['candidate_id']
        with np.load(OUT/'inputs'/s/(cid+'.npz')) as z:
            e={k:z[k].copy() for k in z.files if supervised or k!='y'}
        e.update(scene=s,candidate_id=cid,good=r['loss_good'],source=r['source'])
        e['head']=np.clip((e['head'].astype(np.float32)-norm['mean'])/norm['std'],-10,10)
        data[s,cid]=e
    norm.close();return data

def guard(stage,c):
    allowed=set(c['scenes'])
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(args[0]); scene=next((s for s in c['all_scenes'] if s in p),None)
        if scene and scene not in allowed:raise PermissionError('pilot scene isolation: '+p)
        if any(v in p for v in ('ground_truth','/labels/','/structure_diagnosis/','/prior_diagnosis/')):
            raise PermissionError(stage+' must use frozen pilot inputs, no raw GT: '+p)
    sys.addaudithook(hook)

def prepare():
    assert not OUT.exists(), 'isolated new output root required'
    reserve();dc=read(DIAG/'contract.json');ab=read(diagnostic.AB/'contract.json')
    assert set(dc['scenes'])<=set(ab['splits']['fit'])
    rows=[];entries={};bindings={};gt_bindings={}
    for s in dc['scenes']:
        gt=metrics.load_gt(dc,s);gt_bindings[dc['jobs'][s]['GT']]=sha(dc['jobs'][s]['GT'])
        ids,sizes=metrics.valid_targets(gt,dc['class_ids'])
        paths=[SOURCE/'features'/s/'data.npz',LABELS/s/'data.npz',LABELS/s/'meta.json',DIAG/'priors'/s/'data.npz',DIAG/'associations'/s/'links.json',DIAG/'associations'/s/'data.npz',DIAG/'structure'/s/'data.npz']
        bindings.update({str(p):sha(p) for p in paths})
        meta={r['candidate_id']:r for r in read(LABELS/s/'meta.json')}
        links={r['candidate_id']:r for r in read(paths[4]) if r['prior']=='P1'}
        with np.load(paths[0]) as z,np.load(paths[1]) as lab,np.load(paths[3]) as pr,np.load(paths[6]) as st:
            for r in dc['parents'][s]:
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];visible=z[cid+'_gd10_selected6_groups'];y=lab[cid+'_y'].astype(np.int64)
                im=metrics.identity(parent,gt,ids,sizes)
                assert np.array_equal(parent,st[cid+'_parent']) and np.array_equal(halo,st[cid+'_halo'])
                assert np.array_equal(visible,st[cid+'_visible'])
                expected=np.full(len(halo),2,np.int64)
                if im['stable']: expected[gt[halo]>=1000]=1;expected[gt[halo]==im['target']]=0
                assert np.array_equal(y,expected), (s,cid,'same original-parent labels')
                assert meta[cid]['stable']==im['stable'] and (not im['stable'] or meta[cid]['target']==im['target'])
                member=np.isin(halo,parent);can=visible>=2
                add=int((~member&can&(y==0)).sum());delete=int((member&can&(y==1)).sum())
                delete=delete if len(parent)-delete>=100 else 0
                row=dict(scene=s,candidate_id=cid,source=r['source'],stable=im['stable'],target=im['target'],parent_iou=im['parent_iou'],purity=im['purity'],good=im['stable'] and im['parent_iou']>=.9,loss_good=meta[cid]['good'],addition_opportunity=add,deletion_opportunity=delete,halo_points=len(halo),parent_points=len(parent),parent_key=r['parent'])
                rows.append(row)
                if not im['stable']:continue
                ix=np.searchsorted(pr['point_ids'],halo);assert np.array_equal(pr['point_ids'][ix],halo)
                q=links[cid]['parent']['top'][0]; row['query_id']=q
                base=z[cid+'_gd10_selected6_x'].astype(np.float32);assert np.array_equal(base[:,9]>.5,member)
                entries[s,cid]=dict(base=base,head=pr['P1_head_roi'][ix],query=pr['P1_probability'][ix,q],neighbors=z[cid+'_neighbors'].astype(np.int64),weights=z[cid+'_neighbor_weights'].astype(np.float32),parent=parent,halo=halo,visible=visible,y=y)
    selected=[];used=set();counts={}
    for source in ('native','track'):
        for category,n in (('good',2),('deletion',3),('addition',3)):
            candidates=[r for r in rows if r['source']==source and r['stable'] and (r['good'] if category=='good' else r[category+'_opportunity']>0)]
            counts[source+':'+category]=len(candidates)
            candidates.sort(key=lambda r:hashlib.sha256(f'query-pilot-20261014:{source}:{category}:{r["scene"]}:{r["candidate_id"]}'.encode()).hexdigest())
            chosen=[r for r in candidates if (r['scene'],r['candidate_id']) not in used][:n]
            assert len(chosen)==n, ('insufficient distinct stratum',source,category)
            for r in chosen:r=dict(r,category=category);selected.append(r);used.add((r['scene'],r['candidate_id']))
    OUT.mkdir(parents=True);outputs={};su=np.zeros(128,np.float64);sq=su.copy();points=0
    for r in selected:
        e=entries[r['scene'],r['candidate_id']];p=OUT/'inputs'/r['scene']/(r['candidate_id']+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,**e);outputs[str(p)]=sha(p)
        r.update(labels_sha256=digest(e['y']),ignore_mask_sha256=digest(np.zeros(len(e['y']),bool)),known_dice_mask_sha256=digest(e['y']!=2),parent_membership_sha256=digest(np.isin(e['halo'],e['parent'])),query_sha256=digest(e['query']))
        v=e['head'].astype(np.float64);su+=v.sum(0);sq+=(v*v).sum(0);points+=len(v)
    mean=su/points;std=np.sqrt(np.maximum(sq/points-mean**2,1e-6));np.savez_compressed(OUT/'normalization.npz',mean=mean.astype(np.float32),std=std.astype(np.float32));outputs[str(OUT/'normalization.npz')]=sha(OUT/'normalization.npz')
    # Numerical continuation criteria are registered before any network execution.
    gates=dict(mean_B_minus_A_min=.01,parents_B_minus_A_ge02_min=3,source_mean_B_minus_A_min=0,correct_action_points_min=20,correct_action_parents_min=2,add_precision_min=.9,delete_precision_min=.9,target_loss_max=.01,unknown_fraction_max=.1,good_loss_max=.02,good_B_minus_A_loss_max=.005,drift_max=0,intervention_mean_iou_drop_min=.005)
    files=[Path(__file__).resolve(),ROOT/'research/geometry/code/tests/test_fi1_query_same_parent_pilot.py',ROOT/'research/geometry/code/tools/fi1_candidate_local_segmenter.py',ROOT/'research/geometry/code/tools/fi1_deletion_block_core.py',ROOT/'research/geometry/code/tools/evaluate_fi1_nt_region_identity.py',ROOT/'research/geometry/code/tools/run_fi1_nt_region_identity.py',ROOT/'research/geometry/code/tools/fi1_frontend_protocol.py',ROOT/'research/geometry/code/tools/run_fi1_query_same_parent_pilot.sh']
    files=[(ROOT/'tools/fi1_frontend_protocol.py').resolve() if p.name=='fi1_frontend_protocol.py' else p for p in files]
    code={str(p):sha(p) for p in files}
    c=dict(version=OUT.name,seed=SEED,samples=selected,scenes=sorted({r['scene'] for r in selected}),all_scenes=ab['scenes'],splits=ab['splits'],jobs={s:dc['jobs'][s] for s in dc['scenes']},class_ids=dc['class_ids'],input_bindings=bindings,GT_inputs=gt_bindings,outputs=outputs,code=code,source_counts=counts,epochs=64,batch=4,max_batch_points=60000,learning_rate=.001,weight_decay=.0001,clip=5,loss='shared corrected_loss, parent-other3 CE + known Dice.5 + existing good .25; candidate equal weighting',network='CandidateLocalSegmenter(193,64), shared head128; A query0/B P1 sigmoid',selection='fixed final epoch64; no model/threshold selection',query_selection='existing frozen GT-free full-scene soft-IoU top1 P1 stable ID tie',shuffle_seeds=SHUFFLE_SEEDS,gates=gates,threshold=.8,min_groups=2,min_points=100,region_baseline_threshold=.5,scope='training ability pilot, no generalized/AP claim',sample_selection='good/delete/add, hash order, no query quality filtering',frozen_SL_models=dc['frozen_SL_models'])
    seal(OUT/'sampling_audit.json',dict(all_training_strata=rows,selection_uses_query_accuracy=False))
    seal(OUT/'contract.json',c)
    for p in files:
        dst=OUT/'code_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    print('prepared',len(selected),'parents',points,'halo points',counts,flush=True)

def train():
    c=read(OUT/'contract.json');diagnostic.verify(c['code']);diagnostic.verify(c['outputs']);reserve();guard('train',c)
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True);data=load();keys=sorted(data);receipts={}
    for arm in ('A','B'):
        d=OUT/'models'/arm;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():diagnostic.verify(read(d/'receipt.json')['outputs']);receipts.update(read(d/'receipt.json')['outputs']);continue
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);net=CandidateLocalSegmenter(193,64).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest();history=[];first=0
        if (d/'checkpoint.pt').exists():
            state=torch.load(d/'checkpoint.pt',weights_only=False,map_location='cuda');net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);history=state['history'];first=state['epoch'];torch.set_rng_state(state['rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        for epoch in range(first,c['epochs']):
            net.train();order=np.random.default_rng(SEED+epoch).permutation(len(keys));losses=[];tic=time.time();order_log=[]
            for start in range(0,len(order),4):
                batchkeys=[keys[i] for i in order[start:start+4]]
                batches=[batchkeys] if sum(len(data[k]['halo']) for k in batchkeys)<=60000 else [[k] for k in batchkeys]
                for kk in batches:
                    ee=[data[k] for k in kk];x,nn,w,parts=collate(ee,arm,'cuda');opt.zero_grad(set_to_none=True);logits=net(x,nn,w,parts)
                    loss=sum(corrected_loss(logits[a:b],torch.tensor(e['y'],device='cuda'),x[a:b,9]>.5,e['good']) for e,(a,b) in zip(ee,parts))/len(ee)
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(float(loss.detach()));order_log.append([':'.join(k) for k in kk])
            h=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.time()-tic,steps=len(losses),batch_order=order_log);history.append(h)
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),d/'checkpoint.pt')
            print('training',arm,json.dumps(h),flush=True);reserve()
        torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},d/'model.pt')
        seal(d/'history.json',dict(arm=arm,initial_state_sha256=initial,epochs=history,selected_epoch=64,parameters=sum(p.numel() for p in net.parameters()),no_model_selection=True))
        oo={str(d/n):sha(d/n) for n in ('model.pt','history.json','checkpoint.pt')};seal(d/'receipt.json',dict(complete=True,outputs=oo));receipts.update(oo);del net,opt;torch.cuda.empty_cache()
    ha,hb=[read(OUT/'models'/a/'history.json') for a in ('A','B')]
    assert ha['initial_state_sha256']==hb['initial_state_sha256'] and all(a['batch_order']==b['batch_order'] for a,b in zip(ha['epochs'],hb['epochs']))
    seal(OUT/'models_frozen.json',dict(complete=True,outputs=receipts,final_epoch=64))

def predict():
    c=read(OUT/'contract.json');diagnostic.verify(c['code']);diagnostic.verify(read(OUT/'models_frozen.json')['outputs']);guard('predict',c);torch.set_num_threads(4)
    data=load(False);outputs={}
    for arm in ('A','B'):
        net=CandidateLocalSegmenter(193,64).cuda().eval();net.load_state_dict(torch.load(OUT/'models'/arm/'model.pt',weights_only=True,map_location='cuda'))
        for (s,cid),e in sorted(data.items()):
            d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True);arrays={}
            for mode in (('A',) if arm=='A' else MODES[1:]):
                with torch.inference_mode():
                    x,nn,w,parts=collate([e],mode,'cuda');p=net(x,nn,w,parts).softmax(1).cpu().numpy().astype(np.float32)
                points,_=repair(e['parent'],e['halo'],p,e['visible'])
                arrays[mode+'_probability']=p;arrays[mode+'_points']=points
            path=d/(cid+'_'+arm+'.npz');np.savez_compressed(path,**arrays);outputs[str(path)]=sha(path)
        del net;torch.cuda.empty_cache()
    seal(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,GT_read=False,interventions_no_retraining=True))

def decisions(c,rows):
    g=c['gates'];by={(r['scene'],r['candidate_id'],r['mode']):r for r in rows};paired=[]
    for r in c['samples']:
        k=r['scene'],r['candidate_id'];a=by[*k,'A'];b=by[*k,'B']
        paired.append(dict(scene=k[0],candidate_id=k[1],source=r['source'],category=r['category'],advantage=b['repaired_iou']-a['repaired_iou'],zero_drop=b['repaired_iou']-by[*k,'B_zero']['repaired_iou'],shuffle_drop=b['repaired_iou']-np.mean([by[*k,f'B_shuffle_{i}']['repaired_iou'] for i in range(3)])))
    bb=[r for r in rows if r['mode']=='B'];m=metrics.summarize_rows(bb);good=[r for r in bb if r['good_parent']]
    checks=dict(mean_advantage=np.mean([r['advantage'] for r in paired])>=g['mean_B_minus_A_min'],multi_parent_advantage=sum(r['advantage']>=.02-1e-12 for r in paired)>=g['parents_B_minus_A_ge02_min'],both_sources=all(np.mean([r['advantage'] for r in paired if r['source']==s])>=g['source_mean_B_minus_A_min'] for s in ('native','track')),safe_add=(m['add_target_precision'] or 0)>=g['add_precision_min'],safe_delete=(m['delete_pollution_precision'] or 0)>=g['delete_precision_min'],target_preservation=(m['target_loss'] if m['target_loss'] is not None else 1)<=g['target_loss_max'],unknown=(m['unknown_fraction'] if m['unknown_fraction'] is not None else 1)<=g['unknown_fraction_max'],good_preservation=m['worst_good_loss']<=g['good_loss_max'],good_not_worse_than_A=all(r['good_loss']-by[r['scene'],r['candidate_id'],'A']['good_loss']<=g['good_B_minus_A_loss_max'] for r in good),no_drift=m['drift']<=g['drift_max'],zero_dependency=np.mean([r['zero_drop'] for r in paired])>=g['intervention_mean_iou_drop_min'],shuffle_dependency=np.mean([r['shuffle_drop'] for r in paired])>=g['intervention_mean_iou_drop_min'])
    for name,field in (('addition','added_target'),('deletion','deleted_other')):
        checks[name+'_learned']=sum(r[field] for r in bb)>=g['correct_action_points_min'] and sum(r[field]>0 for r in bb)>=g['correct_action_parents_min']
    return dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],paired=paired,mean_advantage=float(np.mean([r['advantage'] for r in paired])),mean_zero_drop=float(np.mean([r['zero_drop'] for r in paired])),mean_shuffle_drop=float(np.mean([r['shuffle_drop'] for r in paired])),continue_scene_isolated=all(checks.values()))

def evaluate():
    c=read(OUT/'contract.json');assert (OUT/'predictions_frozen.json').exists();diagnostic.verify(read(OUT/'predictions_frozen.json')['outputs']);diagnostic.verify(c['GT_inputs']);rows=[];pool=[];classes=[]
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=metrics.valid_targets(gt,c['class_ids']);_,nt=diagnostic.load_nt(c['jobs'][s]);baseline=metrics.pool_base(nt,gt,ids,sizes)
        best={m:baseline.copy() for m in (*MODES,'region_average','original_NT')};original={digest(p.astype('<i8')) for p in nt.values()}
        with np.load(DIAG/'associations'/s/'data.npz') as region:
            for r in (r for r in c['samples'] if r['scene']==s):
                cid=r['candidate_id']
                with np.load(OUT/'inputs'/s/(cid+'.npz')) as inp:
                    parent=inp['parent'];im=metrics.identity(parent,gt,ids,sizes)
                    assert im['target']==r['target'];y=inp['y'];member=np.isin(inp['halo'],parent);editable=inp['visible']>=2
                    for arm in ('A','B','region_average'):
                        z=None if arm=='region_average' else np.load(OUT/'predictions'/s/(cid+'_'+arm+'.npz'))
                        for mode in (('region_average',) if arm=='region_average' else ('A',) if arm=='A' else MODES[1:]):
                            p=region[cid+'_P1_region'] if z is None else z[mode+'_points'];row,after=metrics.outcome(parent,p,gt,ids,sizes,im,baseline)
                            row.update(scene=s,candidate_id=cid,source=r['source'],mode=mode,category=r['category'],duplicate_original=row['changed'] and digest(p.astype('<i8')) in original,duplicate_generated=False)
                            rows.append(row);best[mode]=np.maximum(best[mode],after)
                            if z is not None:
                                prob=z[mode+'_probability'];pred=prob.argmax(1);classes.append(dict(scene=s,candidate_id=cid,mode=mode,points=len(y),accuracy=float((pred==y).mean()),known_accuracy=float((pred[y!=2]==y[y!=2]).mean()),visible_add_target_ge08=int((~member&editable&(y==0)&(prob[:,0]>=.8)).sum()),visible_delete_other_ge08=int((member&editable&(y==1)&(prob[:,1]>=.8)).sum()),loss=float(corrected_loss(torch.tensor(np.log(np.maximum(prob,1e-30))),torch.tensor(y),torch.tensor(member),r['loss_good']))))
                        if z is not None:z.close()
        for mode in best:pool+=metrics.pool_rows(s,mode,baseline,best[mode],ids)
    decision=decisions(c,rows);summary={m:dict(**metrics.summarize_rows([r for r in rows if r['mode']==m]),**metrics.pool_summary([r for r in pool if r['mode']==m])) for m in (*MODES,'region_average')}
    grouped={f'{m}/{source}/{cat}':metrics.summarize_rows([r for r in rows if r['mode']==m and r['source']==source and (cat=='all' or r['category']==cat)]) for m in (*MODES,'region_average') for source in ('native','track') for cat in ('all','addition','deletion','good')}
    d=OUT/'evaluation';d.mkdir(exist_ok=True)
    for name,rr in (('parents',rows),('pool',pool),('classification',classes),('paired',decision['paired'])):metrics.csv_write(d/(name+'.csv'),rr)
    seal(OUT/'decision.json',decision);seal(d/'summary.json',dict(modes=summary,grouped=grouped,decision=decision,training_only=True))
    print(json.dumps(diagnostic.plain(dict(decision=decision,modes=summary)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','train','predict','evaluate'));a=p.parse_args();globals()[a.mode]()
