#!/usr/bin/env python3
"""Nested cross-fitted, fixed-observation deletion-block experiment."""
from pathlib import Path
from collections import Counter
import argparse,sys,time,shutil
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_local_edit_ablation as previous
from tools import fi1_deletion_block_core as core
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
SOURCE=previous.SOURCE;PREVIOUS=previous.OUT
OUT=ROOT/'.local/fi1_deletion_blocks_development200_20261013_v1'
ARMS=('B_raw','C_gate','G_raw','B_summary','B_relation','G_summary','G_relation')
read=previous.read;OBS=previous.OBS


def plain(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def seal(path,value):freeze_json(path,plain(value))


def stage(name,**kw):
    write_json(OUT/'execution.json',dict(stage=name,updated=time.time(),**plain(kw)));print(name,plain(kw),flush=True)


class Guard:
    def __init__(self,c,mode,allowed=()):self.c,self.mode,self.allowed,self.outer=c,mode,set(allowed),None
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,Path)):return
        p=Path(args[0]).resolve();name=str(p)
        if p.suffix in ('.py','.pyc','.so'):return
        if 'independent_analysis' in name or '/diagnosis/' in name or p.name in ('candidate_diagnosis.csv','sampling_diagnosis.csv','component_probe.json'):
            raise PermissionError('posthoc diagnostic inputs excluded')
        raw=any(part.startswith('ground_truth') for part in p.parts)
        if raw:
            if self.mode=='labels':allowed=name in self.c['GT_inputs']
            elif self.mode=='block_labels':allowed=any(name==self.c['jobs'][s]['GT'] for s in self.allowed)
            elif self.mode=='evaluate':allowed=(OUT/'predictions_frozen.json').exists() and name in self.c['GT_inputs']
            else:allowed=False
            if not allowed:raise PermissionError('raw GT forbidden for this stage/fold')
        if '/labels/' in name and any(str(root.resolve()) in name for root in (OUT,PREVIOUS,SOURCE)):
            if self.mode not in ('labels','train','block_labels','evaluate'):raise PermissionError('inference label cache forbidden')
            if self.mode in ('train','block_labels') and not any('/'+s+'/' in name for s in self.allowed):raise PermissionError('heldout label cache forbidden')
        if '/training_blocks/' in name:
            if self.mode not in ('verifier_train','evaluate'):raise PermissionError('block targets forbidden in inference')
            if self.mode=='verifier_train' and '/outer'+str(self.outer)+'/' not in name:raise PermissionError('another outer fold block labels forbidden')


def prepare():
    assert not OUT.exists();assert shutil.disk_usage(ROOT).free>=50*1024**3
    c=read(PREVIOUS/'contract.json');assert read(PREVIOUS/'completion_verified.json')['complete']
    bindings={str(PREVIOUS/name):sha(PREVIOUS/name) for name in ('contract.json','conditions_frozen.json','labels_frozen.json','predictions_frozen.json','completion_verified.json')}
    for name in ('conditions','labels','predictions'):bindings.update(read(PREVIOUS/(name+'_frozen.json'))['outputs'])
    for name in ('plan','features'):bindings.update(read(SOURCE/(name+'_frozen.json'))['outputs'])
    files=[ROOT/'tools'/n for n in ('run_fi1_deletion_blocks.py','fi1_deletion_block_core.py','evaluate_fi1_deletion_blocks.py','verify_fi1_deletion_blocks.py','run_fi1_deletion_blocks.sh')]
    files+=[ROOT/'tests/test_fi1_deletion_blocks.py',ROOT/'docs/FI1_DELETION_BLOCK_PROTOCOL_20261013.md']
    code=dict(c['code']);code.update({str(p):sha(p) for p in files})
    contract={k:c[k] for k in ('scenes','folds','groups','fixed10','class_ids','jobs','GT_inputs','inputs','frozen_SL_models','gate')}
    contract.update(version=OUT.name,parameters=core.PARAMETERS,arms=ARMS,code=code,source_bindings=bindings,
                    nested_proposals='20 inner fits: outer-training original 4 folds, train3/predict1; 5 outer fits train4/predict outer. Verifiers never train on globally cross-fitted old B.',
                    frozen_B_transfer='B test probabilities produced by original outer B models. New verifiers train on corrected G nested proposals; this input distribution shift is explicit, not an isolated B-only recipe.',
                    GT_free_partition='Existing frozen undirected 8-NN components of parent visible>=2 and p(other)>=.8; no target-based split or candidate filtering.',
                    identity_supervision='Best valid target only if parent purity>=.6. Other parents fully unknown, not forced target.',
                    sampling='Every point once, pollution CE cost bounded3, full-domain Dice/preservation; every block once each epoch; per-scene/per-parent weights, positive cost<=5.',
                    action='Deletion only; all blocks p(safe)>=.9 jointly removed, final min100 fallback; no outer-fold calibration.',
                    success='Original conservative geometry gate and >=1 pure deletion NT-pool gain>=.02; reduced actions alone do not pass.',
                    SL_modified=False,AP_run=False,new_frames=False,new_backbone=False)
    OUT.mkdir();seal(OUT/'contract.json',contract)
    for p in files:
        target=OUT/'code_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    stage('prepared',scenes=200,parents=1599)


def labels():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'labels').hook);outputs={};counts=Counter()
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);metadata={r['candidate_id']:r for r in read(SOURCE/'labels'/s/'meta.json')};arrays={};rows=[]
        with np.load(SOURCE/'features'/s/'data.npz') as z:
            for row in read(SOURCE/'plan'/s/'candidates.json'):
                cid=row['candidate_id'];y,meta=core.stable_supervision(gt,z[cid+'_parent'],z[cid+'_halo'],metadata[cid],c['class_ids']);arrays[cid+'_y']=y
                rows.append(dict(meta,candidate_id=cid));counts['stable' if meta['stable'] else 'unknown']+=1
                counts['good']+=meta['good']
        d=OUT/'labels'/s;d.mkdir(parents=True);np.savez_compressed(d/'data.npz',**arrays);seal(d/'meta.json',rows)
        outputs.update({str(d/n):sha(d/n) for n in ('data.npz','meta.json')})
    seal(OUT/'labels_frozen.json',dict(complete=True,outputs=outputs,counts=dict(counts)));stage('corrected_supervision_frozen',**dict(counts))


def examples(scenes,supervised=False):
    data={}
    for s in scenes:
        with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(PREVIOUS/'conditions'/s/'data.npz') as cond:
            lab=np.load(OUT/'labels'/s/'data.npz') if supervised else None
            meta={r['candidate_id']:r for r in read(OUT/'labels'/s/'meta.json')} if supervised else None
            for row in read(SOURCE/'plan'/s/'candidates.json'):
                cid=row['candidate_id'];e=dict(x=z[cid+'_'+OBS+'_x'].astype(np.float32),neighbors=z[cid+'_neighbors'].astype(np.int64),weights=z[cid+'_neighbor_weights'].astype(np.float32),parent=z[cid+'_parent'],halo=z[cid+'_halo'],visible=z[cid+'_'+OBS+'_groups'])
                e['condition']={name:cond[cid+'_'+name] for name in ('membership','distance','valid','overlap','representatives','own_distance')}
                if supervised:e.update(y=lab[cid+'_y'].astype(np.int64),meta=meta[cid])
                data[s,cid]=e
            if lab is not None:lab.close()
    return data


def forward(model,ee,device='cuda'):
    x,nn,w,parts=previous.original.collate(ee,device=device);return model(x,nn,w,parts),x,parts


def fit_proposal(c,guard,fold,inner):
    exclude={fold} if inner is None else {fold,inner};train=[s for s in c['scenes'] if c['folds'][s] not in exclude]
    test=[s for s in c['scenes'] if c['folds'][s]==(fold if inner is None else inner)]
    d=OUT/'generators'/f'outer{fold}'/('full' if inner is None else f'inner{inner}');d.mkdir(parents=True,exist_ok=True)
    model=previous.original.core.CandidateLocalSegmenter().cuda()
    if (d/'receipt.json').exists():
        previous.original.audit.prior.old.verify_files(read(d/'receipt.json')['outputs']);model.load_state_dict(torch.load(d/'model.pt',weights_only=True));return model,test,d
    guard.mode='train';guard.allowed=set(train);data=examples(train,True);keys=sorted(data)
    seed=20261013+fold*100+(5 if inner is None else inner);torch.manual_seed(seed)
    model=previous.original.core.CandidateLocalSegmenter().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[];scene_counts=Counter(s for s,cid in keys);epoch_start=0
    if (d/'checkpoint.pt').exists():
        state=torch.load(d/'checkpoint.pt',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];epoch_start=state['epoch'];torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    for epoch in range(epoch_start,12):
        model.train();losses=[];tic=time.time();order=np.random.default_rng(seed+100000+epoch).permutation(len(keys))
        for start in range(0,len(order),4):
            kk=[keys[k] for k in order[start:start+4]];batches=[kk] if sum(len(data[k]['x']) for k in kk)<=60000 else [[k] for k in kk]
            for batch in batches:
                ee=[data[k] for k in batch];optimizer.zero_grad(set_to_none=True);logits,x,parts=forward(model,ee);loss=0
                for k,e,(lo,hi) in zip(batch,ee,parts):
                    y=torch.tensor(e['y'],device='cuda');value=core.corrected_loss(logits[lo:hi],y,x[lo:hi,9]>.5,e['meta']['good'])
                    loss=loss+value*(len(keys)/len(scene_counts)/scene_counts[k[0]])/len(ee)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.time()-tic))
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,epoch=epoch+1,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),d/'checkpoint.pt')
        stage('nested_proposal_fit',outer=fold,inner=inner,**history[-1])
    torch.save(model.state_dict(),d/'model.pt');seal(d/'history.json',dict(train_scenes=train,predict_scenes=test,epochs=history,outer=fold,inner=inner,points_sampled_with_replacement=False,model_parameters=sum(p.numel() for p in model.parameters())))
    seal(d/'receipt.json',dict(complete=True,outputs={str(d/n):sha(d/n) for n in ('model.pt','history.json')}));del data;torch.cuda.empty_cache();return model,test,d


def pack(blocks):
    return (np.concatenate(blocks).astype(np.int32) if blocks else np.empty(0,np.int32),np.concatenate(([0],np.cumsum([len(b) for b in blocks]))).astype(np.int64))


def unpack(z,cid,prefix=''):
    flat=z[cid+prefix+'_block_indices'];offset=z[cid+prefix+'_block_offsets'];return [flat[a:b] for a,b in zip(offset[:-1],offset[1:])]


def propose(c,guard,model,scenes,d,outer=False):
    guard.mode='predict';guard.allowed=set();model.eval();outputs={}
    for s in scenes:
        target=d/'proposals'/s;target.mkdir(parents=True,exist_ok=True)
        if (target/'receipt.json').exists():outputs.update(read(target/'receipt.json')['outputs']);continue
        data=examples([s]);arrays={};records=[]
        old=np.load(PREVIOUS/'predictions'/s/'predictions.npz') if outer else None
        for key,e in sorted(data.items()):
            cid=key[1]
            with torch.inference_mode():logits,x,parts=forward(model,[e]);prob=logits.softmax(1).cpu().numpy().astype(np.float16)
            sources={'G':prob}
            if outer:sources['B']=old[cid+'_B_probability'];arrays[cid+'_C_gate_points']=np.setdiff1d(e['parent'],np.setdiff1d(e['parent'],old[cid+'_C_points']))
            for source,pr in sources.items():
                blocks=core.deletion_blocks(e['parent'],e['halo'],pr,e['visible'],e['neighbors']);flat,offset=pack(blocks)
                arrays.update({cid+'_'+source+'_probability':pr,cid+'_'+source+'_block_indices':flat,cid+'_'+source+'_block_offsets':offset,cid+'_'+source+'_regions':core.region_features(e,pr,blocks).astype(np.float16)})
                records.append(dict(candidate_id=cid,source=source,blocks=len(blocks),proposed_points=len(flat)))
        if old is not None:old.close()
        np.savez_compressed(target/'data.npz',**arrays);seal(target/'meta.json',records);oo={str(target/n):sha(target/n) for n in ('data.npz','meta.json')};seal(target/'receipt.json',dict(complete=True,GT_read=False,outputs=oo));outputs.update(oo)
        stage('nested_GT_free_proposals',scene=s,generator=str(d.relative_to(OUT)),blocks=sum(r['blocks'] for r in records))
    seal(d/'proposals_frozen.json',dict(complete=True,GT_read=False,outputs=outputs,model_sha256=sha(d/'model.pt')))


def generators():
    c=read(OUT/'contract.json');guard=Guard(c,'train');sys.addaudithook(guard.hook);torch.set_num_threads(4);assert torch.cuda.is_available()
    for mapping in (c['code'],c['source_bindings'],c['frozen_SL_models']):previous.original.audit.prior.old.verify_files(mapping)
    for fold in range(5):
        for inner in [i for i in range(5) if i!=fold]+[None]:
            model,test,d=fit_proposal(c,guard,fold,inner);propose(c,guard,model,test,d,inner is None);del model;torch.cuda.empty_cache()
    outputs={}
    for p in sorted((OUT/'generators').glob('outer*/*/*frozen.json')):outputs.update(read(p)['outputs'])
    for p in sorted((OUT/'generators').glob('outer*/*/receipt.json')):outputs.update(read(p)['outputs'])
    seal(OUT/'generators_frozen.json',dict(complete=True,outputs=outputs,models=25,inner_models=20,outer_models=5))


def block_labels():
    c=read(OUT/'contract.json');guard=Guard(c,'block_labels');sys.addaudithook(guard.hook);outputs={}
    for fold in range(5):
        scenes=[s for s in c['scenes'] if c['folds'][s]!=fold];guard.allowed=set(scenes);xs=[];ys=[];keys=[];details=[]
        for s in scenes:
            inner=c['folds'][s];d=OUT/'generators'/f'outer{fold}'/f'inner{inner}';assert read(d/'proposals_frozen.json')['complete']
            gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);metadata={r['candidate_id']:r for r in read(OUT/'labels'/s/'meta.json')}
            with np.load(d/'proposals'/s/'data.npz') as z,np.load(SOURCE/'features'/s/'data.npz') as features:
                for r in read(SOURCE/'plan'/s/'candidates.json'):
                    cid=r['candidate_id'];blocks=unpack(z,cid,'_G');labels=core.consequences(gt,features[cid+'_parent'],features[cid+'_halo'],blocks,metadata[cid])
                    for k,(y,detail) in enumerate(labels):xs.append(z[cid+'_G_regions'][k]);ys.append(y);keys.append([s,cid,k]);details.append(dict(detail,scene=s,candidate_id=cid,block=k))
        d=OUT/'training_blocks'/f'outer{fold}';d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'data.npz',x=np.asarray(xs,np.float16).reshape(-1,7,140),y=np.asarray(ys,np.float32).reshape(-1,5));seal(d/'meta.json',dict(keys=keys,details=details,scenes=scenes))
        oo={str(d/n):sha(d/n) for n in ('data.npz','meta.json')};outputs.update(oo);stage('inner_crossfit_action_labels',outer=fold,blocks=len(keys),safe=sum(y[0] for y in ys))
    seal(OUT/'training_blocks_frozen.json',dict(complete=True,outputs=outputs,outer_heldout_GT_read=False))


def fit_verifiers():
    c=read(OUT/'contract.json');guard=Guard(c,'verifier_train');sys.addaudithook(guard.hook);torch.set_num_threads(4);outputs={}
    for fold in range(5):
        guard.outer=fold
        path=OUT/'training_blocks'/f'outer{fold}';meta=read(path/'meta.json');keys=meta['keys'];counts=Counter((s,cid) for s,cid,k in keys);parents=Counter(s for s,cid in counts)
        with np.load(path/'data.npz') as z:x=z['x'].astype(np.float32);y=z['y']
        # Equal mass per scene and parent, capped positive cost estimated only here.
        weights=np.array([1/(parents[s]*counts[s,cid]) for s,cid,k in keys],np.float32);weights/=weights.mean()
        positive=float(weights[y[:,0]==1].sum());negative=float(weights[y[:,0]==0].sum());poscost=min(5.,negative/max(positive,1.))
        weights[y[:,0]==1]*=poscost;weights/=weights.mean()
        xt=torch.tensor(x,device='cuda');yt=torch.tensor(y,device='cuda');wt=torch.tensor(weights,device='cuda')
        for kind in ('summary','relation'):
            d=OUT/'verifiers'/f'outer{fold}'/kind;d.mkdir(parents=True,exist_ok=True)
            if (d/'receipt.json').exists():outputs.update(read(d/'receipt.json')['outputs']);continue
            seed=20261013+fold;torch.manual_seed(seed);model=core.BlockVerifier(kind).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[]
            for epoch in range(40):
                model.train();order=np.random.default_rng(seed+10000+epoch).permutation(len(keys));losses=[]
                for start in range(0,len(order),128):
                    indices=order[start:start+128];optimizer.zero_grad(set_to_none=True);loss=core.verifier_loss(model(xt[indices]),yt[indices],wt[indices]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
                history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
                if (epoch+1)%10==0:stage('block_relation_fit',outer=fold,kind=kind,**history[-1])
            model.eval();pr=[]
            with torch.inference_mode():
                for start in range(0,len(keys),512):pr.append(core.decoded(model(xt[start:start+512])).cpu().numpy())
            pred=np.concatenate(pr);selected=pred[:,0]>=.9;detail=meta['details'];tp=sum(r['pollution_deleted'] for r,yes in zip(detail,selected) if yes);fp=sum(r['target_deleted'] for r,yes in zip(detail,selected) if yes)
            fit=dict(blocks=len(keys),safe_blocks=int(y[:,0].sum()),selected_blocks=int(selected.sum()),selected_safe=int(((y[:,0]==1)&selected).sum()),selected_pollution=tp,selected_target_loss=fp,known_precision=tp/(tp+fp) if tp+fp else None,
                     safe_precision=float(y[selected,0].mean()) if selected.any() else None,consequence_MAE=np.abs(pred[:,1:4]-y[:,1:4]).mean(0).tolist(),delta_MAE=float(np.abs(pred[:,4]-y[:,4]/10).mean()))
            torch.save(model.state_dict(),d/'model.pt');seal(d/'history.json',dict(train_scenes=meta['scenes'],epochs=history,positive_cost=poscost,training_fit=fit,model_parameters=sum(p.numel() for p in model.parameters()),each_block_once_per_epoch=True,used_to_select_hyperparameters=False));oo={str(d/n):sha(d/n) for n in ('model.pt','history.json')};seal(d/'receipt.json',dict(complete=True,outputs=oo));outputs.update(oo);stage('training_action_fit_checked',outer=fold,kind=kind,**fit)
            del model
        del xt,yt,wt;torch.cuda.empty_cache()
    seal(OUT/'verifiers_frozen.json',dict(complete=True,outputs=outputs,models=10))


def predict():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'predict').hook);torch.set_num_threads(4);outputs={}
    for fold in range(5):
        models={}
        for kind in ('summary','relation'):
            model=core.BlockVerifier(kind).cuda().eval();model.load_state_dict(torch.load(OUT/'verifiers'/f'outer{fold}'/kind/'model.pt',weights_only=True));models[kind]=model
        for s in [s for s in c['scenes'] if c['folds'][s]==fold]:
            d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True);arrays={};decisions=[]
            with np.load(OUT/'generators'/f'outer{fold}'/'full'/'proposals'/s/'data.npz') as z,np.load(SOURCE/'features'/s/'data.npz') as features:
                for r in read(SOURCE/'plan'/s/'candidates.json'):
                    cid=r['candidate_id'];p=features[cid+'_parent'];halo=features[cid+'_halo'];cp=z[cid+'_C_gate_points'];arrays[cid+'_C_gate_points']=cp if len(cp)>=100 else p
                    for source in ('B','G'):
                        blocks=unpack(z,cid,'_'+source);arrays[cid+'_'+source+'_raw_points']=core.delete(p,halo,blocks)
                        region=z[cid+'_'+source+'_regions'].astype(np.float32)
                        for kind,model in models.items():
                            with torch.inference_mode():
                                chunks=[core.decoded(model(torch.tensor(region[start:start+512],device='cuda'))).cpu().numpy() for start in range(0,len(region),512)]
                            probability=(np.concatenate(chunks) if chunks else np.empty((0,5),np.float32)).astype(np.float16)
                            selected=probability[:,0].astype(np.float32)>=.9;points=core.delete(p,halo,blocks,selected)
                            arrays.update({cid+'_'+source+'_'+kind+'_points':points,cid+'_'+source+'_'+kind+'_block_prediction':probability})
                            decisions.append(dict(candidate_id=cid,arm=source+'_'+kind,proposed_blocks=len(blocks),selected_blocks=int(selected.sum()),deleted=len(p)-len(points),added=0,minimum_mask_fallback=bool(selected.any() and np.array_equal(p,points))))
            np.savez_compressed(d/'predictions.npz',**arrays);seal(d/'decisions.json',decisions);oo={str(d/n):sha(d/n) for n in ('predictions.npz','decisions.json')};outputs.update(oo);stage('strict_outer_block_prediction',scene=s,outer=fold)
    seal(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,OOF_pointsets=1599*len(ARMS),GT_read=False));stage('all_predictions_sealed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','labels','generators','block_labels','verifiers','predict'));args=parser.parse_args()
    globals()['fit_verifiers' if args.mode=='verifiers' else args.mode]()
