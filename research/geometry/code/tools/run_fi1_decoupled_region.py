#!/usr/bin/env python3
"""Fixed-input independent instance-relation and parent-editability experiment."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,itertools,shutil,sys,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_region_affiliation as legacy
from tools import fi1_decoupled_region_core as core
from tools.fi1_frontend_protocol import write_json
OLD=legacy.OUT;SOURCE=legacy.SOURCE;REGIONS=legacy.REGIONS;CONDITIONS=legacy.CONDITIONS
OUT=ROOT/'.local/fi1_decoupled_region_internal_development200_20261013_v1'
read=legacy.read;seal=legacy.seal;sha=legacy.sha;integrity=legacy.integrity;OBS=legacy.OBS
ARMS=('A_sealed','D_decoupled')


def stage(name,**kw):write_json(OUT/'execution.json',dict(stage=name,updated=time.time(),**legacy.prior.old.plain(kw)));print(name,legacy.prior.old.plain(kw),flush=True)


class Guard:
    def __init__(self,c,mode):self.c,self.mode=c,mode
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,Path)):return
        p=Path(args[0]).resolve();name=str(p)
        if p.suffix in ('.py','.pyc','.so'):return
        if 'independent_analysis' in name or '/diagnosis/' in name or p.name in ('candidate_diagnosis.csv','sampling_diagnosis.csv','geometry_quality.json','target_census.json','overlap.npz'):
            raise PermissionError('historical error/oracle inputs forbidden')
        if self.mode in ('contexts','labels_train','train','predict') and str(OLD) in name and ('/probabilities/' in name or '/predictions/' in name or '/calibration/' in name):raise PermissionError('old learned outputs cannot feed new training/prediction')
        raw=any(q.startswith('ground_truth') for q in p.parts);label='/labels/' in name
        if raw or label:
            s=next((s for s in self.c['scenes'] if '/'+s+'/' in name or p.name==s+'.txt'),None)
            allowed=self.c['splits']['train'] if self.mode in ('labels_train','train') else self.c['splits']['calibration'] if self.mode=='calibrate' else self.c['scenes'] if self.mode in ('evaluate','verify') and (OUT/'predictions_frozen.json').exists() else []
            if s not in allowed:raise PermissionError('split/stage label isolation')
            if raw and name not in self.c['GT_inputs']:raise PermissionError('raw GT whitelist')
            if self.mode=='train' and raw:raise PermissionError('training cache only')
            if self.mode in ('labels_train','calibrate') and not (OUT/'contexts_frozen.json').exists():raise PermissionError('GT-free contexts before labels')


def prepare():
    assert not OUT.exists();assert shutil.disk_usage(ROOT).free>=50*2**30
    c0=read(OLD/'contract.json');assert read(OLD/'delivery_verified.json')['complete'];integrity(c0['code'])
    files=[ROOT/'tools'/n for n in ('fi1_decoupled_region_core.py','run_fi1_decoupled_region.py','evaluate_fi1_decoupled_region.py','verify_fi1_decoupled_region.py','run_fi1_decoupled_region.sh')]
    files+=[ROOT/'tests/test_fi1_decoupled_region.py',ROOT/'docs/FI1_DECOUPLED_REGION_INTERNAL_20261013.md']
    bindings=dict(c0['source_bindings']);bindings.update(read(OLD/'features_frozen.json')['outputs']);bindings.update(read(OLD/'predictions_frozen.json')['outputs'])
    for n in ('contract','features_frozen','models_frozen','rules_frozen','predictions_frozen','completion_verified','delivery_verified'):bindings[str(OLD/(n+'.json'))]=sha(OLD/(n+'.json'))
    c={k:c0[k] for k in ('scenes','splits','folds','groups','class_ids','jobs','GT_inputs','inputs','frozen_SL_models')}
    c.update(version=OUT.name,code=dict(c0['code'],**{str(p):sha(p) for p in files}),source_bindings=bindings,parameters=core.PARAMETERS,arms=ARMS,
             fixed='Same outer0 train160 and112/24/24 physical split, same96 regions/GD10 six views, old GT-free96 token IDs reused. Holdout previously evaluated; new fit cannot read its labels, not blind replication.',
             relation_supervision='Full region GT identity histogram versus fixed reference-anchor histogram: random-point-pair same/different/unknown proportions. >=1000 known IDs include out-of-evaluation-scope annotated identities. No parent target, class whitelist or editability conditioning. Mixed references remain fractional; missing references excluded from loss.',
             relation_representation='Shared57-dim centered geometry/RGB/normals/frozen observation point graph, no parent membership/source roles as identity input. Symmetric sums/abs/products and actual closest boundary point edges; six references individually compared. C local mask match features disabled.',
             parent_supervision='Independent five-state model edit/keep/out_of_scope/unannotated/ambiguous. Stable valid parent .6purity/20hits has edit label iff current96 coverage contains known pollution and ideal delete keeps>=100; clean/no-capacity stable parents keep. No pool-value labels.',
             independence='Separate models/optimizers/losses. Parent classifier consumes GT-free parent/neighbor point features, never relation predictions, avoiding in-sample proposal calibration leakage. No gate-to-relation supervision masking.',
             execution='Only p(edit) and region score=min(parent-reference p(different), strongest neighbor-reference p(same)) thresholds,16 preregistered new-model rules. Whole-region union, overlap dedup, derived same-parent conflict check and min100. No safe head/gain regression/old A or C threshold search.',
             calibration='Same prior conservative precision .9, unknown<=.1, targetloss<=.01, goodlossrate<=.02, drift0,>=3actions/2scenes. Train-only parameters, calibration-only thresholds; none qualify=>abstain. A final pointsets reused unchanged.',
             relation_gate='For same and different separately at probability>=.9, pure known GT >=.95 precision>=.95 plus100 predictions,10parents,5scenes. Ranking/fit are diagnostics, actual NT pool increments and damage decide geometry.',
             stop='If cross-scene high-precision instance relation fails, stop this from-scratch module recipe; next route needs pretrained instance segmentation or new boundary information and permits both addition/deletion. Geometry failure alone does not negate learned relations, but cannot enter SL or automatic five-fold. Do not run another old-output threshold search.',
             new_training=True,C_enabled=False,new_frames=False,new_SAM=False,new_DINO=False,SL_modified=False,AP_run=False)
    integrity(bindings);integrity(c['frozen_SL_models']);OUT.mkdir();seal(OUT/'contract.json',c)
    for p in files:
        q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    stage('prepared',splits={k:len(v) for k,v in c['splits'].items()},old_threshold_search=False)


def context_scene(s):
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'contexts').hook);d=OUT/'contexts'/s
    if (d/'receipt.json').exists():return read(d/'receipt.json')
    raw=np.load(c['jobs'][s]['paths']['processed'],mmap_mode='r');xyz=np.asarray(raw[:,:3],np.float32);arrays={};meta=[]
    with np.load(OLD/'features'/s/'data.npz') as f,np.load(SOURCE/'features'/s/'data.npz') as z,np.load(CONDITIONS/'conditions'/s/'data.npz') as cond:
        for row in read(OLD/'features'/s/'meta.json'):
            cid=row['candidate_id'];n=row['regions'];data=[core.relation_context(f[cid+'_tokens'][j],f[cid+'_mask'][j],f[cid+'_roles'][j],f[cid+'_neighbors'][j],f[cid+'_weights'][j]) for j in range(n)]
            for k,(name,shape,dtype) in enumerate((('nodes',(n,96,57),np.float16),('group',(n,96),np.int8),('weights',(n,96,4),np.float16),('cross',(n,6,48,2),np.int16),('cross_valid',(n,6,48),bool),('distance',(n,6,48),np.float16),('pair_stats',(n,6,12),np.float32))):arrays[cid+'_'+name]=np.asarray([a[k] for a in data],dtype=dtype).reshape(shape)
            arrays[cid+'_valid']=np.asarray([[np.any(a[1]==k+1) for k in range(6)] for a in data],bool).reshape(n,6)
            xx=z[cid+'_'+OBS+'_x'].astype(np.float32);parent=z[cid+'_parent'];halo=z[cid+'_halo'];xx[:,:3]=xyz[halo]-xyz[parent].mean(0)
            parentcontext=core.parent_context(xx,xyz[halo],parent,halo,cond[cid+'_membership'],z[cid+'_'+OBS+'_groups'])
            for name,value in zip(('parent_x','parent_mask','parent_group','parent_neighbors','parent_weights','parent_stats','parent_indices'),parentcontext):arrays[cid+'_'+name]=value
            meta.append(row)
    d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'data.npz',**arrays);seal(d/'meta.json',meta);receipt=dict(complete=True,scene=s,GT_read=False,C_read=False,regions=sum(r['regions'] for r in meta),outputs={str(d/n):sha(d/n) for n in ('data.npz','meta.json')});seal(d/'receipt.json',receipt);return receipt


def contexts():
    c=read(OUT/'contract.json');outputs={};n=0
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(context_scene,s) for s in c['scenes']])):
            r=f.result();outputs.update(r['outputs']);n+=r['regions']
            if (i+1)%20==0:stage('GT_free_decoupled_contexts',completed=i+1,total=160);assert shutil.disk_usage(ROOT).free>=50*2**30
    seal(OUT/'contexts_frozen.json',dict(complete=True,GT_read=False,outputs=outputs,regions=n,C_enabled=False));stage('contexts_sealed',regions=n)


def supervision(s,c):
    old_labels,meta,gt,ids,sizes=legacy.supervision(s,c);arrays={};records=[]
    with np.load(OLD/'features'/s/'data.npz') as f,np.load(OUT/'contexts'/s/'data.npz') as context,np.load(SOURCE/'features'/s/'data.npz') as z,np.load(REGIONS/'regions'/s/'regions.npz') as bank:
        for m in meta:
            cid=m['candidate_id'];allblocks=legacy.prior.old.unpack(bank,cid);blocks=[allblocks[j] for j in bank[cid+'_budget96']];halo=z[cid+'_halo'];parent=z[cid+'_parent'];valid=context[cid+'_valid'];truth=np.zeros((len(blocks),6,3),np.float32)
            for j,b in enumerate(blocks):
                group=context[cid+'_group'][j];token_ids=f[cid+'_token_ids'][j]
                for k in range(6):
                    reference=np.unique(token_ids[group==k+1]);truth[j,k]=core.pair_truth(gt[halo[b]],gt[halo[reference]])
            if m['stable']:
                coverage=legacy.prior.core.union(blocks);vals=gt[halo[coverage]];other=int(((vals>=1000)&(vals!=m['target'])).sum());parent_label=0 if other>0 and len(parent)-other>=100 else 1
            else:parent_label={'out_of_scope':2,'unannotated':3,'ambiguous':4}[m['scope']]
            categories=np.where(truth[:,:,2]>=.1,3,np.where(truth[:,:,0]>=.95,0,np.where(truth[:,:,1]>=.95,1,2))).astype(np.uint8)
            arrays.update({cid+'_pair_truth':truth,cid+'_pair_category':categories,cid+'_pair_valid':valid,cid+'_parent_label':np.int64(parent_label),cid+'_safe':old_labels[cid+'_safe']})
            records.append(dict(m,parent_state=core.PARENT_STATES[parent_label],parent_label=parent_label))
    return arrays,records,gt,ids,sizes


def labels_train():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'labels_train').hook);outputs={};counts=Counter()
    for s in c['splits']['train']:
        arrays,records,*_=supervision(s,c);d=OUT/'labels'/s;d.mkdir(parents=True)
        np.savez_compressed(d/'data.npz',**arrays);seal(d/'meta.json',records)
        for row in records:
            cid=row['candidate_id'];counts['parents']+=1;counts[row['parent_state']]+=1;counts['valid_relations']+=int(arrays[cid+'_pair_valid'].sum());counts['valid_relations_out_of_scope_parent']+=int(arrays[cid+'_pair_valid'].sum()) if row['scope']=='out_of_scope' else 0
        outputs.update({str(d/n):sha(d/n) for n in ('data.npz','meta.json')})
    seal(OUT/'labels_train_frozen.json',dict(complete=True,outputs=outputs,counts=dict(counts)));stage('independent_supervision_sealed',**dict(counts))


PAIR_KEYS=('nodes','group','weights','cross','cross_valid','distance','pair_stats','valid')
PARENT_KEYS=('parent_x','parent_mask','parent_group','parent_neighbors','parent_weights','parent_stats')


def load_data(scenes,supervised=False):
    pair={k:[] for k in (*PAIR_KEYS,'mask','neighbors')};parent={k:[] for k in PARENT_KEYS};pt={k:[] for k in ('pair_truth','pair_category','pair_valid')};parenttruth=[];region_items=[];parent_items=[]
    for s in scenes:
        with np.load(OUT/'contexts'/s/'data.npz') as z,np.load(OLD/'features'/s/'data.npz') as f:
            labels=np.load(OUT/'labels'/s/'data.npz') if supervised else None
            for row in read(OUT/'contexts'/s/'meta.json'):
                cid=row['candidate_id'];n=row['regions']
                for k in PAIR_KEYS:pair[k].append(z[cid+'_'+k])
                for k in ('mask','neighbors'):pair[k].append(f[cid+'_'+k])
                for k in PARENT_KEYS:parent[k].append(z[cid+'_'+k][None])
                if supervised:
                    for k in pt:pt[k].append(labels[cid+'_'+k])
                    parenttruth.append(int(labels[cid+'_parent_label']))
                region_items.extend((s,cid,j) for j in range(n));parent_items.append((s,cid))
            if labels is not None:labels.close()
    pair={k:np.concatenate(v) for k,v in pair.items()};parent={k:np.concatenate(v) for k,v in parent.items()}
    if supervised:pair.update({k:np.concatenate(v) for k,v in pt.items()});parent['truth']=np.array(parenttruth,np.int64)
    pair['items']=region_items;parent['items']=parent_items;return pair,parent


def pair_forward(model,data,idx,device='cuda'):
    args=[]
    for k in ('nodes','mask','group','neighbors','weights','cross','cross_valid','distance','pair_stats'):
        dtype=torch.bool if k in ('mask','cross_valid') else torch.long if k in ('group','neighbors','cross') else torch.float32
        args.append(torch.as_tensor(data[k][idx],device=device,dtype=dtype))
    return model(*args)


def parent_forward(model,data,idx,device='cuda'):
    args=[]
    for k in PARENT_KEYS:
        dtype=torch.bool if k=='parent_mask' else torch.long if k in ('parent_group','parent_neighbors') else torch.float32
        args.append(torch.as_tensor(data[k][idx],device=device,dtype=dtype))
    return model(*args)


def train():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'train').hook);assert torch.cuda.is_available();torch.set_num_threads(4);pair,parent=load_data(c['splits']['train'],True);outputs={}
    for kind,cls,epochs,batch in (('relations',core.InstanceRelation,24,256),('parent',core.ParentEditability,60,32)):
        d=OUT/'models'/kind;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():outputs.update(read(d/'receipt.json')['outputs']);continue
        seed=20261013+(kind=='parent');torch.manual_seed(seed);model=cls().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);data=pair if kind=='relations' else parent
        categories=pair['pair_category'][pair['pair_valid']] if kind=='relations' else parent['truth'];cc=Counter(categories.tolist());median=np.median(list(cc.values()));cw={k:float(np.clip(median/n,.2,10)) for k,n in cc.items()}
        pc=Counter((s,cid) for s,cid,j in pair['items']) if kind=='relations' else Counter(parent['items']);sc=Counter(s for s,cid in pc)
        baseweight=np.array([1/pc[item[0],item[1]]/sc[item[0]] for item in data['items']],np.float32);baseweight/=baseweight.mean();history=[];first=0
        if (d/'checkpoint.pt').exists():
            state=torch.load(d/'checkpoint.pt',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);first=state['epoch'];history=state['history'];torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        for epoch in range(first,epochs):
            tic=time.time();model.train();order=np.random.default_rng(seed+epoch).permutation(len(data['items']));values=[]
            for start in range(0,len(order),batch):
                idx=order[start:start+batch];optimizer.zero_grad(set_to_none=True)
                if kind=='relations':
                    logit=pair_forward(model,pair,idx);truth=torch.tensor(pair['pair_truth'][idx],device='cuda');w=np.zeros((len(idx),6),np.float32)
                    for k,value in cw.items():w[pair['pair_category'][idx]==k]=value
                    w*=pair['pair_valid'][idx];w*=baseweight[idx,None];weight=torch.tensor(w,device='cuda');loss=(-(truth*logit.log_softmax(-1)).sum(-1)*weight).sum()/weight.sum().clamp_min(1)
                else:
                    logit=parent_forward(model,parent,idx);truth=torch.tensor(parent['truth'][idx],device='cuda');weight=torch.tensor([cw[int(v)] for v in parent['truth'][idx]],device='cuda')*torch.tensor(baseweight[idx],device='cuda');loss=(torch.nn.functional.cross_entropy(logit,truth,reduction='none')*weight).sum()/weight.sum()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();values.append(float(loss.detach()))
            h=dict(epoch=epoch+1,loss=float(np.mean(values)),updates=len(values),seconds=time.time()-tic);history.append(h)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),d/'checkpoint.pt')
            if kind=='relations' or (epoch+1)%5==0:stage('decoupled_fit',model=kind,**h)
        torch.save(model.state_dict(),d/'model.pt');seal(d/'history.json',dict(train_scenes=c['splits']['train'],epochs=history,parameters=sum(p.numel() for p in model.parameters()),class_counts=dict(cc),class_weights=cw,other_model_predictions_used=False,region_replacement=False))
        oo={str(d/n):sha(d/n) for n in ('model.pt','history.json')};seal(d/'receipt.json',dict(complete=True,outputs=oo));outputs.update(oo);del model;torch.cuda.empty_cache()
    seal(OUT/'models_frozen.json',dict(complete=True,outputs=outputs));stage('independent_models_sealed')


def predict():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'predict').hook);assert torch.cuda.is_available();torch.set_num_threads(4)
    relation=core.InstanceRelation().cuda().eval();relation.load_state_dict(torch.load(OUT/'models/relations/model.pt',weights_only=True));parentmodel=core.ParentEditability().cuda().eval();parentmodel.load_state_dict(torch.load(OUT/'models/parent/model.pt',weights_only=True));outputs={}
    for i,s in enumerate(c['scenes']):
        d=OUT/'probabilities'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():outputs.update(read(d/'receipt.json')['outputs']);continue
        pair,parent=load_data([s]);parts=[]
        with torch.inference_mode():
            for lo in range(0,len(pair['items']),256):parts.append(pair_forward(relation,pair,np.arange(lo,min(lo+256,len(pair['items'])))).softmax(-1).cpu().numpy())
            pp=parent_forward(parentmodel,parent,np.arange(len(parent['items']))).softmax(-1).cpu().numpy()
        probability=np.concatenate(parts) if parts else np.empty((0,6,3),np.float32);arrays={};lo=0
        for k,row in enumerate(read(OUT/'contexts'/s/'meta.json')):
            cid=row['candidate_id'];n=row['regions'];arrays.update({cid+'_relations':probability[lo:lo+n],cid+'_valid':pair['valid'][lo:lo+n],cid+'_parent':pp[k]});lo+=n
        np.savez_compressed(d/'data.npz',**arrays);oo={str(d/'data.npz'):sha(d/'data.npz')};seal(d/'receipt.json',dict(complete=True,GT_read=False,outputs=oo));outputs.update(oo)
        if (i+1)%20==0:stage('GT_free_decoupled_predictions',completed=i+1,total=160)
    seal(OUT/'probabilities_frozen.json',dict(complete=True,GT_read=False,outputs=outputs));stage('probabilities_sealed_before_calibration')


def calibration_metrics(examples,rule):
    counts=Counter();scenes=set()
    for e in examples:
        q,_=core.execute(e['parent'],e['halo'],e['blocks'],e['prediction'],rule);removed=np.setdiff1d(e['parent'],q);counts['deleted']+=len(removed)
        if not len(removed):continue
        scenes.add(e['scene']);counts['actions']+=1;m=e['meta'];gt=e['gt']
        if not m['stable']:counts['unknown']+=len(removed);counts['unknown_identity_actions']+=1;continue
        gid=m['target'];target=int((gt[removed]==gid).sum());other=int(((gt[removed]>=1000)&(gt[removed]!=gid)).sum());counts.update(target=target,other=other,unknown=len(removed)-target-other,parent_target=int((gt[e['parent']]==gid).sum()))
        after=legacy.original.audit.prior.ious(q,gt,e['ids'],e['sizes']);k=int(np.flatnonzero(e['ids']==gid)[0]);counts['good_degraded']+=m['parent_iou']>=.9 and after[k]-m['parent_iou']<-.02-1e-12
        best=int(np.argmax(after));counts['drift']+=e['ids'][best]!=gid and (gt[q]==e['ids'][best]).sum()>=20
    counts['good_parents']=sum(e['meta']['good'] for e in examples);counts['action_scenes']=len(scenes);n=counts['deleted']
    counts['precision']=counts['other']/n if n else None;counts['target_loss']=counts['target']/max(1,counts['parent_target']);counts['unknown_fraction']=counts['unknown']/max(1,n);counts['good_rate']=counts['good_degraded']/max(1,counts['good_parents']);return dict(counts)


def calibration_examples(c):
    examples=[]
    for s in c['splits']['calibration']:
        _,meta,gt,ids,sizes=supervision(s,c);meta={m['candidate_id']:m for m in meta}
        with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(REGIONS/'regions'/s/'regions.npz') as bank,np.load(OUT/'probabilities'/s/'data.npz') as pr:
            for row in read(OUT/'contexts'/s/'meta.json'):
                cid=row['candidate_id'];allblocks=legacy.prior.old.unpack(bank,cid);blocks=[allblocks[j] for j in bank[cid+'_budget96']]
                examples.append(dict(scene=s,parent=z[cid+'_parent'],halo=z[cid+'_halo'],blocks=blocks,prediction={k:pr[cid+'_'+k] for k in ('relations','parent','valid')},meta=meta[cid],gt=gt,ids=ids,sizes=sizes))
    return examples


def calibrate():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'calibrate').hook);assert read(OUT/'probabilities_frozen.json')['complete'];examples=calibration_examples(c);grid=[];qualified=[]
    for edit,relation in itertools.product(core.PARAMETERS['edit_thresholds'],core.PARAMETERS['relation_thresholds']):
        rule=dict(edit=edit,relation=relation,abstain=False);m=calibration_metrics(examples,rule)
        ok=m.get('actions',0)>=3 and m['action_scenes']>=2 and m['precision']>=.9 and m['target_loss']<=.01 and m['unknown_fraction']<=.1 and m['good_rate']<=.02 and m.get('drift',0)==0
        row=dict(rule=rule,metrics=m,qualified=bool(ok));grid.append(row)
        if ok:qualified.append(row)
    chosen=max(qualified,key=lambda r:(r['metrics']['other'],-r['metrics'].get('target',0),-r['metrics'].get('unknown',0),r['rule']['edit'],r['rule']['relation'])) if qualified else dict(rule={'abstain':True},metrics=calibration_metrics(examples,{'abstain':True}),qualified=False)
    seal(OUT/'calibration.json',dict(complete=True,grid=grid,chosen=chosen,scenes=c['splits']['calibration']));seal(OUT/'rules_frozen.json',dict(complete=True,chosen=chosen,calibration_sha256=sha(OUT/'calibration.json'),holdout_labels_read=False,old_A_recalibrated=False));stage('new_model_rule_locked',chosen=chosen,qualified=len(qualified))


def apply():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'apply').hook);rule=read(OUT/'rules_frozen.json')['chosen']['rule'];outputs={};counts=Counter()
    for s in c['scenes']:
        d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True);arrays={};actions=[]
        with np.load(OUT/'probabilities'/s/'data.npz') as pr,np.load(SOURCE/'features'/s/'data.npz') as z,np.load(REGIONS/'regions'/s/'regions.npz') as bank,np.load(OLD/'predictions'/s/'data.npz') as baseline:
            old_actions={r['candidate_id']:r for r in read(OLD/'predictions'/s/'actions.json') if r['arm']=='A'}
            for row in read(OUT/'contexts'/s/'meta.json'):
                cid=row['candidate_id'];allblocks=legacy.prior.old.unpack(bank,cid);blocks=[allblocks[j] for j in bank[cid+'_budget96']];prediction={k:pr[cid+'_'+k] for k in ('relations','parent','valid')}
                q,act=core.execute(z[cid+'_parent'],z[cid+'_halo'],blocks,prediction,rule);arrays[cid+'_D_decoupled_points']=q;arrays[cid+'_A_sealed_points']=baseline[cid+'_A_points'];counts['actions']+=act['deleted']>0
                actions.extend([dict(candidate_id=cid,arm='D_decoupled',**act),dict(candidate_id=cid,arm='A_sealed',reason=old_actions[cid]['reason'],selected=old_actions[cid]['selected'],deleted=old_actions[cid]['deleted'])])
        np.savez_compressed(d/'data.npz',**arrays);seal(d/'actions.json',actions);outputs.update({str(d/n):sha(d/n) for n in ('data.npz','actions.json')})
    seal(OUT/'predictions_frozen.json',dict(complete=True,GT_read=False,outputs=outputs,rules_sha256=sha(OUT/'rules_frozen.json'),A_final_outputs_unchanged=True));stage('actual_pointsets_sealed',**dict(counts))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','contexts','labels_train','train','predict','calibrate','apply'));args=parser.parse_args();globals()[args.mode]()
