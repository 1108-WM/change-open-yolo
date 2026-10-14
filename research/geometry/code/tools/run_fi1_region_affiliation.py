#!/usr/bin/env python3
"""One preregistered physical-scene train/calibration/holdout test, A versus C."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import argparse,itertools,json,sys,time,shutil
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_local_region_supply as prior
from tools import run_fi1_candidate_local_repair as original
from tools import fi1_region_affiliation_core as core
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
SOURCE=prior.SOURCE;REGIONS=prior.OUT;CONDITIONS=prior.old.PREVIOUS
OUT=ROOT/'.local/fi1_region_affiliation_internal_development200_20261013_v1'
read=prior.read;seal=prior.seal;OBS=prior.OBS;ARMS=('A','C')


def stage(name,**kw):write_json(OUT/'execution.json',dict(stage=name,updated=time.time(),**prior.old.plain(kw)));print(name,prior.old.plain(kw),flush=True)


class Guard:
    def __init__(self,c,mode):self.c,self.mode=c,mode
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,Path)):return
        p=Path(args[0]).resolve();name=str(p)
        if p.suffix in ('.py','.pyc','.so'):return
        if 'independent_analysis' in name or '/diagnosis/' in name or p.name in ('candidate_diagnosis.csv','sampling_diagnosis.csv','geometry_quality.json','target_census.json','overlap.npz'):
            raise PermissionError('historical/posthoc GT diagnostic inputs excluded')
        s=next((s for s in self.c['scenes'] if '/'+s+'/' in name or p.name==s+'.txt'),None)
        raw=any(q.startswith('ground_truth') for q in p.parts)
        label='/labels/' in name
        if raw or label:
            allowed=self.c['splits']['train'] if self.mode in ('labels_train','train') else self.c['splits']['calibration'] if self.mode=='calibrate' else self.c['scenes'] if self.mode in ('evaluate','verify') and (OUT/'predictions_frozen.json').exists() else []
            if s not in allowed:raise PermissionError('label access forbidden for stage/split')
            if raw and name not in self.c['GT_inputs']:raise PermissionError('GT whitelist')
            if self.mode=='train' and raw:raise PermissionError('training reads train supervision cache only')
            if self.mode in ('labels_train','calibrate') and not (OUT/'features_frozen.json').exists():raise PermissionError('features must be frozen before supervision')


def integrity(mapping):
    for p,h in mapping.items():assert sha(Path(p))==h,p


def prepare():
    assert not OUT.exists();assert shutil.disk_usage(ROOT).free>=50*2**30
    c0=read(REGIONS/'contract.json');assert read(REGIONS/'delivery_verified.json')['complete']
    splits=core.split_scenes(c0['scenes'],c0['folds']);scenes=sorted(sum(splits.values(),[]))
    assert [len(splits[k]) for k in ('train','calibration','holdout')]==[112,24,24]
    files=[ROOT/'tools'/n for n in ('fi1_region_affiliation_core.py','run_fi1_region_affiliation.py','evaluate_fi1_region_affiliation.py','verify_fi1_region_affiliation.py','run_fi1_region_affiliation.sh')]
    files+=[ROOT/'tests/test_fi1_region_affiliation.py',ROOT/'docs/FI1_REGION_AFFILIATION_INTERNAL_20261013.md']
    bindings={str(REGIONS/n):sha(REGIONS/n) for n in ('contract.json','regions_frozen.json','delivery_verified.json')}
    for root,name in ((SOURCE,'plan'),(SOURCE,'features'),(CONDITIONS,'conditions'),(REGIONS,'regions')):
        for p,h in read(root/(name+'_frozen.json'))['outputs'].items():
            if any('/'+s+'/' in p for s in scenes):bindings[p]=h
    c={k:c0[k] for k in ('folds','groups','class_ids','frozen_SL_models')}
    c.update(version=OUT.name,scenes=scenes,splits=splits,jobs={s:c0['jobs'][s] for s in scenes},
             GT_inputs={c0['jobs'][s]['GT']:c0['GT_inputs'][c0['jobs'][s]['GT']] for s in scenes},
             inputs={p:h for p,h in c0['inputs'].items() if any('/'+s+'/' in p or Path(p).name==s+'.json' for s in scenes)},
             code=dict(c0['code'],**{str(p):sha(p) for p in files}),source_bindings=bindings,parameters=core.PARAMETERS,arms=ARMS,
             supervision='Stable parent=best valid instance >=20hits and >=.6 purity. Soft target/other/unknown point fractions plus pure-known-other classification. Identity-unknown all unknown; mixed never safe. No gain regression.',
             sampling='Each same-source96 region once each epoch, no replacement; train-only categorical weight clip .2..10 and per-scene/per-parent balance. Fixed24 epochs, final checkpoint only.',
             representation='96 spatial point tokens:24 region,24 retained raw-8NN boundary,24 retained parent,6 per competing NT instance. Raw graph2 messages with .15m cutoff; four roles and competitor IDs. Original64 point features frozen. No GT/SP columns.',
             C='20 per-view explicit local instance-mask relations in same selected6 GD10 frames. Match on retained parent or competing anchors, >=3 visible hits and local IoU>=.05; top-two .9ratio/.02margin ambiguity explicit. Missing observations explicit. No whole-candidate GVC score.',
             execution='Soft fractions plus pure-pollution probability, fixed calibration grid. Selected complete regions unioned; per-point max predicted target/unknown over selected covering regions; recompute mean union risk and min100. No GT in execution, additions, gain head or dedup refilling.',
             calibration='Choose rule by most known pollution deleted, subject precision>=.9 including unknown, targetloss<=.01, unknownfraction<=.1, gooddegradation<=.02, drift0 and >=3 actions/2scenes. If none qualify abstain. Grid/method fixed before any new labels; holdout never read.',
             scope='One original outer0 train160 split112/24/24 by physical hash. Historical development and GT supply diagnostics exposed; internal unseen-to-this-fit scenes, not whole-system blind/OOF5.',
             waived_old_gate='User explicitly authorizes learning current small96 regions; min_safe256 not a gate in this test.',
             new_training=True,new_frames=False,new_backbone=False,new_SAM=False,new_DINO=False,SL_modified=False,AP_run=False)
    integrity(c['code']);integrity(bindings);integrity(c['frozen_SL_models'])
    OUT.mkdir();seal(OUT/'contract.json',c)
    for p in files:
        q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    stage('prepared',scenes=len(scenes),splits={k:len(v) for k,v in splits.items()},outer_fold=0)


def feature_scene(s):
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'features').hook);d=OUT/'features'/s
    if (d/'receipt.json').exists():return read(d/'receipt.json')
    job=c['jobs'][s];raw=np.load(job['paths']['processed'],mmap_mode='r');xyz=np.asarray(raw[:,:3],np.float32)
    plans=read(SOURCE/'plan'/s/'candidates.json');frame_ids=sorted({f for r in plans for f in r['views'][OBS]['frames']})
    first=json.loads(Path(job['paths']['observations']).read_text().splitlines()[0]);bankroot=Path(first['point_indices_path']).parent.parent
    frame_banks={};input_hashes={}
    for f in frame_ids:
        bank=bankroot/f'frame{f:06d}';rows,points,_=original.bank_points(bank,len(raw))
        frame_banks[f]=(rows,points);input_hashes[str(bank/'receipt.json')]=sha(bank/'receipt.json')
    arrays={};records=[];counts=Counter()
    with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(CONDITIONS/'conditions'/s/'data.npz') as cond,np.load(REGIONS/'regions'/s/'regions.npz') as bank,np.load(SOURCE/'plan'/s/'evidence.npz') as evidence:
        roi=evidence['roi'];visible=original.audit.unpack(evidence,'reliable');fi={int(f):k for k,f in enumerate(evidence['frames'])}
        for row in plans:
            cid=row['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];xx=z[cid+'_'+OBS+'_x'].astype(np.float32)
            xx[:,:3]=xyz[halo]-xyz[parent].mean(0);member=np.isin(halo,parent);membership=cond[cid+'_membership']
            allblocks=prior.old.unpack(bank,cid);region_ids=bank[cid+'_budget96'];blocks=[allblocks[i] for i in region_ids]
            selected=row['views'][OBS]['frames'];assert len(selected)<=6
            frames=[]
            for f in selected:
                observed=visible[fi[f],np.searchsorted(roi,halo)];rows,points=frame_banks[f]
                masks=np.column_stack([np.isin(halo,p) for p in points]) if points else np.empty((len(halo),0),bool)
                frames.append((observed,masks,np.asarray([r['predicted_iou'] for r in rows],np.float32),core.view_context(member,membership,observed,masks)))
            tokens=[];masks=[];roles=[];neighbors=[];weights=[];stats=[];views=[];token_ids=[]
            for b in blocks:
                t,mask,role,nn,w,st,tid=core.tokens(xx,xyz[halo],member,z[cid+'_neighbors'],membership,b)
                vv=np.zeros((6,20),np.float32);vv[:,19]=1
                for k,(seen,inst,q,context) in enumerate(frames):vv[k]=core.local_view(b,member,membership,seen,inst,q,context)
                tokens.append(t);masks.append(mask);roles.append(role);neighbors.append(nn);weights.append(w);stats.append(st);views.append(vv);token_ids.append(tid)
                counts.update(regions=1,view_slots=6,selected_views=len(frames),region_seen_views=int((vv[:,1]>0).sum()),parent_matches=int(vv[:,5].sum()),parent_ambiguous=int(vv[:,6].sum()),neighbor_matches=int(vv[:,7].sum()),neighbor_ambiguous=int(vv[:,8].sum()),missing_views=int(vv[:,19].sum()),usable_parent_regions=int(((vv[:,0]>.5)&(vv[:,1]>0)&(vv[:,5]>.5)&(vv[:,6]<.5)).sum()>=2))
            n=len(blocks)
            for name,data,shape,dtype in (('tokens',tokens,(n,96,75),np.float16),('mask',masks,(n,96),bool),('roles',roles,(n,96),np.int8),('neighbors',neighbors,(n,96,4),np.int16),('weights',weights,(n,96,4),np.float16),('stats',stats,(n,16),np.float32),('views',views,(n,6,20),np.float16),('token_ids',token_ids,(n,96),np.int32)):
                arrays[cid+'_'+name]=np.asarray(data,dtype=dtype).reshape(shape)
            arrays[cid+'_region_ids']=region_ids
            records.append(dict(candidate_id=cid,source=row['source'],regions=n,frames=selected,parent_points=len(parent),competitors=int(cond[cid+'_valid'].sum())))
    d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'data.npz',**arrays);seal(d/'meta.json',records)
    receipt=dict(complete=True,scene=s,GT_read=False,new_observations=False,counts=dict(counts),inputs=input_hashes,outputs={str(d/n):sha(d/n) for n in ('data.npz','meta.json')});seal(d/'receipt.json',receipt);return receipt


def features():
    c=read(OUT/'contract.json');outputs={};inputs={};totals={k:Counter() for k in c['splits']};split={s:k for k,ss in c['splits'].items() for s in ss}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(feature_scene,s) for s in c['scenes']])):
            r=f.result();outputs.update(r['outputs']);inputs.update(r['inputs']);totals[split[r['scene']]].update(r['counts'])
            if (i+1)%10==0:stage('GT_free_same_source_features',completed=i+1,total=160,disk_free_GiB=round(shutil.disk_usage(ROOT).free/2**30,2));assert shutil.disk_usage(ROOT).free>=50*2**30
    seal(OUT/'features_frozen.json',dict(complete=True,GT_read=False,outputs=outputs,inputs=inputs))
    coverage={}
    for k,m in totals.items():
        coverage[k]=dict(m,usable_parent_region_fraction=m['usable_parent_regions']/max(1,m['regions']),parent_matching_ambiguity=m['parent_ambiguous']/max(1,m['parent_matches']),neighbor_matching_ambiguity=m['neighbor_ambiguous']/max(1,m['neighbor_matches']))
    seal(OUT/'evidence_preflight.json',dict(complete=True,GT_read=False,splits=coverage,view_names=core.VIEW_NAMES,missing_is_unknown=True,coverage_not_a_GT_filter=True))
    stage('features_and_C_coverage_sealed',coverage=coverage)


def supervision(s,c):
    job=c['jobs'][s];assert sha(job['GT'])==c['GT_inputs'][job['GT']];gt=np.loadtxt(job['GT'],dtype=np.int64)
    ids,sizes=original.audit.prior.valid_targets(gt,c['class_ids']);arrays={};records=[]
    from tools.evaluate_fi1_deletion_blocks import iou
    with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(REGIONS/'regions'/s/'regions.npz') as bank:
        for row in read(SOURCE/'plan'/s/'candidates.json'):
            cid=row['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];scores=iou(parent,gt,ids,sizes);idx=int(np.argmax(scores)) if len(ids) and scores.max()>0 else None
            gid=None if idx is None else int(ids[idx]);hits=0 if gid is None else int((gt[parent]==gid).sum());purity=hits/len(parent);stable=gid is not None and hits>=20 and purity>=.6
            dom,cnt=np.unique(gt[parent],return_counts=True);dominant=int(dom[np.argmax(cnt)]);fraction=cnt.max()/len(parent)
            scope='stable' if stable else 'out_of_scope' if dominant>=1000 and dominant//1000 not in c['class_ids'] and fraction>=.6 else 'unannotated' if (gt[parent]<1000).mean()>=.5 else 'ambiguous'
            blocks=prior.old.unpack(bank,cid);selected=bank[cid+'_budget96'];fractions=[];safe=[];categories=[];rawcounts=[]
            for rid in selected:
                vals=gt[halo[blocks[rid]]];target=int((vals==gid).sum()) if stable else 0;other=int(((vals>=1000)&(vals!=gid)).sum()) if stable else 0;unknown=len(vals)-target-other
                fractions.append([target/len(vals),other/len(vals),unknown/len(vals)]);safe.append(stable and other==len(vals) and len(parent)-len(vals)>=100)
                categories.append(0 if target==len(vals) else 1 if other==len(vals) else 2 if target and other else 3 if stable else 4);rawcounts.append([target,other,unknown])
            n=len(selected);arrays.update({cid+'_fractions':np.asarray(fractions,np.float32).reshape(n,3),cid+'_safe':np.asarray(safe,np.float32),cid+'_category':np.asarray(categories,np.uint8),cid+'_counts':np.asarray(rawcounts,np.int32).reshape(n,3)})
            records.append(dict(candidate_id=cid,stable=bool(stable),target=gid if stable else None,original_target=gid,purity=float(purity),scope=scope,parent_iou=float(scores.max(initial=0)),good=bool(stable and scores[idx]>=.9)))
    return arrays,records,gt,ids,sizes


def labels_train():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'labels_train').hook);outputs={};counts=Counter()
    for s in c['splits']['train']:
        arrays,records,*_=supervision(s,c);d=OUT/'labels'/s;d.mkdir(parents=True)
        np.savez_compressed(d/'data.npz',**arrays);seal(d/'meta.json',records)
        for row in records:
            cid=row['candidate_id'];counts.update(regions=len(arrays[cid+'_safe']),safe=int(arrays[cid+'_safe'].sum()),parents=1);counts[row['scope']]+=1
        outputs.update({str(d/n):sha(d/n) for n in ('data.npz','meta.json')})
    seal(OUT/'labels_train_frozen.json',dict(complete=True,train_scenes=c['splits']['train'],outputs=outputs,counts=dict(counts)));stage('train_only_supervision_sealed',**dict(counts))


KEYS=('tokens','mask','roles','neighbors','weights','stats','views')


def load_data(scenes,supervised=False):
    values={k:[] for k in KEYS};targets={k:[] for k in ('fractions','safe','category')};items=[]
    for s in scenes:
        with np.load(OUT/'features'/s/'data.npz') as z:
            lab=np.load(OUT/'labels'/s/'data.npz') if supervised else None
            for row in read(OUT/'features'/s/'meta.json'):
                cid=row['candidate_id'];n=row['regions']
                for k in KEYS:values[k].append(z[cid+'_'+k])
                if supervised:
                    for k in targets:targets[k].append(lab[cid+'_'+k])
                items.extend((s,cid,j) for j in range(n))
            if lab is not None:lab.close()
    data={k:np.concatenate(v) for k,v in values.items()}
    if supervised:data.update({k:np.concatenate(v) for k,v in targets.items()})
    data['items']=items;return data


def batch_data(data,ids,arm,device='cuda'):
    floats=('tokens','weights','stats','views');bools=('mask',)
    out={k:torch.as_tensor(data[k][ids],device=device,dtype=torch.float32 if k in floats else torch.bool if k in bools else torch.long) for k in KEYS}
    if arm=='A':out['views'].zero_()
    return out


def logits(model,batch):return model(batch['tokens'],batch['mask'],batch['roles'],batch['neighbors'],batch['weights'],batch['stats'],batch['views'])


def train():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'train').hook);assert torch.cuda.is_available();torch.set_num_threads(4)
    data=load_data(c['splits']['train'],True);counts=Counter(data['category'].tolist());median=np.median(list(counts.values()));categoryweight={k:float(np.clip(median/v,.2,10)) for k,v in counts.items()}
    parentcount=Counter((s,cid) for s,cid,_ in data['items']);scenecount=Counter(s for s,cid in parentcount)
    weight=np.asarray([categoryweight[int(y)]/parentcount[s,cid]/scenecount[s] for (s,cid,j),y in zip(data['items'],data['category'])],np.float32);weight/=weight.mean()
    outputs={}
    for arm in ARMS:
        d=OUT/'models'/arm;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():outputs.update(read(d/'receipt.json')['outputs']);continue
        torch.manual_seed(core.PARAMETERS['seed']);model=core.RegionAffiliation().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[];first=0
        if (d/'checkpoint.pt').exists():
            state=torch.load(d/'checkpoint.pt',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);first=state['epoch'];history=state['history'];torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        for epoch in range(first,core.PARAMETERS['epochs']):
            tic=time.time();model.train();order=np.random.default_rng(core.PARAMETERS['seed']+epoch).permutation(len(data['items']));losses=[]
            for start in range(0,len(order),core.PARAMETERS['batch']):
                ids=order[start:start+core.PARAMETERS['batch']];batch=batch_data(data,ids,arm);optimizer.zero_grad(set_to_none=True)
                pred=logits(model,batch);value=core.loss(pred,torch.tensor(data['fractions'][ids],device='cuda'),torch.tensor(data['safe'][ids],device='cuda'),torch.tensor(weight[ids],device='cuda'))
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(value.detach()))
            row=dict(epoch=epoch+1,loss=float(np.mean(losses)),updates=len(losses),seconds=time.time()-tic);history.append(row)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),d/'checkpoint.pt');stage('same_source_region_fit',arm=arm,**row)
        torch.save(model.state_dict(),d/'model.pt');seal(d/'history.json',dict(train_scenes=c['splits']['train'],epochs=history,regions=len(data['items']),category_counts=dict(counts),category_weights=categoryweight,parameters=sum(p.numel() for p in model.parameters()),region_replacement=False))
        oo={str(d/n):sha(d/n) for n in ('model.pt','history.json')};seal(d/'receipt.json',dict(complete=True,outputs=oo));outputs.update(oo);del model;torch.cuda.empty_cache()
    seal(OUT/'models_frozen.json',dict(complete=True,outputs=outputs));stage('two_models_sealed')


def predict_probabilities():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'predict').hook);assert torch.cuda.is_available();torch.set_num_threads(4);outputs={}
    for arm in ARMS:
        model=core.RegionAffiliation().cuda();model.load_state_dict(torch.load(OUT/'models'/arm/'model.pt',weights_only=True));model.eval()
        for i,s in enumerate(c['scenes']):
            d=OUT/'probabilities'/arm/s;d.mkdir(parents=True,exist_ok=True)
            if (d/'receipt.json').exists():outputs.update(read(d/'receipt.json')['outputs']);continue
            data=load_data([s]);parts=[]
            with torch.inference_mode():
                for start in range(0,len(data['items']),256):parts.append(core.decode(logits(model,batch_data(data,np.arange(start,min(start+256,len(data['items']))),arm))).cpu().numpy())
            pr=np.concatenate(parts) if parts else np.empty((0,4),np.float32);arrays={};lo=0
            for row in read(OUT/'features'/s/'meta.json'):
                n=row['regions'];arrays[row['candidate_id']+'_probability']=pr[lo:lo+n];lo+=n
            np.savez_compressed(d/'data.npz',**arrays);oo={str(d/'data.npz'):sha(d/'data.npz')};seal(d/'receipt.json',dict(complete=True,GT_read=False,outputs=oo));outputs.update(oo)
            if (i+1)%20==0:stage('GT_free_region_probabilities',arm=arm,completed=i+1,total=160)
        del model;torch.cuda.empty_cache()
    seal(OUT/'probabilities_frozen.json',dict(complete=True,GT_read=False,outputs=outputs));stage('raw_probabilities_sealed_before_calibration')


def calibration_metrics(examples,rule):
    counts=Counter();scenes=set()
    for e in examples:
        q,action=core.execute(e['parent'],e['halo'],e['blocks'],e['probability'],rule);removed=np.setdiff1d(e['parent'],q,assume_unique=True);counts['deleted']+=len(removed)
        if not len(removed):continue
        counts['actions']+=1;scenes.add(e['scene']);m=e['meta'];gt=e['gt']
        if not m['stable']:counts['unknown']+=len(removed);counts['unknown_identity_actions']+=1;continue
        gid=m['target'];wrong=int((gt[removed]==gid).sum());other=int(((gt[removed]>=1000)&(gt[removed]!=gid)).sum());unknown=len(removed)-wrong-other
        counts.update(target=wrong,other=other,unknown=unknown,parent_target=int((gt[e['parent']]==gid).sum()))
        before=m['parent_iou'];total=int((gt==gid).sum());hit=int((gt[q]==gid).sum());after=hit/(len(q)+total-hit)
        counts['good_degraded']+=before>=.9 and after-before<-.02-1e-12
        values,cc=np.unique(gt[q],return_counts=True);h=dict(zip(values,cc));best=e['ids'][np.argmax([h.get(g,0)/(len(q)+n-h.get(g,0)) for g,n in zip(e['ids'],e['sizes'])])] if len(e['ids']) else gid
        counts['drift']+=best!=gid and h.get(best,0)>=20
    counts['good_parents']=sum(e['meta']['good'] for e in examples);counts['action_scenes']=len(scenes)
    n=counts['deleted'];counts['precision']=counts['other']/n if n else None;counts['target_loss']=counts['target']/max(1,counts['parent_target']);counts['unknown_fraction']=counts['unknown']/max(1,n);counts['good_rate']=counts['good_degraded']/max(1,counts['good_parents'])
    return dict(counts)


def calibrate():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'calibrate').hook);assert read(OUT/'probabilities_frozen.json')['complete'];allrules={};outputs={}
    for arm in ARMS:
        examples=[]
        for s in c['splits']['calibration']:
            _,meta,gt,ids,sizes=supervision(s,c);mm={r['candidate_id']:r for r in meta}
            with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(REGIONS/'regions'/s/'regions.npz') as bank,np.load(OUT/'probabilities'/arm/s/'data.npz') as pr:
                for row in read(OUT/'features'/s/'meta.json'):
                    cid=row['candidate_id'];blocks=prior.old.unpack(bank,cid);blocks=[blocks[k] for k in bank[cid+'_budget96']]
                    examples.append(dict(scene=s,parent=z[cid+'_parent'],halo=z[cid+'_halo'],blocks=blocks,probability=pr[cid+'_probability'],meta=mm[cid],gt=gt,ids=ids,sizes=sizes))
        candidates=[];qualified=[]
        for safe,target,unknown,risk in itertools.product(core.PARAMETERS['calibration_safe'],core.PARAMETERS['calibration_target'],core.PARAMETERS['calibration_unknown'],core.PARAMETERS['calibration_union_risk']):
            rule=dict(safe=safe,target=target,unknown=unknown,union_risk=risk,abstain=False);m=calibration_metrics(examples,rule)
            ok=m['actions']>=3 and m['action_scenes']>=2 and m['precision']>=.9 and m['target_loss']<=.01 and m['unknown_fraction']<=.1 and m['good_rate']<=.02 and m['drift']==0
            row=dict(rule=rule,metrics=m,qualified=bool(ok));candidates.append(row)
            if ok:qualified.append(row)
        if qualified:
            best=max(qualified,key=lambda r:(r['metrics']['other'],-r['metrics']['target'],-r['metrics']['unknown'],r['rule']['safe'],-r['rule']['target'],-r['rule']['unknown'],-r['rule']['union_risk']))
        else:best=dict(rule={'abstain':True},metrics=calibration_metrics(examples,{'abstain':True}),qualified=False)
        allrules[arm]=best;seal(OUT/'calibration'/f'{arm}.json',dict(complete=True,scenes=c['splits']['calibration'],chosen=best,grid=candidates));outputs[str(OUT/'calibration'/f'{arm}.json')]=sha(OUT/'calibration'/f'{arm}.json');stage('calibration_rule_locked',arm=arm,chosen=best,qualifying_rules=len(qualified))
    seal(OUT/'rules_frozen.json',dict(complete=True,rules=allrules,outputs=outputs,holdout_labels_read=False))


def apply():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'apply').hook);rules=read(OUT/'rules_frozen.json')['rules'];outputs={};counts=Counter()
    for s in c['scenes']:
        d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True);arrays={};records=[]
        with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(REGIONS/'regions'/s/'regions.npz') as bank:
            pp={a:np.load(OUT/'probabilities'/a/s/'data.npz') for a in ARMS}
            for row in read(OUT/'features'/s/'meta.json'):
                cid=row['candidate_id'];allblocks=prior.old.unpack(bank,cid);blocks=[allblocks[k] for k in bank[cid+'_budget96']]
                for arm in ARMS:
                    q,act=core.execute(z[cid+'_parent'],z[cid+'_halo'],blocks,pp[arm][cid+'_probability'],rules[arm]['rule']);arrays[cid+'_'+arm+'_points']=q;records.append(dict(candidate_id=cid,arm=arm,**act));counts[arm+'_actions']+=act['deleted']>0
            for pr in pp.values():pr.close()
        np.savez_compressed(d/'data.npz',**arrays);seal(d/'actions.json',records);outputs.update({str(d/n):sha(d/n) for n in ('data.npz','actions.json')})
    seal(OUT/'predictions_frozen.json',dict(complete=True,GT_read=False,outputs=outputs,rules_sha256=sha(OUT/'rules_frozen.json')));stage('actual_union_pointsets_sealed',**dict(counts))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','features','labels_train','train','predict_probabilities','calibrate','apply'));args=parser.parse_args();globals()[args.mode]()
