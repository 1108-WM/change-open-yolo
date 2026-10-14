#!/usr/bin/env python3
"""Reuse frozen GD10 observations; isolate correction sampling and instance identity."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,sys,time,shutil,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
import torch
from tools import run_fi1_candidate_local_repair as original
from tools import fi1_local_edit_ablation_core as learn
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha,digest
SOURCE=original.OUT;OUT=ROOT/'.local/fi1_local_edit_ablation_development200_20261012_v1'
AUDIT=ROOT/'.local/fi1_local_repair_independent_analysis_20261012'
ARMS=('A','A0','B','C');OBS='gd10_selected6';read=original.read


def stage(name,**kw):write_json(OUT/'execution.json',dict(stage=name,updated=time.time(),**kw));print(name,kw,flush=True)


class Guard:
    def __init__(self,c,mode,fold=None):self.c,self.mode,self.fold=c,mode,fold
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,Path)):return
        p=str(Path(args[0]).resolve())
        if Path(p).suffix in ('.py','.pyc'):return
        if '/fi1_local_repair_independent_analysis_' in p:raise PermissionError('posthoc error/oracle lists excluded from new learner')
        if any(q in p for q in ('/val312/','/reserve100/','fi1_sl_single200_dev55')):raise PermissionError('outside original development200')
        if any(q.startswith('ground_truth') for q in Path(p).parts):
            allowed=self.mode=='labels' and (OUT/'conditions_frozen.json').exists() or self.mode=='evaluate' and (OUT/'predictions_frozen.json').exists()
            if not allowed or p not in self.c['GT_inputs']:raise PermissionError('GT only supervision creation or sealed evaluation')
        if str(OUT/'labels')+'/' in p or str(SOURCE/'labels')+'/' in p:
            if self.mode not in ('labels','train','evaluate','pilot'):raise PermissionError('inference cannot read label cache')
            if self.mode in ('train','pilot'):
                s=next((s for s in self.c['scenes'] if '/'+s+'/' in p),None)
                if s is None or self.c['folds'][s]==self.fold:raise PermissionError('outer-heldout labels forbidden')
        if self.mode in ('conditions','predict') and (Path(p).name in ('summary.json','candidate_diagnosis.csv','geometry_quality.json','target_census.json','overlap.npz') or '/diagnosis/' in p):raise PermissionError('GT posthoc diagnosis excluded')


def prepare():
    assert not OUT.exists();assert shutil.disk_usage(ROOT).free>=50*1024**3
    c0=read(SOURCE/'contract.json');assert read(SOURCE/'delivery_verified.json')['complete']
    original.audit.prior.old.verify_files(c0['code']);original.audit.prior.old.verify_files(c0['frozen_SL_models'])
    verify=read(AUDIT/'verification.json')
    for name,h in verify['artifacts'].items():assert sha(AUDIT/name)==h
    assert verify['source_contract_sha256']==sha(SOURCE/'contract.json')
    files=[ROOT/'tools'/name for name in ('run_fi1_local_edit_ablation.py','fi1_local_edit_ablation_core.py','evaluate_fi1_local_edit_ablation.py','verify_fi1_local_edit_ablation.py','run_fi1_local_edit_ablation.sh')]
    files += [ROOT/'tests/test_fi1_local_edit_ablation.py',ROOT/'docs/FI1_LOCAL_EDIT_ABLATION_20261012.md']
    code=dict(c0['code']);code.update({str(p):sha(p) for p in files})
    bindings={str(SOURCE/name):sha(SOURCE/name) for name in ('contract.json','plan_frozen.json','features_frozen.json','labels_frozen.json','models_frozen.json','predictions_frozen.json','evaluation_frozen.json','delivery_verified.json')}
    for name in ('plan','features','labels','models','predictions'):bindings.update(read(SOURCE/(name+'_frozen.json'))['outputs'])
    c=dict(version=OUT.name,scenes=c0['scenes'],folds=c0['folds'],groups=c0['groups'],fixed10=c0['fixed10'],class_ids=c0['class_ids'],jobs=c0['jobs'],GT_inputs=c0['GT_inputs'],inputs=c0['inputs'],code=code,frozen_SL_models=c0['frozen_SL_models'],source_bindings=bindings,
           audit_report_sha256=sha(AUDIT/'report.md'),parameters=learn.PARAMETERS,arms=list(ARMS),source_observation_arm=OBS,gate=c0['gate'],
           control='A original frozen GD10 predictions, trained on original two arms; A0 GD10-only original uniform supervision; B same A0 features/network/prior/12epochs with balanced strata; C B+explicit parent and neighbor conditioning and six-class edit gate.',
           candidate_targets='Maximum valid parent GT IoU >=20 intersections, same base labels. Only training folds get strata and GT scope labels. Stable target purity>=.6; visible ideal parent delta>=.005=>edit else keep; GT outside scope/unannotated/ambiguous/small are separately supervised only in C and posthoc evaluation.',
           pilot='Per outer-fold, training-only hash4/stratum up to20 parents B/C40 fixed epochs. Diagnostic only, no parameter selection or warm start; final fits independently from fixed seed.',
           point_action='Same .8 probability, >=2 measured-visible groups and min100; C extra learned candidate p(edit)>=.7. No GT-driven acceptance, no threshold calibration or sweep.',
           success='Original frozen conservative geometry gate plus >=1 deletion-only pool-gain>=.02 target. Paired B-A0 and C-B, baseline A; preserve additions, reduce neighbor additions and severe degradation. Both labels and diagnosis remain exposed development, not whole-system blind.',
           SL_modified=False,production_modified=False,new_frames=False,new_SAM=False,new_DINO_features=False)
    OUT.mkdir();freeze_json(OUT/'contract.json',c)
    (OUT/'features').symlink_to(SOURCE/'features',target_is_directory=True);(OUT/'plan').symlink_to(SOURCE/'plan',target_is_directory=True)
    for p in files:
        d=OUT/'code_snapshot'/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
    stage('prepared',scenes=200,parents=1599,variants=list(ARMS))


def condition_scene(s):
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'conditions').hook);d=OUT/'conditions'/s
    if (d/'receipt.json').exists():r=read(d/'receipt.json');original.audit.prior.old.verify_files(r['outputs']);return r
    job=c['jobs'][s];geometry,points=original.audit.prior.old.load_nt(job);xyz=np.asarray(np.load(job['paths']['processed'],mmap_mode='r')[:,:3]);arrays={};meta=[]
    with np.load(SOURCE/'features'/s/'data.npz') as z:
        for row in read(SOURCE/'plan'/s/'candidates.json'):
            cid=row['candidate_id'];p=z[cid+'_parent'];halo=z[cid+'_halo'];x=z[cid+'_'+OBS+'_x'].astype(np.float32);parent=np.isin(halo,p)
            lookup=np.full(len(xyz),-1,np.int32);lookup[halo]=np.arange(len(halo));ranked=[]
            for g in geometry:
                key=g['geometry_key']
                if key==row['parent']:continue
                local=lookup[points[key]];local=local[local>=0]
                if len(local)<20:continue
                overlap=int(parent[local].sum());iou=overlap/(len(p)+len(points[key])-overlap)
                if iou>=.8:continue
                contact=int((x[local,10]<=1/3).sum());novel=int((~parent[local]).sum())
                if contact<5:continue
                score=min(contact,256)+.5*min(novel,256);ranked.append((-score,-len(local),key,local))
            chosen=[]
            for _,_,key,local in sorted(ranked,key=lambda q:q[:3]):
                if any(len(np.intersect1d(local,previous))/len(np.union1d(local,previous))>=.8 for _,previous in chosen):continue
                chosen.append((key,local))
                if len(chosen)==4:break
            member=np.zeros((len(halo),4),bool);distance=np.full((len(halo),4),.15,np.float16);valid=np.zeros(4,bool);overlap=np.zeros(4,np.float32)
            for k,(key,local) in enumerate(chosen):
                member[local,k]=True;valid[k]=True;dd,_=cKDTree(xyz[halo[local]]).query(xyz[halo]);distance[:,k]=np.minimum(dd,4).astype(np.float16);overlap[k]=parent[local].mean()
            rawx=x.copy();rawx[:,:3]=xyz[halo]-xyz[p].mean(0);reps=learn.representatives(rawx,parent)
            own_distance=np.linalg.norm(xyz[halo,None]-xyz[halo[reps]][None],axis=-1).clip(0,4).astype(np.float16)
            arrays.update({cid+'_membership':member,cid+'_distance':distance,cid+'_valid':valid,cid+'_overlap':overlap,cid+'_representatives':reps,cid+'_own_distance':own_distance})
            meta.append(dict(candidate_id=cid,competitors=[key for key,_ in chosen],parent_representatives=halo[reps].tolist()))
    d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'data.npz',**arrays);freeze_json(d/'meta.json',meta)
    receipt=dict(complete=True,GT_read=False,scene=s,outputs={str(d/name):sha(d/name) for name in ('data.npz','meta.json')});freeze_json(d/'receipt.json',receipt);return receipt


def conditions():
    c=read(OUT/'contract.json');original.audit.prior.old.verify_files(c['code']);original.audit.prior.old.verify_files(c['source_bindings']);outputs={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(condition_scene,s) for s in c['scenes']])):
            outputs.update(f.result()['outputs']);stage('GT_free_competition_conditions',completed=i+1,total=200)
    freeze_json(OUT/'conditions_frozen.json',dict(complete=True,GT_read=False,outputs=outputs,original_features_and_views_unchanged=True))


def candidate_scope(gt,parent,halo,y,visible,metadata,classes):
    values,counts=np.unique(gt[parent],return_counts=True);i=int(np.argmax(counts));dominant=int(values[i]);fraction=float(counts[i]/len(parent));target=metadata['target']
    sizes=Counter(gt.tolist())
    if dominant>=1000 and dominant//1000 not in classes and fraction>=.6:return 2
    if float((gt[parent]<1000).mean())>=.5:return 3
    if dominant>=1000 and dominant//1000 in classes and sizes[dominant]<100 and fraction>=.6:return 5
    purity=0 if target is None else float((gt[parent]==target).mean())
    if target is None or purity<.6:return 4
    state=np.isin(halo,parent);known=y!=2;seen=visible>=2
    corrected=state.copy();corrected[seen&known]=(y[seen&known]==0)
    before_hit=int((gt[parent]==target).sum());old=before_hit/(len(parent)+sizes[target]-before_hit)
    new_hit=int((gt[halo[corrected]]==target).sum());new=new_hit/(corrected.sum()+sizes[target]-new_hit)
    return 0 if new-old>=.005 else 1


def labels():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'labels').hook);outputs={};stats=Counter()
    for s in c['scenes']:
        job=c['jobs'][s];assert sha(job['GT'])==c['GT_inputs'][job['GT']];gt=np.loadtxt(job['GT'],dtype=np.int64);base_meta={r['candidate_id']:r for r in read(SOURCE/'labels'/s/'meta.json')};arrays={};rows=[]
        with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(SOURCE/'labels'/s/'labels.npz') as labels,np.load(OUT/'conditions'/s/'data.npz') as cond:
            for r in read(SOURCE/'plan'/s/'candidates.json'):
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];y=labels[cid+'_y'];x=z[cid+'_'+OBS+'_x'];visible=z[cid+'_'+OBS+'_groups'];meta=base_meta[cid]
                groups=learn.strata(y,np.isin(halo,parent),visible,x);scope=candidate_scope(gt,parent,halo,y,visible,meta,c['class_ids']);membership=cond[cid+'_membership'];relations=np.full(4,2,np.uint8)
                if meta['target'] is not None:
                    for k in np.flatnonzero(cond[cid+'_valid']):
                        raw=gt[halo[membership[:,k]]];values,counts=np.unique(raw,return_counts=True);i=int(np.argmax(counts));dominant=int(values[i])
                        if dominant>=1000 and counts[i]/len(raw)>=.6:relations[k]=0 if dominant==meta['target'] else 1
                arrays[cid+'_y']=y;arrays[cid+'_relation']=relations
                for k,g in enumerate(groups):arrays[cid+'_stratum'+str(k)]=g.astype(np.int32)
                rows.append(dict(meta,scope=scope,scope_name=learn.SCOPE[scope],stratum_points={name:len(g) for name,g in zip(learn.STRATA,groups)}));stats[learn.SCOPE[scope]]+=1
        d=OUT/'labels'/s;d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'data.npz',**arrays);freeze_json(d/'meta.json',rows);outputs.update({str(d/name):sha(d/name) for name in ('data.npz','meta.json')})
    freeze_json(OUT/'labels_frozen.json',dict(complete=True,outputs=outputs,scope_counts=dict(stats)));stage('supervision_sealed',**dict(stats))


def examples(scenes,supervised=True):
    result={}
    for s in scenes:
        with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(OUT/'conditions'/s/'data.npz') as cond:
            if supervised:lab=np.load(OUT/'labels'/s/'data.npz');meta={r['candidate_id']:r for r in read(OUT/'labels'/s/'meta.json')}
            for r in read(SOURCE/'plan'/s/'candidates.json'):
                cid=r['candidate_id'];e=dict(x=z[cid+'_'+OBS+'_x'].astype(np.float32),neighbors=z[cid+'_neighbors'].astype(np.int64),weights=z[cid+'_neighbor_weights'].astype(np.float32),parent=z[cid+'_parent'],halo=z[cid+'_halo'],visible=z[cid+'_'+OBS+'_groups'])
                e['condition']={name:cond[cid+'_'+name] for name in ('membership','distance','valid','overlap','representatives','own_distance')}
                if supervised:e.update(y=lab[cid+'_y'].astype(np.int64),groups=[lab[cid+'_stratum'+str(k)] for k in range(5)],relation=lab[cid+'_relation'].astype(np.int64),meta=meta[cid])
                result[s,cid]=e
            if supervised:lab.close()
    return result


def tensor_conditions(ee,device='cuda'):
    result=[]
    for e in ee:
        result.append({k:torch.tensor(v,device=device,dtype=torch.long if k=='representatives' else torch.bool if k in ('membership','valid') else torch.float32) for k,v in e['condition'].items()})
    return result


def forward(model,variant,ee,device='cuda'):
    x,nn,w,parts=original.collate(ee,device=device)
    if variant=='C':logits,candidate,relations=model(x,nn,w,parts,tensor_conditions(ee,device));return x,parts,logits,candidate,relations
    return x,parts,model(x,nn,w,parts),None,None


def fit(variant,data,keys,epochs,seed,d,fold,pilot=False):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=learn.model(variant).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[];scene_counts=Counter(s for s,cid in keys)
    scope_counts=Counter(data[key]['meta']['scope'] for key in keys);class_weights=np.array([1/np.sqrt(max(scope_counts[k],1)) for k in range(6)]);class_weights=class_weights/(sum(scope_counts[k]*class_weights[k] for k in range(6))/len(keys));class_weights=torch.tensor(class_weights,device='cuda',dtype=torch.float32)
    epoch_start=0
    if (d/'checkpoint.pt').exists():
        state=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);epoch_start=state['epoch'];history=state['history']
    for epoch in range(epoch_start,epochs):
        model.train();order=np.random.default_rng(seed+100000+epoch).permutation(len(keys));losses=[];tic=time.time()
        for start in range(0,len(order),4):
            batch_keys=[keys[i] for i in order[start:start+4]];batch=[data[k] for k in batch_keys]
            batches=[list(zip(batch_keys,batch))] if sum(len(e['x']) for e in batch)<=60000 else [[(k,e)] for k,e in zip(batch_keys,batch)]
            for pairs in batches:
                kk,ee=zip(*pairs);optimizer.zero_grad(set_to_none=True);x,parts,logits,candidate,relations=forward(model,variant,ee);loss=0
                for i,(key,e,(lo,hi)) in enumerate(zip(kk,ee,parts)):
                    y=torch.tensor(e['y'],device='cuda');parent=x[lo:hi,9]>.5
                    if variant=='A0':indices=torch.tensor(rng.choice(hi-lo,min(4096,hi-lo),replace=False),device='cuda');value=original.core.segmentation_loss(logits[lo:hi],y,parent,e['meta']['good'],indices)
                    else:value=learn.balanced_loss(logits[lo:hi],y,parent,e['meta']['good'],e['groups'],rng)
                    if variant=='C':
                        value=value+.5*torch.nn.functional.cross_entropy(candidate[i:i+1],torch.tensor([e['meta']['scope']],device='cuda'),weight=class_weights,reduction='sum')
                        valid=torch.tensor(e['condition']['valid'],device='cuda');truth=torch.tensor(e['relation'],device='cuda')
                        if valid.any():value=value+.25*torch.nn.functional.cross_entropy(relations[i][valid],truth[valid])
                    loss=loss+value*(len(keys)/len(scene_counts)/scene_counts[key[0]])/len(ee)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        entry=dict(epoch=epoch+1,mean_training_loss=float(np.mean(losses)),steps=len(losses),seconds=time.time()-tic);history.append(entry)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),d/'checkpoint.pt')
        stage('training_fit' if pilot else 'training_fivefold',variant=variant,fold=fold,**entry)
    model.eval();return model,history


def pilot_summary(model,variant,data,keys):
    totals=Counter();classification=Counter()
    for key in keys:
        e=data[key]
        with torch.inference_mode():x,parts,logits,candidate,relation=forward(model,variant,[e]);prob=logits.softmax(1).cpu().numpy().astype(np.float16);cp=None if candidate is None else candidate.softmax(1)[0].cpu().numpy().astype(np.float16)
        predicted,decision=learn.repair(e['parent'],e['halo'],prob,e['visible'],variant,cp);removed=np.isin(e['halo'],np.setdiff1d(e['parent'],predicted));added=np.isin(e['halo'],np.setdiff1d(predicted,e['parent']));truth=e['y'];seen=e['visible']>=2;present=np.isin(e['halo'],e['parent'])
        totals.update(delete_opportunity=int((present&(truth==1)&seen).sum()),add_opportunity=int((~present&(truth==0)&seen).sum()),correct_delete=int((removed&(truth==1)).sum()),correct_add=int((added&(truth==0)).sum()),wrong_delete=int((removed&(truth==0)).sum()),wrong_add=int((added&(truth==1)).sum()),unknown_edits=int(((removed|added)&(truth==2)).sum()),actions=decision['accepted'])
        if cp is not None:classification['correct']+=int(cp.argmax()==e['meta']['scope']);classification['total']+=1
    return dict(totals),dict(classification)


def pilot():
    c=read(OUT/'contract.json');guard=Guard(c,'pilot',0);sys.addaudithook(guard.hook);assert torch.cuda.is_available();torch.set_num_threads(4);outputs={}
    for fold in range(5):
        guard.fold=fold;scenes=[s for s in c['scenes'] if c['folds'][s]!=fold];eligible=[]
        for s in scenes:
            for r in read(OUT/'labels'/s/'meta.json'):eligible.append((s,r))
        selected=set()
        for category in learn.STRATA:
            pool=[(s,r['candidate_id']) for s,r in eligible if r['stratum_points'][category]>0]
            selected.update(sorted(pool,key=lambda k:(digest(['pilot-20261012',fold,category,*k]),k))[:4])
        keys=sorted(selected);data=examples(sorted({s for s,cid in keys}));data={key:data[key] for key in keys}
        for variant in ('B','C'):
            d=OUT/'pilot'/variant/f'fold{fold}';d.mkdir(parents=True,exist_ok=True)
            if (d/'receipt.json').exists():outputs.update(read(d/'receipt.json')['outputs']);continue
            model,history=fit(variant,data,keys,40,20261012+fold,d,fold,True);metrics,classification=pilot_summary(model,variant,data,keys)
            torch.save(model.cpu().state_dict(),d/'model.pt');freeze_json(d/'fit.json',dict(training_only=True,fold=fold,keys=keys,epochs=history,metrics=metrics,candidate_classification=classification,used_for_parameter_choice=False,used_for_full_training_warm_start=False));oo={str(d/name):sha(d/name) for name in ('model.pt','fit.json')};freeze_json(d/'receipt.json',dict(complete=True,outputs=oo));outputs.update(oo);del model;torch.cuda.empty_cache()
            stage('training_inner_fit_checked',fold=fold,variant=variant,metrics=metrics)
        del data
    freeze_json(OUT/'pilot_frozen.json',dict(complete=True,outputs=outputs,no_hyperparameter_adjustment=True))


def train():
    c=read(OUT/'contract.json');guard=Guard(c,'train',0);sys.addaudithook(guard.hook);assert torch.cuda.is_available();torch.set_num_threads(4);outputs={};original.audit.prior.old.verify_files(c['code'])
    assert read(OUT/'pilot_frozen.json')['complete']
    for fold in range(5):
        guard.fold=fold;scenes=[s for s in c['scenes'] if c['folds'][s]!=fold];data=examples(scenes);keys=sorted(data)
        for variant in learn.VARIANTS:
            d=OUT/'models'/variant/f'fold{fold}';d.mkdir(parents=True,exist_ok=True)
            if (d/'receipt.json').exists():r=read(d/'receipt.json');original.audit.prior.old.verify_files(r['outputs']);outputs.update(r['outputs']);continue
            model,history=fit(variant,data,keys,12,20261012+fold,d,fold);torch.save(model.cpu().state_dict(),d/'model.pt')
            freeze_json(d/'history.json',dict(train_scenes=scenes,test_scenes=[s for s in c['scenes'] if c['folds'][s]==fold],epochs=history,model_parameters=sum(p.numel() for p in model.parameters()),GD10_only=True,uniform_supervision=variant=='A0',full_training_from_scratch=True))
            oo={str(d/name):sha(d/name) for name in ('model.pt','history.json')};freeze_json(d/'receipt.json',dict(complete=True,outputs=oo,heldout_labels_read=False));outputs.update(oo);del model;torch.cuda.empty_cache()
        del data
    freeze_json(OUT/'models_frozen.json',dict(complete=True,models=15,outputs=outputs,no_heldout_labels=True));stage('fifteen_models_sealed')


def predict():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'predict').hook);assert torch.cuda.is_available();torch.set_num_threads(4);outputs={};original.audit.prior.old.verify_files(read(OUT/'models_frozen.json')['outputs'])
    for fold in range(5):
        models={v:learn.model(v).cuda().eval() for v in learn.VARIANTS}
        for v,m in models.items():m.load_state_dict(torch.load(OUT/'models'/v/f'fold{fold}'/'model.pt',map_location='cuda',weights_only=True))
        for s in c['scenes']:
            if c['folds'][s]!=fold:continue
            d=OUT/'predictions'/s
            if (d/'receipt.json').exists():r=read(d/'receipt.json');original.audit.prior.old.verify_files(r['outputs']);outputs.update(r['outputs']);continue
            data=examples([s],False);arrays={};records=[];old_decisions={r['candidate_id']:r for r in read(SOURCE/'predictions'/s/'decisions.json')}
            with np.load(SOURCE/'predictions'/s/'predictions.npz') as old:
                for row in read(SOURCE/'plan'/s/'candidates.json'):
                    cid=row['candidate_id'];e=data[s,cid];record=dict(candidate_id=cid,source=row['source'],arms={'A':old_decisions[cid]['arms'][OBS]})
                    arrays[cid+'_A_points']=old[cid+'_'+OBS+'_points'];arrays[cid+'_A_probability']=old[cid+'_'+OBS+'_probability']
                    for v,m in models.items():
                        with torch.inference_mode():x,parts,logits,can,rel=forward(m,v,[e]);p=logits.softmax(1).cpu().numpy().astype(np.float16);cp=None if can is None else can.softmax(1)[0].cpu().numpy().astype(np.float16)
                        repaired,decision=learn.repair(e['parent'],e['halo'],p,e['visible'],v,cp);arrays[cid+'_'+v+'_points']=repaired;arrays[cid+'_'+v+'_probability']=p
                        if cp is not None:arrays[cid+'_C_candidate_probability']=cp;arrays[cid+'_C_relation_probability']=rel[0].softmax(1).cpu().numpy().astype(np.float16)
                        decision['points_sha256']=hashlib.sha256(repaired.astype('<i8').tobytes()).hexdigest();record['arms'][v]=decision
                    records.append(record)
            d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'predictions.npz',**arrays);freeze_json(d/'decisions.json',records);oo={str(d/name):sha(d/name) for name in ('predictions.npz','decisions.json')};freeze_json(d/'receipt.json',dict(complete=True,fold=fold,GT_read=False,outputs=oo));outputs.update(oo);stage('GT_free_heldout_predictions',fold=fold,scene=s)
            del data
        del models;torch.cuda.empty_cache()
    freeze_json(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,GT_read=False,A_exact_frozen_reuse=True,all_new_predictions_OOF=True));stage('all_four_arms_sealed')


def evaluate():
    from tools.evaluate_fi1_local_edit_ablation import evaluate as run
    run()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','conditions','labels','pilot','train','predict','evaluate'));a=p.parse_args();globals()[a.mode]()

if __name__=='__main__':main()
