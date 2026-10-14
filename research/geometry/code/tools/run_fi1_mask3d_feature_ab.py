#!/usr/bin/env python3
"""Fixed A/B feature experiment; physical model selection, then calibration, then check."""
from pathlib import Path
from collections import Counter
import argparse,copy,json,shutil,sys,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import fi1_mask3d_feature_ab_core as core
from tools import run_fi1_candidate_local_repair as original
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
OUT=ROOT/'.local/fi1_mask3d_feature_ab_internal_20261013_v1'
SOURCE=original.OUT
LABELS=ROOT/'.local/fi1_deletion_blocks_development200_20261013_v1'
PRIOR=ROOT/'.local/fi1_decoupled_region_internal_development200_20261013_v1'
REGIONS=ROOT/'.local/fi1_local_region_supply_development200_20261013_v1'
OBS='gd10_selected6'
def read(p):return json.loads(Path(p).read_text())
def plain(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v
def seal(p,v):freeze_json(p,plain(v))
def stage(name,**kw):write_json(OUT/'execution.json',dict(stage=name,updated=time.time(),**plain(kw)));print(name,plain(kw),flush=True)
def reserve():
    for p in (ROOT,OUT.resolve()):
        if shutil.disk_usage(p).free<50*2**30:raise RuntimeError('50GiB reserve reached: '+str(p))
def integrity(mapping):
    for p,h in mapping.items():assert sha(p)==h,p

class Guard:
    def __init__(self,c,mode,allowed=()):self.c,self.mode,self.allowed=c,mode,set(allowed)
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,Path)):return
        p=Path(args[0]).resolve();name=str(p)
        if p.suffix in ('.py','.pyc','.so'):return
        if 'independent_analysis' in name or '/diagnosis/' in name or p.name in ('candidate_diagnosis.csv','sampling_diagnosis.csv','geometry_quality.json','target_census.json','overlap.npz'):raise PermissionError('posthoc diagnostic input forbidden')
        israw=any(k.startswith('ground_truth') for k in p.parts);islabel='/labels/' in name
        if israw or islabel:
            s=next((s for s in self.c['scenes'] if '/'+s+'/' in name or p.name==s+'.txt'),None)
            if self.mode=='train':ok=islabel and s in self.allowed
            elif self.mode=='calibrate':ok=israw and s in self.c['splits']['calibration'] and (OUT/'probabilities_frozen.json').exists()
            elif self.mode in ('evaluate','verify'):ok=s in self.c['scenes'] and (OUT/'predictions_frozen.json').exists()
            else:ok=False
            if not ok:raise PermissionError('split/stage GT and label isolation')
            if israw and name not in self.c['GT_inputs']:raise PermissionError('GT whitelist')

def prepare():
    assert OUT.exists() and not (OUT/'contract.json').exists();reserve()
    old=read(PRIOR/'contract.json');ss=core.selection_split(old['splits']['train']);assert len(ss['fit'])==96 and len(ss['selection'])==16
    split=copy.deepcopy(old['splits']);split.update(ss)
    physical=[{s.split('_')[0] for s in split[k]} for k in ('fit','selection','calibration','holdout')]
    assert all(not physical[i]&physical[j] for i in range(4) for j in range(i))
    files=[ROOT/'tools'/n for n in ('fi1_mask3d_feature_ab_core.py','run_fi1_mask3d_feature_ab.py','export_fi1_mask3d_instance_features.py','evaluate_fi1_mask3d_feature_ab.py','verify_fi1_mask3d_feature_ab.py','run_fi1_mask3d_feature_ab.sh')]
    files += [ROOT/'tests/test_fi1_mask3d_feature_ab.py',ROOT/'docs/FI1_MASK3D_FEATURE_AB_INTERNAL_20261013.md',ROOT/'tools/fi1_candidate_local_segmenter.py',ROOT/'tools/fi1_deletion_block_core.py',ROOT/'utils/utils_3d.py',ROOT/'models/Mask3D/mask3d/__init__.py',ROOT/'models/Mask3D/mask3d/models/mask3d.py']
    bindings={str(PRIOR/'contract.json'):sha(PRIOR/'contract.json')}
    for root,names in ((SOURCE,('plan','features')),(LABELS,('labels',)),(REGIONS,('regions',))):
        for name in names:
            for p,h in read(root/(name+'_frozen.json'))['outputs'].items():
                if any('/'+s+'/' in p for s in old['scenes']):bindings[p]=h
    c={k:old[k] for k in ('scenes','folds','groups','class_ids','jobs','GT_inputs','inputs','frozen_SL_models')}
    c.update(version=OUT.name,seed=core.SEED,splits=split,arms=['A','B'],source_bindings=bindings,code={str(p):sha(p) for p in files},
        mask3d_checkpoint_sha256=sha(ROOT/'pretrained/checkpoints/scannet200_val.ckpt'),maximum_epochs=24,
        model_selection='minimum per-scene corrected loss on selection16, earliest tie, no refit112',
        network='existing CandidateLocalSegmenter(192,64), same initialization/order/optimizer; A extra128 channels zero AFTER normalization',
        supervision='sealed corrected full-domain stable-parent labels; every original point once; pollution cost3, known-domain Dice.5, good-preservation.25',
        export='read only processed[:,:9], no segments/labels; frozen mask_features_head128, explicit sparse-coordinate+inverse map, hook output invariance every scene',
        thresholds=list(core.THRESHOLDS),generation_threshold=.8,physical_split_preserved=True,SL_modified=False,AP_run=False,new_frames=False,new_quality_heads=False,
        gate=dict(min_value_targets=1,min_actions=3,min_action_scenes=2,min_add_precision=.9,min_delete_precision=.9,max_target_loss=.01,max_unknown=.1,max_good_rate=.02,max_good_tail_loss=.1,max_drift=0),
        source_observation_arm=OBS,point_output=True,previous_holdout_exposed=True)
    integrity(c['code']);integrity(bindings);integrity(c['frozen_SL_models']);seal(OUT/'contract.json',c)
    for p in files:
        q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    stage('prepared',scenes=160,fit=96,selection=16,calibration=24,holdout=24)

def load_scenes(scenes,supervised=False):
    result={}
    for s in scenes:
        with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(OUT/'instance_features'/s/'data.npz') as prior:
            metadata={r['candidate_id']:r for r in read(LABELS/'labels'/s/'meta.json')} if supervised else {}
            lab=np.load(LABELS/'labels'/s/'data.npz') if supervised else None
            roi=prior['point_ids'];feat=prior['features']
            for r in read(SOURCE/'plan'/s/'candidates.json'):
                cid=r['candidate_id'];halo=z[cid+'_halo'];idx=np.searchsorted(roi,halo);assert np.array_equal(roi[idx],halo)
                e=dict(base=z[cid+'_'+OBS+'_x'].astype(np.float32),prior=feat[idx].astype(np.float32),neighbors=z[cid+'_neighbors'].astype(np.int64),weights=z[cid+'_neighbor_weights'].astype(np.float32),parent=z[cid+'_parent'],halo=halo,visible=z[cid+'_'+OBS+'_groups'],source=r['source'])
                if supervised:e.update(y=lab[cid+'_y'].astype(np.int64),meta=metadata[cid])
                result[s,cid]=e
            if lab is not None:lab.close()
    return result

def normalization(data):
    # Same prior standardization for both arms, fitted only on actual fitting scenes.
    total=0;su=np.zeros(128,np.float64);square=su.copy()
    for e in data.values():
        v=e['prior'];total+=len(v);su+=v.sum(0,dtype=np.float64);square+=np.square(v.astype(np.float64)).sum(0)
    mean=su/total;std=np.sqrt(np.maximum(square/total-mean**2,1e-6));return mean.astype(np.float32),std.astype(np.float32)

def forward(net,ee,arm,mean,std,device='cuda'):
    examples=[dict(x=core.inputs(e['base'],e['prior'],mean,std,arm),neighbors=e['neighbors'],weights=e['weights']) for e in ee]
    x,nn,w,parts=original.collate(examples,device=device);return net(x,nn,w,parts),x,parts

def validation(net,data,arm,mean,std):
    net.eval();by_scene={};accuracy=Counter()
    with torch.no_grad():
        for (s,cid),e in sorted(data.items()):
            logits,x,parts=forward(net,[e],arm,mean,std);y=torch.tensor(e['y'],device='cuda')
            value=core.corrected_loss(logits,y,x[:,9]>.5,e['meta']['good']);by_scene.setdefault(s,[]).append(float(value))
            pred=logits.argmax(1);accuracy['points']+=len(y);accuracy['correct']+=int((pred==y).sum());accuracy['known']+=int((y!=2).sum());accuracy['known_correct']+=int(((y!=2)&(pred==y)).sum())
    return float(np.mean([np.mean(v) for v in by_scene.values()])),dict(accuracy)

def train():
    c=read(OUT/'contract.json');assert read(OUT/'instance_features_frozen.json')['complete'];reserve();integrity(c['code']);integrity(c['frozen_SL_models'])
    guard=Guard(c,'train',c['splits']['fit']+c['splits']['selection']);sys.addaudithook(guard.hook)
    torch.set_num_threads(4);assert torch.cuda.is_available();fit=load_scenes(c['splits']['fit'],True);selection=load_scenes(c['splits']['selection'],True);mean,std=normalization(fit)
    if not (OUT/'normalization.npz').exists():np.savez_compressed(OUT/'normalization.npz',mean=mean,std=std)
    else:
        with np.load(OUT/'normalization.npz') as z:assert np.array_equal(z['mean'],mean) and np.array_equal(z['std'],std)
    keys=sorted(fit);scene_counts=Counter(s for s,cid in keys);outputs={}
    for arm in ('A','B'):
        d=OUT/'models'/arm;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():integrity(read(d/'receipt.json')['outputs']);outputs.update(read(d/'receipt.json')['outputs']);continue
        torch.manual_seed(core.SEED);net=core.model().cuda();optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        history=[];first=0;best=float('inf');selected_epoch=None
        initial_hash=__import__('hashlib').sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
        if (d/'checkpoint.pt').exists():
            state=torch.load(d/'checkpoint.pt',map_location='cuda',weights_only=False);net.load_state_dict(state['model']);optim.load_state_dict(state['optimizer']);history=state['history'];first=state['epoch'];best=state['best'];selected_epoch=state['selected_epoch'];torch.set_rng_state(state['rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        for epoch in range(first,c['maximum_epochs']):
            net.train();order=np.random.default_rng(core.SEED+epoch).permutation(len(keys));losses=[];tic=time.time()
            for start in range(0,len(order),4):
                kk=[keys[i] for i in order[start:start+4]];batches=[kk] if sum(len(fit[k]['base']) for k in kk)<=60000 else [[k] for k in kk]
                for batch in batches:
                    ee=[fit[k] for k in batch];optim.zero_grad(set_to_none=True);logits,x,parts=forward(net,ee,arm,mean,std);loss=0
                    for k,e,(lo,hi) in zip(batch,ee,parts):
                        y=torch.tensor(e['y'],device='cuda');value=core.corrected_loss(logits[lo:hi],y,x[lo:hi,9]>.5,e['meta']['good']);loss+=value*(len(keys)/len(scene_counts)/scene_counts[k[0]])/len(ee)
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optim.step();losses.append(float(loss.detach()))
            val,acc=validation(net,selection,arm,mean,std)
            if val<best-1e-10:best=val;selected_epoch=epoch+1;torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},d/'model.pt')
            h=dict(epoch=epoch+1,training_loss=float(np.mean(losses)),selection_loss=val,selection_points=acc,selected_epoch=selected_epoch,steps=len(losses),seconds=time.time()-tic);history.append(h)
            torch.save(dict(model=net.state_dict(),optimizer=optim.state_dict(),epoch=epoch+1,history=history,best=best,selected_epoch=selected_epoch,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),d/'checkpoint.pt')
            stage('AB_training',arm=arm,**h);reserve()
        seal(d/'history.json',dict(arm=arm,fit_scenes=c['splits']['fit'],selection_scenes=c['splits']['selection'],epochs=history,selected_epoch=selected_epoch,best_selection_loss=best,initial_state_sha256=initial_hash,parameters=sum(p.numel() for p in net.parameters()),replacement=False,calibration_or_holdout_labels_read=False))
        oo={str(d/n):sha(d/n) for n in ('model.pt','history.json')};seal(d/'receipt.json',dict(complete=True,outputs=oo));outputs.update(oo);del net,optim;torch.cuda.empty_cache()
    outputs[str(OUT/'normalization.npz')]=sha(OUT/'normalization.npz');seal(OUT/'models_frozen.json',dict(complete=True,outputs=outputs,arms=2,Mask3D_trained=False,selection_only_inside112=True));stage('AB_models_locked')

def predict():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'predict').hook);torch.set_num_threads(4);integrity(read(OUT/'models_frozen.json')['outputs'])
    with np.load(OUT/'normalization.npz') as z:mean,std=z['mean'],z['std']
    models={}
    for arm in ('A','B'):
        net=core.model().cuda().eval();net.load_state_dict(torch.load(OUT/'models'/arm/'model.pt',map_location='cuda',weights_only=True));models[arm]=net
    outputs={}
    for i,s in enumerate(c['scenes']):
        d=OUT/'probabilities'/s;d.mkdir(parents=True,exist_ok=True)
        if (d/'receipt.json').exists():integrity(read(d/'receipt.json')['outputs']);outputs.update(read(d/'receipt.json')['outputs']);continue
        data=load_scenes([s]);arrays={}
        for (s,cid),e in sorted(data.items()):
            for arm in ('A','B'):
                with torch.inference_mode():logits,_,_=forward(models[arm],[e],arm,mean,std);p=logits.softmax(1).cpu().numpy().astype(np.float16)
                arrays[cid+'_'+arm+'_probability']=p
                arrays[cid+'_'+arm+'_generated']=core.repair(e['parent'],e['halo'],p,e['visible'],.8)
        np.savez_compressed(d/'data.npz',**arrays);oo={str(d/'data.npz'):sha(d/'data.npz')};seal(d/'receipt.json',dict(complete=True,GT_read=False,outputs=oo));outputs.update(oo);reserve();stage('GT_free_AB_prediction',completed=i+1,total=160,scene=s)
    seal(OUT/'probabilities_frozen.json',dict(complete=True,outputs=outputs,GT_read=False,all_generated_pointsets_frozen=True));stage('raw_AB_outputs_sealed_before_calibration')

def calibrate():
    from tools.evaluate_fi1_mask3d_feature_ab import evaluate_scenes
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'calibrate').hook);assert read(OUT/'probabilities_frozen.json')['complete'];grid=[];chosen={}
    results=evaluate_scenes(c,c['splits']['calibration'],[(a,t) for a in ('A','B') for t in core.THRESHOLDS],calibration=True)
    for arm in ('A','B'):
        qualified=[]
        for t in core.THRESHOLDS:
            m=results[f'{arm}:{t}']['metrics'];protect=core.protection(m);ok=all(protect.values());row=dict(arm=arm,threshold=t,metrics=m,protection=protect,qualified=ok);grid.append(row)
            if ok:qualified.append(row)
        best=max(qualified,key=lambda r:(r['metrics']['pool_value_targets'],r['metrics']['pool_value_scenes'],r['metrics']['pool_delta_sum'],-r['metrics'].get('deleted_target',0)-r['metrics'].get('added_other',0),r['threshold'])) if qualified else None
        chosen[arm]=dict(threshold=best['threshold'],abstain=False,qualified=True) if best else dict(abstain=True,qualified=False)
    seal(OUT/'calibration.json',dict(complete=True,grid=grid,chosen=chosen,scenes=c['splits']['calibration']));seal(OUT/'rules_frozen.json',dict(complete=True,chosen=chosen,holdout_labels_read=False,calibration_sha256=sha(OUT/'calibration.json')));stage('AB_execution_rules_locked',chosen=chosen)

def apply():
    c=read(OUT/'contract.json');sys.addaudithook(Guard(c,'apply').hook);rules=read(OUT/'rules_frozen.json')['chosen'];outputs={}
    for s in c['scenes']:
        d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True);arrays={}
        with np.load(SOURCE/'features'/s/'data.npz') as z,np.load(OUT/'probabilities'/s/'data.npz') as pr:
            for r in read(SOURCE/'plan'/s/'candidates.json'):
                cid=r['candidate_id']
                for arm in ('A','B'):
                    rule=rules[arm];arrays[cid+'_'+arm+'_accepted']=z[cid+'_parent'].copy() if rule['abstain'] else core.repair(z[cid+'_parent'],z[cid+'_halo'],pr[cid+'_'+arm+'_probability'],z[cid+'_'+OBS+'_groups'],rule['threshold'])
        np.savez_compressed(d/'data.npz',**arrays);outputs[str(d/'data.npz')]=sha(d/'data.npz');reserve()
    seal(OUT/'predictions_frozen.json',dict(complete=True,outputs=outputs,GT_read=False,rules_sha256=sha(OUT/'rules_frozen.json')));stage('accepted_pointsets_sealed_before_holdout_GT')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','train','predict','calibrate','apply'));a=p.parse_args();globals()[a.mode]()
