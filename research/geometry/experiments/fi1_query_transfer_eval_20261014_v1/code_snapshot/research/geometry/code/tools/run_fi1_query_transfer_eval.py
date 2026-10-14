#!/usr/bin/env python3
"""Evaluation only of sealed epoch64 pilot models on the original selection16."""
from pathlib import Path
import argparse, hashlib, shutil, sys, time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_nt_region_identity as diag
import run_fi1_query_same_parent_pilot as pilot
import evaluate_fi1_nt_region_identity as met

OUT=ROOT/'research/geometry/experiments/fi1_query_transfer_eval_20261014_v1'
PILOT=pilot.OUT
MODES=(*pilot.MODES,'region_average')
def seal(p,v):diag.seal(p,v)
def read(p):return diag.read(p)
def step(name,**kw):
    print(name,diag.plain(kw),flush=True)
def guard():
    allowed=set(read(OUT/'contract.json')['scenes'])
    allscenes=read(diag.AB/'contract.json')['scenes']
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(args[0]);scene=next((s for s in allscenes if s in p),None)
        if scene and scene not in allowed:raise PermissionError('selection16 only: '+p)
        if any(v in p for v in ('ground_truth','/labels/','/diagnosis/','/structure_diagnosis/','/prior_diagnosis/','/evaluation/')):
            raise PermissionError('GT forbidden until predictions sealed: '+p)
    sys.addaudithook(hook)

def prepare():
    assert not OUT.exists();pilot.reserve()
    old=read(diag.AB/'contract.json');pc=read(PILOT/'contract.json');dc=read(diag.OUT/'contract.json')
    scenes=old['splits']['selection'];trainphys={s.split('_')[0] for s in pc['scenes']}
    assert len(scenes)==len({s.split('_')[0] for s in scenes})==16
    assert not trainphys&{s.split('_')[0] for s in scenes}
    assert not set(scenes)&set(old['splits']['holdout'])
    parents={s:read(diag.SOURCE/'plan'/s/'candidates.json') for s in scenes};assert all(len(v)==8 for v in parents.values())
    bindings={str(PILOT/'contract.json'):diag.sha(PILOT/'contract.json'),str(PILOT/'normalization.npz'):diag.sha(PILOT/'normalization.npz'),str(PILOT/'models_frozen.json'):diag.sha(PILOT/'models_frozen.json')}
    bindings.update(read(PILOT/'models_frozen.json')['outputs'])
    inputs={}
    for s in scenes:
        j=old['jobs'][s]
        paths=[j['paths']['processed'],j['paths']['observations'],j['binding']['geometry'],j['binding']['points'],diag.SOURCE/'features'/s/'data.npz',diag.SOURCE/'plan'/s/'candidates.json',diag.SOURCE/'plan'/s/'evidence.npz']
        inputs.update({str(p):diag.sha(p) for p in paths})
    tools=Path(__file__).resolve().parent
    files=[Path(__file__).resolve(),tools/'verify_fi1_query_transfer_eval.py',tools/'run_fi1_query_transfer_eval.sh',tools/'run_fi1_nt_region_identity.py',tools/'fi1_nt_region_identity_core.py',tools/'run_fi1_query_same_parent_pilot.py',tools/'fi1_candidate_local_segmenter.py',tools/'evaluate_fi1_nt_region_identity.py',ROOT/'research/geometry/code/tests/test_fi1_query_transfer_eval.py']
    OUT.mkdir(parents=True)
    # Safety criteria decide interpretation, never alter prediction rules or select a model.
    protection=dict(add_precision_min=.9,delete_precision_min=.9,target_loss_max=.01,unknown_fraction_max=.1,good_loss_gt02_rate_max=.02,worst_good_loss_max=.1,drift_max=0,source_mean_parent_delta_min=0)
    c=dict(version=OUT.name,scenes=scenes,parents=parents,jobs={s:old['jobs'][s] for s in scenes},class_ids=old['class_ids'],GT_inputs={old['jobs'][s]['GT']:old['GT_inputs'][old['jobs'][s]['GT']] for s in scenes},inputs=inputs,model_bindings=bindings,code={str(p):diag.sha(p) for p in files},checkpoint=dc['checkpoint'],checkpoint_sha256=dc['checkpoint_sha256'],prior_inputs={},frozen_SL_models=dc['frozen_SL_models'],pilot_training_scenes=pc['scenes'],physical_isolation=True,existing_usage='original feature AB selection16 repeatedly used for model selection; current pilot never trained/selected on these scenes; upstream and project exposed development data',modes=MODES,threshold=.8,region_threshold=.5,normalization=str(PILOT/'normalization.npz'),model_paths={a:str(PILOT/'models'/a/'model.pt') for a in ('A','B')},epoch=64,training=False,model_selection=False,calibration=False,AP_run=False,production_modified=False,shuffle_seeds=pilot.SHUFFLE_SEEDS,prior_export='existing frozen paired exporter, XYZ P1; P0 retained solely for hook/interface invariance; no prior retraining',region_baseline='same full NT top4 membership connected regions and parent top1 P1 mean .5 as frozen diagnostic',scope='all 8 original sampled parents per scene, 4N/4T, no GT/query filtering; full original NT is pool baseline, no all-NT repair claim',interpretation=dict(protection=protection,min_pool_gain02_scenes=2,min_B_exclusive_gain02_scenes=2,increment='B expanded full NT IoU minus NT+A >=.02 on targets in >=2 physical scenes; no training saturation absolute mean requirement',dependency='paired pool continuous gain and target set versus B_zero and three fixed shuffle rules; only supporting evidence, no pointwise significance',ambiguous='too small or conflicting scene-level tradeoffs => undetermined, do not expand'))
    diag.verify(bindings);diag.verify(inputs);diag.verify(c['frozen_SL_models']);seal(OUT/'contract.json',c)
    for p in files:
        q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    step('prepared',scenes=16,parents=128,training=False)

def inherited(mode):
    guard();diag.OUT=OUT
    getattr(diag,mode)()

def examples(s):
    c=read(OUT/'contract.json');norm=np.load(c['normalization'])
    with np.load(diag.SOURCE/'features'/s/'data.npz') as z,np.load(OUT/'priors'/s/'data.npz') as pr:
        links={r['candidate_id']:r for r in read(OUT/'associations'/s/'links.json') if r['prior']=='P1'}
        for r in c['parents'][s]:
            cid=r['candidate_id'];halo=z[cid+'_halo'];ix=np.searchsorted(pr['point_ids'],halo);assert np.array_equal(pr['point_ids'][ix],halo)
            q=links[cid]['parent']['top'][0];head=pr['P1_head_roi'][ix].astype(np.float32)
            yield dict(scene=s,candidate_id=cid,source=r['source'],base=z[cid+'_gd10_selected6_x'].astype(np.float32),head=np.clip((head-norm['mean'])/norm['std'],-10,10).astype(np.float32),query=pr['P1_probability'][ix,q],neighbors=z[cid+'_neighbors'].astype(np.int64),weights=z[cid+'_neighbor_weights'].astype(np.float32),parent=z[cid+'_parent'],halo=halo,visible=z[cid+'_gd10_selected6_groups'],query_id=q)
    norm.close()

def predict():
    import torch
    c=read(OUT/'contract.json');guard();diag.verify(c['code']);diag.verify(c['model_bindings']);diag.verify(read(OUT/'associations_frozen.json')['outputs']);torch.set_num_threads(4);assert torch.cuda.is_available()
    models={}
    for arm in ('A','B'):
        net=pilot.CandidateLocalSegmenter(193,64).cuda().eval();net.load_state_dict(torch.load(c['model_paths'][arm],weights_only=True,map_location='cuda'));models[arm]=net
    outputs={}
    for s in c['scenes']:
        d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True);arrays={};links=[]
        for e in examples(s):
            cid=e['candidate_id'];links.append(dict(candidate_id=cid,query_id=e['query_id']))
            for mode in pilot.MODES:
                with torch.inference_mode():
                    x,nn,w,parts=pilot.collate([e],mode,'cuda');p=models['A' if mode=='A' else 'B'](x,nn,w,parts).softmax(1).cpu().numpy().astype(np.float32)
                points,_=pilot.repair(e['parent'],e['halo'],p,e['visible'])
                arrays[cid+'_'+mode+'_probability']=p;arrays[cid+'_'+mode+'_points']=points
        np.savez_compressed(d/'data.npz',**arrays);seal(d/'query_ids.json',links)
        outputs.update({str(p):diag.sha(p) for p in d.glob('*')});pilot.reserve();step('prediction',scene=s)
    seal(OUT/'predictions_frozen.json',dict(complete=True,GT_read=False,models_unchanged=True,outputs=outputs))

def safety(rows,c):
    m=met.summarize_rows(rows);g=c['interpretation']['protection'];good=m['good_parents']
    # Undefined action precision is recorded as no evidence; it is not a harmful edit.
    checks=dict(add_precision=m['add_target_precision'] is None or m['add_target_precision']>=g['add_precision_min'],delete_precision=m['delete_pollution_precision'] is None or m['delete_pollution_precision']>=g['delete_precision_min'],target_loss=m['target_loss'] is None or m['target_loss']<=g['target_loss_max'],unknown=m['unknown_fraction'] is None or m['unknown_fraction']<=g['unknown_fraction_max'],good_rate=(m['good_loss_gt_02']/good if good else 0)<=g['good_loss_gt02_rate_max'],good_tail=m['worst_good_loss']<=g['worst_good_loss_max'],drift=m['drift']<=g['drift_max'],source_means=all(np.mean([r['parent_delta'] for r in rows if r['source']==s and r['stable']])>=g['source_mean_parent_delta_min'] for s in ('native','track')))
    return dict(passed=all(checks.values()),checks=checks)

def interpret(c,rows,pool):
    mm={m:met.pool_summary([r for r in pool if r['mode']==m and r['source']=='all']) for m in MODES}
    protections={m:safety([r for r in rows if r['mode']==m],c) for m in ('A','B')}
    pp={(r['scene'],r['gt_id'],r['mode']):r for r in pool if r['source']=='all'}
    exclusive=[]
    for (s,g,m),b in pp.items():
        if m=='B':
            a=pp[s,g,'A'];delta=b['expanded']-a['expanded']
            if delta>=.02-1e-12:exclusive.append(dict(scene=s,gt_id=g,B_minus_A_pool_delta=delta,A=a['expanded'],B=b['expanded']))
    evidence={m:protections[m]['passed'] and mm[m]['pool_value_scenes']>=2 and len({r['scene'] for r in pool if r['mode']==m and r['source']=='all' and r['gain02']})>=2 for m in ('A','B')}
    if not evidence['A'] and not evidence['B']:action='pause_current_repairer_expansion'
    elif evidence['A'] and not (evidence['B'] and len({r['scene'] for r in exclusive})>=2):action='prefer_A_for_further_research_no_query_increment_support'
    elif evidence['B'] and len({r['scene'] for r in exclusive})>=2:action='query_has_cross_scene_evidence_new_protocol_required'
    else:action='undetermined_do_not_expand'
    return dict(action=action,protection=protections,multi_scene_safe_pool_evidence=evidence,B_exclusive_gain02=exclusive,B_exclusive_scenes=len({r['scene'] for r in exclusive}),pool=mm,training=False,threshold_changed=False,model_selected=False)

def evaluate():
    c=read(OUT/'contract.json');assert (OUT/'predictions_frozen.json').exists();diag.verify(read(OUT/'predictions_frozen.json')['outputs']);diag.verify(c['GT_inputs']);rows=[];pool=[];opportunities=[];distributions=[]
    train=pilot.load();trainbase=np.concatenate([e['base'] for e in train.values()]);lo,hi=trainbase.min(0),trainbase.max(0)
    norm=np.load(c['normalization']);training_rows=list(__import__('csv').DictReader((PILOT/'evaluation/parents.csv').open()))
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=met.valid_targets(gt,c['class_ids']);_,nt=diag.load_nt(c['jobs'][s]);baseline=met.pool_base(nt,gt,ids,sizes)
        best={(m,source):baseline.copy() for m in ('original_NT',*MODES) for source in ('all','native','track')};seen={m:set() for m in MODES};original={p.astype('<i8').tobytes() for p in nt.values()}
        with np.load(OUT/'predictions'/s/'data.npz') as pr,np.load(OUT/'associations'/s/'data.npz') as reg,np.load(OUT/'priors'/s/'data.npz') as pri:
            for e in examples(s):
                cid=e['candidate_id'];parent=e['parent'];halo=e['halo'];member=np.isin(halo,parent);visible=e['visible']>=2;im=met.identity(parent,gt,ids,sizes);g=im['target'];y=np.full(len(halo),2,np.int64)
                if im['stable']:y[gt[halo]>=1000]=1;y[gt[halo]==g]=0
                ix=np.searchsorted(pri['point_ids'],halo);rawhead=pri['P1_head_roi'][ix].astype(np.float32);scaled=(rawhead-norm['mean'])/norm['std']
                distributions.append(dict(scene=s,candidate_id=cid,source=e['source'],stable=im['stable'],scope=im['scope'],parent_iou=im['parent_iou'],parent_points=len(parent),halo_points=len(halo),visible_fraction=float(visible.mean()),head_clipped_fraction=float((np.abs(scaled)>10).mean()),base_outside_training_minmax_fraction=float(((e['base']<lo)|(e['base']>hi)).mean()),query_mean=float(e['query'].mean()),query_parent_mean=float(e['query'][member].mean()),query_id=e['query_id']))
                feasible=parent.copy();upper=parent.copy()
                if im['stable']:
                    feasible=halo[np.where(visible,np.where(y==2,member,y==0),member)];upper=halo[np.where(visible,gt[halo]==g,member)]
                    if len(feasible)<100:feasible=parent.copy()
                    if len(upper)<100:upper=parent.copy() # not a tight upper bound in this exceptional case
                fi=met.ious(feasible,gt,ids,sizes);ui=met.ious(upper,gt,ids,sizes)
                for mode in MODES:
                    p=reg[cid+'_P1_region'] if mode=='region_average' else pr[cid+'_'+mode+'_points'];row,after=met.outcome(parent,p,gt,ids,sizes,im,baseline)
                    blob=p.astype('<i8').tobytes();row.update(scene=s,candidate_id=cid,source=e['source'],mode=mode,duplicate_original=row['changed'] and blob in original,duplicate_generated=row['changed'] and blob in seen[mode]);seen[mode].add(blob)
                    rows.append(row);best[mode,'all']=np.maximum(best[mode,'all'],after);best[mode,e['source']]=np.maximum(best[mode,e['source']],after)
                    if im['stable']:
                        idx=im['index'];opportunities.append(dict(scene=s,candidate_id=cid,source=e['source'],mode=mode,target=g,parent_iou=im['parent_iou'],repaired_iou=float(after[idx]),known_feasible_iou=float(fi[idx]),generous_feasible_iou=float(ui[idx]),known_remaining=float(fi[idx]-after[idx]),addition_opportunity=int((~member&visible&(y==0)).sum()),deletion_opportunity=int((member&visible&(y==1)).sum()),addition_recall=row['added_target']/max(1,int((~member&visible&(y==0)).sum())),deletion_recall=row['deleted_other']/max(1,int((member&visible&(y==1)).sum())),upper_min100_exception=bool(len(halo[np.where(visible,gt[halo]==g,member)])<100)))
        for (mode,source),expanded in best.items():
            rr=met.pool_rows(s,mode,baseline,expanded,ids)
            for r in rr:r['source']=source
            pool+=rr
        step('postseal_evaluation',scene=s)
    summary={m:dict(**met.summarize_rows([r for r in rows if r['mode']==m]),**met.pool_summary([r for r in pool if r['mode']==m and r['source']=='all'])) for m in MODES}
    grouped={f'{m}/{source}/{scope}':met.summarize_rows([r for r in rows if r['mode']==m and r['source']==source and (scope=='all' or r['scope']==scope)]) for m in MODES for source in ('native','track') for scope in ('all','stable','ambiguous','unannotated','out_of_scope')}
    scene_summary=[]
    for s in c['scenes']:
        for m in MODES:
            scene_summary.append(dict(scene=s,mode=m,**met.summarize_rows([r for r in rows if r['scene']==s and r['mode']==m]),**met.pool_summary([r for r in pool if r['scene']==s and r['mode']==m and r['source']=='all'])))
    for m in ('A','B'):
        summary[m]['training_reference']=read(PILOT/'evaluation/summary.json')['modes'][m]
    decision=interpret(c,rows,pool);d=OUT/'evaluation';d.mkdir(exist_ok=True)
    for name,rr in (('parents',rows),('pool',pool),('opportunities',opportunities),('input_distribution',distributions),('scenes',scene_summary)):met.csv_write(d/(name+'.csv'),rr)
    seal(d/'summary.json',dict(modes=summary,grouped=grouped,source_pool={f'{m}/{source}':met.pool_summary([r for r in pool if r['mode']==m and r['source']==source]) for m in MODES for source in ('native','track')},decision=decision))
    seal(OUT/'decision.json',decision);step('evaluation_complete',decision=decision)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','structures','export','associate','predict','evaluate'));a=p.parse_args()
    if a.mode in ('structures','export','associate'):inherited(a.mode)
    else:globals()[a.mode]()
