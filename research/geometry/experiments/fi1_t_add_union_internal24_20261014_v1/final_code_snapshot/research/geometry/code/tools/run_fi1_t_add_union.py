#!/usr/bin/env python3
"""Frozen calibration24 internal verification of two T-only union additions."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_nt_region_identity as diag
import evaluate_fi1_nt_region_identity as met
import fi1_t_add_union_core as core
OUT=ROOT/'research/geometry/experiments/fi1_t_add_union_internal24_20261014_v1'
TOOLS=Path(__file__).resolve().parent
def read(p):return diag.read(p)
def seal(p,x):diag.seal(p,x)
def physical(s):return s.split('_')[0]
def reserve():diag.reserve()
def guard():
    c=read(OUT/'contract.json');allowed=set(c['scenes']);known=set(c['original_scene_universe'])
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        p=str(args[0])
        if any(v in p for v in ('ground_truth','/labels/','/diagnosis/','/structure_diagnosis/','/prior_diagnosis/','/evaluation/','audit_results','failure_diagnosis')):raise PermissionError('GT/results forbidden before predictions: '+p)
        s=next((s for s in known if '/'+s+'/' in p or '/'+s+'.' in p),None)
        if s and s not in allowed:raise PermissionError('outside fixed24: '+p)
    sys.addaudithook(hook)

def prepare():
    assert not OUT.exists();reserve()
    old=read(diag.AB/'contract.json');local=read(ROOT/'research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/contract.json');pilot=read(ROOT/'research/geometry/experiments/fi1_query_same_parent_pilot_20261014_v1/contract.json');previous=read(diag.OUT/'contract.json')
    scenes=old['splits']['calibration'];assert len(scenes)==len({physical(s) for s in scenes})==24
    excluded=set(local['jobs'])|set(pilot['scenes'])|set(old['splits']['holdout'])|set(old['groups']['fold0'])
    assert not {physical(s) for s in scenes}&{physical(s) for s in excluded}
    inputs={};parents={};usage=[]
    for s in scenes:
        j=old['jobs'][s];meta=read(diag.SOURCE/'plan'/s/'candidates.json');parents[s]=[r for r in meta if r['source']=='track']
        paths=[j['paths']['processed'],j['paths']['observations'],j['binding']['geometry'],j['binding']['points'],diag.SOURCE/'features'/s/'data.npz',diag.SOURCE/'plan'/s/'candidates.json',diag.SOURCE/'plan'/s/'evidence.npz']
        for p in paths:assert Path(p).is_file();inputs[str(p)]=diag.sha(p)
        usage.append(dict(scene=s,physical=physical(s),previous_split='calibration24',previous_development_use=True,group='historical90' if s in old['groups']['historical90'] else 'new110',T_parents=len(parents[s]),N_parents_generated=0,not_final_holdout=True))
    sourcefiles=[Path(__file__),TOOLS/'fi1_t_add_union_core.py',TOOLS/'verify_fi1_t_add_union.py',TOOLS/'export_fi1_t_add_union_priors.py',TOOLS/'run_fi1_t_add_union_host.sh',TOOLS/'run_fi1_nt_region_identity.py',TOOLS/'fi1_nt_region_identity_core.py',TOOLS/'evaluate_fi1_nt_region_identity.py',ROOT/'research/geometry/code/tests/test_fi1_t_add_union.py']
    OUT.mkdir(parents=True)
    c=dict(version=OUT.name,scenes=scenes,parents=parents,jobs={s:old['jobs'][s] for s in scenes},class_ids=old['class_ids'],original_scene_universe=old['groups']['development200'],inputs=inputs,GT_inputs={old['jobs'][s]['GT']:old['GT_inputs'][old['jobs'][s]['GT']] for s in scenes},code={str(p):diag.sha(p) for p in sourcefiles},prior_inputs={},checkpoint=previous['checkpoint'],checkpoint_sha256=previous['checkpoint_sha256'],frozen_SL_models=previous['frozen_SL_models'],parameters=diag.core.PARAMETERS,usage=usage,excluded_physical_scenes=sorted({physical(s) for s in excluded}),sampling='exact original calibration24, no replacement; original GT-free sampled track parents, usually4 per scene, one scene3; not full-T repair',scope='original full deduplicated N+T retained, plus at most one added union candidate per sampled T parent per arm; no N edits or T deletions; A/B pools separate',region_domain='same membership-region method, top4 fullNT neighbors within .15m/overlap, raw8NN edges<=.05m; outside parent and visible_groups>=2; all nonempty blocks no new size/query/view/geometry filters; no query boundary splitting',rules=dict(A='parent selected query mean over entire editable block >=.5',B='A AND own mean - greatest other-query mean >=.1; competitor query ID not assumed separate object'),query='fixed checkpoint XYZ P1; original full-scene softIoU top1 deterministic ID tie; no GT association; seed20261014+fixed scene index',primary='all selected whole blocks union for each T parent, no GT filtering/reselection; preserve original T; empty union no extra candidate',single_blocks='diagnostic only, not acceptance or GT-picked output',dataset_usage='original feature AB calibration24 used for prior threshold calibration; development200 seen by upstream/B26 and historical geometry fits/evaluations; independent of local-edit fit12/exposed16 and pilot but NOT whole-project blind test',predeclared_assessment=dict(min_same_parent_strict_pool_gain02_scenes=2,pool_gain_min=.02,stable_parent='original valid best IoU target >=20 hits/purity>=.6',strict_witness='same parent target; union IoU improvement>=.02 over complete original NT; added_target_fraction>=.9 and known_annotation_fraction>=.9',protection=dict(add_known_target_precision_min=.9,all_edit_unknown_fraction_max=.1,good_parent_iou_min=.9,good_loss_gt02_rate_max=.02,worst_good_loss_max=.1,stable_mean_parent_delta_min=0,drift_max=0),interpretation='gates describe safety/value, never add acceptance filters; no zero-action success; B needs lower wrong/unknown/good-parent risk and still >=2 strict gain scenes; less action alone not success'),training=False,AP_run=False,production_modified=False,SL_modified=False,new_frames=False,GT_read_for_selection=False)
    diag.verify(inputs);diag.verify(c['frozen_SL_models']);seal(OUT/'contract.json',c);seal(OUT/'scenes.json',dict(scenes=scenes,usage=usage))
    for p in sourcefiles:
        q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    met.csv_write(OUT/'data_usage.csv',usage)
    print('prepared',len(scenes),'scenes',sum(map(len,parents.values())),'T parents',flush=True)

def structures():
    guard();diag.OUT=OUT;diag.structures()

def predict():
    c=read(OUT/'contract.json');guard();diag.verify(c['code']);diag.verify(c['inputs']);diag.verify(read(OUT/'priors_frozen.json')['outputs']);diag.verify(read(OUT/'structure_frozen.json')['outputs']);outputs={};totals=[]
    for s,j in c['jobs'].items():
        d=OUT/'predictions'/s;d.mkdir(parents=True,exist_ok=True);records=[];unions=[];arrays={};geom,nt=diag.load_nt(j);ncover=np.zeros(j['binding']['point_count'],bool);tcover=ncover.copy()
        for g in geom:
            sources={a['source'] for a in diag.original_aliases(g)}
            if 'native' in sources:ncover[nt[g['geometry_key']]]=True
            if 'track' in sources:tcover[nt[g['geometry_key']]]=True
        with np.load(OUT/'structure'/s/'data.npz') as st,np.load(OUT/'priors'/s/'data.npz') as pr:
            roi=pr['point_ids'];probability=pr['P1_probability'];volume=pr['P1_full_volume']
            for meta in c['parents'][s]:
                cid=meta['candidate_id'];parent=st[cid+'_parent'];halo=st[cid+'_halo'];labels=st[cid+'_labels'];visible=st[cid+'_visible'];ix=np.searchsorted(roi,halo);assert np.array_equal(roi[ix],halo);p=probability[ix]
                link=diag.core.association(p,volume,np.flatnonzero(np.isin(halo,parent)));query=link['top'][0];selected={'A':[],'B':[]};ids={'A':[],'B':[]}
                for b in core.blocks(parent,halo,labels,visible,p,query):
                    pts=halo[b.pop('indices')];bid=f'{cid}_r{b["region"]:04d}';arrays[bid+'_block']=pts
                    row=dict(b,scene=s,candidate_id=cid,parent_geometry=meta['parent'],block_id=bid,points=len(pts),query_id=query,query_ambiguous=link['ambiguous'],query_gap=link['gap'],points_sha256=hashlib.sha256(pts.astype('<i8').tobytes()).hexdigest(),full_N_fraction=float(ncover[pts].mean()),full_T_fraction=float(tcover[pts].mean()),full_neither_fraction=float((~ncover[pts]&~tcover[pts]).mean()));records.append(row)
                    for arm in ('A','B'):
                        if b[arm]:selected[arm].append(pts);ids[arm].append(bid)
                for arm in ('A','B'):
                    result,added=core.union(parent,selected[arm]);arrays[cid+'_'+arm+'_points']=result;arrays[cid+'_'+arm+'_added']=added
                    unions.append(dict(scene=s,candidate_id=cid,source='track',parent_geometry=meta['parent'],arm=arm,query_id=query,query_ambiguous=link['ambiguous'],query_gap=link['gap'],parent_points=len(parent),added=len(added),points=len(result),blocks=ids[arm],selected_blocks=len(ids[arm]),extra_candidate=bool(len(added)),points_sha256=hashlib.sha256(result.astype('<i8').tobytes()).hexdigest(),added_sha256=hashlib.sha256(added.astype('<i8').tobytes()).hexdigest()))
                    assert np.isin(parent,result).all()
                assert np.isin(arrays[cid+'_B_added'],arrays[cid+'_A_added']).all()
        np.savez_compressed(d/'data.npz',**arrays);seal(d/'blocks.json',records);seal(d/'unions.json',unions);met.csv_write(d/'blocks.csv',records);met.csv_write(d/'unions.csv',unions)
        seal(d/'receipt.json',dict(complete=True,GT_read=False,outputs={str(p):diag.sha(p) for p in d.glob('*')},parents=len(c['parents'][s]),blocks=len(records),A_extra=sum(r['extra_candidate'] for r in unions if r['arm']=='A'),B_extra=sum(r['extra_candidate'] for r in unions if r['arm']=='B')))
        outputs.update({str(p):diag.sha(p) for p in d.glob('*')});reserve();totals.append(dict(scene=s,blocks=len(records),parents=len(c['parents'][s])));print('predicted',s,len(records),'blocks',flush=True)
    seal(OUT/'predictions_frozen.json',dict(complete=True,GT_read=False,outputs=outputs,scenes=totals,original_NT_preserved=True,N_edits=0,T_deletions=0,at_most_one_union_per_T_per_arm=True,A_B_separate=True))

def evaluate():
    c=read(OUT/'contract.json');diag.verify(read(OUT/'predictions_frozen.json')['outputs']);diag.verify(c['GT_inputs']);rows=[];blockrows=[];pools=[];witnesses=[];scenerows=[]
    for s,j in c['jobs'].items():
        gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=met.valid_targets(gt,c['class_ids']);geom,nt=diag.load_nt(j);baseline=met.pool_base(nt,gt,ids,sizes);best={a:baseline.copy() for a in ('A','B')};single={a:baseline.copy() for a in ('A','B')}
        original={v.astype('<i8').tobytes() for v in nt.values()};blocks=read(OUT/'predictions'/s/'blocks.json');unions=read(OUT/'predictions'/s/'unions.json')
        with np.load(OUT/'structure'/s/'data.npz') as st,np.load(OUT/'predictions'/s/'data.npz') as pred:
            identities={}
            for meta in c['parents'][s]:
                cid=meta['candidate_id'];parent=st[cid+'_parent'];identities[cid]=identity(parent,gt,ids,sizes,c['class_ids'])
            for u in unions:
                cid=u['candidate_id'];arm=u['arm'];parent=st[cid+'_parent'];result=pred[cid+'_'+arm+'_points'];add=pred[cid+'_'+arm+'_added'];im=identities[cid]
                row,after=met.outcome(parent,result,gt,ids,sizes,im,baseline);row.update(u);row.update(unknown_annotation_added=int((gt[add]<1000).sum()),unknown_identity_added=0 if im['stable'] else len(add),duplicate_original=bool(len(add) and result.astype('<i8').tobytes() in original),known_annotation_fraction=float((gt[add]>=1000).mean()) if len(add) else None)
                if im['stable']:
                    row.update(added_target_fraction=float((gt[add]==im['target']).mean()) if len(add) else None,known_target_precision=float((gt[add]==im['target']).sum()/max(1,(gt[add]>=1000).sum())) if len(add) else None)
                rows.append(row)
                if len(add):best[arm]=np.maximum(best[arm],after)
                for k in np.flatnonzero(after-baseline>=.02-1e-12):
                    same=bool(im['stable'] and im['target']==int(ids[k]));strict=bool(same and len(add) and row['added_target_fraction']>=.9 and row['known_annotation_fraction']>=.9)
                    witnesses.append(dict(row,pool_target=int(ids[k]),same_parent_target=same,strict_witness=strict,pool_baseline=float(baseline[k]),pool_after=float(after[k]),pool_gain=float(after[k]-baseline[k])))
            for b in blocks:
                cid=b['candidate_id'];parent=st[cid+'_parent'];pts=pred[b['block_id']+'_block'];result=np.union1d(parent,pts);row,after=met.outcome(parent,result,gt,ids,sizes,identities[cid],baseline);row.update(b);row['diagnostic_only']=True;blockrows.append(row)
                for arm in ('A','B'):
                    if b[arm]:single[arm]=np.maximum(single[arm],after)
        for mode,v in [('NT',baseline)]+[(a,best[a]) for a in ('A','B')]+[(a+'_single_diagnostic',single[a]) for a in ('A','B')]:pools+=met.pool_rows(s,mode,baseline,v,ids)
        scenerows.append(dict(scene=s,T_parents=len(c['parents'][s]),original_NT_candidates=len(nt),valid_GT_targets=len(ids),A_extra=sum(r['extra_candidate'] for r in unions if r['arm']=='A'),B_extra=sum(r['extra_candidate'] for r in unions if r['arm']=='B')))
        print('evaluated',s,len(ids),'GT targets',flush=True)
    d=OUT/'evaluation';d.mkdir(exist_ok=True)
    for name,rr in [('parents',rows),('single_blocks_diagnostic',blockrows),('pool',pools),('pool_witnesses',witnesses),('scenes',scenerows)]:met.csv_write(d/(name+'.csv'),rr)
    seal(d/'rows.json',rows);seal(d/'witnesses.json',witnesses)
    summaries={}
    for arm in ('A','B'):
        rr=[r for r in rows if r['arm']==arm];stable=[r for r in rr if r['stable']];changed=[r for r in stable if r['added']];good=[r for r in stable if r['good_parent']];ww=[w for w in witnesses if w['arm']==arm];pp=[p for p in pools if p['mode']==arm]
        target=sum(r.get('added_target',0) for r in rr);wrong=sum(r.get('added_other',0) for r in rr);unknown=sum(r['unknown_annotation_added'] if r['stable'] else r['added'] for r in rr);edits=sum(r['added'] for r in rr);strict={(w['scene'],w['pool_target']) for w in ww if w['strict_witness']}
        summaries[arm]=dict(parents=len(rr),stable_parents=len(stable),extra_candidates=sum(r['extra_candidate'] for r in rr),changed_stable=len(changed),selected_blocks=sum(r['selected_blocks'] for r in rr),added_points=edits,added_target=target,added_other=wrong,unknown_added=unknown,unannotated_added=sum(r['unknown_annotation_added'] for r in rr),unstable_added=sum(r['unknown_identity_added'] for r in rr),known_target_precision=target/(target+wrong) if target+wrong else None,unknown_fraction=unknown/edits if edits else None,stable_mean_delta=float(np.mean([r['parent_delta'] for r in stable])) if stable else None,improved=sum(r['parent_delta']>.02-1e-12 for r in stable),harmed=sum(r['parent_delta']<-.02-1e-12 for r in stable),good_parents=len(good),good_loss_gt02=sum(r['good_loss']>.02 for r in good),worst_good_loss=max((r['good_loss'] for r in good),default=0),drift=sum(r['drift'] for r in stable),pool_gain02_targets=sum(p['gain02'] for p in pp),pool_gain02_scenes=len({p['scene'] for p in pp if p['gain02']}),same_parent_pool_targets=len({(w['scene'],w['pool_target']) for w in ww if w['same_parent_target']}),strict_pool_targets=len(strict),strict_pool_scenes=len({s for s,g in strict}),strict_pool_target_ids=sorted(strict),full_GT_denominator=len(pp),pool_continuous_gain=sum(p['delta'] for p in pp),pool_mean_gain=sum(p['delta'] for p in pp)/max(1,len(pp)))
        m=summaries[arm];g=c['predeclared_assessment']['protection'];checks=dict(value=m['strict_pool_scenes']>=2,known_precision=m['known_target_precision'] is not None and m['known_target_precision']>=g['add_known_target_precision_min'],unknown=m['unknown_fraction'] is not None and m['unknown_fraction']<=g['all_edit_unknown_fraction_max'],good_rate=m['good_loss_gt02']/max(1,m['good_parents'])<=g['good_loss_gt02_rate_max'],good_tail=m['worst_good_loss']<=g['worst_good_loss_max'],stable_mean=m['stable_mean_delta'] is not None and m['stable_mean_delta']>=0,drift=m['drift']==0,nonempty=m['extra_candidates']>0);m['checks']=checks;m['passed']=all(checks.values())
    a,b=summaries['A'],summaries['B'];risk_lower=b['added_other']<a['added_other'] and b['unknown_added']<=a['unknown_added'] and b['good_loss_gt02']<=a['good_loss_gt02'] and b['worst_good_loss']<=a['worst_good_loss']
    seal(d/'summary.json',dict(arms=summaries,B_reduces_measured_risk=risk_lower,B_preserves_valuable_repair=b['strict_pool_scenes']>=2,B_supported=risk_lower and b['passed'],single_block_not_acceptance=True,whole_project_blind=False))
    seal(OUT/'evaluation_frozen.json',dict(complete=True,GT_after_predictions=True,outputs={str(p):diag.sha(p) for p in d.glob('*')}))

def identity(parent,gt,ids,sizes,classes):
    im=met.identity(parent,gt,ids,sizes)
    if not im['stable']:
        v,n=np.unique(gt[parent],return_counts=True);g=int(v[n.argmax()]);fraction=n.max()/len(parent)
        im['scope']='out_of_scope' if g>=1000 and g//1000 not in classes and fraction>=.6 else 'small_instance' if g>=1000 and g//1000 in classes and g not in ids and fraction>=.6 else 'unannotated' if (gt[parent]<1000).mean()>=.5 else 'ambiguous'
    return im

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','structures','predict','evaluate'));a=p.parse_args();globals()[a.mode]()
