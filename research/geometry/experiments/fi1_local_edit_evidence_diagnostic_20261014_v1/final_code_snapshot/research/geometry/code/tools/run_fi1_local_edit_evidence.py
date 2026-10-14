#!/usr/bin/env python3
"""Generate/freeze GT-free edits and evidence, then post-hoc GT diagnostics."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,sys,time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import fi1_local_edit_evidence_core as core
import run_fi1_nt_region_identity as diag
import evaluate_fi1_nt_region_identity as met
OUT=ROOT/'research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1'
SOURCES=dict(fit12=diag.OUT,exposed16=ROOT/'research/geometry/experiments/fi1_query_transfer_eval_20261014_v1')
def seal(p,v):diag.seal(p,v)
def read(p):return diag.read(p)
def reserve():
    for p in (ROOT,diag.AB.resolve()):assert shutil.disk_usage(p).free>=50*2**30
def guard():
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        name=str(args[0])
        if any(w in name for w in ('ground_truth','/labels/','/diagnosis/','/structure_diagnosis/','/prior_diagnosis/','/evaluation/','audit_results','failure_diagnosis')):raise PermissionError('No GT/results when generating evidence: '+name)
    sys.addaudithook(hook)
def prepare():
    assert not OUT.exists();reserve();jobs={};cohorts={};inputs={};gt={};classes=None
    for name,root in SOURCES.items():
        c=read(root/'contract.json');cohorts[name]=c['scenes'];classes=c['class_ids']
        for s in c['scenes']:
            j=c['jobs'][s];jobs[s]=dict(j,cohort=name,asset_root=str(root))
            gt[j['GT']]=c['GT_inputs'][j['GT']]
            for key in ('geometry','points'):p=j['binding'][key];inputs[p]=diag.sha(p)
            for p in [j['paths']['processed'],j['paths']['observations'],diag.SOURCE/'plan'/s/'evidence.npz',root/'structure'/s/'data.npz',root/'structure'/s/'candidates.json',root/'priors'/s/'data.npz',root/'associations'/s/'links.json',root/'associations'/s/'data.npz']:
                inputs[str(p)]=diag.sha(p)
    assert not set(cohorts['fit12'])&set(cohorts['exposed16'])
    files=[Path(__file__).resolve(),Path(__file__).with_name('fi1_local_edit_evidence_core.py'),Path(__file__).with_name('verify_fi1_local_edit_evidence.py'),ROOT/'research/geometry/code/tests/test_fi1_local_edit_evidence.py']
    OUT.mkdir(parents=True)
    c=dict(version=OUT.name,cohorts=cohorts,jobs=jobs,class_ids=classes,inputs=inputs,GT_inputs=gt,code={str(p):diag.sha(p) for p in files},proposal='all nonempty visible>=2 member regions; add outside parent/delete inside; no minsize filter; also same-region connected query .5/margin0 state pieces when region state mixed; neither N nor T support retained',query='fixed P1 GT-free full-scene softIoU parent top1, competitor greatest region-piece mean among other query IDs; query IDs not distinct GT identities',views='original selected <=6 frames, same depth visibility and frozen automatic masks; delete reference excludes edit; coverage>=.5 defines descriptive common/conflict frames, no GT',geometry='piece bbox/size, nearest retained parent distance and original region contact graph; no new frames',screens='all; query direction .5; margin .1; confident .8/.2 and contrast; >=2 common(add)/conflict(delete) frames; contrast AND views; descriptive bins only, no learned/executed rules',interpretation='Report per cohort/N-T/add-delete/proposal family AUROC and scene-bootstrap, empirical correctness and harmful edits and pooled NT value. Evidence screens are descriptive on exposed data, not independent rule validation; no GT chooses deployed actions.',good='parent IoU>=.9; labels from same original valid GT parent match >=20 hits/purity>=.6; class scope uses full contract classes',statistical='gain>1e-9 vs harm<-1e-9, neutral separate; point pure90 and known90 separately; proposal weighted and macro physical-scene AUROC with200 scene-bootstrap seed20261014; no points as independent samples',training=False,AP_run=False,production_modified=False,frozen_SL_models=read(diag.OUT/'contract.json')['frozen_SL_models'])
    seal(OUT/'contract.json',c)
    for p in files:q=OUT/'code_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    print('prepared',len(jobs),'scenes',flush=True)

def generate():
    c=read(OUT/'contract.json');guard();diag.verify(c['code']);diag.verify(c['inputs']);outputs={}
    for s,j in c['jobs'].items():
        root=Path(j['asset_root']);d=OUT/'proposals'/s;d.mkdir(parents=True,exist_ok=True)
        xyz=diag.core.safe_input(np.load(j['paths']['processed'],mmap_mode='r'))[:,:3];metas=read(root/'structure'/s/'candidates.json');links={r['candidate_id']:r for r in read(root/'associations'/s/'links.json') if r['prior']=='P1'}
        from tools.fi1_r0_r1_protocol import original_aliases
        geometry,nt=diag.load_nt(j);ncover=np.zeros(len(xyz),bool);tcover=ncover.copy()
        for geom in geometry:
            sources={a['source'] for a in original_aliases(geom)}
            if 'native' in sources:ncover[nt[geom['geometry_key']]]=True
            if 'track' in sources:tcover[nt[geom['geometry_key']]]=True
        obs=[json.loads(l) for l in Path(j['paths']['observations']).read_text().splitlines()];bankroot=Path(obs[0]['point_indices_path']).parent.parent;banks={};bank_bindings={}
        for f in sorted({f for r in metas for f in r['views'][diag.OBS]['frames']}):
            path=bankroot/f'frame{f:06d}'/'observations.json';bank_bindings[str(path)]=diag.sha(path);banks[f]=[]
            for r in read(path):
                path=Path(r['point_indices_path']);bank_bindings[str(path)]=diag.sha(path)
                with np.load(path) as z:banks[f].append(z['point_indices'].copy())
        records=[];frames=[];arrays={}
        with np.load(root/'structure'/s/'data.npz') as st,np.load(root/'priors'/s/'data.npz') as pr,np.load(diag.SOURCE/'features'/s/'data.npz') as base,np.load(diag.SOURCE/'plan'/s/'evidence.npz') as ev:
            fullvisible=np.unpackbits(ev['reliable'],axis=1,bitorder='little')[:,:len(ev['roi'])].astype(bool);frameidx={int(f):i for i,f in enumerate(ev['frames'])}
            for r in metas:
                cid=r['candidate_id'];halo=st[cid+'_halo'];parent=st[cid+'_parent'];member=np.isin(halo,parent);groups=st[cid+'_visible'];ix=np.searchsorted(pr['point_ids'],halo);assert np.array_equal(pr['point_ids'][ix],halo);p=pr['P1_probability'][ix];q=links[cid]['parent']['top'][0];labels=st[cid+'_labels'];nn=base[cid+'_neighbors'];membership=st[cid+'_membership'];localxyz=xyz[halo];fm={}
                own=np.flatnonzero(member)
                for f in r['views'][diag.OBS]['frames']:
                    vis=fullvisible[frameidx[f],np.searchsorted(ev['roi'],halo)]
                    masks=np.column_stack([np.isin(halo,b) for b in banks[f]]) if banks[f] else np.zeros((len(halo),0),bool)
                    fm[f]=(vis,masks)
                for region,family,part,edit in core.pieces(labels,member,groups,p,q,nn,localxyz):
                    action='delete' if member[edit[0]] else 'add';pid=f'{cid}_r{region:04d}_{family}_{part:04d}';reference=np.setdiff1d(own,edit) if action=='delete' else own
                    pointset=halo[edit];arrays[pid]=pointset
                    mv=membership[edit[0]];ns=bool(member[edit[0]] and r['source']=='native');ts=bool(member[edit[0]] and r['source']=='track')
                    for k,nb in enumerate(r['neighbors']):
                        if mv[k+1]:ns|='native' in nb['sources'];ts|='track' in nb['sources']
                    row=dict(scene=s,cohort=j['cohort'],candidate_id=cid,source=r['source'],proposal_id=pid,region=region,family=family,part=part,action=action,points=len(edit),full_region_points=int((labels==region).sum()),parent_points=len(parent),retained_reference_points=len(reference),editable_region_fraction=len(edit)/int((labels==region).sum()),native_support=ns,track_support=ts,support='both' if ns and ts else 'N_only' if ns else 'T_only' if ts else 'neither',member_vector=','.join(map(str,mv.astype(int))),point_ids_sha256=hashlib.sha256(pointset.astype('<i8').tobytes()).hexdigest(),query_id=q,parent_query_ambiguous=links[cid]['parent']['ambiguous'],parent_query_gap=links[cid]['parent']['gap'],omitted_neighbor_count=len(r['omitted']),minimum100_possible=len(parent)-len(edit)>=100 if action=='delete' else True,**core.query_evidence(p[edit],q))
                    row.update(full_N_fraction=float(ncover[pointset].mean()),full_T_fraction=float(tcover[pointset].mean()),full_both_fraction=float((ncover[pointset]&tcover[pointset]).mean()),full_neither_fraction=float((~ncover[pointset]&~tcover[pointset]).mean()))
                    box=np.ptp(localxyz[edit],axis=0);row.update(bbox_x=float(box[0]),bbox_y=float(box[1]),bbox_z=float(box[2]),visible_group_mean=float(groups[edit].mean()))
                    if len(reference):
                        distances=cKDTree(localxyz[reference]).query(localxyz[edit],k=1)[0];row.update(parent_min_distance=float(distances.min()),parent_mean_distance=float(distances.mean()),parent_contact05_fraction=float((distances<=.05).mean()))
                    else:row.update(parent_min_distance=None,parent_mean_distance=None,parent_contact05_fraction=None)
                    fe=[]
                    for f,(vis,masks) in fm.items():
                        fr=core.frame_evidence(edit,reference,vis,masks);fr.update(scene=s,proposal_id=pid,frame=f);frames.append(fr);fe.append(fr)
                    row.update(selected_frames=len(fe),valid_frames=sum(not f['missing'] for f in fe),missing_frames=sum(f['missing'] for f in fe),common_frames=sum(f['common'] for f in fe),conflict_frames=sum(f['conflict'] for f in fe),bestmask_common_frames=sum(f['common_best_mask'] for f in fe))
                    row.update({'screen_'+k:v for k,v in core.screens(row).items()});records.append(row)
        np.savez_compressed(d/'edits.npz',**arrays);seal(d/'evidence.json',records);met.csv_write(d/'evidence.csv',records);met.csv_write(d/'frames.csv',frames)
        oo={str(path):diag.sha(path) for path in d.glob('*')};seal(d/'receipt.json',dict(complete=True,GT_read=False,outputs=oo,mask_bank_inputs=bank_bindings,proposals=len(records)));outputs.update(oo);outputs[str(d/'receipt.json')]=diag.sha(d/'receipt.json');reserve();print('generated',s,len(records),flush=True)
    seal(OUT/'proposals_frozen.json',dict(complete=True,GT_read=False,outputs=outputs))

def identity(parent,gt,ids,sizes,classes):
    im=met.identity(parent,gt,ids,sizes)
    # Scope classification uses full class list, preserving small-instance distinction.
    if not im['stable']:
        v,n=np.unique(gt[parent],return_counts=True);g=int(v[n.argmax()]);frac=n.max()/len(parent)
        im['scope']='out_of_scope' if g>=1000 and g//1000 not in classes and frac>=.6 else 'small_instance' if g>=1000 and g//1000 in classes and g not in ids and frac>=.6 else 'unannotated' if (gt[parent]<1000).mean()>=.5 else 'ambiguous'
    return im

def evaluate():
    c=read(OUT/'contract.json');diag.verify(read(OUT/'proposals_frozen.json')['outputs']);diag.verify(c['GT_inputs']);rows=[];pool=[];unionrows=[];baselines=[]
    for s,j in c['jobs'].items():
        root=Path(j['asset_root']);gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=met.valid_targets(gt,c['class_ids']);_,nt=diag.load_nt(j);baseline=met.pool_base(nt,gt,ids,sizes);evidence=read(OUT/'proposals'/s/'evidence.json')
        best={};unions={};parentmeta={};pointsets={};original={p.astype('<i8').tobytes() for p in nt.values()}
        with np.load(OUT/'proposals'/s/'edits.npz') as edits,np.load(root/'structure'/s/'data.npz') as st:
            for cid in sorted({r['candidate_id'] for r in evidence}):
                parent=st[cid+'_parent'];parentmeta[cid]=identity(parent,gt,ids,sizes,c['class_ids']);pointsets[cid]=parent
            for e in evidence:
                cid=e['candidate_id'];parent=pointsets[cid];im=parentmeta[cid];points=edits[e['proposal_id']];proposed=np.union1d(parent,points) if e['action']=='add' else np.setdiff1d(parent,points);fallback=len(proposed)<100;p=parent.copy() if fallback else proposed
                row,after=met.outcome(parent,p,gt,ids,sizes,im,baseline);row.update(e,minimum100_fallback=fallback,duplicate_original=bool(row['changed'] and p.astype('<i8').tobytes() in original))
                if im['stable']:
                    target=(gt[points]==im['target']);other=(gt[points]>=1000)&~target;unknown=(gt[points]<1000);correct=other if e['action']=='delete' else target
                    row.update(affected_correct=int(correct.sum()),affected_wrong=int((target if e['action']=='delete' else other).sum()),affected_unknown=int(unknown.sum()),known_fraction=float((~unknown).mean()),correct_known_fraction=float(correct.sum()/max(1,(~unknown).sum())),pure90=bool(correct.mean()>=.9),fully_known_correct=bool(correct.all()),net_gain=row['parent_delta']>1e-9,net_harm=row['parent_delta']<-1e-9,neutral=abs(row['parent_delta'])<=1e-9)
                rows.append(row)
                for screen,keep in core.screens(e).items():
                    if not keep:continue
                    mode=f'{e["family"]}/{e["source"]}/{e["action"]}/{screen}';key=mode
                    best.setdefault(key,baseline.copy());best[key]=np.maximum(best[key],after)
                    uk=(cid,e['family'],screen,e['action']);unions.setdefault(uk,set()).update(map(int,points))
            for (cid,family,screen,action),pts in unions.items():
                parent=pointsets[cid];edit=np.array(sorted(pts),np.int64);p=np.union1d(parent,edit) if action=='add' else np.setdiff1d(parent,edit)
                if len(p)<100:p=parent.copy()
                row,after=met.outcome(parent,p,gt,ids,sizes,parentmeta[cid],baseline);source=next(r['source'] for r in evidence if r['candidate_id']==cid);mode=f'{family}/{source}/{action}/{screen}/union';row.update(scene=s,cohort=j['cohort'],candidate_id=cid,source=source,family=family,screen=screen,action=action,mode=mode);unionrows.append(row);best.setdefault(mode,baseline.copy());best[mode]=np.maximum(best[mode],after)
        for mode,expanded in best.items():
            rr=met.pool_rows(s,mode,baseline,expanded,ids)
            for r in rr:r['cohort']=j['cohort']
            pool+=rr
        # Existing fixed region-average, A/B records remain context, not retuned baselines.
        for cid,im in parentmeta.items():baselines.append(dict(scene=s,cohort=j['cohort'],candidate_id=cid,**im))
        print('evaluated',s,len(evidence),flush=True)
    d=OUT/'evaluation';d.mkdir(exist_ok=True)
    for name,rr in (('proposals',rows),('pool',pool),('diagnostic_unions',unionrows),('parents',baselines)):met.csv_write(d/(name+'.csv'),rr)
    seal(d/'rows.json',rows);seal(d/'unions.json',unionrows)
    seal(OUT/'evaluation_frozen.json',dict(complete=True,outputs={str(p):diag.sha(p) for p in d.glob('*')},GT_after_proposals=True))

def auc(y,score):
    y=np.asarray(y,bool);score=np.asarray(score,float)
    if not y.any() or y.all():return None
    ranks=rankdata(score);n=y.sum();return float((ranks[y].sum()-n*(n+1)/2)/(n*(len(y)-n)))
def analysis():
    c=read(OUT/'contract.json');rows=read(OUT/'evaluation/rows.json');unions=read(OUT/'evaluation/unions.json');pool=list(csv.DictReader((OUT/'evaluation/pool.csv').open()));comparisons=[];bins=[];summary={}
    features=('query_direction','query_contrast','query_std','query_entropy','competitor_winner_fraction','views_direction','valid_frames','geometry_contact','region_size')
    for cohort in c['cohorts']:
      for source in ('native','track'):
       for action in ('add','delete'):
        for family in ('region','query_boundary'):
          rr=[r for r in rows if r['cohort']==cohort and r['source']==source and r['action']==action and r['family']==family];known=[r for r in rr if r['stable']];binary=[r for r in known if not r['minimum100_fallback'] and not r['neutral']]
          key=f'{cohort}/{source}/{action}/{family}';summary[key]=dict(proposals=len(rr),stable=len(known),unstable=len(rr)-len(known),gain=sum(r['net_gain'] for r in known),harm=sum(r['net_harm'] for r in known),pure90=sum(r['pure90'] for r in known),affected_unknown=sum(r.get('affected_unknown',r['points']) for r in rr),parents=len({r['candidate_id']+r['scene'] for r in rr}),scenes=len({r['scene'] for r in rr}))
          def value(r,f):
            return {'query_direction':r['query_mean'] if action=='add' else 1-r['query_mean'],'query_contrast':r['query_margin_mean'] if action=='add' else -r['query_margin_mean'],'query_std':-r['query_std'],'query_entropy':-r['query_entropy'],'competitor_winner_fraction':-r['competitor_winner_fraction'] if action=='add' else r['competitor_winner_fraction'],'views_direction':r['common_frames']-r['conflict_frames'] if action=='add' else r['conflict_frames']-r['common_frames'],'valid_frames':r['valid_frames'],'geometry_contact':r['parent_contact05_fraction'] if r['parent_contact05_fraction'] is not None else 0,'region_size':np.log1p(r['points'])}[f]
          for f in features:
            ys=[r['net_gain'] for r in binary];scores=[value(r,f) for r in binary];aa=auc(ys,scores);scene_aucs=[auc([r['net_gain'] for r in binary if r['scene']==s],[value(r,f) for r in binary if r['scene']==s]) for s in sorted({r['scene'] for r in binary})];scene_aucs=[a for a in scene_aucs if a is not None]
            bs=[];scenes=sorted({r['scene'] for r in binary});rng=np.random.default_rng(20261014)
            for _ in range(200):
                if not scenes:break
                selected=rng.choice(scenes,len(scenes));br=[r for s in selected for r in binary if r['scene']==s];a=auc([r['net_gain'] for r in br],[value(r,f) for r in br]);
                if a is not None:bs.append(a)
            comparisons.append(dict(group=key,feature=f,gain=sum(ys),harm=len(ys)-sum(ys),auc=aa,macro_scene_auc=float(np.mean(scene_aucs)) if scene_aucs else None,scene_auc_supported=len(scene_aucs),bootstrap_p025=float(np.quantile(bs,.025)) if bs else None,bootstrap_p975=float(np.quantile(bs,.975)) if bs else None,good_mean=float(np.mean([v for y,v in zip(ys,scores) if y])) if any(ys) else None,bad_mean=float(np.mean([v for y,v in zip(ys,scores) if not y])) if ys and not all(ys) else None))
          for screen in core.screens(dict(action=action,query_mean=0,query_margin_mean=0,competitor_mean=0,common_frames=0,conflict_frames=0)):
            br=[r for r in rr if r['screen_'+screen]];sr=[r for r in br if r['stable']];mode=f'{family}/{source}/{action}/{screen}';pr=[r for r in pool if r['cohort']==cohort and r['mode']==mode];ur=[r for r in unions if r['cohort']==cohort and r['source']==source and r['action']==action and r['family']==family and r['screen']==screen];up=[r for r in pool if r['cohort']==cohort and r['mode']==mode+'/union']
            bins.append(dict(group=key,screen=screen,proposals=len(br),stable=len(sr),gain=sum(r['net_gain'] for r in sr),harm=sum(r['net_harm'] for r in sr),gain_parent_scenes=len({r['scene'] for r in sr if r['net_gain']}),correct_points=sum(r.get('affected_correct',0) for r in br),wrong_points=sum(r.get('affected_wrong',0) for r in br),unknown_points=sum(r.get('affected_unknown',r['points']) for r in br),pool_gain02=sum(r['gain02']=='True' for r in pr),pool_gain02_scenes=len({r['scene'] for r in pr if r['gain02']=='True'}),union_pool_gain02=sum(r['gain02']=='True' for r in up),union_pool_gain02_scenes=len({r['scene'] for r in up if r['gain02']=='True'}),union_good_loss_gt02=sum(r.get('good_loss',0)>.02 for r in ur),union_worst_good_loss=max((r.get('good_loss',0) for r in ur),default=0)))
    met.csv_write(OUT/'analysis/feature_comparison.csv',comparisons);met.csv_write(OUT/'analysis/evidence_bins.csv',bins);seal(OUT/'analysis/summary.json',dict(groups=summary,comparisons=comparisons,bins=bins,no_fitted_rules=True,no_independent_acceptance=True))
    print('analyzed',len(rows),'proposals',len(comparisons),'comparisons',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','generate','evaluate','analysis'));a=p.parse_args();globals()[a.mode]()
