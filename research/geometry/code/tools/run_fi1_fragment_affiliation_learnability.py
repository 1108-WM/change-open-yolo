#!/usr/bin/env python3
"""Five-fold learnability test for complementary-fragment affiliation.

The retrieval and SL scorer are frozen.  This experiment compares a fixed
geometry/mask summary with the same summary augmented by region appearance,
surface and local-shape statistics.  It reports held-out affiliation metrics
and the geometry value/pollution of accepted unions; it never appends output.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, shutil, sys, time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from tools import run_fi1_complementary_fragment_diagnostic as diag
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha

OUT=ROOT/'.local/fi1_fragment_affiliation_learnability_development200_20261011_v1'
SOURCE=ROOT/'.local/fi1_complementary_fragment_retrieval_identity_development200_20261011_v1'

def stage(name,**kw):
    write_json(OUT/'execution.json',dict(status='running',stage=name,updated=time.time(),**kw));print(name,kw,flush=True)

def prepare():
    assert not OUT.exists() and SOURCE.exists() and diag.read(SOURCE/'independent_verified.json')['complete']
    c=diag.read(SOURCE/'contract.json'); files=[Path(__file__).resolve(),ROOT/'tools/run_fi1_complementary_fragment_diagnostic.py',ROOT/'tools/fi1_complementary_fragment_evidence.py',ROOT/'tests/test_fi1_complementary_diagnostic_audit.py']
    files += [ROOT/'tools/fi1_fragment_region_visual_evidence.py',ROOT/'tools/ncs5_proxyclip_core.py',ROOT/'tools/run_fi1_fragment_affiliation_learnability.sh',ROOT/'tests/test_fi1_fragment_region_evidence.py']
    checkpoint=ROOT/'pretrained/checkpoints/dinov2_vits14_pretrain.pth'
    contract=dict(version=OUT.name,source_root=str(SOURCE),scenes=c['scenes'],folds=c['folds'],groups=c['groups'],thresholds=c['thresholds'],parameters=c['parameters'],code={str(p):sha(p) for p in files},source_files={str(SOURCE/q):sha(SOURCE/q) for q in ('contract.json','features_frozen.json','labels_frozen.json','summary.json','independent_verified.json')},model_checkpoint=str(checkpoint),model_sha256=sha(checkpoint),
        retrieval='Use only rows with combined retrieval from frozen GT-free feature seal; no threshold change.',
        summary_features='pair IoU, exclusive sizes, boundary contacts/colour/normal, witness group counts and projection gaps.',
        visual_features='Frozen DINOv2-S14 full-frame dense tokens pooled over projected visible exclusive regions, at most3 distinct posegroups chosen withoutGT by jointvisiblefraction. Same-frame and crossview similarities, surface RGB/normal/centered-shape summaries, actual measureddepth pathjump/missing and region occlusion residual. Full-frame token context may include neighbors; densepooling is not new independent instance evidence.',
        labels='same_instance vs different_instances from sealed post-GT identities; mixed_or_unreliable excluded, never relabeled negative.',
        model='same fixed HGBClassifier(max_iter=200,max_leaf_nodes=15,max_depth=4,min_samples_leaf=20,learning_rate=.05,l2=1,early_stopping=False,seed20261011). Equal total scene weight, natural class prevalence, no balancing or calibration scan.',
        acceptance='Report train and fold-out ROC-AUC and pair average precision (not instance AP), fixed thresholds .5 and .9 declared before labels; .9 is conservative action diagnostic, not90% safety guarantee. Unknown never counted as negative; unknown predictedscore recorded and actions diagnostics include it. Gate: augmented pooled pairAP>A baseline, at least3fold AP improves, reliable action precision>=.9 with>=20 accepted reliable pairs,>=2new075NTtargets, reliable additions majority target and <=.5pollution ratio. No output append this round.',
        region_view_count=3,depth_path_samples=32,depth_path_count=8,DINO_resolution=[336,448],excluded=['dev55','val312','reserve100'],
        new_training=True,new_AP=False,SL_modified=False,production_modified=False)
    OUT.mkdir();freeze_json(OUT/'contract.json',contract)
    for p in files:
        d=OUT/'code_snapshot'/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
    stage('prepared')

def _sample(x,limit=256):
    x=np.asarray(x);return x[np.linspace(0,len(x)-1,min(len(x),limit),dtype=int)] if len(x)>limit else x

def _stats(raw,idx):
    x=raw[idx];xyz=x[:,:3];rgb=np.clip(x[:,3:6]/255.,0,1);n=x[:,6:9]
    nm=np.linalg.norm(n,axis=1,keepdims=True);nn=np.divide(n,nm,out=np.zeros_like(n),where=nm>1e-12)
    # fixed 8-bin luminance/color evidence, independent of semantic labels.
    hist=np.concatenate([np.histogram(rgb[:,j],bins=8,range=(0,1),density=False)[0] for j in range(3)]).astype(float);hist/=max(1,len(x))
    center=xyz.mean(0);return np.r_[rgb.mean(0),rgb.std(0),nn.mean(0),nn.std(0),xyz.std(0),xyz.min(0)-center,xyz.max(0)-center,hist]

def _row_feature(raw,points,row):
    a=np.asarray(points[row['left']],dtype=np.int64);b=np.asarray(points[row['right']],dtype=np.int64);common=np.intersect1d(a,b,assume_unique=True);aa=np.setdiff1d(a,common,assume_unique=True);bb=np.setdiff1d(b,common,assume_unique=True)
    bd=row['boundary'];v=row['views'] or {};w=v.get('witnesses',[])
    gaps=[q['projection_bbox_gap'] for q in w if q.get('projection_bbox_gap') is not None]
    summary=np.asarray([row['pair_iou'],np.log1p(len(a)),np.log1p(len(b)),np.log1p(len(aa)),np.log1p(len(bb)),len(common)/max(1,len(np.union1d(a,b))),bd.get('contacts10',0),bd.get('contacts05',0),bd.get('distance_min') or .2,bd.get('color_difference') if bd.get('color_difference') is not None else .5,bd.get('normal_difference') if bd.get('normal_difference') is not None else .5,bd.get('normal_valid_fraction',0),len(v.get('eligible_groups',[])),len(v.get('joint_groups',[])),len(v.get('compact_groups',[])),len(v.get('separated_groups',[])),len(v.get('projection_groups',[])),len(v.get('depth_groups',[])),len(gaps),np.median(gaps) if gaps else 99,np.min(gaps) if gaps else 99],dtype=float)
    sa,sb=_stats(raw,_sample(aa)),_stats(raw,_sample(bb));visual=np.r_[(sa+sb)/2,np.abs(sa-sb)]
    # Summary endpoints also symmetric; no direction or scene coordinate shortcuts.
    summary[[1,2]]=sorted(summary[[1,2]]);summary[[3,4]]=sorted(summary[[3,4]])
    return summary,visual

def features():
    from tools.fi1_fragment_region_visual_evidence import visual_scene,load_model
    c=diag.read(OUT/'contract.json');source=Path(c['source_root']);rows=[];summary=[];visual=[];counts=Counter();parent=diag.read(source/'contract.json')
    # This process must not open sealed GT labels or rawGT while collecting evidence.
    sys.addaudithook(diag.Guard(parent).hook)
    for p,h in c['code'].items():assert sha(p)==h
    model=load_model(c);outputs={}
    for i,s in enumerate(c['scenes']):
        stage('GT_free_region_features',completed=i,scene=s,total=200)
        nt,pts=diag.load_nt(parent['jobs'][s]);keys=list(pts);raw=np.asarray(np.load(parent['jobs'][s]['paths']['processed'],mmap_mode='r')[:,:9])
        selected=[row for row in diag.read(source/'features'/s/'pairs.json') if row['retrieval']['combined']]
        rv,receipts=visual_scene(c,parent,s,raw,pts,selected,model)
        arows=[];vrows=[];meta=[]
        for row,evidence,receipt in zip(selected,rv,receipts):
            a,b=_row_feature(raw,pts,row);b=np.r_[b,evidence];arows.append(a);vrows.append(b);summary.append(a);visual.append(b);r=dict(scene=s,left=row['left'],right=row['right']);rows.append(r);meta.append(dict(**r,views=receipt));counts['pairs']+=1
        d=OUT/'scenes'/s;d.mkdir(exist_ok=True,parents=True);np.savez_compressed(d/'features.npz',summary=np.asarray(arows).reshape(-1,21),visual=np.asarray(vrows).reshape(-1,len(visual[-1]) if visual else 112))
        freeze_json(d/'region_views.json',meta)
        for q in ('features.npz','region_views.json','region_embeddings.npz'):outputs[str(d/q)]=sha(d/q)
    summary=np.asarray(summary);visual=np.asarray(visual);assert len(summary)==len(rows) and np.isfinite(summary).all() and np.isfinite(visual).all()
    np.savez_compressed(OUT/'features.npz',summary=summary,visual=visual)
    freeze_json(OUT/'feature_rows.json',rows);outputs.update({str(OUT/q):sha(OUT/q) for q in ('features.npz','feature_rows.json')});freeze_json(OUT/'features_frozen.json',dict(complete=True,pairs=len(rows),summary_dim=int(summary.shape[1]),visual_dim=int(visual.shape[1]),counts=dict(counts),GT_used=False,outputs=outputs));stage('features_sealed',pairs=len(rows))

def _load_labels(c):
    items=diag.read(OUT/'feature_rows.json');lookup={};source=Path(c['source_root'])
    for scene in c['scenes']:
        for y in diag.read(source/'labels'/scene/'pairs.json'):
            lookup[(scene,y['left'],y['right'])]=y
    keys=[(r['scene'],r['left'],r['right']) for r in items]
    labels=np.array([1 if lookup[k]['identity_label']=='same_instance' else 0 if lookup[k]['identity_label']=='different_instances' else -1 for k in keys])
    return keys,labels,lookup


def metrics(y,score):
    return dict(pairs=len(y),same=int(y.sum()),different=int((y==0).sum()),roc_auc=float(roc_auc_score(y,score)),pair_average_precision=float(average_precision_score(y,score)))


def action_diagnostics(items,score,labels,lookup,threshold):
    selected=score>=threshold;stat=Counter(accepted=int(selected.sum()),same=int(((labels==1)&selected).sum()),different=int(((labels==0)&selected).sum()),unknown=int(((labels<0)&selected).sum()))
    targets={t:set() for t in ('0.50','0.75','0.90')};majority=0;count=0;targetpoints=0;pollution=0;unique_unions=set()
    for key,ok in zip(items,selected):
        if not ok:continue
        y=lookup[key]
        for t,hits in y['new_hits'].items():targets[t].update((key[0],g) for g in hits)
        for direction in y['directions']:
            if direction['anchor']['reliable']:
                count+=1;majority+=direction['target_majority'];targetpoints+=direction['added_target_points'];pollution+=direction['added_pollution_points']
    denom=stat['same']+stat['different']
    return dict(**stat,reliable_precision=stat['same']/denom if denom else None,reliable_accepts=denom,new_over_complete_NT={t:len(v) for t,v in targets.items()},reliable_anchor_directions=count,target_majority_directions=majority,target_majority_rate=majority/count if count else 0,added_target_points=targetpoints,added_pollution_points=pollution,pollution_ratio=pollution/max(1,pollution+targetpoints))


def train():
    import joblib
    from threadpoolctl import threadpool_limits
    c=diag.read(OUT/'contract.json');assert diag.read(OUT/'features_frozen.json')['complete']
    with np.load(OUT/'features.npz') as z:summary=z['summary'];visual=z['visual']
    items,labels,lookup=_load_labels(c);assert len(items)==len(labels)==len(summary)
    stage('training',pairs=len(items),same=int((labels==1).sum()),different=int((labels==0).sum()),unknown=int((labels<0).sum()))
    splits=np.array([c['folds'][s] for s,_,_ in items]);known=labels>=0;results={};models={};predictions={}
    for modelname,X in (('summary',summary),('summary_plus_visual',np.c_[summary,visual])):
        allpred=np.full(len(labels),np.nan);foldmetrics={}
        for fold in range(5):
            test=splits==fold;tr=(~test)&known;te=test&known
            training_scenes=Counter(items[i][0] for i in np.flatnonzero(tr));weights=np.array([1/training_scenes[items[i][0]] for i in np.flatnonzero(tr)]);weights*=len(weights)/weights.sum()
            model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,max_depth=4,min_samples_leaf=20,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=20261011)
            with threadpool_limits(4):
                model.fit(X[tr],labels[tr],sample_weight=weights);allpred[test]=model.predict_proba(X[test])[:,1];trainprob=model.predict_proba(X[tr])[:,1]
            d=OUT/'models'/modelname;d.mkdir(parents=True,exist_ok=True);joblib.dump(model,d/f'fold{fold}.joblib')
            foldmetrics[str(fold)]=dict(training=metrics(labels[tr],trainprob),heldout=metrics(labels[te],allpred[te]),train_scenes=sorted(training_scenes),test_scenes=sorted({items[i][0] for i in np.flatnonzero(test)}),training_weight_sum=float(weights.sum()),feature_dim=X.shape[1])
            print(modelname,fold,foldmetrics[str(fold)]['heldout'],flush=True)
        assert np.isfinite(allpred).all();np.save(OUT/(modelname+'_oof.npy'),allpred);predictions[modelname]=allpred
        results[modelname]=dict(folds=foldmetrics,pooled=metrics(labels[known],allpred[known]))
    # Freeze every heldout score from BOTH arms before inspecting geometry outcomes.
    frozen={str(OUT/(name+'_oof.npy')):sha(OUT/(name+'_oof.npy')) for name in predictions}
    frozen.update({str(q):sha(q) for q in (OUT/'models').glob('*/*.joblib')})
    freeze_json(OUT/'predictions_frozen.json',dict(complete=True,outputs=frozen,all_test_scores_out_of_fold=True,GT_inputs_to_model=False,unknown_labels_excluded_from_training=True))
    for name,pred in predictions.items():
        results[name]['actions']={str(th):action_diagnostics(items,pred,labels,lookup,th) for th in (.5,.9)}
        for fold in range(5):
            idx=np.flatnonzero(splits==fold);results[name]['folds'][str(fold)]['actions']={str(th):action_diagnostics([items[i] for i in idx],pred[idx],labels[idx],lookup,th) for th in (.5,.9)}
    base=results['summary']['pooled'];aug=results['summary_plus_visual']['pooled'];action=results['summary_plus_visual']['actions']['0.9'];better=sum(results['summary_plus_visual']['folds'][str(f)]['heldout']['pair_average_precision']>results['summary']['folds'][str(f)]['heldout']['pair_average_precision'] for f in range(5))
    improve=aug['pair_average_precision']>base['pair_average_precision'] and aug['roc_auc']>base['roc_auc'];advance=improve and better>=3 and action['reliable_accepts']>=20 and (action['reliable_precision'] or 0)>=.9 and action['new_over_complete_NT']['0.75']>=2 and action['target_majority_rate']>.5 and action['pollution_ratio']<=.5
    freeze_json(OUT/'results.json',dict(complete=True,results=results,visual_improves_both=improve,folds_pair_AP_improved=better,geometry_value_gate=advance,own_semantic_or_instance_AP=False,SL_modified=False))
    lines=['# 实例归属可学习性实验','', '固定无重叠邻接检索与SL；比较同容量HGB的摘要特征和摘要＋区域视觉证据。后者包含冻结DINOv2可见独有区域的token池化、三姿态组对应、点RGB／法向／局部形状和实测深度路径／遮挡。所有区域特征先封存，训练折实例身份作为监督，混合或身份不足不当负例。', '', f"摘要 pooled ROC-AUC={base['roc_auc']:.6f}，配对average precision={base['pair_average_precision']:.6f}；加入视觉 ROC-AUC={aug['roc_auc']:.6f}，配对average precision={aug['pair_average_precision']:.6f}。配对AP不是ScanNet实例AP。",'',f'折外双指标改善={improve}，{better}/5折配对AP改善；预注册0.9动作诊断门={advance}。后验0.9不自动意味着90%安全率。未知配对仍预测但不按负例评测，动作诊断单列未知。','', '|模型|折|训练配对AP|留出配对AP|留出ROC-AUC|.9接受|可靠precision|未知|新增>.75目标|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('summary','summary_plus_visual'):
        for fold,v in results[name]['folds'].items():
            g=v['actions']['0.9'];lines.append(f"|{name}|{fold}|{v['training']['pair_average_precision']:.6f}|{v['heldout']['pair_average_precision']:.6f}|{v['heldout']['roc_auc']:.6f}|{g['accepted']}|{g['reliable_precision']}|{g['unknown']}|{g['new_over_complete_NT']['0.75']}|")
    lines+=['','动作只作后置几何诊断，未生成第二版补全输出、获取自身类别证据或接回SL；没有新ScanNetAP、开发55调参、阈值搜索、模型容量搜索或裁剪。summary中的旧相机delta-z与近邻距离非独立身份信息；aug使用传感器实测路径变化与遮挡残差。所有帧来自既有30帧，dense token含邻域上下文，视觉相似也非身份保证。','', '新增目标按全部NT池检查，方向污染以可靠锚实例为参照；同实例标签仅指独有区域的可靠身份，不保证整个父候选纯净，完整并集质量在独立补充中检查。详见contract.json、features_frozen.json、predictions_frozen.json、results.json。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    freeze_json(OUT/'decision.json',dict(complete=True,visual_improves_both=improve,folds_pair_AP_improved=better,geometry_value_gate=advance,own_semantic_or_instance_AP=False,SL_modified=False,production_modified=False))
    stage('training_and_diagnostic_complete',geometry_value_gate=advance)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','features','train'));a=p.parse_args()
    with threadpool_limits(4):globals()[a.mode]()
