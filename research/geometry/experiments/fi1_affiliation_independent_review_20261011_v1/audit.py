from pathlib import Path
from collections import Counter
import hashlib,json,sys
import numpy as np
import joblib
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'.local/fi1_fragment_affiliation_learnability_development200_20261011_v1'
BASE=ROOT/'.local/fi1_complementary_fragment_retrieval_identity_development200_20261011_v1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=read(SRC/'contract.json');rows=read(SRC/'feature_rows.json')
with np.load(SRC/'features.npz') as z:S=z['summary'];V=z['visual']
result=dict(scope='Cached-only independent audit: no raw GT, training, GPU inference or instance AP.',checks={},metrics={},action_diagnostics={},limitations=[])
bound={}
for name in ['predictions_frozen.json']:
    bound.update(read(SRC/name)['outputs'])
bound.update(c['code']);bound.update(c['source_files'])
for name in ['features.npz','feature_rows.json']:
    path=SRC/name;bound[str(path)]=read(SRC/'features_frozen.json')['outputs'][str(path)]
for p,h in bound.items():assert sha(p)==h,p
result['checks']['hashes']=len(bound)
by={s:{} for s in c['scenes']}
for i,r in enumerate(rows):by[r['scene']][(r['left'],r['right'])]=i
y=np.full(len(rows),-9);labelrows=[None]*len(rows);counts=Counter()
for scene,ix in by.items():
    for r in read(BASE/'labels'/scene/'pairs.json'):
        index=ix.get((r['left'],r['right']))
        if index is None:continue
        y[index]={'same_instance':1,'different_instances':0,'mixed_or_unreliable':-1}[r['identity_label']];labelrows[index]=r
    for r in read(BASE/'features'/scene/'pairs.json'):
        index=ix.get((r['left'],r['right']))
        if index is None:continue
        distance=r['boundary'].get('distance_min')
        if distance==0:
            counts['zero_distance_rows']+=1
            assert S[index,8]==.2
    for r in read(SRC/'scenes'/scene/'region_views.json'):
        views=r['views'];counts['pairs_with_no_visual_views']+=not views
        for v in views:
            counts['selected_pair_views']+=1
            counts['either_region_one_token']+=min(v['left_token_cells'],v['right_token_cells'])==1
            counts['both_regions_one_token']+=v['left_token_cells']==v['right_token_cells']==1
assert not (y==-9).any()
folds=np.array([c['folds'][r['scene']] for r in rows]);known=y>=0
result['checks'].update(pairs=len(rows),same=int((y==1).sum()),different=int((y==0).sum()),unknown=int((y<0).sum()),summary_dim=S.shape[1],visual_dim=V.shape[1],feature_observations=dict(counts))
original=read(SRC/'results.json')['results']
with threadpool_limits(4):
    for name,X in [('summary',S),('summary_plus_visual',np.c_[S,V])]:
        pp=np.load(SRC/(name+'_oof.npy'));fm=[]
        for f in range(5):
            test=folds==f;tr=(folds!=f)&known
            train_scene={rows[i]['scene'][:9] for i in np.flatnonzero(tr)};test_scene={rows[i]['scene'][:9] for i in np.flatnonzero(test)}
            assert not train_scene&test_scene
            model=joblib.load(SRC/'models'/name/f'fold{f}.joblib')
            replay=model.predict_proba(X[test])[:,1];assert np.array_equal(pp[test],replay)
            assert model.n_features_in_==X.shape[1] and model.n_iter_==200
            ap=float(average_precision_score(y[test&known],pp[test&known]));assert abs(ap-original[name]['folds'][str(f)]['heldout']['pair_average_precision'])<1e-14
            fm.append(ap)
        ap=float(average_precision_score(y[known],pp[known]));auc=float(roc_auc_score(y[known],pp[known]))
        assert abs(ap-original[name]['pooled']['pair_average_precision'])<1e-14
        assert abs(auc-original[name]['pooled']['roc_auc'])<1e-14
        result['metrics'][name]=dict(pair_AP=ap,ROC_AUC=auc,fold_AP=fm)
        result['action_diagnostics'][name]={}
        for threshold in [.5,.9]:
            ids=np.flatnonzero(pp>=threshold);cc=Counter(accepted=len(ids),same=int((y[ids]==1).sum()),different=int((y[ids]==0).sum()),unknown=int((y[ids]<0).sum()))
            new={t:set() for t in ['0.50','0.75','0.90']}
            for i in ids:
                r=labelrows[i]
                for t,hits in r['new_hits'].items():new[t].update((rows[i]['scene'],g) for g in hits)
            saved=original[name]['actions'][str(threshold)]
            for k,v in cc.items():assert v==saved[k]
            nn={t:len(ss) for t,ss in new.items()};assert nn==saved['new_over_complete_NT']
            result['action_diagnostics'][name][str(threshold)]=dict(cc,new_coverage=nn)
pos=np.flatnonzero(y==1)
result['same_instance_objective']=dict(positive_pairs=len(pos),positive_pairs_with_new075=sum(bool(labelrows[i]['new_hits']['0.75']) for i in pos),positive_pairs_without_new075=sum(not labelrows[i]['new_hits']['0.75'] for i in pos))
result['checks']['exact_OOF_scores']=2*len(rows)
result['limitations']=['No independent rerun of DINO GPU feature extraction or pixel registration in this audit.','Cached GT-derived labels used for audit, no raw GT reads.','Identity classification of exclusive regions does not establish full-union quality or new coverage.','OOF evaluations exclude unknown labels; thresholds on unknown pairs are not safety guarantees.','Exact zero distance is encoded as .2 by truthiness fallback in both arms; quantified, not changed in frozen experiment.']
result['script_sha256']=sha(__file__)
assert not (OUT/'audit.json').exists()
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
