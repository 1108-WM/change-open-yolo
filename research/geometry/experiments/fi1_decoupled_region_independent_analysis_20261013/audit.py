"""Posthoc CSV audit only. No new rules, models, or executable predictions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'.local/fi1_decoupled_region_internal_development200_20261013_v1'
OUT=Path(__file__).resolve().parent
d=pd.read_csv(R/'evaluation/relations.csv')
for col in d.columns:
    if col.endswith(('_truth','_probability')):d[col]=d[col].astype(np.float32)
safe=pd.read_csv(ROOT/'.local/fi1_region_affiliation_internal_development200_20261013_v1/evaluation/regions.csv')
safe=safe[safe.arm=='A'];keys=['scene','candidate_id','region']
q=d[d.reference=='retained_parent'][keys+['split','different_probability','different_truth']].rename(columns={'different_probability':'p_diff_parent','different_truth':'gt_diff_parent'})
k=d[d.reference.str.startswith('competitor')].groupby(keys).agg(p_same_neighbor=('same_probability','max'),gt_same_neighbor=('same_truth','max')).reset_index()
q=q.merge(k,on=keys,how='left').fillna({'p_same_neighbor':0.,'gt_same_neighbor':0.})
q['pred_score']=np.minimum(q.p_diff_parent,q.p_same_neighbor)
q['truth_score']=np.minimum(q.gt_diff_parent,q.gt_same_neighbor)
q=q.merge(safe[keys+['safe']],on=keys,validate='one_to_one')
summary=json.loads((R/'summary.json').read_text());result={'posthoc_only':True,'splits':{}}
for split in ['train','calibration','holdout']:
    s=d[(d.split==split)&(d.reference=='retained_parent')];high=s[s.different_probability>=.9]
    u=q[(q.split==split)&q.safe]
    row={'parent_reference_pure_different':int((s.different_truth>=.95).sum()),'parent_reference_high_different':len(high),'parent_reference_high_different_correct':int((high.different_truth>=.95).sum()),'safe_regions':len(u),'safe_pred_score_max':float(u.pred_score.max()),'threshold_counts':{},'high_confidence_agreement':{}}
    for threshold in [.7,.8,.9,.95]:row['threshold_counts'][str(threshold)]={'safe_model_pass':int((u.pred_score>=threshold).sum()),'safe_exact_relation_pass':int((u.truth_score>=threshold).sum())}
    for name in ['same','different']:
        v=d[(d.split==split)&(d[name+'_probability']>=.9)][name+'_truth']
        pure=float((v>=.95).mean());agreement=float(v.mean())
        assert np.isclose(pure,summary['splits'][split]['relations']['classes'][name]['pure_precision'])
        assert np.isclose(agreement,summary['splits'][split]['relations']['classes'][name]['expected_GT_agreement'])
        row['high_confidence_agreement'][name]={'count':len(v),'pure_precision':pure,'mean_point_pair_agreement':agreement,'truth_under_half':int((v<.5).sum())}
    result['splits'][split]=row
grid=json.loads((R/'calibration.json').read_text())['grid']
result['calibration']={'rules':len(grid),'with_actions':[x for x in grid if x['metrics'].get('actions',0)>0],'without_actions':sum(x['metrics'].get('actions',0)==0 for x in grid)}
c=json.loads((R/'contract.json').read_text());verified=[]
for name in ['fi1_decoupled_region_core.py','run_fi1_decoupled_region.py','evaluate_fi1_decoupled_region.py']:
    p=ROOT/'tools'/name;h=hashlib.sha256(p.read_bytes()).hexdigest()
    assert h==c['code'][str(p)]==hashlib.sha256((R/'code_snapshot/tools'/name).read_bytes()).hexdigest();verified.append(name)
result['source_contract_snapshot_verified']=verified
result['no_new_training']=result['no_new_rules']=True
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
