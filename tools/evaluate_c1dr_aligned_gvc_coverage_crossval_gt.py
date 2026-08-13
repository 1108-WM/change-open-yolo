#!/usr/bin/env python3
"""Final action-aligned C1d-R strict-coverage GVC audit (GT-only)."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'tools'):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
from build_track_native_competition_ledger import _read_scenes
from construct_c1c_global_feasible_ap_oracle_gt import AP25,OFFICIAL,_build_scene_cache,_objective,_summary_records
from construct_c1d_attribution_oracle_gt import FROZEN_SCORE,_baseline_state,_records_for_state
from construct_c1d_global_feasible_replacement_oracle_gt import _canonical_native_map,_read_actions
from diagnose_gvc_class_agnostic_ap import _class_agnostic_gt_ids,_configure_scannet200_instance_eval,instance_eval

EPS=1e-12
FAMILIES={"single_covered":"replace_native_covered_099_with_track","all_covered":"replace_native_covered_099_with_all_tracks"}
FEATURES=("aligned_gvc_only","raw_score_difference_only","equal_aligned_gvc_raw_score_difference")
BUDGETS=(.50,.25,.10)
def _resolve(p): return p if p.is_absolute() else ROOT/p
def _rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def _state(x): return {int(k):v for k,v in x.items()}

def _action_features(scene,actions,pair_root,relation_root,cache,native_root):
 pairs={(int(x['track_id']),int(x['native_candidate_id'])):x for x in _rows(pair_root/scene/'pair_relative_gvc.jsonl')}
 rel={(int(x['proposal_id']),int(x['native_candidate_id'])):x for x in _rows(relation_root/scene/'track_native_relations.jsonl')}
 scores=np.asarray(np.load(native_root/f'{scene}_pred_scores.npy'),dtype=float); out=[]
 for cid,group in actions.items():
  for a in group:
   family=next((n for n,k in FAMILIES.items() if a['action_kind']==k),None)
   if family is None: continue
   removed=sorted(set(map(int,a['component_native_candidate_ids']))-set(map(int,a['kept_native_candidate_ids'])))
   tracks=list(map(int,a['kept_track_ids'])); per_native=[]
   for n in removed:
    cover=[t for t in tracks if (t,n) in rel and float(rel[t,n]['native_inside_track_ratio'])>=.99]
    if not cover: raise ValueError(f'{scene} {a["action_name"]}: removed native has no covering track')
    raw=max(float(cache['tracks'][t]['mean_node_quality'])-float(scores[n]) for t in cover)
    public=[]
    for t in cover:
     pair=pairs.get((t,n))
     if pair is not None and int(pair['selected_public_common_view_count'])>0:
      public.append(float(pair['track_minus_native_gvc']))
    per_native.append((max(public) if public else None,raw))
   # Every removed native needs independent pair evidence for aligned GVC.
   gvc=None if not per_native or any(v[0] is None for v in per_native) else min(v[0] for v in per_native)
   raw=None if not per_native else min(v[1] for v in per_native)
   out.append({'scene_name':scene,'component_id':cid,'action_name':a['action_name'],'action_kind':a['action_kind'],'family':family,'removed_native_ids':removed,'kept_track_ids':tracks,'gvc':gvc,'raw':raw})
 return out

def _standard(rows):
 g=np.asarray([r['gvc'] for r in rows if r['gvc'] is not None and r['raw'] is not None],float); q=np.asarray([r['raw'] for r in rows if r['gvc'] is not None and r['raw'] is not None],float)
 if not len(g): return None
 return (float(g.mean()),max(float(g.std()),1e-12),float(q.mean()),max(float(q.std()),1e-12))
def _value(row,feature,st):
 if feature=='aligned_gvc_only': return row['gvc']
 if feature=='raw_score_difference_only': return row['raw']
 if row['gvc'] is None or row['raw'] is None or st is None:return None
 return .5*((row['gvc']-st[0])/st[1]+(row['raw']-st[2])/st[3])
def _plan(scene,actions,features,family,feature,st,direction,threshold):
 state=_baseline_state(actions); chosen=[]; sign=1 if direction=='higher' else -1
 by=defaultdict(list)
 for r in features:
  if r['family']==family:
   v=_value(r,feature,st)
   if v is not None: by[r['component_id']].append((sign*v,v,r))
 for cid,vals in by.items():
  _,v,row=max(vals,key=lambda x:(x[0],x[2]['action_name']))
  if sign*v>=sign*threshold-EPS: state[cid]=row['action_name']; chosen.append({**row,'feature_value':v})
 return state,chosen
def _records(scenes,states,actions,caches,canonical):
 d={}
 for s in scenes:d[s],_= _records_for_state(s,actions[s],states[s],caches[s],canonical[s],'raw',FROZEN_SCORE)
 return d
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('scene_list','pair_root','relation_root','action_root','fold_root','track_root','native_root','gt_dir','output_root'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
 p.add_argument('--allow-gt-diagnostics',action='store_true');p.add_argument('--max-scenes',type=int);a=p.parse_args()
 if not a.allow_gt_diagnostics:raise SystemExit('--allow-gt-diagnostics is required')
 for n in ('scene_list','pair_root','relation_root','action_root','fold_root','track_root','native_root','gt_dir','output_root'):setattr(a,n,_resolve(getattr(a,n)))
 a.relation_ledger_root=a.relation_root;a.filtered_d2b_track_root=a.track_root;a.native_prediction_cache=a.native_root;a.gt_instance_dir=a.gt_dir
 if a.output_root.exists() and any(a.output_root.iterdir()):raise SystemExit('output root is non-empty')
 scenes=_read_scenes(a.scene_list);scenes=scenes if a.max_scenes is None else scenes[:a.max_scenes]
 if len(scenes)<5:raise SystemExit('need five scenes')
 a.output_root.mkdir(parents=True,exist_ok=True);_configure_scannet200_instance_eval();old=instance_eval.util_3d.load_ids;instance_eval.util_3d.load_ids=lambda f:_class_agnostic_gt_ids(old(f))
 try:
  actions={s:_read_actions(a.action_root,s) for s in scenes};caches={s:_build_scene_cache(s,a) for s in scenes};canonical={s:_canonical_native_map(s,a.fold_root,np.asarray(np.load(a.native_root/f'{s}_pred_scores.npy'),float)) for s in scenes}
  feats={s:_action_features(s,actions[s],a.pair_root,a.relation_root,caches[s],a.native_root) for s in scenes}
  baseline_states={s:_baseline_state(actions[s]) for s in scenes};base=_records(scenes,baseline_states,actions,caches,canonical);base_ap=_objective(base);base_thr=_summary_records(base)
  fold_rows=[];oof=[]
  for feature in FEATURES:
   for family in FAMILIES:
    oof_rec={}
    for fold in range(5):
     train=[s for i,s in enumerate(scenes) if i%5!=fold];test=[s for i,s in enumerate(scenes) if i%5==fold];train_rows=[r for s in train for r in feats[s] if r['family']==family];st=_standard(train_rows) if feature.startswith('equal') else None
     values=[_value(r,feature,st) for r in train_rows];values=[v for v in values if v is not None]
     candidates=[(base_ap if False else _objective(_records(train,{s:_baseline_state(actions[s]) for s in train},actions,caches,canonical)),'noop',0.,None)]
     for direction in ('higher','lower'):
      for budget in BUDGETS:
       th=float(np.quantile(values,1-budget if direction=='higher' else budget)); states={s:_plan(s,actions[s],feats[s],family,feature,st,direction,th)[0] for s in train};candidates.append((_objective(_records(train,states,actions,caches,canonical)),direction,budget,th))
     train_ap,direction,budget,threshold=max(candidates,key=lambda x:(x[0],x[1]=='noop',x[1]=='higher',x[2]))
     states={};dec={}
     for s in test:
      states[s],dec[s]=(_baseline_state(actions[s]),[]) if direction=='noop' else _plan(s,actions[s],feats[s],family,feature,st,direction,threshold)
     rec=_records(test,states,actions,caches,canonical);oof_rec.update(rec);fold_rows.append({'feature':feature,'family':family,'fold':fold,'training_choice':direction,'budget':budget,'threshold':threshold,'training_ap':train_ap,'heldout_ap':_objective(rec),'heldout_threshold_ap':_summary_records(rec),'selected_action_count':sum(map(len,dec.values()))})
     for s,rs in dec.items():
      for r in rs:oof.append({'feature':feature,'family':family,'fold':fold,**r})
    thr=_summary_records(oof_rec);fold_rows.append({'feature':feature,'family':family,'fold':'oof_aggregate','official_ap':_objective(oof_rec),'threshold_ap':thr,'delta_vs_baseline':{'official_ap':_objective(oof_rec)-base_ap,'threshold_ap':{k:thr[k]-base_thr[k] for k in thr}},'selected_action_count':sum(r['feature']==feature and r['family']==family for r in oof)})
 finally:instance_eval.util_3d.load_ids=old
 for n,rows in [('aligned_action_features.jsonl',[r for s in scenes for r in feats[s]]),('fold_results.jsonl',fold_rows),('oof_actions.jsonl',oof)]: (a.output_root/n).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
 aggregate=[r for r in fold_rows if r['fold']=='oof_aggregate'];payload={'diagnostic_type':'GT-only action-aligned strict-coverage C1d-R GVC audit','proposal_materialization_applied':False,'model_training_applied':False,'missing_public_pair_action':'coexist','official_ap_thresholds':list(OFFICIAL),'ap25_threshold':AP25,'fixed_budgets':list(BUDGETS),'baseline':{'official_ap':base_ap,'threshold_ap':base_thr},'oof_aggregate':aggregate,'params':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}};(a.output_root/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps({r['feature']+'::'+r['family']:r['delta_vs_baseline']['official_ap'] for r in aggregate},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
