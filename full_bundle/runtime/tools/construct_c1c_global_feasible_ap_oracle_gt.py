#!/usr/bin/env python3
"""GT-only C1c deterministic multi-start global-AP coordinate construction.

This is a feasible construction, not a mathematical AP upper bound.  It uses
only pre-registered component actions and accepts an action only when the
global evaluator-equivalent AP@0.50:0.05:0.90 strictly improves.
"""
from __future__ import annotations
import argparse,gc,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'tools'):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
from build_track_native_competition_ledger import _read_scenes,_native_cache_contract
from diagnose_d2b_native_component_competition_oracle_gt import _component_members,_load_tracks,_prediction,_record
from diagnose_d2b_native_track_ranking_oracle_gt import _average_precision,_empty_record,_append
from diagnose_gvc_class_agnostic_ap import _class_agnostic_gt_ids,_configure_scannet200_instance_eval,instance_eval

AP25=.25
OFFICIAL=tuple(sorted(round(float(x),2) for x in instance_eval.opt['overlaps'] if float(x)>=.5))
DIAGNOSTIC=(AP25,*OFFICIAL,.95)
EPS=1e-12

def _resolve(p): p=Path(p); return p if p.is_absolute() else ROOT/p
def _read_actions(root,scene):
 rows=[json.loads(x) for x in (root/f'{scene}.jsonl').read_text().splitlines() if x]
 out={}
 for r in rows: out.setdefault(int(r['component_id']),[]).append(r)
 return {k:sorted(v,key=lambda x:x['action_name']) for k,v in out.items()}
def _initial(actions,kind):
 result={}
 for cid,rows in actions.items():
  names={r['action_kind']:r for r in rows}
  if kind=='coexist': pick=names['coexist'] if 'coexist' in names else names['keep_all_tracks']
  elif kind=='native_only': pick=names['native_only'] if 'native_only' in names else names['suppress_all_tracks']
  else:
   singles=[r for r in rows if r['action_kind'] in ('native_plus_one_track','keep_one_track')]
   pick=singles[0] if singles else (names['coexist'] if 'coexist' in names else names['keep_all_tracks'])
  result[cid]=pick['action_name']
 return result
def _candidate_sets(actions,state,native_count):
 keep_n=set(range(native_count)); keep_t=set()
 for cid,rows in actions.items():
  row=next(r for r in rows if r['action_name']==state[cid]); tracks=set(row['component_track_ids']); natives=set(row['component_native_candidate_ids']); kind=row['action_kind']
  # Component native candidates are controlled only by the selected action.
  keep_n.difference_update(natives)
  if kind in ('coexist','native_only','native_plus_one_track'): keep_n.update(natives)
  if kind in ('coexist','keep_all_tracks'): keep_t.update(tracks)
  elif kind in ('native_plus_one_track','keep_one_track'): keep_t.add(int(row['selected_track_id']))
 return keep_n,keep_t
def _build_scene_cache(scene,args):
 tracks=_load_tracks(args.filtered_d2b_track_root,scene); masks=np.load(args.native_prediction_cache/f'{scene}_pred_masks.npy',mmap_mode='r'); scores=np.asarray(np.load(args.native_prediction_cache/f'{scene}_pred_scores.npy',mmap_mode='r'),dtype=np.float32)
 ledger_tracks,ledger_natives=_component_members(args.relation_ledger_root,scene)
 if ledger_tracks!=set(tracks): raise ValueError(f'{scene}: track contract differs')
 full=_prediction(masks,scores,tracks,set(range(masks.shape[1])),set(tracks))
 gt,pred=instance_eval.assign_instances_for_scan(full,str(args.gt_instance_dir/f'{scene}.txt'))
 minimum=int(instance_eval.opt['min_region_sizes'][0]); native_valid=[i for i,size in enumerate(np.count_nonzero(masks,axis=0)) if size>=minimum]; track_valid=[i for i in sorted(tracks) if int(tracks[i]['point_count'])>=minimum]
 expected=[('native',i) for i in native_valid]+[('track',i) for i in track_valid]
 rows=pred['chair']
 if len(rows)!=len(expected): raise ValueError(f'{scene}: full prediction/evaluator mapping differs')
 uuid_by_candidate={key:row['uuid'] for key,row in zip(expected,rows)}
 return {'tracks':tracks,'native_count':masks.shape[1],'gt':gt,'pred':pred,'uuid_by_candidate':uuid_by_candidate}
def _scene_records(scene,actions,state,cache):
 tracks=cache['tracks']
 native_count=cache['native_count']
 keep_n,keep_t=_candidate_sets(actions,state,native_count)
 keep_uuid={cache['uuid_by_candidate'][('native',i)] for i in keep_n if ('native',i) in cache['uuid_by_candidate']}
 keep_uuid.update(cache['uuid_by_candidate'][('track',i)] for i in keep_t if ('track',i) in cache['uuid_by_candidate'])
 pred_matches={'chair':[row for row in cache['pred']['chair'] if row['uuid'] in keep_uuid]}
 gt={'chair':[dict(row,matched_pred=[p for p in row['matched_pred'] if p['uuid'] in keep_uuid]) for row in cache['gt']['chair']]}
 out={str(int(round(t*100))):_record_from_matches(gt,pred_matches,t) for t in DIAGNOSTIC}
 return out,{'kept_native_count':len(keep_n),'kept_track_count':len(keep_t)}
def _record_from_matches(gt,pred,t):
 from diagnose_mv3dis_global_feasible_action_oracle_gt import _scene_ap_records
 return _scene_ap_records(gt,pred,t)
def _objective(records_by_scene):
 values=[]
 for t in OFFICIAL:
  tag=str(int(round(t*100))); total=_empty_record()
  for scene in records_by_scene: _append(total,records_by_scene[scene][tag])
  values.append(_average_precision(total['true'],total['score'],total['fn'],total['has_gt'],total['has_pred']))
 return float(np.mean(values))
def _summary_records(records_by_scene):
 out={}
 for t in DIAGNOSTIC:
  tag=str(int(round(t*100))); total=_empty_record()
  for scene in records_by_scene: _append(total,records_by_scene[scene][tag])
  out[tag]=_average_precision(total['true'],total['score'],total['fn'],total['has_gt'],total['has_pred'])
 return out
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--allow-gt-diagnostics',action='store_true'); p.add_argument('--scene-list',type=Path,required=True); p.add_argument('--relation-ledger-root',type=Path,required=True); p.add_argument('--action-ledger-root',type=Path,required=True); p.add_argument('--filtered-d2b-track-root',type=Path,required=True); p.add_argument('--native-prediction-cache',type=Path,required=True); p.add_argument('--gt-instance-dir',type=Path,default=Path('data/scannet200/ground_truth')); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--max-sweeps',type=int,default=3); p.add_argument('--max-scenes',type=int); a=p.parse_args()
 if not a.allow_gt_diagnostics: raise SystemExit('--allow-gt-diagnostics is required')
 for n in ('scene_list','relation_ledger_root','action_ledger_root','filtered_d2b_track_root','native_prediction_cache','gt_instance_dir','output_root'): setattr(a,n,_resolve(getattr(a,n)))
 all_scenes=_read_scenes(a.scene_list); scenes=all_scenes if a.max_scenes is None else all_scenes[:a.max_scenes]
 if not scenes or a.max_sweeps<1: raise SystemExit('invalid scene/sweep count')
 if a.output_root.exists() and any(a.output_root.iterdir()): raise SystemExit('output root is non-empty')
 a.output_root.mkdir(parents=True); contract=_native_cache_contract(a.native_prediction_cache,len(all_scenes)); _configure_scannet200_instance_eval(); original=instance_eval.util_3d.load_ids; instance_eval.util_3d.load_ids=lambda f:_class_agnostic_gt_ids(original(f))
 try:
  configs=[]
  for start in ('coexist','native_only','one_track'):
   actions={s:_read_actions(a.action_ledger_root,s) for s in scenes}; state={s:_initial(actions[s],start) for s in scenes}; records={}; details={}
   caches={s:_build_scene_cache(s,a) for s in scenes}
   for s in scenes: records[s],details[s]=_scene_records(s,actions[s],state[s],caches[s])
   score=_objective(records); history=[]
   for sweep in range(a.max_sweeps):
    improved=0
    for s in scenes:
     for cid in sorted(actions[s]):
      old=state[s][cid]
      for candidate in actions[s][cid]:
       name=candidate['action_name']
       if name==old: continue
       state[s][cid]=name; trial,trial_detail=_scene_records(s,actions[s],state[s],caches[s]); candidate_records=dict(records); candidate_records[s]=trial; value=_objective(candidate_records)
       if value>score+EPS: records[s]=trial; details[s]=trial_detail; score=value; old=name; improved+=1
       else: state[s][cid]=old
    history.append({'sweep':sweep+1,'accepted_action_count':improved,'global_official_ap':score})
    if not improved: break
   configs.append((score,start,state,records,details,history))
  score,start,state,records,details,history=max(configs,key=lambda x:(x[0],-('coexist','native_only','one_track').index(x[1])))
  # Re-evaluate every final scene from the selected state before reporting.
  final_records={}; final_details={}
  for s in scenes: final_records[s],final_details[s]=_scene_records(s,{s:_read_actions(a.action_ledger_root,s)}[s],state[s],caches[s])
  final_score=_objective(final_records); metrics=_summary_records(final_records)
 finally: instance_eval.util_3d.load_ids=original
 payload={'diagnostic_type':'GT-only global-feasible AP construction by deterministic multi-start coordinate ascent','not_mathematical_ap_upper_bound':True,'official_ap_thresholds':list(OFFICIAL),'ap25_threshold':AP25,'extra_diagnostic_thresholds':[.95],'native_cache_contract':contract,'scene_count':len(scenes),'selected_start':start,'coordinate_ascent_history':history,'official_ap':final_score,'threshold_ap':metrics,'final_scene_candidate_counts':final_details,'final_component_actions':state,'proposal_materialization_applied':False,'params':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
 (a.output_root/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); print(json.dumps({k:payload[k] for k in ('scene_count','selected_start','official_ap','coordinate_ascent_history')},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
