#!/usr/bin/env python3
"""Pre-register no-GT C1c append-only component actions."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from build_track_native_competition_ledger import _read_scenes,_resolve,_write_jsonl

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--scene-list',type=Path,required=True); p.add_argument('--relation-ledger-root',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--max-scenes',type=int); a=p.parse_args()
 for n in ('scene_list','relation_ledger_root','output_root'): setattr(a,n,_resolve(getattr(a,n)))
 scenes=_read_scenes(a.scene_list); scenes=scenes if a.max_scenes is None else scenes[:a.max_scenes]
 if not scenes: raise SystemExit('--max-scenes must be positive')
 if a.output_root.exists() and any(a.output_root.iterdir()): raise SystemExit('output root is non-empty')
 a.output_root.mkdir(parents=True); total=0
 for scene in scenes:
  components=[json.loads(x) for x in (a.relation_ledger_root/scene/'relation_components.jsonl').read_text().splitlines() if x]; actions=[]
  for c in components:
   cid=int(c['component_id']); tracks=[int(x) for x in c['track_ids']]; natives=[int(x) for x in c['native_candidate_ids']]
   if natives:
    kinds=[('coexist',None),('native_only',None)]+[('native_plus_one_track',t) for t in tracks]
   else:
    kinds=[('keep_all_tracks',None),('suppress_all_tracks',None)]+[('keep_one_track',t) for t in tracks]
   for kind,track in kinds: actions.append({'scene_name':scene,'component_id':cid,'action_name':kind if track is None else f'{kind}:{track}','action_kind':kind,'selected_track_id':track,'component_track_ids':tracks,'component_native_candidate_ids':natives,'ground_truth_usage':'none','proposal_materialization_applied':False,'decision_state':'pre-registered action only; not selected or applied'})
  _write_jsonl(a.output_root/f'{scene}.jsonl',actions); total+=len(actions)
 summary={'diagnostic_type':'no-GT C1c pre-registered append-only component action ledger','scene_count':len(scenes),'action_count':total,'mixed_component_actions':['coexist','native_only','native_plus_one_track(track_id)'],'track_only_component_actions':['keep_all_tracks','keep_one_track(track_id)','suppress_all_tracks'],'replacement_diagnostic_excluded':'track_only(track_id)','ground_truth_usage':'none','proposal_materialization_applied':False,'params':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
 (a.output_root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True))
if __name__=='__main__': main()
