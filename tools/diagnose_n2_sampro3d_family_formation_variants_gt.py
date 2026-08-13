#!/usr/bin/env python3
"""GT-only N2a oracle comparing four frozen no-GT family-formation variants."""
import argparse,json,os,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from diagnose_n1_sampro3d_candidate_space_oracle_gt import _resolve,_scenes,_load_gt,_sp_gt_counts,_geometry_for_superpoints,_ious,_load_track_iou,_load_native_iou,maximum_cardinality_matching
CONTRACT="GT-only N2a fixed-variant oracle; no geometry/score/proposal inference mutation."
VARIANTS=("sam_top_single_observation_control","cross_view_consistency_medoid","two_view_superpoint_consensus","compatible_view_union_high_recall_diagnostic")
def _scene(scene,args):
 gt_ids,gt_sizes=_load_gt(args.gt_instance_dir/f"{scene}.txt",args.min_region_size);p=np.load(args.processed_scene_root/scene/f"{scene.replace('scene','')}.npy",mmap_mode="r");sp=np.asarray(p[:,9],dtype=np.int64);ids,c=np.unique(sp,return_counts=True);sizes={int(i):int(n) for i,n in zip(ids,c)};spgt=_sp_gt_counts(sp,gt_ids,gt_sizes);d2b=_load_track_iou(args.d2b_track_root,scene,gt_ids,gt_sizes);native=_load_native_iou(args.native_prediction_cache,scene,gt_ids,gt_sizes)
 rows=[json.loads(x) for x in (args.variant_ledger_root/scene/"family_formation_variant_ledger.jsonl").read_text().splitlines() if x];by=defaultdict(list)
 for r in rows:by[r["variant_name"]].append(r)
 summaries=[];detail=[]
 for variant in VARIANTS:
  family_iou={};
  for r in by[variant]:
   _,n,inter=_geometry_for_superpoints(r["superpoint_ids"],sizes,spgt);family_iou[r["candidate_family_key"]]=_ious(n,inter,gt_sizes)
  result={"scene_name":scene,"variant_name":variant,"family_count":len(family_iou)}
  for t in (.25,.5):
   tag=str(int(t*100));match=maximum_cardinality_matching(family_iou,t);matched=set(match.values());result[f"matched_gt_iou{tag}"]=len(matched);result[f"new_vs_d2b_native_union_iou{tag}"]=sum(gt in matched and d2b[gt]<t and native[gt]<t for gt in gt_sizes);result[f"duplicate_d2b_native_union_iou{tag}"]=sum(gt in matched and (d2b[gt]>=t or native[gt]>=t) for gt in gt_sizes);result[f"baseline_union_covered_iou{tag}"]=sum(d2b[g]>=t or native[g]>=t for g in gt_sizes)
   for gt in gt_sizes:detail.append({"scene_name":scene,"variant_name":variant,"gt_instance_id":gt,f"matched_iou{tag}":gt in matched,f"new_vs_union_iou{tag}":bool(gt in matched and d2b[gt]<t and native[gt]<t),f"d2b_iou":d2b[gt],f"native_iou":native[gt],"ground_truth_usage":"offline_diagnostic_only","proposal_materialization_applied":False})
  summaries.append(result)
 return summaries,detail
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--allow-gt-diagnostics",action="store_true");p.add_argument("--scene-list",type=Path,required=True);p.add_argument("--variant-ledger-root",type=Path,required=True);p.add_argument("--d2b-track-root",type=Path,required=True);p.add_argument("--native-prediction-cache",type=Path,required=True);p.add_argument("--processed-scene-root",type=Path,default=Path("data/scannet200"));p.add_argument("--gt-instance-dir",type=Path,default=Path("data/scannet200/ground_truth"));p.add_argument("--output-root",type=Path,required=True);p.add_argument("--min-region-size",type=int,default=100);p.add_argument("--max-scenes",type=int);a=p.parse_args()
 if not a.allow_gt_diagnostics:raise SystemExit("--allow-gt-diagnostics is required")
 for n in("scene_list","variant_ledger_root","d2b_track_root","native_prediction_cache","processed_scene_root","gt_instance_dir","output_root"):setattr(a,n,_resolve(getattr(a,n)))
 if a.output_root.exists() and any(a.output_root.iterdir()):raise SystemExit("output root is non-empty")
 scenes=_scenes(a.scene_list)[:a.max_scenes];a.output_root.mkdir(parents=True,exist_ok=True);allsum=[]
 for i,s in enumerate(scenes,1):
  sums,detail=_scene(s,a);stage=a.output_root/f".{s}.tmp.{os.getpid()}";stage.mkdir();
  with (stage/"variant_gt_oracle.jsonl").open("w") as h:
   for x in detail:h.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
  (stage/"summary.json").write_text(json.dumps(sums,ensure_ascii=False,indent=2,sort_keys=True)+"\n");os.replace(stage,a.output_root/s);allsum+=sums;print(f"[N2a oracle] {i}/{len(scenes)} {s}",flush=True)
 total=[]
 for v in VARIANTS:
  x={"variant_name":v}
  for k in ("family_count","matched_gt_iou25","matched_gt_iou50","new_vs_d2b_native_union_iou25","new_vs_d2b_native_union_iou50","duplicate_d2b_native_union_iou25","duplicate_d2b_native_union_iou50","baseline_union_covered_iou25","baseline_union_covered_iou50"):x[k]=sum(r.get(k,0) for r in allsum if r["variant_name"]==v)
  total.append(x)
 root={"diagnostic_type":"N2a GT-only fixed family-formation variant oracle","decision_constraint":CONTRACT,"scene_count":len(scenes),"proposal_materialization_applied":False,"ap_computed":False,"variant_summaries":total,"params":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
 (a.output_root/"summary.json").write_text(json.dumps(root,ensure_ascii=False,indent=2,sort_keys=True)+"\n");print(json.dumps(root,ensure_ascii=False,indent=2,sort_keys=True))
if __name__=="__main__":main()
