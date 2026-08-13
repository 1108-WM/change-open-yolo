#!/usr/bin/env python3
"""GT-only class-agnostic AP for fixed native, D2b, and N2-medoid caches."""
import argparse,gc,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from diagnose_gvc_class_agnostic_ap import _resolve,_read_scenes,_class_agnostic_gt_ids,_configure_scannet200_instance_eval,_merge_scan_matches,_track_prediction,instance_eval,UNIFIED_PREDICTED_CLASS
CONTRACT="GT-only fixed-cache AP diagnostic; GT cannot alter N2 geometry, medoid scores, D2b/native scores, or competition decisions."
def n2_prediction(scene,cache,variant,point_count):
 score={r["candidate_family_key"]:float(r["medoid_cross_view_mean_jaccard"]) for r in (json.loads(x) for x in (variant/scene/"family_formation_variant_ledger.jsonl").read_text().splitlines() if x) if r["variant_name"]=="cross_view_consistency_medoid"}
 rows=json.loads((cache/scene/"n2_medoid_candidates.json").read_text())["candidates"];m=np.zeros((point_count,len(rows)),dtype=bool);s=np.zeros(len(rows),dtype=np.float32)
 for i,r in enumerate(rows):
  pts=np.asarray(np.load(r["points_path"])["point_indices"],dtype=np.int64);pts=pts[(pts>=0)&(pts<point_count)];m[pts,i]=True;s[i]=max(0.,score[r["canonical_family_key"]])
 return {"pred_masks":m,"pred_scores":s,"pred_classes":np.full(len(rows),UNIFIED_PREDICTED_CLASS,dtype=np.int64)}
def ap(name,components,scenes,a):
 _configure_scannet200_instance_eval();matches={};orig=instance_eval.util_3d.load_ids;instance_eval.util_3d.load_ids=lambda f:_class_agnostic_gt_ids(orig(f))
 try:
  for i,scene in enumerate(scenes,1):
   prefix=a.native_prediction_cache/f"{scene}_pred_";nm=np.load(str(prefix)+"masks.npy",mmap_mode="r");ns=np.asarray(np.load(str(prefix)+"scores.npy"),dtype=np.float32);native={"pred_masks":nm,"pred_scores":ns,"pred_classes":np.full(len(ns),UNIFIED_PREDICTED_CLASS,dtype=np.int64)};preds=[]
   for c in components:
    if c=="native":preds.append(native)
    elif c=="d2b":preds.append(_track_prediction(a.d2b_track_root,scene,nm.shape[0],"mean_node_quality"))
    else:preds.append(n2_prediction(scene,a.n2_cache_root,a.variant_ledger_root,nm.shape[0]))
   gt_file=str(a.gt_instance_dir/f"{scene}.txt");g,p=instance_eval.assign_instances_for_scan(preds[0],gt_file)
   for q in preds[1:]:g2,p2=instance_eval.assign_instances_for_scan(q,gt_file);g,p=_merge_scan_matches(g,p,g2,p2)
   matches[os.path.abspath(gt_file)]={"gt":g,"pred":p};print(f"[N2 AP {name}] {i}/{len(scenes)} {scene}",flush=True);del preds,native,nm,g,p;gc.collect()
 finally:instance_eval.util_3d.load_ids=orig
 scores,*_=instance_eval.evaluate_matches(matches);avg=instance_eval.compute_averages(scores);instance_eval.write_result_file(avg,str(a.output_dir/f"{name}.csv"));x=avg["classes"]["chair"];return {"ap":float(x["ap"]),"ap50":float(x["ap50%"]),"ap25":float(x["ap25%"]) }
def main():
 p=argparse.ArgumentParser();p.add_argument("--allow-gt-diagnostics",action="store_true");p.add_argument("--scene-list",type=Path,required=True);p.add_argument("--native-prediction-cache",type=Path,required=True);p.add_argument("--d2b-track-root",type=Path,required=True);p.add_argument("--n2-cache-root",type=Path,required=True);p.add_argument("--variant-ledger-root",type=Path,required=True);p.add_argument("--gt-instance-dir",type=Path,default=Path("data/scannet200/ground_truth"));p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
 if not a.allow_gt_diagnostics:raise SystemExit("--allow-gt-diagnostics required")
 for n in("scene_list","native_prediction_cache","d2b_track_root","n2_cache_root","variant_ledger_root","gt_instance_dir","output_dir"):setattr(a,n,_resolve(getattr(a,n)))
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise SystemExit("output dir non-empty")
 a.output_dir.mkdir(parents=True);scenes=_read_scenes(a.scene_list);out={"native":ap("native",("native",),scenes,a),"native_plus_d2b":ap("native_plus_d2b",("native","d2b"),scenes,a),"native_plus_n2_medoid":ap("native_plus_n2_medoid",("native","n2"),scenes,a),"native_plus_d2b_plus_n2_medoid":ap("native_plus_d2b_plus_n2_medoid",("native","d2b","n2"),scenes,a)};res={"diagnostic_type":"GT-only fixed-cache class-agnostic AP","decision_constraint":CONTRACT,"scene_count":len(scenes),"n2_score_field":"medoid_cross_view_mean_jaccard","results":out,"proposal_materialization_applied":True};(a.output_dir/"summary.json").write_text(json.dumps(res,ensure_ascii=False,indent=2)+"\n");print(json.dumps(res,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
