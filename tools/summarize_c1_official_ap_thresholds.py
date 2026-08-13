#!/usr/bin/env python3
"""Reaggregate frozen C1 diagnostics using evaluator-defined official AP thresholds."""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnose_gvc_class_agnostic_ap import instance_eval

def main():
 p=argparse.ArgumentParser(); p.add_argument('--c1a',type=Path,required=True); p.add_argument('--c1b',type=Path,required=True); p.add_argument('--c1c',type=Path,required=True); p.add_argument('--parity',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 def load(x): return json.loads(x.read_text())
 c1a,c1b,c1c,parity=map(load,(a.c1a,a.c1b,a.c1c,a.parity))
 official=[str(int(round(float(x)*100))) for x in sorted(instance_eval.opt['overlaps']) if float(x)>=.5]
 def mean(source,key): return float(np.mean([source['threshold_metrics'][tag][key] for tag in official]))
 totals=c1a['threshold_totals']; valid=sum(totals[tag]['valid_gt_instance_count'] for tag in official)
 c1a_inc=float(np.mean([totals[tag]['track_filter_oracle_increment_vs_native']/totals[tag]['valid_gt_instance_count'] for tag in official]))
 out={'official_ap_thresholds':[round(float(x),2) for x in sorted(instance_eval.opt['overlaps']) if float(x)>=.5],'ap25_threshold':.25,'extra_diagnostic_thresholds':[.95],'parity':parity,'c1a':{'coverage_increment_like_ap':c1a_inc,'coverage50_increment':totals['50']['track_filter_oracle_increment_vs_native']},'c1b':{'frozen_track_score_ap':mean(c1b,'c1a_filter_frozen_track_score_ap'),'track_rank_permutation_ap':mean(c1b,'c1a_filter_gt_quality_track_rank_permutation_ap'),'track_rank_permutation_gain':mean(c1b,'gt_quality_track_rank_permutation_gain_vs_filter'),'gt_iou_score_rule_gain':mean(c1b,'gt_iou_track_score_rule_gain_vs_filter')},'c1c_coverage_matching_diagnostic':{'fixed_score_gain_vs_native':mean(c1c,'fixed_score_gain_vs_native'),'coverage_increment_like_ap':float(np.mean([c1c['threshold_metrics'][tag]['coverage_increment_vs_native']/c1c['threshold_metrics'][tag]['valid_gt'] for tag in official])),'not_ap_upper_bound':True}}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True))
if __name__=='__main__': main()
