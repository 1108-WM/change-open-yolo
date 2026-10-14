#!/usr/bin/env python3
"""Post-seal diagnostic: why GT-picked pair opportunities escaped one action."""
from pathlib import Path
from collections import Counter
import sys,json
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_witnessed_local_completion as run
from tools.fi1_witnessed_local_completion import witnessed_addition
from tools.fi1_frontend_protocol import freeze_json

def main():
    c=run.read(run.OUT/'contract.json');rr=[json.loads(x) for x in (run.diag.OUT/'targets.jsonl').read_text().splitlines()];rows=[];counts=Counter()
    for s in c['scenes']:
        opportunities=[r for r in rr if r['scene']==s and r['missing075'] and r['NT_top12_pair_union_oracle'] and r['NT_top12_pair_union_oracle']['iou']>c['thresholds'][1]]
        if not opportunities:continue
        print(s,len(opportunities),flush=True);_,pts,_=run.diag.prior.src.ctl.base.load_geometry(c['jobs'][s]['binding']);fs,_=run.diag.load_frames(c,s);raw=np.load(c['jobs'][s]['paths']['processed'],mmap_mode='r');xyz=np.asarray(raw[:,:3]).copy();del raw;gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64)
        new=run.read(run.OUT/'geometry'/s/'geometry.json');newkeys={r['geometry_key'] for r in new}
        for r in opportunities:
            pair=r['NT_top12_pair_union_oracle'];left,right=pair['left'],pair['right'];a,b=pts[left],pts[right];inter=len(np.intersect1d(a,b));ov=inter/(len(a)+len(b)-inter);directions=[]
            for parent,donor in ((left,right),(right,left)):
                extra=np.setdiff1d(pts[donor],pts[parent]);reason=None;result=pts[parent]
                if not c['parameters']['min_pair_iou']<ov<c['parameters']['max_pair_iou']:reason='pair_overlap_excluded'
                elif len(extra)>len(pts[parent]):reason='anchor_identity_budget'
                else:result,receipt=witnessed_addition(pts[parent],pts[donor],fs,xyz,c['parameters']);reason=receipt['reason']
                hit=int(np.sum(gt[result]==r['gt_id']));size=int(np.sum(gt==r['gt_id']));iou=hit/(len(result)+size-hit) if len(result)>=100 else 0
                digest=run.hashlib.sha256(result.astype('<i8').tobytes()).hexdigest();key=s+':geometry:'+digest;emitted=key in newkeys
                direction=dict(parent=parent,donor=donor,reason=reason,result_iou=iou,emitted=emitted);directions.append(direction)
            useful=any(d['result_iou']>c['thresholds'][1] for d in directions);emitted=any(d['result_iou']>c['thresholds'][1] and d['emitted'] for d in directions)
            # Parallel failure summaries are not a causal AP attribution.
            counts['opportunities']+=1;counts['action_possible_repair']+=useful;counts['action_emitted_repair']+=emitted
            for reason in {d['reason'] for d in directions}:counts['has_direction_'+reason]+=1
            rows.append(dict(scene=s,gt_id=r['gt_id'],kind=r['kind'],pair_iou=ov,oracle_union_iou=pair['iou'],directions=directions))
    assert counts['opportunities']==60
    freeze_json(run.OUT/'opportunity_action_gap.json',dict(complete=True,counts=dict(counts),rows=rows,diagnostic_only=True,parameters_changed=False,new_predictions=False))

if __name__=='__main__':main()
