"""Meaningful duplicate/ignore trace check against frozen official evaluator."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import unittest
import numpy as np
import analyze_fi1_t_add_downstream as a
class TestTrace(unittest.TestCase):
    def test_duplicate_and_ignore_official_ap(self):
        ev=a.run.ctl.base.evaluator();label=ev.CLASS_LABELS[0];gid=int(ev.VALID_CLASS_IDS[0])*1000+1
        g=dict(instance_id=gid,label_id=int(ev.VALID_CLASS_IDS[0]),vert_count=200,med_dist=0.,dist_conf=1.)
        ps=[]
        for n,score in enumerate([.7,.9]):ps.append(dict(uuid=str(n),pred_id=n,vert_count=200,confidence=score,void_intersection=0,label_id=g['label_id'],matched_gt=[dict(g,intersection=200)]))
        void=dict(uuid='void',pred_id=2,vert_count=200,confidence=.95,void_intersection=200,label_id=g['label_id'],matched_gt=[])
        match=dict(gt={l:[] for l in ev.CLASS_LABELS},pred={l:[] for l in ev.CLASS_LABELS});match['gt'][label]=[dict(g,matched_pred=[dict({k:v for k,v in p.items() if k!='matched_gt'},intersection=200) for p in ps])];match['pred'][label]=ps+[void]
        official=ev.evaluate_matches({'s':match})[0]
        for i,t in enumerate(ev.opt['overlaps']):
            slots,fn,pres,cov=a.trace(match,ev,t);self.assertEqual([r['uuid'] for r in slots if r['kind']=='TP'],['1']);self.assertEqual([r['uuid'] for r in slots if r['kind']=='duplicate_FP'],['0']);self.assertEqual(cov,{gid})
            calc=a.integrate(slots,fn,pres,ev.CLASS_LABELS);np.testing.assert_array_equal(calc,official[0,:,i])
if __name__=='__main__':unittest.main()
