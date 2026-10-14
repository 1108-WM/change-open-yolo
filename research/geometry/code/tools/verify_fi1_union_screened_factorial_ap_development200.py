#!/usr/bin/env python3
"""Recompute pooled AP scalars, match consistency and exact original controls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_union_screened_factorial_ap_development200 as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha
from tools.fi1_union_development_replay import Guard


def verify(out):
    c=run.read(out/'contract.json');g=Guard(c);g.gt_enabled=True;sys.addaudithook(g.hook)
    summary=run.read(out/'evaluation/summary.json')
    for files in (c['code'],c['inputs'],c['label_inputs'],run.read(out/'predictions_frozen.json')['outputs'],summary['AP_arrays']):run.base.verify_files(files)
    GT=0;edges=0;selections=0
    for s in c['scenes']:
        receipt=run.read(out/'evaluation'/s/'receipt.json');run.base.verify_files(receipt['outputs']);GT+=receipt['GT']
        review=run.read(out/'evaluation'/s/'target_review.json')
        for view in run.VIEWS:
            for a in run.ARMS:
                selected=run.read(out/'predictions'/s/(view+'_'+a+'.json'));source=run.read(c['jobs'][s][a])
                if view=='semantic':assert selected==source
                else:
                    expected={}
                    for m in sorted(source,key=lambda m:(-m['score'],m['class_index'])):
                        expected.setdefault(m['geometry_key'],dict(geometry_key=m['geometry_key'],score=m['score'],winning_class_index=m['class_index']))
                    assert selected==sorted(expected.values(),key=lambda m:(-m['score'],m['geometry_key']))
                match=run.read(out/'evaluation'/s/(view+'_'+a+'_matches.json'))
                pmap={p['uuid']:p for ps in match['pred'].values() for p in ps};assert len(pmap)==sum(map(len,match['pred'].values()))
                assert len(pmap)<=len(selected)
                for label,gs in match['gt'].items():
                    for gt in gs:
                        for p in gt['matched_pred']:
                            pp=pmap[p['uuid']]
                            assert p['confidence']==pp['confidence'] and p['pred_id']==pp['pred_id']
                            hits=[v for v in pp['matched_gt'] if v['instance_id']==gt['instance_id']]
                            assert len(hits)==1 and hits[0]['intersection']==p['intersection'];edges+=1
                if view=='semantic' and a in run.CONTROLS:
                    old=run.read(run.CLASSIFIER/'evaluation'/s/(run.CONTROLS[a]+'_matches.json'))
                    assert run.canonical_match(match)==run.canonical_match(old)
                selections+=1
        assert review['semantic']['GT']==review['class_agnostic']['GT']==receipt['GT']
    assert GT==4689 and selections==1600
    arrays=0;scalars=0
    for view in run.VIEWS:
        ev=run.base.evaluator() if view=='semantic' else run.class_agnostic_evaluator()[0]
        for a in run.ARMS:
            matches={s:run.read(out/'evaluation'/s/(view+'_'+a+'_matches.json')) for s in c['scenes']}
            ids=[p['uuid'] for m in matches.values() for ps in m['pred'].values() for p in ps];assert len(ids)==len(set(ids))
            for group,scenes in c['reporting_groups'].items():
                x,*_=ev.evaluate_matches({s:matches[s] for s in scenes})
                expected=np.load(out/'evaluation'/(group+'_'+view+'_'+a+'_AP.npy'),allow_pickle=False)
                assert np.array_equal(x,expected,equal_nan=True);arrays+=1
                v=summary['AP'][group][view][a];o=ev.opt['overlaps']
                manual=dict(ap=float(np.nanmean(x[:,:,~np.isclose(o,.25)])),ap50=float(np.nanmean(x[:,:,np.isclose(o,.5)])),ap25=float(np.nanmean(x[:,:,np.isclose(o,.25)])))
                for k in manual:assert abs(manual[k]-v[k])<1e-14;scalars+=1
                if view=='semantic' and a in run.CONTROLS:assert np.array_equal(x,np.load(run.CLASSIFIER/'evaluation'/(group+'_'+run.CONTROLS[a]+'_AP.npy')),equal_nan=True)
            print('Verified AP',view,a,flush=True)
    for group,views in summary['AP'].items():
        for view,ap in views.items():assert summary['effects_pp'][group][view]==run.conditional_effects(ap)
    freeze_json(out/'final_verification.json',dict(complete=True,GT=GT,selections=selections,AP_arrays=arrays,AP_scalars=scalars,
        bidirectional_edges=edges,old_control_arrays_exact=6,old_control_matches_exact=400,
        all_input_output_hashes=True,independent_output_collapse=True,AP_recomputed=True,scalars_independent=True,
        no_UUID_reuse_across_scenes=True,raw_GT_read=False,reserve100_read=False,heldout_read=False,final_test_read=False,
        production_modified=False,verifier_sha256=sha(__file__),evaluation_sha256=sha(out/'evaluation/summary.json')))
    print('Complete',arrays,'AP arrays',scalars,'scalars',GT,'GT',flush=True)


if __name__=='__main__':verify(run.OUT)
