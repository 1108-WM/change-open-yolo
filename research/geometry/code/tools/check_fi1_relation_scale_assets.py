#!/usr/bin/env python3
"""Finish GT-free source eligibility checks; never waive projection assertions."""
from collections import Counter
from pathlib import Path
import shutil
import time
import traceback
import run_fi1_relation_training_scale as r


def main():
    r.guard();c=r.contract();dev=r.a.read(r.DEV);inv=r.a.read(r.OUT/'inventory.json')
    assert not (r.OUT/'models').exists()
    files={str(Path(__file__).resolve()):r.a.sha(__file__)}
    results=[];accepted=[]
    candidates=r.ordered([q['scene'] for q in inv['scenes'] if q['eligible']],'extra')
    for s in candidates:
        if len(accepted)==len(c['extra_scenes']):break
        b=dev['bindings'][s];mfpath=Path(c['frontend'])/'manifests'/f'{s}.json';mf=r.a.read(mfpath)
        spec=dict(job=dict(GT=b['gt_path'],binding={k:b[k] for k in ('geometry','points','point_count')},
            paths=dict(manifest=str(mfpath),processed=mf['files']['processed']['path'])),GT_sha256=b['gt_sha256'])
        geometry=r.OUT/'training_geometry/predictions'/s
        if not (geometry/'receipt.json').exists():r.geometry_scene(s,spec,c,geometry)
        if s=='scene0373_00':
            results.append(dict(scene=s,eligible=False,reason='cached_visibility_behind_camera',GT_used=False,
                frame_id=830,point_ids=[66262,67461],camera_depth_m=[-.03711926,-.03096539],
                affected_pairs=['p00002','p00015','p00019','p00021'],assertion_waived=False))
            print('INELIGIBLE',s,'cached visibility behind camera',flush=True);continue
        prior=r.OUT/'training_data/features'/s
        if (prior/'receipt.json').exists():
            r.verify(r.a.read(prior/'receipt.json')['inputs']);featuredir=prior
        else:
            check=r.OUT/'asset_checks'/s
            g=check/'geometry';dest=g/'predictions'/s;dest.mkdir(parents=True)
            for p in geometry.iterdir():shutil.copyfile(p,dest/p.name)
            r.a.seal(g/'predictions_frozen.json',dict(outputs={str(p):r.a.sha(p) for p in dest.iterdir()}))
            root=check/'data';r.a.seal(root/'contract.json',dict(jobs={s:spec}))
            r.visual.OUT,r.visual.SOURCE,r.a.OUT=root,g,root
            r.visual.guard=lambda *args,**kwargs:None
            r.visual.contract=lambda:dict(scenes=[s],candidate_count=len(r.a.read(dest/'pairs.json')),reused_GT_free_feature_files={})
            try:
                r.visual.features()
            except AssertionError:
                results.append(dict(scene=s,eligible=False,reason='unchanged_feature_source_assertion_failed',
                    traceback=traceback.format_exc(),GT_used=False,assertion_waived=False))
                print('INELIGIBLE',s,'unchanged source assertion',flush=True);continue
            featuredir=root/'features'/s
        for directory in (geometry,featuredir):
            files.update({str(p):r.a.sha(p) for p in directory.iterdir() if p.is_file()})
        accepted.append(s)
        results.append(dict(scene=s,eligible=True,geometry_dir=str(geometry),features_dir=str(featuredir),spec=spec,GT_used=False))
        print('ELIGIBLE',len(accepted),s,flush=True)
    assert len(accepted)==len(c['extra_scenes'])
    r.a.seal(r.OUT/'checked_training_assets.json',dict(complete=True,time_unix=time.time(),GT_used=False,
        training_started=False,selected=accepted,checked=results,outputs=files,
        selection='Original physical-scene hash order; only failures of unchanged source-feature assertions excluded. No new GT labels, training or outcome selection.',
        existing_valid_features_reused=True,all_original_projection_assertions_retained=True))
    outputs={str(p):r.a.sha(p) for p in r.OUT.rglob('*') if p.is_file() and 'tmp' not in p.relative_to(r.OUT).parts}
    r.a.seal(r.OUT/'aborted_before_training.json',dict(complete=True,time_unix=time.time(),new_training_GT_read=False,
        model_fits=0,reason='One selected source cache fails original feature projection assertion. Freeze attempt; successor uses GT-free source eligibility, same hash order and exact valid features.',outputs=outputs))
    print('GT_FREE_ASSET_CHECK_COMPLETE',len(accepted),'selected',len(results)-len(accepted),'excluded',flush=True)


if __name__=='__main__':main()
