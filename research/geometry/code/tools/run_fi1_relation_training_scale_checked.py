#!/usr/bin/env python3
"""Execute the same scale protocol using GT-free source-checked training assets."""
import argparse
import copy
from pathlib import Path
import shutil
import time
import run_fi1_relation_training_scale as r

PREDECESSOR=r.OUT
r.OUT=r.a.EXP/'fi1_relation_training_scale_exposed52_20261015_v2'


def prepare():
    old=r.a.read(PREDECESSOR/'contract.json')
    r.guard(allowed_labels=[PREDECESSOR/'pilot_data/labels'/f'{old["scenes"][0]}.json'])
    r.a.reserve();assert not r.OUT.exists()
    check=r.a.read(PREDECESSOR/'checked_training_assets.json')
    abort=r.a.read(PREDECESSOR/'aborted_before_training.json')
    assert abort['model_fits']==0 and not abort['new_training_GT_read'] and check['GT_used'] is False
    r.verify(abort['outputs']);r.verify(check['outputs']);r.verify(old['code']);r.verify(old['source_files'])
    c=copy.deepcopy(old);c.update(version=r.OUT.name,time_unix=time.time(),extra_scenes=check['selected'])
    c['predecessor']=dict(path=str(PREDECESSOR),reason=abort['reason'],new_GT_before_abort=False,model_fits_before_abort=0)
    c['jobs']={q['scene']:q['spec'] for q in check['checked'] if q['eligible']}
    for split in c['splits'].values():split['double']=split['original']+c['extra_scenes'][:len(split['original'])]
    c['source_files'].update({str(PREDECESSOR/name):r.a.sha(PREDECESSOR/name) for name in ('checked_training_assets.json','aborted_before_training.json','pilot_parity.json')})
    c['source_files'].update(check['outputs'])
    c['code'][str(Path(__file__).resolve())]=r.a.sha(__file__)
    c['asset_eligibility']='Same original candidate/features/labels. Require original feature assertions to pass on all used views. Exclude source failures before any new labels or fitting, preserve original physical hash order, no frame deletion/visibility repair or threshold relaxation.'
    r.a.seal(r.OUT/'contract.json',c)
    inv=r.a.read(PREDECESSOR/'inventory.json')
    invalid={q['scene']:q['reason'] for q in check['checked'] if not q['eligible']}
    for q in inv['scenes']:
        if q['scene'] in invalid:q.update(eligible=False,reason=invalid[q['scene']])
    inv.update(selected=c['extra_scenes'],available=sum(q['eligible'] for q in inv['scenes']),asset_check=str(PREDECESSOR/'checked_training_assets.json'))
    r.a.seal(r.OUT/'inventory.json',inv)
    for q in check['checked']:
        if not q['eligible']:continue
        for key,target in (('geometry_dir',r.OUT/'training_geometry/predictions'/q['scene']),('features_dir',r.OUT/'feature_cache/features'/q['scene'])):
            target.mkdir(parents=True)
            for p in Path(q[key]).iterdir():shutil.copyfile(p,target/p.name)
    r.a.seal(r.OUT/'pilot_parity.json',dict(**r.a.read(PREDECESSOR/'pilot_parity.json'),
        original_record=str(PREDECESSOR/'pilot_parity.json'),original_sha256=r.a.sha(PREDECESSOR/'pilot_parity.json')))
    for path in c['code']:
        p=Path(path);shutil.copyfile(p,r.OUT/('code_'+p.name))
    r.a.seal(r.OUT/'label_writer_addendum_before_training.json',dict(before_any_model_fitting=True,
        code={str(Path(__file__).resolve()):r.a.sha(__file__)},reason='Explicit new-scene output label paths added to frozen guard allowlist; only selected training GT readable.'))
    (r.OUT/'tmp').mkdir()
    print('CHECKED_PROTOCOL',len(c['extra_scenes']),'new sites',len(invalid),'source failures excluded',flush=True)


def features():
    r.guard();c=r.contract();seal=r.a.read(r.OUT/'training_geometry_frozen.json');r.verify(seal['outputs'])
    geometry=r.OUT/'training_geometry';root=r.OUT/'training_data'
    r.a.seal(geometry/'predictions_frozen.json',seal);r.a.seal(root/'contract.json',dict(jobs=c['jobs']))
    cache=r.OUT/'feature_cache';reused={str(p):r.a.sha(p) for p in cache.rglob('*') if p.is_file()}
    r.visual.OUT,r.visual.SOURCE,r.visual.PREDECESSOR,r.a.OUT=root,geometry,cache,root
    r.visual.guard=lambda *args,**kwargs:None
    paircount=sum(len(r.a.read(geometry/'predictions'/s/'pairs.json')) for s in c['extra_scenes'])
    r.visual.contract=lambda:dict(scenes=c['extra_scenes'],candidate_count=paircount,reused_GT_free_feature_files=reused)
    r.visual.features()


def labels():
    c=r.contract();original=r.guard
    outputs=[r.OUT/'training_data/labels'/f'{s}.json' for s in c['extra_scenes']]
    def guard(allowed_labels=(),gt=False):
        assert not gt
        original(allowed_labels=list(allowed_labels)+outputs,gt=False)
    r.guard=guard;r.labels()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','generate','features','labels','fit','predict','evaluate','verify'))
    p.add_argument('--fold',type=int,choices=range(5));args=p.parse_args()
    if args.mode in ('prepare','features','labels'):globals()[args.mode]()
    elif args.mode=='fit':assert args.fold is not None;r.fit(args.fold)
    elif args.mode=='evaluate':
        import evaluate_fi1_relation_training_scale as e
        e.main()
    elif args.mode=='verify':
        import verify_fi1_relation_training_scale as v
        v.main()
    else:getattr(r,args.mode)()
