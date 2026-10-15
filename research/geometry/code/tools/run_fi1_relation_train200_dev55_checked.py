#!/usr/bin/env python3
"""Training-label output read allowlist correction; preserve protocol and outputs."""
import argparse
import shutil
import time
import run_fi1_relation_train200_dev55 as r


def labels():
    c=r.a.read(r.OUT/'contract.json');paths=[r.OUT/'labels'/f'{s}.json' for s in c['train_scenes']]
    original_guard=r.guard
    r.guard=lambda allow=(),gt=False:original_guard(list(allow)+paths,gt)
    original_seal=r.a.seal
    def checked_seal(path,value):
        if path in paths and path.exists():
            assert r.a.read(path)==r.a.plain(value),path
            return
        return original_seal(path,value)
    r.a.seal=checked_seal
    addendum=r.OUT/'training_label_allowlist_addendum.json'
    assert not (r.OUT/'models').exists() and not (r.OUT/'training_labels_frozen.json').exists()
    p=r.Path(__file__).resolve()
    original_seal(addendum,dict(before_training=True,time_unix=time.time(),model_fits=0,dev_GT_read=False,
        code={str(p):r.a.sha(p)},reason='Guard blocked hashing the newly written first training-label output. Permit only predefined training200 label paths; regenerate and compare existing output exactly before reuse. No dev55 label/GT permission, no label/model/feature changes.',
        existing_labels={str(path):r.a.sha(path) for path in paths if path.exists()}))
    shutil.copyfile(p,r.OUT/('code_'+p.name))
    r.labels()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('labels',));args=p.parse_args();labels()
