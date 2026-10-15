#!/usr/bin/env python3
"""Allow the new label stage's explicit output paths through the frozen guard."""
import argparse
from pathlib import Path
import run_fi1_relation_training_scale as r


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    c=r.contract()
    addendum=r.a.read(r.OUT/'label_writer_addendum_before_training.json')
    r.verify(addendum['code'])
    root=r.OUT/('pilot_data' if args.pilot else 'training_data')
    scenes=[r.a.read(r.OUT/'pilot_geometry_frozen.json')['scene']] if args.pilot else c['extra_scenes']
    outputs=[root/'labels'/f'{s}.json' for s in scenes]
    original=r.guard
    def guard(allowed_labels=(),gt=False):
        assert not gt
        original(allowed_labels=list(allowed_labels)+outputs,gt=False)
    r.guard=guard
    r.labels(args.pilot)


if __name__=='__main__':main()
