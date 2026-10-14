#!/usr/bin/env python3
"""Reuse each worker's mutable audit scope; no partition/diagnosis change."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from tools import run_fi1_local_region_supply as run
BaseGuard=run.Guard
_WORKER_GUARD=None


def shared_guard(c,mode,fold=None):
    global _WORKER_GUARD
    if _WORKER_GUARD is None:_WORKER_GUARD=BaseGuard(c,mode,fold)
    else:_WORKER_GUARD.c=c;_WORKER_GUARD.mode=mode;_WORKER_GUARD.fold=fold
    # Existing audit callbacks all reference this same mutable object. Python
    # audit hooks cannot be removed when a ProcessPool worker receives a new job.
    return _WORKER_GUARD


run.Guard=shared_guard


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('diagnose','verify'));mode=parser.parse_args().mode
    amendment=run.read(run.OUT/'execution_amendment_v2.json');run.old.previous.original.audit.prior.old.verify_files(amendment['code'])
    if mode=='diagnose':
        from tools.diagnose_fi1_local_region_supply import diagnose
        diagnose()
    else:
        from tools.verify_fi1_local_region_supply import verify
        verify()
