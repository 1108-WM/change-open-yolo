#!/usr/bin/env python3
"""Dispatch amendment only: features calls the existing extract_features function."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from tools import run_fi1_candidate_local_repair as run
MODES={name:getattr(run,'extract_features' if name=='features' else name) for name in ('plan','features','labels','train','predict','evaluate')}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=MODES);a=p.parse_args()
    amendment=run.read(run.OUT/'execution_amendment_v2.json')
    run.audit.prior.old.verify_files(amendment['code'])
    assert run.sha(run.OUT/'contract.json')==amendment['original_contract_sha256']
    MODES[a.mode]()

if __name__=='__main__':main()
