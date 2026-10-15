#!/usr/bin/env python3
"""Canonicalize allowlist only; preserve original sealed prediction code/data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_t_add_downstream as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha
BaseGuard=run.ref.Guard
class CanonicalGuard(BaseGuard):
    def __init__(self,c,labels=False):
        super().__init__({**c,'GT_inputs':{str(Path(p).resolve()):h for p,h in c['GT_inputs'].items()}},labels)
if __name__=='__main__':
    c=run.read(run.OUT/'contract.json')
    if not (run.OUT/'evaluation_adapter_frozen.json').exists():
        freeze_json(run.OUT/'evaluation_adapter_frozen.json',dict(code={str(Path(__file__).resolve()):sha(Path(__file__))},reason='GT allowlist needs same canonical-path normalization as audit hook; failed previous evaluation before GT opened',prediction_seal_sha256=sha(run.OUT/'predictions_frozen.json'),predictions_changed=False,thresholds_changed=False))
    run.ref.Guard=CanonicalGuard
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):run.evaluate(c)
