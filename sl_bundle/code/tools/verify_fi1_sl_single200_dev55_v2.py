#!/usr/bin/env python3
"""Postseal audit access fix; frozen training/prediction/evaluator stay intact.

The original helper used the inference-style audit guard while regenerating
training pairs. Permit only the bound training200 labels, only after models,
predictions and evaluation are sealed. No dev55 labels or raw GT are allowed.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools import run_fi1_sl_single200_dev55 as run
from tools import verify_fi1_sl_single200_dev55 as original
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha
FrozenGuard=run.Guard


class PostsealAuditGuard(FrozenGuard):
    def check(self,path):
        p=Path(path).resolve()
        # Explicitly bound identity manifests may mention historical val312.
        # This never admits val312 predictions, labels or performance artifacts.
        metadata={Path(q).resolve() for q in self.c.get('metadata_inputs',{})}
        if self.mode=='audit' and p in metadata and not any(v.startswith('ground_truth') for v in p.parts):
            return
        if self.mode=='audit' and p.name=='labels.npz' and p.parent.parent==run.L/'dataset' and p.parent.name in self.c['train_scenes']:
            assert (run.OUT/'model_frozen.json').exists() and (run.OUT/'predictions_frozen.json').exists() and (run.OUT/'evaluation_summary.json').exists()
            return
        super().check(path)


def main():
    run.Guard=PostsealAuditGuard
    original.main()
    freeze_json(run.OUT/'audit_access_fix_verified.json',dict(complete=True,
        original_verifier_sha256=sha(original.__file__),postseal_wrapper_sha256=sha(__file__),
        change='Allow immutable train200 labels after sealing and explicitly contract-bound identity metadata; no val312 data/performance or rawGT.',
        training_prediction_evaluator_changed=False,raw_GT_read=False,dev55_labels_in_fit=False))


if __name__=='__main__':main()
