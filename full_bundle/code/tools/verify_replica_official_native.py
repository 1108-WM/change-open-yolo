#!/usr/bin/env python3
"""Verify sealed native masks are the author's exact sets, never inferred masks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '.local/replica_official_masks_full_20261011_v1'
MASKS = ROOT / '.local/official_openyolo3d_assets_20261011/extracted/replica_masks'


def main():
    scenes = json.loads((OUT / 'contract.json').read_text())['scenes']
    rows = []
    for scene in scenes:
        masks, scores = torch.load(MASKS / (scene + '.pt'), map_location='cpu', weights_only=False)
        masks = masks.bool().numpy()
        baseline = np.load(OUT / 'baseline' / scene / (scene + '_pred_masks.npy'), mmap_mode='r')
        np.testing.assert_array_equal(baseline, masks)
        official_hashes = {hashlib.sha256(row.tobytes()).hexdigest()
                           for row in np.packbits(masks, axis=0, bitorder='little').T}
        full = np.load(OUT / 'records' / scene / 'native_cache' / (scene + '_pred_masks.npy'), mmap_mode='r')
        full_hashes = {hashlib.sha256(row.tobytes()).hexdigest()
                      for row in np.packbits(full, axis=0, bitorder='little').T}
        assert full_hashes <= official_hashes, scene
        for method, path in [('baseline', OUT / 'baseline' / scene), ('Full', OUT / 'records' / scene / 'native_cache')]:
            classes = np.load(path / (scene + '_pred_classes.npy'))
            confidences = np.load(path / (scene + '_pred_scores.npy'))
            assert ((classes >= 0) & (classes <= 48)).all()
            assert np.isfinite(confidences).all() and ((confidences >= 0) & (confidences <= 1)).all()
        rows.append({'scene': scene, 'official_proposals': masks.shape[1],
                     'baseline_exact_author_masks_and_order': True, 'Full_native_members': full.shape[1],
                     'Full_native_unique': len(full_hashes), 'Full_native_only_exact_author_point_sets': True})
        print(scene, rows[-1], flush=True)
        del masks, baseline, full
    (OUT / 'native_domain_verified.json').write_text(json.dumps({'complete': True, 'GT_read': False,
        'no_new_prediction': True, 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
