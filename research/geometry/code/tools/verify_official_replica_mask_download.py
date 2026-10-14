#!/usr/bin/env python3
"""Extract and verify author's Replica masks without reading GT or evaluating."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[4]
ASSET = ROOT / '.local/official_openyolo3d_assets_20261011'
SCENES = ['office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    archive = ASSET / 'download/OpenYOLO3D.zip'
    destination = ASSET / 'extracted/replica_masks'
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        inventory = [{'name': i.filename, 'size': i.file_size,
                      'compressed_size': i.compress_size, 'crc32': f'{i.CRC:08x}'}
                     for i in zipped.infolist()]
        (ASSET / 'archive_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        members = {}
        for scene in SCENES:
            matches = [i for i in zipped.infolist()
                       if i.filename.endswith('/output/replica/replica_masks/' + scene + '.pt')
                       or i.filename == 'output/replica/replica_masks/' + scene + '.pt']
            assert len(matches) == 1, (scene, [i.filename for i in matches])
            members[scene] = matches[0]
        assert shutil.disk_usage(ROOT).free - sum(i.file_size for i in members.values()) > 50 * 1024**3
        rows = []
        for scene, member in members.items():
            path = destination / (scene + '.pt')
            assert not path.exists(), f'Refusing to overwrite: {path}'
            with zipped.open(member) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
            # Reading each member to EOF verifies its ZIP CRC.
            value = torch.load(path, map_location='cpu', weights_only=False)
            assert isinstance(value, (tuple, list)) and len(value) == 2, (scene, type(value))
            masks, scores = value
            assert isinstance(masks, torch.Tensor) and isinstance(scores, torch.Tensor)
            point_count = len(PlyData.read(ROOT / 'data/replica' / scene / (scene + '_mesh.ply'))['vertex'].data)
            assert masks.ndim == 2 and masks.shape[0] == point_count, (scene, masks.shape, point_count)
            assert scores.ndim == 1 and masks.shape[1] == len(scores)
            assert torch.isfinite(scores).all() and torch.isfinite(masks).all()
            if masks.dtype != torch.bool:
                assert ((masks == 0) | (masks == 1)).all()
            counts = masks.count_nonzero(dim=0)
            row = {'scene': scene, 'archive_member': member.filename,
                   'path': str(path), 'sha256': sha(path), 'points': point_count,
                   'proposals': len(scores), 'mask_dtype': str(masks.dtype),
                   'min_proposal_points': int(counts.min()), 'max_proposal_points': int(counts.max()),
                   'score_range': [float(scores.min()), float(scores.max())],
                   'CRC_verified': True, 'point_domain_count_matches': True}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del value, masks, scores
    receipt = {'complete': True, 'source': 'official upstream scripts/get_class_agn_masks.sh',
               'drive_file_id': '1FneLaYaClWDO51L9lIvlTQbheh5SfOFD',
               'url': 'https://drive.google.com/file/d/1FneLaYaClWDO51L9lIvlTQbheh5SfOFD/view',
               'archive': str(archive), 'archive_size': archive.stat().st_size,
               'archive_sha256': sha(archive), 'extracted_directory': str(destination),
               'GT_read': False, 'new_prediction_or_evaluation': False,
               'point_order_note': 'Full point counts verified against original meshes; mask files contain no XYZ, so count alone is not an independent proof of index order.',
               'scenes': rows}
    (ASSET / 'verified.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print('All eight official Replica mask files extracted and verified.', flush=True)


if __name__ == '__main__':
    main()
