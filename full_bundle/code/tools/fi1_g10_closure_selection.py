"""Strict full-dataset selection adapter; leaves frozen frontend source untouched."""
import json
from pathlib import Path

from tools.fi1_frontend_protocol import sha256_file

ROOT = Path(__file__).resolve().parents[3]
ROLES = {'train597_g10_backend': ('train', 597),
         'dev55_g10_backend': ('development_validation', 55),
         'val312_g10_backend': ('val312', 312)}
NATIVE_BINDING_ROLES = (*ROLES, 'additional_train199', 'additional_train198',
                        'additional_train_g10_pilot30')


def validate_selection(selection, expected):
    role = selection['dataset_role']
    if role not in ROLES:
        raise ValueError('unknown full dataset role')
    scenes = selection['scenes']
    if (len(scenes) != ROLES[role][1] or len(set(scenes)) != len(scenes)
            or scenes != sorted(expected) or selection['stride'] != 10
            or selection['active_arms'] != ['D']
            or selection['frontend'] != 'grounded_sam'):
        raise ValueError('full dataset identity/stride/arms differ')
    return selection


def fixed_selection(root):
    root = Path(root)
    selection = json.loads((root / 'selection.json').read_text())
    if selection.get('dataset_role') not in ROLES:
        from tools.run_fi1_frontend_comparison import fixed_selection as legacy
        return legacy(root)
    receipt = json.loads((root / 'selection_receipt.json').read_text())
    if not receipt['complete']:
        raise ValueError('selection receipt incomplete')
    for path, expected in receipt['inputs'].items():
        if sha256_file(path) != expected:
            raise ValueError('selection source changed: ' + path)
    for name in ('selection.json', 'scenes_execution_order.txt'):
        if sha256_file(root / name) != receipt['outputs'][name]:
            raise ValueError('selection output changed: ' + name)
    manifest_path = ROOT / '资料/训练597场景清单_20260917.json'
    val_path = ROOT / 'models/Mask3D/third_party/ScanNet/Tasks/Benchmark/scannetv2_val.txt'
    if not {str(manifest_path), str(val_path)} <= set(receipt['inputs']):
        raise ValueError('canonical train/development/validation lists must be bound')
    manifest = json.loads(manifest_path.read_text())
    group = ROLES[selection['dataset_role']][0]
    expected = val_path.read_text().split() if group == 'val312' else manifest['roles'][group]
    if (root / 'scenes_execution_order.txt').read_text().split() != selection['scenes']:
        raise ValueError('selection order differs')
    return validate_selection(selection, expected)
