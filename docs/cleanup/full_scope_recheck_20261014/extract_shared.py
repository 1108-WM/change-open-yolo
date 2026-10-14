"""Extract unchanged shared formulas and retire the obsolete CLI sources once."""
from pathlib import Path
import ast
import hashlib
import json
import os
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CODE = ROOT / 'full_bundle/code/tools'
ARCHIVE = ROOT / 'archive/retired_20261014/full_shared_formula_extraction.tar.gz'

RETIRED = [
    'construct_c1c_global_feasible_ap_oracle_gt',
    'diagnose_d2b_native_component_competition_oracle_gt',
    'diagnose_d2b_native_track_ranking_oracle_gt',
    'diagnose_mv3dis_global_feasible_action_oracle_gt',
    'diagnose_mv3dis_boundary_action_oracle_gt',
    'diagnose_n1_sampro3d_candidate_space_oracle_gt',
    'diagnose_n2_medoid_candidate_oracle_gt',
    'diagnose_gvc_class_agnostic_ap',
    'build_automatic_sam_track_growth_ledger',
    'export_sam_automatic_observations',
]
MAPPING = {
    'construct_c1c_global_feasible_ap_oracle_gt': 'fi1_instance_eval_shared',
    'diagnose_d2b_native_track_ranking_oracle_gt': 'fi1_instance_eval_shared',
    'diagnose_gvc_class_agnostic_ap': 'fi1_instance_eval_shared',
    'build_automatic_sam_track_growth_ledger': 'fi1_superpoint_geometry',
    'export_sam_automatic_observations': 'fi1_mask_projection',
}
SELECTIONS = {
    'fi1_instance_eval_shared': {
        'diagnose_gvc_class_agnostic_ap': [
            'UNIFIED_PREDICTED_CLASS', 'UNIFIED_GT_CLASS', 'VALID_GT_CLASSES',
            '_class_agnostic_gt_ids', '_configure_scannet200_instance_eval', '_merge_scan_matches',
        ],
        'diagnose_mv3dis_global_feasible_action_oracle_gt': ['_scene_ap_records', '_average_precision'],
        'diagnose_d2b_native_track_ranking_oracle_gt': ['_empty_record', '_append'],
        'construct_c1c_global_feasible_ap_oracle_gt': ['AP25', 'OFFICIAL', 'DIAGNOSTIC'],
    },
    'fi1_superpoint_geometry': {
        'build_automatic_sam_track_growth_ledger': ['_normalize_rgb', '_normalize_normals', '_raw_superpoint_context'],
    },
    'fi1_mask_projection': {
        'export_sam_automatic_observations': [
            'select_frame_indices', 'mask_to_visible_points', 'encode_binary_mask_rle',
            'decode_binary_mask_rle', 'exact_mask_relation', '_select_masks',
        ],
    },
}
HEADERS = {
    'fi1_instance_eval_shared': '''"""Shared instance matching/AP formulas for Full training and evaluation.

GT helpers belong to supervision or post-prediction evaluation; inference
callers do not invoke them. No oracle selection CLI is retained here.
"""
import numpy as np
from evaluate.scannet200 import eval_semantic_instance as instance_eval
''',
    'fi1_superpoint_geometry': '''"""Raw superpoint normalization and contact geometry used by frozen D-v3."""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
''',
    'fi1_mask_projection': '''"""Shared mask RLE, visible-point projection and fixed observation ordering."""
import numpy as np
''',
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def selected_nodes(stem, names):
    source = (CODE / (stem + '.py')).read_text()
    nodes = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name): nodes[target.id] = node
    return [(name, nodes[name], ast.get_source_segment(source, nodes[name])) for name in names]


def main():
    if ARCHIVE.exists(): raise SystemExit('Already archived; do not repeat extraction')
    updates = {}
    for base in ['full_bundle/code', 'sl_bundle/code', 'research/geometry/code']:
        for p in (ROOT/base).rglob('*.py'):
            if p.stem in RETIRED: continue
            old = p.read_text()
            new = old
            for before, after in MAPPING.items(): new = new.replace(before, after)
            if new != old: updates[p] = new
    originals = [CODE/(n+'.py') for n in RETIRED] + list(updates)
    test_roots = ['research/geometry/code/tests/test_export_sam_automatic_observations.py',
                  'research/geometry/code/tests/test_build_automatic_sam_track_growth_ledger.py',
                  'research/geometry/code/tests/test_diagnose_gvc_class_agnostic_ap.py']
    originals += [ROOT/p for p in test_roots if ROOT/p not in originals]
    before = {str(p.relative_to(ROOT)): sha(p) for p in originals}
    with tarfile.open(ARCHIVE, 'w:gz') as tar:
        for p in originals: tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        for name, digest in before.items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == digest
    formulas = []
    for dest, sources in SELECTIONS.items():
        pieces = [HEADERS[dest]]
        for stem, names in sources.items():
            for name, node, segment in selected_nodes(stem, names):
                pieces.append(segment)
                formulas.append(dict(original=stem+'.py', symbol=name, destination=dest+'.py',
                                     original_ast=ast.dump(node, include_attributes=False)))
        if dest == 'fi1_instance_eval_shared':
            pieces.append('def _record_from_matches(gt, pred, t):\n    return _scene_ap_records(gt, pred, t)')
        target = CODE/(dest+'.py')
        target.write_text('\n\n'.join(pieces)+'\n')
        tree = ast.parse(target.read_text())
        result_nodes = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef): result_nodes[node.name] = node
            elif isinstance(node, ast.Assign):
                for n in node.targets:
                    if isinstance(n, ast.Name): result_nodes[n.id] = node
        for row in formulas:
            if row['destination'] == target.name:
                assert row['original_ast'] == ast.dump(result_nodes[row['symbol']], include_attributes=False)
                row['ast_unchanged'] = True
                del row['original_ast']
        alias = ROOT/'tools'/target.name
        assert not alias.exists()
        alias.symlink_to(os.path.relpath(target, alias.parent))
    for p, text in updates.items(): p.write_text(text)
    # Keep only tests of retained formulas, move them alongside their actual scope.
    tests = [
        ('test_export_sam_automatic_observations', 'test_fi1_mask_projection', None),
        ('test_build_automatic_sam_track_growth_ledger', 'test_fi1_superpoint_geometry', 'test_growth_frontier_requires_cross_view_graph_evidence'),
        ('test_diagnose_gvc_class_agnostic_ap', 'test_fi1_instance_eval_shared', 'test_append_tracks_preserves_native_predictions'),
    ]
    for old_stem, new_stem, cutoff in tests:
        p = ROOT/'research/geometry/code/tests'/(old_stem+'.py')
        content = p.read_text()
        if cutoff:
            tree = ast.parse(content)
            if old_stem == 'test_build_automatic_sam_track_growth_ledger':
                first_retired = next(n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name==cutoff)
                content = '\n'.join(content.splitlines()[:first_retired-1])+'\n'
            else:
                n = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name==cutoff)
                content = '\n'.join(content.splitlines()[:n.lineno-1]+content.splitlines()[n.end_lineno:])+'\n'
        dest = ROOT/'full_bundle/code/tests'/(new_stem+'.py')
        dest.write_text(content)
        alias = ROOT/'tests'/(new_stem+'.py')
        alias.symlink_to(os.path.relpath(dest, alias.parent))
        p.unlink()
        old_alias = ROOT/'tests'/(old_stem+'.py')
        if old_alias.is_symlink(): old_alias.unlink()
    for stem in RETIRED:
        p = CODE/(stem+'.py')
        p.unlink()
        alias = ROOT/'tools'/p.name
        if alias.is_symlink(): alias.unlink()
        for q in [CODE/'__pycache__', ROOT/'tools/__pycache__']:
            if q.is_dir():
                for cache in q.glob(stem+'.*.pyc'): cache.unlink()
    result = dict(complete=True, archive=str(ARCHIVE.relative_to(ROOT)), original_hashes=before,
                  retired=[str((CODE/(n+'.py')).relative_to(ROOT)) for n in RETIRED],
                  new_modules=[str((CODE/(n+'.py')).relative_to(ROOT)) for n in SELECTIONS],
                  formulas=formulas, modified_callers=[str(p.relative_to(ROOT)) for p in updates],
                  retired_tests=test_roots, formulas_ast_unchanged=True)
    (OUT/'shared_extraction_executed.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(retired=len(RETIRED), new_modules=len(SELECTIONS), unchanged_formulas=len(formulas), modified_callers=len(updates))))


if __name__ == '__main__': main()
