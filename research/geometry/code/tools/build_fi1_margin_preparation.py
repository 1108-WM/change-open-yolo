#!/usr/bin/env python3
"""Freeze representative samples, score actions and complete available evidence without GT."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fi1_gain_attribution_core import Bindings, read_rows, sha, unique, write_json, write_rows, points_from_population
from tools.fi1_evaluator_delta_core import load_population
from tools.fi1_scene_reference import validate_reference, validate_partition_requests, digest
from tools.fi1_margin_preparation import (
    VERSION, ARMS, VISUAL_FEATURES, VisualIndex, stream, protocol, order_key,
    representative_geometries, parent_index, candidates_from_selection,
    verify_quality, expanded_features, fit_score_anchors, action_requests, action_features,
)


def prepare(args):
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'run_state.json', {'status': 'running'})
    # The protocol is saved before reading evidence and before any possible GT labeling.
    write_json(out / 'protocol.json', protocol())
    bind = Bindings()
    try:
        refs = {}; predictions = {}; geometries = {}
        for fold in range(5):
            for role in ('fit', 'heldout'):
                folder = args.references_root / f'fold_{fold}' / role
                ref = json.loads(bind.check(folder / 'reference.json').read_text())
                validate_reference(ref)
                if ref['role'] != role or ref['outer_fold_index'] != fold:
                    raise ValueError('reference location/identity differs')
                refs[fold, role] = ref
                predictions[fold, role] = read_rows(bind.check(folder / 'predictions.jsonl', ref['predictions_sha256']))
                geometries[fold, role] = unique(read_rows(bind.check(folder / 'geometry_index.jsonl', ref['geometry_index_sha256'])), 'geometry_key')
        ref0 = refs[0, 'fit']
        if any(r['split_id'] != ref0['split_id'] or r['class_names'] != ref0['class_names'] or
               r['population_root'] != ref0['population_root'] for r in refs.values()):
            raise ValueError('references use different split/vocabulary/population')
        pop_root = Path(ref0['population_root']); population, ps = load_population(pop_root, bind)
        b1 = sorted(predictions[0, 'fit'] + predictions[0, 'heldout'], key=lambda r: r['prediction_id'])
        unique(b1, 'prediction_id')
        for (fold, role), rows in predictions.items():
            if rows != [r for r in b1 if r['scene_name'] in refs[fold, role]['reference_scenes']]:
                raise ValueError('B1 changed across references')
            for key, geo in geometries[fold, role].items():
                p = population[key]
                for field in ('points_sha256', 'point_count', 'points_file_sha256', 'scene_name', 'sources', 'is_new_geometry'):
                    if geo[field] != p[field]:
                        raise ValueError('reference/population identity differs')
        vs = json.loads(bind.check(args.visual_root / 'summary.json').read_text())
        schema = json.loads(bind.check(args.visual_root / 'feature_schema.json', vs['hashes']['feature_schema.json']).read_text())
        if schema['feature_names'] != list(VISUAL_FEATURES) or vs['class_count'] != 198:
            raise ValueError('visual schema differs')
        visual_path = bind.check(args.visual_root / 'class_scoring_features.jsonl', vs['hashes']['class_scoring_features.jsonl'])
        visual = VisualIndex(stream(visual_path), population, ref0['class_names'])
        print(f'Validated {len(visual.values)} complete visual geometries', flush=True)
        aliases = read_rows(bind.check(pop_root / 'refinement_aliases.jsonl', ps['hashes']['refinement_aliases.jsonl']))
        parents = parent_index(aliases, population)
        selected, inventory = representative_geometries(population, b1, visual, ref0['split'], protocol()['geometries_per_split_fold_source'])
        candidates = candidates_from_selection(selected, population, b1, visual, parents)
        write_rows(out / 'selected_geometries.jsonl', selected)
        write_rows(out / 'candidates.jsonl', candidates)
        write_json(out / 'sampling_inventory.json', inventory)
        needed = {r['geometry_key'] for r in selected}
        needed |= {a['parent_geometry_key'] for k in list(needed) for a in parents[k]}
        point_cache = {}
        def get_points(key):
            if key not in point_cache:
                point_cache[key] = points_from_population(pop_root, population[key], bind)
            return point_cache[key]
        for key in needed:
            get_points(key)
        metadata = read_rows(bind.check(pop_root / 'legacy_stage_a_quality_metadata.jsonl', ps['hashes']['legacy_stage_a_quality_metadata.jsonl']))
        source_path = args.stage_a_root.resolve() / 'oof_quality_predictions.jsonl'
        source_rows = read_rows(bind.check(source_path, ps['input_provenance'][str(source_path)]))
        quality = verify_quality(needed, metadata, source_rows, population, get_points)
        expanded = expanded_features(candidates, population, b1, visual, parents, quality, get_points)
        write_rows(out / 'candidate_evidence.jsonl', [{'prediction_id': pid, **row} for pid, row in expanded.items()])
        edges = read_rows(bind.check(pop_root / 'geometry_overlap_edges.jsonl', ps['hashes']['geometry_overlap_edges.jsonl']))
        manifests = []
        for fold in range(5):
            anchors = fit_score_anchors(predictions[fold, 'fit'], set(refs[fold, 'fit']['reference_scenes']))
            anchor_receipt = {'outer_fold_index': fold, 'fit_reference_id': refs[fold, 'fit']['reference_id'],
                              'fit_predictions_sha256': refs[fold, 'fit']['predictions_sha256'],
                              'fit_reference_scenes': refs[fold, 'fit']['reference_scenes'], 'anchors': anchors,
                              'gt_used': False}
            folder = out / f'fold_{fold}'; folder.mkdir()
            write_json(folder / 'score_anchors.json', anchor_receipt)
            for role in ('fit', 'heldout'):
                ref = refs[fold, role]; folder = out / f'fold_{fold}' / role; folder.mkdir()
                part = [c for c in candidates if c['scene_name'] in ref['reference_scenes']]
                smoke_ids = set()
                for source in protocol()['source_strata']:
                    group = [c for c in part if c['source_stratum'] == source]
                    if group:
                        key = min({c['geometry_key'] for c in group}, key=lambda k: order_key('smoke', k))
                        chosen = min((c for c in group if c['geometry_key'] == key), key=lambda c: order_key('smoke', c['prediction_id']))
                        smoke_ids.add(chosen['prediction_id'])
                for arm in ARMS:
                    requests = action_requests(part, arm, anchors)
                    validate_partition_requests(requests, predictions[fold, role], geometries[fold, role], ref)
                    features = action_features(requests, predictions[fold, role], edges, population, expanded)
                    name = arm + '.requests.jsonl'; feature_name = arm + '.features.jsonl'
                    write_rows(folder / name, requests); write_rows(folder / feature_name, features)
                    write_rows(folder / (arm + '.smoke.jsonl'), [r for r in requests if r['prediction_id'] in smoke_ids])
                    manifest = {'version': VERSION, 'reference_id': ref['reference_id'], 'split_id': ref['split_id'],
                        'outer_fold_index': fold, 'partition_role': role, 'score_arm': arm,
                        'reference_root': str((args.references_root / f'fold_{fold}' / role).resolve()),
                        'protocol_sha256': sha(out / 'protocol.json'),
                        'score_anchors_sha256': sha(out / f'fold_{fold}' / 'score_anchors.json'),
                        'requests_file': name, 'requests_sha256': sha(folder / name),
                        'features_file': feature_name, 'features_sha256': sha(folder / feature_name),
                        'candidate_evidence_sha256': sha(out / 'candidate_evidence.jsonl'),
                        'row_count': len(requests), 'gt_used_for_selection_or_features': False,
                        'historical_supervised_quality_used': True, 'whole_system_oof': False,
                        'labels_generated': False, 'training_ready': False, 'model_fitted': False}
                    write_json(folder / (arm + '.manifest.json'), manifest)
                    manifests.append({k: manifest[k] for k in ('outer_fold_index', 'partition_role', 'score_arm', 'row_count')})
        feature_names = sorted(next(iter(expanded.values()))['features']) if expanded else []
        if any(sorted(r['features']) != feature_names for r in expanded.values()):
            raise ValueError('inconsistent expanded feature schema')
        summary = {'version': VERSION, 'selected_geometry_count': len(selected), 'candidate_count': len(candidates),
            'selected_scene_count': len({c['scene_name'] for c in candidates}),
            'candidate_source_counts': dict(Counter(c['source_stratum'] for c in candidates)),
            'selected_source_counts': dict(Counter(s['source_stratum'] for s in selected)),
            'visual_geometry_count': len(visual.values), 'point_verified_geometry_count': len(point_cache),
            'quality_verified_geometry_count': sum(q is not None for q in quality.values()),
            'candidate_quality_present_count': sum(r['quality_metadata'] is not None for r in expanded.values()),
            'candidate_parent_present_count': sum(bool(r['parent_relations']) for r in expanded.values()),
            'evidence_feature_names': feature_names, 'manifest_counts': manifests,
            'input_sha256': bind.files, 'gt_cache_payloads_read': False, 'ap_arrays_read': False,
            'labels_generated': False, 'training_ready': False, 'model_fitted': False, 'production_ready': False,
            'whole_system_oof': False, 'region_semantics': 'whole child/parent supports plus exact retained/removed/added point fractions; no new region-only visual inference',
            'limitations': ['availability-conditioned pilot sample, not an unbiased full-population training sample',
                'new refined quality is missing; parent quality is separate and never substituted',
                'historical FI1/B1 supervision reused; downstream isolation only',
                'single-action targets non-additive; joint AP remains required', 'raw real-scene GT association not revalidated'],
            'code_sha256': {n: sha(ROOT / n) for n in ('tools/build_fi1_margin_preparation.py',
                'tools/fi1_margin_preparation.py', 'tools/fi1_scene_reference.py', 'tools/build_fi1_margin_dataset.py')}}
        summary['hashes'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run_state.json'}
        write_json(out / 'summary.json', summary)
        write_json(out / 'run_state.json', {'status': 'complete', 'summary_sha256': sha(out / 'summary.json')})
        print(json.dumps({k: summary[k] for k in ('selected_geometry_count', 'candidate_count', 'selected_scene_count', 'selected_source_counts', 'candidate_source_counts', 'quality_verified_geometry_count')}, indent=2))
    except Exception as exc:
        write_json(out / 'run_state.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('references-root', 'visual-root', 'stage-a-root', 'output-root'):
        p.add_argument('--' + name, type=Path, required=True)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
