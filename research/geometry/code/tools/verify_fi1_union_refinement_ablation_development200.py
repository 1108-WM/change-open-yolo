#!/usr/bin/env python3
"""Verify all source interventions, both match directions, pooled AP and controls."""
from pathlib import Path
from collections import Counter
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools import run_fi1_union_refinement_ablation_development200 as run
from tools.fi1_frontend_protocol import freeze_json, sha256_file as sha
from tools.fi1_union_development_replay import Guard


def verify(out):
    c = run.read(out / 'contract.json'); guard = Guard(c); guard.gt_enabled = True; sys.addaudithook(guard.hook)
    summary = run.read(out / 'evaluation/summary.json')
    seal = run.read(out / 'predictions_frozen.json')
    for files in (c['code'], c['inputs'], c['label_inputs'], seal['outputs'], seal['scene_receipts'], summary['AP_arrays']):
        run.base.verify_files(files)
    GT = edges = selections = 0
    inventory = Counter(); details = []
    for s in c['scenes']:
        spec = c['bindings'][s]
        gs = run.read(spec['geometry']); sizes = {g['geometry_key']: g['point_count'] for g in gs}
        keep = set(run.read(c['jobs'][s]['gate'])['kept'])
        links = run.read(out / 'predictions' / s / 'refinement_links.json')
        reconstructed = run.refinement_links(gs, keep)
        assert [{k: v for k, v in x.items() if k != 'outputs'} for x in links] == reconstructed
        inventory['refinement_aliases'] += len(links)
        inventory['refinement_scenes'] += bool(links)
        inventory['unique_refinement_pointsets'] += len({x['geometry_key'] for x in links})
        inventory['refinement_only_pointsets'] += len({x['geometry_key'] for x in links if x['independent_geometry']})
        inventory['rejected_parent_aliases'] += sum(not x['parent_U_admitted'] for x in links)
        inventory['admitted_parent_aliases'] += sum(x['parent_U_admitted'] for x in links)
        receipt = run.read(out / 'evaluation' / s / 'receipt.json'); run.base.verify_files(receipt['outputs']); GT += receipt['GT']
        pr = run.read(out / 'predictions' / s / 'receipt.json')
        matches = {v: {} for v in run.VIEWS}
        for a in run.ARMS:
            pool = run.candidate_pool(gs, a, keep)
            # Independent alias expectations, including shared N/T/U-prime masks.
            expected = {}
            for g in gs:
                allowed = run.SOURCES.get(a, run.SOURCES['Full'] if a == 'screened_Full' else run.SOURCES['NTU'])
                aliases = [al for al in run.original_aliases(g) if al['source'] in allowed
                    and (not a.startswith('screened_') or al['source'] != 'pair_union' or g['geometry_key'] in keep)]
                if aliases: expected[g['geometry_key']] = aliases
            assert {g['geometry_key']: run.original_aliases(g) for g in pool} == expected
            assert pr['stats'][a]['geometry'] == len(pool)
            semantic = run.read(out / 'predictions' / s / ('semantic_' + a + '.json'))
            assert len(semantic) == 800 and semantic == run.base.rank(semantic)
            if a in run.CONTROLS: assert semantic == run.read(c['jobs'][s]['controls'][a])
            for link in links:
                assert link['outputs'][a]['present_in_pool'] == (link['geometry_key'] in expected)
                assert link['outputs'][a]['selected_classes'] == [m['class_index'] for m in semantic if m['geometry_key'] == link['geometry_key']]
            for view in run.VIEWS:
                selected = run.read(out / 'predictions' / s / (view + '_' + a + '.json'))
                if view == 'class_agnostic':
                    collapsed = {}
                    for m in sorted(semantic, key=lambda m: (-m['score'], m['class_index'])):
                        collapsed.setdefault(m['geometry_key'], dict(geometry_key=m['geometry_key'], score=m['score'], winning_class_index=m['class_index']))
                    assert selected == sorted(collapsed.values(), key=lambda m: (-m['score'], m['geometry_key']))
                match = run.read(out / 'evaluation' / s / (view + '_' + a + '_matches.json'))
                matches[view][a] = match
                pmap = {p['uuid']: p for ps in match['pred'].values() for p in ps}
                assert len(pmap) == sum(map(len, match['pred'].values())) == sum(sizes[m['geometry_key']] >= 100 for m in selected)
                for label, gs_ in match['gt'].items():
                    for gt in gs_:
                        for p in gt['matched_pred']:
                            pp = pmap[p['uuid']]
                            assert p['confidence'] == pp['confidence'] and p['pred_id'] == pp['pred_id']
                            hits = [v for v in pp['matched_gt'] if v['instance_id'] == gt['instance_id']]
                            assert len(hits) == 1 and hits[0]['intersection'] == p['intersection']; edges += 1
                if view == 'semantic' and a in run.CONTROLS:
                    old = run.read(run.CLASSIFIER / 'evaluation' / s / (run.CONTROLS[a] + '_matches.json'))
                    assert run.shared.canonical_match(match) == run.shared.canonical_match(old)
                selections += 1
        # Describe retained U-prime geometries and missing-parent pathways using
        # already sealed class-agnostic matches, no raw labels or new decisions.
        for link in links:
            row = dict(scene=s, **link, class_agnostic={})
            for a in ('NTUp', 'Full', 'screened_Full'):
                selected = run.read(out / 'predictions' / s / ('class_agnostic_' + a + '.json'))
                selected = [m for m in selected if sizes[m['geometry_key']] >= 100]
                ps = next(iter(matches['class_agnostic'][a]['pred'].values()))
                assert len(selected) == len(ps)
                hit = next((p for m, p in zip(selected, ps) if m['geometry_key'] == link['geometry_key']), None)
                row['class_agnostic'][a] = None if hit is None else dict(score=hit['confidence'],
                    best_iou=max((g['intersection'] / (g['vert_count'] + hit['vert_count'] - g['intersection'])
                                  for g in hit['matched_gt'] if g['instance_id'] >= 1000 and g['vert_count'] >= 100), default=0))
            details.append(row)
    assert GT == 4689 and selections == 2400
    arrays = scalars = 0
    for view in run.VIEWS:
        ev = run.base.evaluator() if view == 'semantic' else run.class_agnostic_evaluator()[0]
        for a in run.ARMS:
            matches = {s: run.read(out / 'evaluation' / s / (view + '_' + a + '_matches.json')) for s in c['scenes']}
            ids = [p['uuid'] for m in matches.values() for ps in m['pred'].values() for p in ps]
            assert len(ids) == len(set(ids))
            for group, scenes in c['reporting_groups'].items():
                x, *_ = ev.evaluate_matches({s: matches[s] for s in scenes})
                assert np.array_equal(x, np.load(out / 'evaluation' / (group + '_' + view + '_' + a + '_AP.npy')), equal_nan=True)
                arrays += 1
                overlaps = ev.opt['overlaps']
                manual = dict(ap=float(np.nanmean(x[:, :, ~np.isclose(overlaps, .25)])),
                    ap50=float(np.nanmean(x[:, :, np.isclose(overlaps, .5)])), ap25=float(np.nanmean(x[:, :, np.isclose(overlaps, .25)])))
                for k, v in manual.items(): assert abs(v - summary['AP'][group][view][a][k]) < 1e-14; scalars += 1
                if view == 'semantic' and a in run.CONTROLS:
                    assert np.array_equal(x, np.load(run.CLASSIFIER / 'evaluation' / (group + '_' + run.CONTROLS[a] + '_AP.npy')), equal_nan=True)
                if view == 'class_agnostic' and a in ('NT', 'screened_NTU'):
                    oldarm = 'A' if a == 'NT' else 'D'
                    assert np.array_equal(x, np.load(run.shared.OUT / 'evaluation' / (group + '_class_agnostic_' + oldarm + '_AP.npy')), equal_nan=True)
            print('Verified', view, a, flush=True)
    for group, views in summary['AP'].items():
        for view, ap in views.items(): assert summary['effects_pp'][group][view] == run.effects(ap)
    freeze_json(out / 'refinement_parent_review.json', dict(complete=True, inventory=dict(inventory),
        records=details, warning='Parent rejection is from the existing U-output gate, not a new U-prime quality label; IoU descriptions are not AP attribution.'))
    freeze_json(out / 'final_verification.json', dict(complete=True, GT=GT, selections=selections,
        AP_arrays=arrays, AP_scalars=scalars, bidirectional_edges=edges, old_semantic_control_arrays_exact=9,
        old_class_agnostic_control_arrays_exact=6, old_semantic_control_matches_exact=600,
        source_aliases_independent=True, parent_links_verified=True, prediction_stage_independent_features=True,
        all_input_output_hashes=True, independent_output_collapse=True, AP_recomputed=True, scalars_independent=True,
        no_UUID_reuse_across_scenes=True, raw_GT_read=False, reserve100_read=False, heldout_read=False,
        final_test_read=False, production_modified=False, verifier_sha256=sha(__file__), evaluation_sha256=sha(out / 'evaluation/summary.json')))
    print('Verified', arrays, 'AP arrays;', scalars, 'scalars;', GT, 'GT.', flush=True)


if __name__ == '__main__': verify(run.OUT)
