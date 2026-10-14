"""Independent final verification of U factorial selections, AP and descriptions."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT))
import numpy as np
from tools.run_fi1_union_loss_attribution_train90 import OLD, OUT, Guard, read, rows, base
from tools.fi1_frontend_protocol import sha256_file as sha, freeze_json


def verify(out):
    c = read(out / 'contract.json'); done = read(out / 'completion_verified.json'); assert done['complete']
    sys.addaudithook(Guard(c['scenes'], [s['gt_path'] for s in c['bindings'].values()]).hook)
    base.verify_files(c['code']); base.verify_files({c['B_model']: c['B_model_sha256']})
    seal = read(out / 'predictions_frozen.json'); assert not seal['GT_read'] and seal['scene_arm_count'] == 360
    base.verify_files(seal['outputs']); summary = read(out / 'evaluation/summary.json')
    assert sha(out / 'evaluation/summary.json') == done['summary_sha256']; base.verify_files(summary['AP_arrays'])
    uids = {a: set() for a in ('A', 'B', 'C', 'D')}; NT = NTU = checks = GT = 0
    def k(m): return m['geometry_key'], m['class_index']
    for scene in c['scenes']:
        d = out / 'scenes' / scene; pr = read(d / 'receipt.json'); er = read(out / 'evaluation' / scene / 'receipt.json')
        base.verify_files(er['outputs']); NT += pr['NT_members']; NTU += pr['NTU_members']; checks += er['intersection_checks']; GT += er['GT']
        m0 = rows(OLD / 'scenes' / scene / 'nt_members.jsonl'); m1 = rows(d / 'ntu_members.jsonl')
        s0 = np.load(OLD / 'scenes' / scene / 'nt_scores.npy', allow_pickle=False); s1 = np.load(d / 'ntu_scores.npy', allow_pickle=False)
        old = {k(m): (m, float(s)) for m, s in zip(m0, s0)}; new = {k(m): (m, float(s)) for m, s in zip(m1, s1)}
        assert len(old) == len(m0) and len(new) == len(m1) and set(old) <= set(new)
        diag = read(out / 'evaluation' / scene / 'diagnosis.json')
        for a in uids:
            expected = []
            for identity, (m, score) in (old.items() if a in ('A', 'C') else new.items()):
                if a == 'B' and identity in old: m, score = old[identity]
                elif a == 'C': score = new[identity][1]
                expected.append({**m, 'score': score})
            expected.sort(key=lambda m: (-m['score'], not m['has_original_native_alias'], m['geometry_key'], m['class_index']))
            selected = read(d / (a + '_selected.json')); assert selected == expected[:800] and len({k(m) for m in selected}) == 800
            match = read(out / 'evaluation' / scene / (a + '_matches.json'))
            current = [p['uuid'] for ps in match['pred'].values() for p in ps]
            assert len(current) == len(set(current)) and not set(current) & uids[a]; uids[a].update(current)
            assert sum(g['instance_id'] >= 1000 and g['vert_count'] >= 100 for gs in match['gt'].values() for g in gs) == er['GT']
            allrows = diag['arms'][a]['all']; assert allrows['members'] == 800 and len(current) == 800-allrows['small_ignored']
            for t in ('.5', '.75', '.9'):
                t = '0'+t; desc = allrows['thresholds'][t]
                assert sum(desc[n] for n in ('correct_range', 'wrong_class_good_range', 'no_range_at_threshold')) == len(current)
                assert desc['redundant_correct_members'] == desc['correct_range']-desc['unique_correct_GT'] >= 0
        for name, before, after in [('B_minus_A','A','B'),('C_minus_A','A','C'),('D_minus_C','C','D'),('D_minus_B','B','D'),('D_minus_A','A','D')]:
            for t in ('0.5','0.75','0.9'):
                cover = diag['comparisons'][name]['coverage'][t]
                assert diag['arms'][after]['all']['thresholds'][t]['unique_correct_GT']-diag['arms'][before]['all']['thresholds'][t]['unique_correct_GT'] == cover['gained_GT']-cover['lost_GT']
    assert GT == summary['GT'] == 2001 and NT == seal['NT_score_rows_checked'] and NTU == seal['NTU_feature_context_rows_checked']
    assert checks == summary['independent_GT_intersections']
    for group in (*c['batches'], 'combined90'):
        for a in uids:
            x = np.load(out / 'evaluation' / (group+'_'+a+'_AP.npy'), allow_pickle=False); assert x.shape == (1,198,10)
            values = dict(ap=float(np.nanmean(x[0,:,:9])), ap50=float(np.nanmean(x[0,:,0])), ap25=float(np.nanmean(x[0,:,9])))
            assert all(abs(values[n]-summary['AP'][group][a][n]) < 1e-14 for n in values)
            if a == 'A': assert np.array_equal(x, np.load(OLD / 'evaluation' / (group+'_nt_only_AP.npy'), allow_pickle=False), equal_nan=True)
        for name, before, after in [('B_minus_A','A','B'),('C_minus_A','A','C'),('D_minus_C','C','D'),('D_minus_B','B','D'),('D_minus_A','A','D')]:
            for metric in ('ap','ap50','ap25'):
                calc = 100*(summary['AP'][group][after][metric]-summary['AP'][group][before][metric])
                assert abs(calc-summary['differences_pp'][group][name][metric]) < 1e-14
        for metric in ('ap','ap50','ap25'):
            ap = summary['AP'][group]; calc = 100*(ap['D'][metric]-ap['B'][metric]-ap['C'][metric]+ap['A'][metric])
            assert abs(calc-summary['differences_pp'][group]['interaction_D_minus_B_minus_C_plus_A'][metric]) < 1e-14
    freeze_json(out / 'final_verification.json', dict(complete=True, scenes=90, predictions=360, GT=GT,
        independently_resorted_all_prebudget_members=True, NT_score_rows=NT, NTU_score_rows=NTU,
        A_parent_AP_arrays_exact=True, AP_scalar_recalculations=36, effect_scalar_recalculations=54,
        coverage_algebra_and_diagnostic_partitions_verified=True, all_code_model_prediction_evaluation_hashes_verified=True,
        no_UUID_reuse_across_scans=True, evaluated_predictions={a:len(v) for a,v in uids.items()},
        GT_payload_read_by_checker=False, heldout_read=False, final_test_read=False,
        summary_sha256=sha(out / 'evaluation/summary.json'), checker_sha256=sha(__file__)))
    print('Independent four-group selections,36AP scalars,54effects and all receipts passed.', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, default=OUT); a=p.parse_args(); verify(a.output.resolve())
