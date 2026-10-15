#!/usr/bin/env python3
"""Append a reporting erratum; all prediction/evaluation and v1 seals stay intact."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_local_block_attribution_selection as run
D=run.diag;OUT=run.OUT


def main():
    old=D.read(OUT/'completion_verified.json');D.verify(old['outputs'])
    case=next(r for r in D.read(OUT/'evaluation/scene0688_00/blocks.json') if r['candidate_id']=='c006' and r['region']==1)
    single=case['C_single']['pool_delta']
    row=next(r for r in D.read(OUT/'evaluation/scene0688_00/unions.json') if r['candidate_id']=='c006' and r['mode']=='C')
    unknown=case['clipped_GT']['unknown'];total=case['clipped_GT']['points']
    original=(OUT/'report.md').read_text()
    bad=f'只处理这块且保留原父时，相对NT的增量{single:.6f}，仍不足0.02，目标点oracle约0.0678。'
    assert bad in original and single>=.02 and row['pool_delta']<.02
    corrected=(f'只处理这块且保留原父时，相对NT的增量{single:+.6f}，已超过0.02；'
        f'但新增点中{unknown}/{total}（{unknown/total:.2%}）为未知，不能视为安全成功。'
        f'实际固定同父C联合全部块的增量仅{row["pool_delta"]:+.6f}，未达0.02；目标点oracle约0.0678。')
    report=original.replace(bad,corrected).replace('日期：2026-10-15。完成；',
        '日期：2026-10-15。报告v2，修正scene0688单块与union解释；所有预测、GT统计及判定不变。完成；')
    path=OUT/'report_v2.md';assert not path.exists();path.write_text(report)
    erratum=dict(reason='Report v1 incorrectly says the +0.042515 single-block C gain is below .02. It is above .02; its 47/147 unannotated additions fail safety. The actual whole-parent C union gains +.011995, below .02.',
        corrected_report=str(path),original_report_sha256=D.sha(OUT/'report.md'),
        single_block_C_gain=single,union_C_gain=row['pool_delta'],single_block_unknown=unknown,
        single_block_points=total,predictions_changed=False,GT_labels_changed=False,
        rules_changed=False,decision_changed=False)
    D.seal(OUT/'report_errata.json',erratum)
    paths=[path,OUT/'report_errata.json',OUT/'completion_verified.json',OUT/'verification.json',OUT/'decision.json',Path(__file__)]
    for name in ('predictions_frozen.json','evaluation_frozen.json','analysis_frozen.json'):
        D.verify(D.read(OUT/name)['outputs']);paths.append(OUT/name)
    D.seal(OUT/'completion_verified_v2.json',dict(complete=True,
        supersedes_report_only=str(OUT/'report.md'),current_report=str(path),
        original_completion_sha256=D.sha(OUT/'completion_verified.json'),
        outputs={str(p.resolve()):D.sha(p) for p in paths},
        prediction_evaluation_seals_unchanged=True,original_completion_still_verifies=True,
        tests='3 passed',AP_run=False,training=False,production_modified=False,reading_entry_modified=False,github_push=False))
    print('reporting correction sealed; original seals intact',flush=True)


if __name__=='__main__':main()
