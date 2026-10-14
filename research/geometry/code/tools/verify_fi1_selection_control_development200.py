#!/usr/bin/env python3
"""Independent sorted-point intersections/selection, original AP and receipts."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from tools import run_fi1_selection_control_development200 as run
from tools.fi1_union_development_replay import Guard
from tools.fi1_frontend_protocol import freeze_json, write_json as write, sha256_file as sha


def independently_select(members, points, budget=800):
    """No production rank, sparse intersection, or NMS function is called."""
    keys = sorted(points)
    sizes = {k: len(points[k]) for k in keys}
    near, pairs, intersections = defaultdict(set), [], 0
    for i, left in enumerate(keys):
        for right in keys[i+1:]:
            lo, hi = sorted((sizes[left], sizes[right]))
            # Exact necessary size bound; equality .95 is never a conflict.
            if hi == 0 or 20*lo <= 19*hi:
                continue
            intersection = int(np.intersect1d(points[left], points[right], assume_unique=True).size)
            intersections += 1
            union = sizes[left] + sizes[right] - intersection
            if union and 20*intersection > 19*union:
                near[left].add(right); near[right].add(left)
                pairs.append(dict(left=left, right=right, intersection=intersection,
                                  union=union, iou=intersection/union))
    ordered = sorted(members, key=lambda m: (-m['score'], not m['has_original_native_alias'],
                                             m['geometry_key'], m['class_index']))
    kept, discarded, class_kept = [], [], defaultdict(list)
    examined = 0
    for position, m in enumerate(ordered, 1):
        examined += 1
        witness = None
        for old in class_kept[m['class_index']]:
            if old['geometry_key'] in near[m['geometry_key']]:
                witness = old
                break
        if witness is None:
            kept.append(m); class_kept[m['class_index']].append(m)
            if len(kept) == budget: break
        else:
            pair = next(p for p in pairs if {p['left'], p['right']} == {m['geometry_key'], witness['geometry_key']})
            discarded.append(dict(member=m, witness=witness, input_rank=position,
                                  intersection=pair['intersection'], union=pair['union'], iou=pair['iou']))
    return ordered[:budget], kept, discarded, pairs, examined, intersections


def selection(out):
    c = run.read(out / 'contract.json'); guard = Guard(c); sys.addaudithook(guard.hook)
    seal = run.read(out / 'predictions_frozen.json')
    for files in (c['inputs'], c['code'], seal['outputs'], seal['scene_receipts']): run.base.verify_files(files)
    start = time.monotonic()
    totals = Counter()
    for n, s in enumerate(c['scenes'], 1):
        run.stage(out, 'independent_selection_verification', scene=s, completed=n-1, total=200)
        spec, job = c['bindings'][s], c['jobs'][s]
        gs, all_points = run.shared.load_points(spec)
        saved_pool = run.read(job['nt_geometry.json'])
        expected = {g['geometry_key']: [a for a in run.original_aliases(g) if a['source'] in ('native', 'track')]
                    for g in gs}
        expected = {k: v for k, v in expected.items() if v}
        assert {g['geometry_key']: run.original_aliases(g) for g in saved_pool} == expected
        points = {k: all_points[k] for k in expected}
        members = run.base.rows(job['nt_members.jsonl']); scores = np.load(job['nt_scores.npy'], allow_pickle=False)
        scored = [{**m, 'score': float(v)} for m, v in zip(members, scores)]
        a, b, suppressed, pairs, examined, intersections = independently_select(scored, points)
        review = run.read(out / 'predictions' / s / 'selection_review.json')
        assert a == run.read(job['control_semantic']) == run.read(out / 'predictions' / s / 'semantic_A.json')
        assert b == run.read(out / 'predictions' / s / 'semantic_B.json')
        assert suppressed == [{k: v for k, v in x.items() if k not in ('source_flow', 'was_in_original_K800')}
                              for x in review['suppressed']]
        assert pairs == [{k: v for k, v in p.items() if k != 'shared_eligible_classes'} for p in review['near_geometry_pairs']]
        assert examined == review['examined'] and len(scored)-examined == review['unexamined']
        metadata = review['sources']; assert set(metadata) == set(points)
        for k, aliases in expected.items():
            sources = {al['source'] for al in aliases}
            tag = 'shared_NT' if len(sources) == 2 else ('N_only' if 'native' in sources else 'T_only')
            assert metadata[k]['source_tag'] == tag and metadata[k]['aliases'] == aliases
            assert metadata[k]['points_sha256'] == hashlib.sha256(np.asarray(points[k], dtype='<i8').tobytes()).hexdigest()
        flows = Counter()
        for record in review['suppressed']:
            left, right = record['member']['geometry_key'], record['witness']['geometry_key']
            flow = metadata[left]['source_tag'] + '->' + metadata[right]['source_tag']
            assert record['source_flow'] == flow
            assert record['member']['class_index'] == record['witness']['class_index']
            assert record['witness']['score'] >= record['member']['score']
            assert record['was_in_original_K800'] == (run.shared.key(record['member']) in {run.shared.key(m) for m in a})
            flows[flow] += 1
        assert dict(flows) == review['suppression_source_flow']
        akeys, bkeys = {run.shared.key(m) for m in a}, {run.shared.key(m) for m in b}
        assert review['entered'] == [m for m in b if run.shared.key(m) not in akeys]
        assert review['exited'] == [m for m in a if run.shared.key(m) not in bkeys]
        for arm, selected in zip(run.ARMS, (a, b)):
            collapsed = {}
            for m in sorted(selected, key=lambda m: (-m['score'], m['class_index'])):
                collapsed.setdefault(m['geometry_key'], dict(geometry_key=m['geometry_key'], score=m['score'], winning_class_index=m['class_index']))
            ca = sorted(collapsed.values(), key=lambda m: (-m['score'], m['geometry_key']))
            assert ca == run.read(out / 'predictions' / s / ('class_agnostic_' + arm + '.json'))
            if arm == 'A': assert ca == run.read(job['control_class_agnostic'])
        totals.update(scenes=1, prebudget_members=len(scored), independent_intersections=intersections,
                      near_geometry_pairs=len(pairs), suppressed=len(suppressed), entered=len(review['entered']), exited=len(review['exited']))
        if n % 10 == 0: print('Independently selected', n, '/200', flush=True)
    assert not guard.gt_opened and totals['scenes'] == 200
    freeze_json(out / 'selection_verified.json', dict(complete=True, counts=dict(totals),
                independent_method='Sorted-array intersect1d with exact size bound; integer strict19/20; independent ordered kept-only greedy selector',
                all_scene_selections_witnesses_and_refill_exact=True, independent_source_aliases=True,
                independent_output_collapse=True, raw_GT_read=False, prediction_seal_sha256=sha(out / 'predictions_frozen.json'),
                verifier_sha256=sha(__file__), elapsed_seconds=time.monotonic()-start,
                process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def final(out):
    c = run.read(out / 'contract.json'); guard = Guard(c); guard.gt_enabled = True; sys.addaudithook(guard.hook)
    summary = run.read(out / 'evaluation/summary.json'); seal = run.read(out / 'predictions_frozen.json')
    selection_receipt = run.read(out / 'selection_verified.json'); assert selection_receipt['complete']
    for files in (c['inputs'], c['code'], c['label_inputs'], seal['outputs'], seal['scene_receipts'], summary['AP_arrays']):
        run.base.verify_files(files)
    edges = GT = 0
    for s in c['scenes']:
        receipt = run.read(out / 'evaluation' / s / 'receipt.json'); run.base.verify_files(receipt['outputs']); GT += receipt['GT']
        sizes = {g['geometry_key']: g['point_count'] for g in run.read(c['bindings'][s]['geometry'])}
        for v in run.VIEWS:
            for a in run.ARMS:
                rows = run.read(out / 'predictions' / s / (v + '_' + a + '.json'))
                match = run.read(out / 'evaluation' / s / (v + '_' + a + '_matches.json'))
                ps = {p['uuid']: p for values in match['pred'].values() for p in values}
                assert len(ps) == sum(sizes[m['geometry_key']] >= 100 for m in rows)
                ev = run.base.evaluator() if v == 'semantic' else run.class_agnostic_evaluator()[0]
                usable = [m for m in rows if sizes[m['geometry_key']] >= 100]
                pred_ids = {(m['geometry_key'], m.get('class_index')): i for i, m in enumerate(usable)}
                for ci, label in enumerate(ev.CLASS_LABELS):
                    expected = [m for m in usable if v == 'class_agnostic' or m['class_index'] == ci]
                    assert len(expected) == len(match['pred'][label])
                    for m, p in zip(expected, match['pred'][label]):
                        assert p['vert_count'] == sizes[m['geometry_key']] and p['confidence'] == m['score']
                        assert p['pred_id'] == pred_ids[(m['geometry_key'], m.get('class_index'))]
                    for g in match['gt'][label]:
                        for p in g['matched_pred']:
                            pp = ps[p['uuid']]
                            assert p['confidence'] == pp['confidence'] and p['pred_id'] == pp['pred_id']
                            hits = [x for x in pp['matched_gt'] if x['instance_id'] == g['instance_id']]
                            assert len(hits) == 1 and hits[0]['intersection'] == p['intersection']
                            edges += 1
                if a == 'A':
                    assert run.shared.canonical_match(match) == run.shared.canonical_match(run.read(c['jobs'][s]['matches_' + v]))
    assert GT == 4689
    arrays = scalars = 0
    old = run.read(run.SOURCE / 'evaluation/summary.json')
    for v in run.VIEWS:
        ev = run.base.evaluator() if v == 'semantic' else run.class_agnostic_evaluator()[0]
        for a in run.ARMS:
            matches = {s: run.read(out / 'evaluation' / s / (v + '_' + a + '_matches.json')) for s in c['scenes']}
            ids = [p['uuid'] for m in matches.values() for ps in m['pred'].values() for p in ps]
            assert len(ids) == len(set(ids))
            for g, scenes in c['reporting_groups'].items():
                x, *_ = ev.evaluate_matches({s: matches[s] for s in scenes})
                assert np.array_equal(x, np.load(out / 'evaluation' / (g + '_' + v + '_' + a + '_AP.npy')), equal_nan=True)
                overlaps = ev.opt['overlaps']
                manual = dict(ap=float(np.nanmean(x[:, :, ~np.isclose(overlaps, .25)])),
                              ap50=float(np.nanmean(x[:, :, np.isclose(overlaps, .5)])),
                              ap25=float(np.nanmean(x[:, :, np.isclose(overlaps, .25)])))
                for k, value in manual.items():
                    assert abs(value - summary['AP'][g][v][a][k]) < 1e-14
                    scalars += 1
                arrays += 1
                if a == 'A':
                    q = run.SOURCE / 'evaluation' / (g + '_' + v + '_NT_AP.npy')
                    assert sha(q) == old['AP_arrays'][str(q)]
                    assert np.array_equal(x, np.load(q), equal_nan=True)
            print('Verified AP', v, a, flush=True)
    for g, views in summary['AP'].items():
        for v, ap in views.items():
            assert summary['effects_pp'][g][v] == {k: 100*(ap['B'][k]-ap['A'][k]) for k in ('ap', 'ap50', 'ap25')}
    aggregate = run.read(out / 'selection_review.json')
    for group, scenes in c['reporting_groups'].items():
        counts, flows = Counter(), Counter()
        for s in scenes:
            pr = run.read(out / 'predictions' / s / 'selection_review.json')
            counts.update(pr['counts']); flows.update(pr['suppression_source_flow'])
        assert dict(counts) == aggregate['groups'][group]['counts']
        assert dict(flows) == aggregate['groups'][group]['suppression_source_flow']
        for v in run.VIEWS:
            ev = run.base.evaluator() if v == 'semantic' else run.class_agnostic_evaluator()[0]
            independent = {f'{t:.2f}': Counter(gained=0, lost=0, shared_score_up=0, shared_score_down=0) for t in ev.opt['overlaps']}
            for s in scenes:
                mm = {a: run.read(out / 'evaluation' / s / (v + '_' + a + '_matches.json')) for a in run.ARMS}
                for label, targets in mm['A']['gt'].items():
                    after = {g['instance_id']: g for g in mm['B']['gt'][label]}
                    for gt in targets:
                        if gt['instance_id'] < 1000 or gt['vert_count'] < 100: continue
                        other = after[gt['instance_id']]
                        for t in ev.opt['overlaps']:
                            def frontier(g):
                                return max((p['confidence'] for p in g['matched_pred']
                                    if p['intersection']/(g['vert_count']+p['vert_count']-p['intersection']) > t), default=None)
                            b, a = frontier(gt), frontier(other)
                            ct = independent[f'{t:.2f}']
                            ct['gained'] += b is None and a is not None; ct['lost'] += b is not None and a is None
                            if b is not None and a is not None:
                                ct['shared_score_up'] += a > b; ct['shared_score_down'] += a < b
            assert {t: dict(ct) for t, ct in independent.items()} == aggregate['groups'][group]['target_changes'][v]
    freeze_json(out / 'final_verification.json', dict(complete=True, GT=GT, selections=800,
                AP_arrays=arrays, AP_scalars=scalars, bidirectional_edges=edges,
                old_two_view_control_matches_exact=400, old_control_arrays_exact=6,
                all_input_output_hashes=True, independent_selection=True, independent_frontiers=True,
                AP_arrays_recomputed=True, scalars_independent=True, raw_GT_read=False,
                reserve100_read=False, heldout_read=False, final_test_read=False, production_modified=False,
                evaluation_sha256=sha(out / 'evaluation/summary.json'), verifier_sha256=sha(__file__)))
    print('Verified', arrays, 'AP arrays;', scalars, 'scalars;', GT, 'GT.', flush=True)


def report(out):
    """Finish the isolated report; project state updates are a separate action."""
    assert run.read(out / 'final_verification.json')['complete']
    summary = run.read(out / 'evaluation/summary.json'); review = run.read(out / 'selection_review.json')
    ap, delta = summary['AP']['development200'], summary['effects_pp']['development200']
    positive = delta['semantic']['ap'] > 0
    decision = dict(complete=True, adopted=False, production_modified=False,
                    semantic_main_AP_improved=positive,
                    class_agnostic_main_AP_improved=delta['class_agnostic']['ap'] > 0,
                    threshold_search=False, new_learning_head_implemented=False,
                    finding='Fixed NT NMS improves semantic main AP on exposed development200.' if positive else
                            'Fixed NT NMS does not improve semantic main AP on exposed development200.',
                    next='Inspect retained-versus-suppressed quality and uncovered errors before freezing any relation/representative method; compare future methods against both NT and NT+standard NMS.',
                    U_Uprime='Optional deferred branch; no automatic generation change.',
                    reserve100_consumed=True, reserve100_read_this_run=False,
                    significance_claim=False, independent_acceptance_passed=False)
    freeze_json(out / 'mechanism_decision.json', decision)
    lines = ['# NT固定同类点mask NMS开发200对照', '',
             '两臂闭环和独立核验已完成。固定既有NT点集、自身语义、预算前成员、B26分数和上下文，仅改变输出选择。',
             'A为原K800；B按原排序，仅由已保留同类成员以真实点IoU严格>0.95抑制，合法补位到最多800。无新训练、视觉推理、点集或阈值搜索。', '',
             '| 集合 | 口径 | A AP% | B AP% | ΔAP pp | A AP50% | B AP50% | A AP25% | B AP25% |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in ('development200', 'historical90', 'new110'):
        for view in run.VIEWS:
            a, b = summary['AP'][group][view]['A'], summary['AP'][group][view]['B']
            lines.append(f'| {group} | {view} | {100*a["ap"]:.6f} | {100*b["ap"]:.6f} | {100*(b["ap"]-a["ap"]):+.6f} | {100*a["ap50"]:.6f} | {100*b["ap50"]:.6f} | {100*a["ap25"]:.6f} | {100*b["ap25"]:.6f} |')
    counts = review['groups']['development200']['counts']
    lines += ['', '## 选择与来源', '',
              f'预算前成员{counts["prebudget_members"]}；实际抑制{counts["suppressed"]}，原K800退出{counts["exited"]}／新增补位{counts["entered"]}。',
              f'原选择{counts["selected_A"]}，NMS后{counts["selected_B"]}；全池近重复几何对{counts["near_geometry_pairs"]}，原已选存在同类近重复关系的成员{counts["original_selected_near_same_class_members"]}。',
              '抑制来源流向（被抑制→保留见证；共享N/T来源单列）：', '']
    for flow, count in sorted(review['groups']['development200']['suppression_source_flow'].items()): lines.append(f'- {flow}: {count}')
    lines += ['', '## 正确覆盖与伤害', '',
              '| 口径 | IoU严格阈值 | 新覆盖目标 | 失覆盖目标 | 共享目标最高分上升 | 共享目标最高分下降 |',
              '|---|---:|---:|---:|---:|---:|']
    for v in run.VIEWS:
        for t in ('.25', '.50', '.75', '.90'):
            ct = review['groups']['development200']['target_changes'][v]['0' + t]
            lines.append(f'| {v} | {t} | {ct["gained"]} | {ct["lost"]} | {ct["shared_score_up"]} | {ct["shared_score_down"]} |')
    lines += ['', '范围描述、潜在重复覆盖和最高分前沿都不是原评测独占TP/FP，也不是可相加AP损失。逐场景完整抑制见证、来源、IoU、双方分数、补位和退出在predictions/*/selection_review.json；事后真实范围描述在evaluation/*/range_review.json。',
              '', '## 结论与后续', '', decision['finding'],
              '标准NMS只作参考，不作为论文创新；不把所有收益归为N/T跨源去重。若研究新关系选择，须区分同物体替代、互补范围和不同物体，并对照原NT及标准NMS。失败不扫0.90/0.85，不自动启动新头或U′生成。',
              '', '## 复用、核验与边界', '',
              '全部200场景预算前NT成员资格重新构建、20维自身特征和冻结B26分数重算一致；26维特征/分数原始字节哈希与10-07联合消融精确一致，A两个视图的400选择精确复用。',
              '独立核验用排序点数组交集与整数19/20判定重做全200选择、全部实际抑制见证和补位；不重复调用生产NMS。800输出及选择核验先封存后GT。原评测AST、198类、void、小实例和严格IoU不改；400旧匹配、6旧AP数组精确复现，12AP数组／36标量重算，双向匹配交集逐点核验。',
              '仅既有开发200（旧90＋新110），4689有效GT；B26见过597，开发200已反复暴露，非整系统盲测或独立验收。保留100已消费，本轮未再读取；开发55、验证312、最终测试和图例未读。生产Full不变。',
              '类别无关为实际所选成员按完全相同点集取最高已选分数、不补位，保留不同近重复点集；不是独立D-v3几何评分。两种AP不能相减当语义损失，不平均子组AP。',
              '', f'200场景预测/控制核验/NMS累计{review["groups"]["development200"]["prediction_seconds"]:.2f}秒；进程峰值内存、独立核验额外耗时见predictions_frozen.json及selection_verified.json。',
              f'仅新增NMS选择累计{review["groups"]["development200"]["nms_seconds"]:.2f}秒（含点集交集与关系统计，不含B26控制重放）。',
              '项目代码入口为tools/fi1_same_class_mask_nms.py、tools/run_fi1_selection_control_development200.py、tools/verify_fi1_selection_control_development200.py。旧冻结代码和回执未修改。', '']
    (out / 'report.md').write_text('\n'.join(lines))
    names = ('contract.json', 'tests_verified.json', 'predictions_frozen.json', 'selection_verified.json',
             'matches_frozen.json', 'evaluation/summary.json', 'selection_review.json', 'final_verification.json',
             'mechanism_decision.json', 'report.md')
    freeze_json(out / 'acceptance.json', dict(complete=True, diagnostic_complete=True, adopted=False,
                artifacts={n: sha(out/n) for n in names}, scenes=200, GT=4689, selections=800,
                AP_arrays=12, AP_scalars=36, resume_required=False, production_modified=False,
                reserve100_read=False, heldout_read=False, final_test_read=False,
                semantic_main_AP_improved=positive, independent_acceptance_passed=False))
    write(out / 'execution.json', dict(status='complete', stage='verified_and_reported', resume_required=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('selection', 'final', 'report'))
    parser.add_argument('--output', type=Path, default=run.OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    try:
        with threadpool_limits(limits=1): globals()[args.phase](out)
    except Exception as error:
        write(out / 'execution.json', dict(status='failed', stage='verification_' + args.phase,
              error_type=type(error).__name__, error=str(error), resume_required=True))
        raise


if __name__ == '__main__': main()
