"""Replay the reviewed verifier without changing its seal; independently audit AUC/bins."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'research/geometry/code/tools'))
import verify_fi1_local_edit_evidence_counts as replay

DEST = Path(__file__).resolve().parent

def main():
    receipts = []
    original_seal = replay.run.seal
    def capture(path, value):
        assert Path(path) == replay.run.OUT/'verification_counts.json'
        receipts.append(value)
    replay.run.seal = capture
    try:
        replay.verify()
    finally:
        replay.run.seal = original_seal
    assert len(receipts) == 1
    base = replay.run.OUT
    rows = json.loads((base/'evaluation/rows.json').read_text())
    unions = json.loads((base/'evaluation/unions.json').read_text())
    stats = []
    for cohort in ('fit12', 'exposed16'):
        for source in ('native', 'track'):
            for action in ('add', 'delete'):
                rr = [r for r in rows if r['cohort']==cohort and r['source']==source and r['action']==action and r['family']=='region']
                binary = [r for r in rr if r['stable'] and not r['minimum100_fallback'] and not r['neutral']]
                positive = np.array([r['query_mean'] if action=='add' else 1-r['query_mean'] for r in binary if r['net_gain']])
                negative = np.array([r['query_mean'] if action=='add' else 1-r['query_mean'] for r in binary if r['net_harm']])
                auc = float(((positive[:,None]>negative).sum()+.5*(positive[:,None]==negative).sum())/(len(positive)*len(negative)))
                for screen in ('query_direction','query_contrast'):
                    selected = [r for r in rr if r['screen_'+screen]]
                    stable = [r for r in selected if r['stable']]
                    ur = [r for r in unions if r['cohort']==cohort and r['source']==source and r['action']==action and r['family']=='region' and r['screen']==screen]
                    stats.append(dict(cohort=cohort,source=source,action=action,screen=screen,query_auc=auc,selected=len(selected),stable=len(stable),gain=sum(r['net_gain'] for r in stable),harm=sum(r['net_harm'] for r in stable),wrong_points=sum(r.get('affected_wrong',0) for r in selected),unknown_points=sum(r.get('affected_unknown',r['points']) for r in selected),union_stable_parents=sum(r['stable'] for r in ur),union_good_parents=sum(r.get('good_parent',False) for r in ur),union_good_loss02=sum(r.get('good_loss',0)>.02 for r in ur),union_worst_good_loss=max((r.get('good_loss',0) for r in ur),default=0)))
    result = dict(complete=True,existing_count_verifier_rerun=receipts[0],independent_pairwise_auc_and_bin_aggregation=stats,limitations=['Reused existing reviewed count verifier and its input loader/screens; not a wholly new independent verifier.', 'No rerun of multiframe visibility/mask generation, training or GPU inference.', 'Original experiment seals preserved; this run writes only this audit result.', 'Local artifacts audited; remote GitHub commit identity not verified.'])
    (DEST/'audit_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
