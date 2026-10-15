#!/usr/bin/env python3
"""Seal the new audit delivery without updating any earlier artifact."""
import ast
import csv
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fi1_t_source_chain as a


def main():
    a.reserve()
    snapshots = a.OUT / 'code_snapshot'
    for name in ('analyze_fi1_t_source_chain.py', 'verify_fi1_t_source_chain.py',
                 'finalize_fi1_t_source_chain.py'):
        path = Path(__file__).with_name(name)
        ast.parse(path.read_text())
        dest = snapshots / name
        if dest.exists() and a.sha(dest) != a.sha(path):
            dest = snapshots / (path.stem + '_' + a.sha(path)[:12] + path.suffix)
        if not dest.exists():
            with dest.open('xb') as f:
                f.write(path.read_bytes())
        assert a.sha(dest) == a.sha(path)
    frozen = a.read(a.OUT / 'source_evidence_frozen.json')
    evaluation = a.read(a.OUT / 'evaluation_frozen.json')
    verified = a.read(a.OUT / 'verification.json')
    for seal in (frozen, evaluation):
        for p, h in seal['outputs'].items():
            assert a.sha(p) == h, p
    assert frozen['code_sha256'] == a.sha(Path(a.__file__))
    assert evaluation['code_sha256'] == a.sha(Path(__file__).with_name('analyze_fi1_t_source_chain.py'))
    assert verified['code_sha256'] == a.sha(Path(__file__).with_name('verify_fi1_t_source_chain.py'))
    with (a.OUT / 'block_source_evidence.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len({(r['scene'],r['block_id']) for r in rows}) == 1308
    blockcounts = {}
    for r in rows:
        key = (r['scene'],r['candidate_id'])
        blockcounts[key] = blockcounts.get(key,0)+1
    scopes = {(r['scene'],r['candidate_id']):r for r in a.read(a.OUT/'evaluation/parent_scopes.json')}
    parentrows = []
    for scene in contract_jobs():
        for p in a.read(a.OUT/'evidence'/scene/'parents.json'):
            key = (scene,p['candidate_id'])
            meta = scopes[key]
            parentrows.append(dict(scene=scene,candidate_id=p['candidate_id'],track_id=p['track_id'],
                parent_geometry=p['geometry_key'],locator=p['locator']['points_path'],
                pointset_sha256=p['points_sha256'],blocks=blockcounts.get(key,0),
                source_track_ids=';'.join(map(str,p['source_track_ids'])),
                observation_ids=';'.join(map(str,p['observation_ids'])),
                frame_ids=';'.join(map(str,p['frame_ids'])),
                parent_scope=meta['scope'],target=meta['target'] if meta['stable'] else None,
                trace=f'evidence/{scene}/parents.json'))
    assert len(parentrows)==len({(r['scene'],r['candidate_id']) for r in parentrows})==207
    assert sum(r['blocks'] for r in parentrows)==1308
    assert [(r['scene'],r['candidate_id']) for r in parentrows if not r['blocks']]==[('scene0627_01','c006')]
    with (a.OUT/'parent_source_evidence.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(parentrows[0]))
        writer.writeheader()
        writer.writerows(parentrows)
    assert len({r['scene'] for r in rows}) == 52
    for path in (a.OUT/'report.md', a.OUT/'README.md'):
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            assert (path.parent/link.split('#')[0]).exists(), (path,link)
    summary = a.read(a.OUT/'evaluation_summary.json')
    assert summary['modes']['d1_initial']['same_parent_gain02']==3
    assert len(summary['modes']['d1_initial']['same_parent_gain02_scenes'])==3
    assert summary['modes']['d1_initial']['good_copy_loss_gt02']==3
    assert summary['modes']['merge_input_target_oracle']['same_parent_gain02']==0
    assert summary['strata']['all']['flags']['D1_removed']==417
    assert summary['strata']['all']['flags']['merge_removed']==17
    assert summary['strata']['opportunity_contribution']['blocks']==32
    assert verified['counts']['blocks']==1308 and verified['counts']['original_input_hashes']==81168
    manifest = {str(p):a.sha(p) for p in sorted(a.OUT.rglob('*')) if p.is_file()}
    a.seal(a.OUT/'delivery_verified.json',dict(complete=True,counts=summary['counts'],
        outputs=manifest, code={str(p):a.sha(p) for p in [Path(a.__file__),
            Path(__file__).with_name('analyze_fi1_t_source_chain.py'),
            Path(__file__).with_name('verify_fi1_t_source_chain.py'), Path(__file__)]},
        source_evidence_precedes_GT=True, report_links_and_csv_scope_checked=True,
        report_key_numbers_checked=True, training=False,new_frames=False,
        old_ownership_thresholds_changed=False,AP=False,production_modified=False,
        next_prototype_implemented=False,old_seals_modified=False))
    print('delivery_sealed',len(manifest),'files',summary['counts'],flush=True)


def contract_jobs():
    return a.read(a.OUT/'contract.json')['jobs']


if __name__=='__main__':
    main()
