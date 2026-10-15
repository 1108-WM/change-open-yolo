"""Read-only replay and supplementary attribution checks; own output directory only."""
from pathlib import Path
from collections import Counter
import sys
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'research/geometry/code/tools'))
import verify_fi1_nt_instance_ownership as verifier
run = verifier.run
D = run.D
OUT = run.OUT

def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

original_seal = D.seal
def redirected(path, value):
    assert Path(path) == OUT / 'verification.json'
    write('replayed_verification.json', value)
D.seal = redirected
verifier.main()
D.seal = original_seal

hash_checks = 0
for name in ('final_delivery_verified.json', 'delivery_verified.json', 'completion_verified.json',
             'failure_diagnosis_frozen.json', 'failure_diagnosis_v2_frozen.json'):
    bindings = D.read(OUT / name)['outputs']
    D.verify(bindings)
    hash_checks += len(bindings)

counts = Counter()
rows = []
block_composition = Counter()
for scene in D.read(OUT / 'contract.json')['scenes']:
    nodes = D.read(OUT / 'predictions' / scene / 'nodes.json')
    parents = {p['candidate_id']: p for p in D.read(OUT / 'predictions' / scene / 'parents.json')}
    groups = D.read(OUT / 'predictions' / scene / 'groups.json')
    evblocks = D.read(OUT / 'evaluation' / scene / 'blocks.json')
    rows.extend(D.read(OUT / 'evaluation' / scene / 'unions.json'))
    with np.load(OUT / 'predictions' / scene / 'data.npz') as z:
        for b in evblocks:
            counts[b['state']] += 1
            if b['state'] != 'accept':
                continue
            members = groups[b['parent_groups'][0]]['members']
            own = parents[b['candidate_id']]['parent_geometry']
            tracks = [n for n in members if nodes[n]['points'] >= 100 and 'track' in nodes[n]['sources']]
            native = [n for n in members if nodes[n]['points'] >= 100 and 'native' in nodes[n]['sources']]
            counts['accept_own_parent_is_only_eligible_T'] += len(tracks) == 1 and nodes[tracks[0]]['key'] == own
            hits = z[b['block_id'] + '_hits']
            supporting_sources = {src for n in members if hits[n] >= .5 * b['points'] for src in nodes[n]['sources']}
            block_composition['+'.join(sorted(supporting_sources))] += 1
            counts['accept_distinct_N_T_both_cover_block'] += any(n != t and hits[n] >= .5*b['points'] and hits[t] >= .5*b['points'] for n in native for t in tracks)
            counts['accept_no_single_native_bridge'] += not bool(b['native_bridge_nodes'])

aggregate = {}
for arm in run.ARMS:
    rr = [r for r in rows if r['arm'] == arm]
    stable = [r for r in rr if r['stable']]
    changed = [r for r in stable if r['changed']]
    added = sum(r['added'] for r in stable)
    target = sum(r['added_target'] for r in stable)
    other = sum(r['added_other'] for r in stable)
    unknown = sum(r['added_GT']['unknown'] for r in stable)
    assert added == target + other + unknown
    aggregate[arm] = dict(actions=sum(r['changed'] for r in rr), stable_actions=len(changed),
        target=target, other=other, unknown=unknown, precision=target/added,
        good_parents=sum(r['parent_iou'] >= .9 for r in stable),
        good_parents_with_action=sum(r['parent_iou'] >= .9 for r in changed),
        good_copy_damage=sum(r['parent_iou'] >= .9 and r['parent_delta'] < -.02-1e-12 for r in stable),
        stable_copy_damage=sum(r['parent_delta'] <= -.02+1e-12 for r in stable),
        bridge_N_redundant=sum(r['bridging_native_already_at_least_as_good'] for r in changed),
        gain02=sum(r['pool_delta'] >= .02-1e-12 for r in stable))
    stored = D.read(OUT / 'summary.json')['modes'][arm]
    assert abs(aggregate[arm]['precision'] - stored['stable_add_precision']) < 1e-12
    assert aggregate[arm]['bridge_N_redundant'] == stored['bridge_native_redundant_actions']

base, new = aggregate['all_C'], aggregate['relation_C']
result = dict(complete=True, receipt_hash_checks=hash_checks, counts=dict(counts),
    accepted_block_support_sources=dict(block_composition), aggregates=aggregate,
    target_retention_vs_all_C=new['target']/base['target'],
    known_neighbor_removal_vs_all_C=1-new['other']/base['other'],
    scope='Reran existing set-based independent verifier, plus new support-source and aggregate census. No new rules, outputs, training or AP.')
write('supplementary_checks.json', result)
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
