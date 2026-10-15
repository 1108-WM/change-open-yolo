#!/usr/bin/env python3
"""Independent point-membership checks and descriptive source-vote census."""
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fi1_t_source_chain as a


def independent_census(pp, gt, target):
    labels, counts = np.unique(gt[pp], return_counts=True)
    known = int(counts[labels >= 1000].sum())
    truth = int(np.count_nonzero(gt[pp] == target)) if target else None
    return dict(points=len(pp), target=truth, other=known-truth if target else None,
                unknown=len(pp)-known)


def main():
    source = a.read(a.OUT / 'source_evidence_frozen.json')
    evaluation = a.read(a.OUT / 'evaluation_frozen.json')
    for frozen in (source, evaluation):
        for p, h in frozen['outputs'].items():
            assert a.sha(p) == h, p
    assert source['GT_used'] is False and evaluation['GT_used'] is True
    contract = a.read(a.OUT / 'contract.json')
    old = a.read(a.OLD / 'contract.json')
    assert a.sha(a.OLD / 'contract.json') == contract['prior_contract_sha256']
    hist = a.OUT / 'historical_source/refine_details_automatic_tracks_consensus.py'
    spec = importlib.util.spec_from_file_location('historical_consensus', hist)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    inputhashes = {}
    counts = Counter()
    votes, opportunities, failure_groups, accepted_negatives = [], [], [], []
    all_union_rows = a.read(a.OUT / 'evaluation/all_unions.json')
    for si, (s, entry) in enumerate(contract['jobs'].items()):
        print('independent_membership_checks', si+1, 52, s, flush=True)
        ed = a.OUT / 'evidence' / s
        receipt = a.read(ed / 'receipt.json')
        for p, h in receipt['inputs'].items():
            assert p not in inputhashes or inputhashes[p] == h
            inputhashes[p] = h
        job = entry['job']
        arm = Path(entry['parents'][0]['source_root']).parent.parent
        rawobs = a.jsonlines(arm / 'hierarchy' / s / 'original_automatic_observations.jsonl')
        rawtracks = a.read(arm / 'tracks' / s / 'automatic_tracks.json')['tracks']
        d1tracks = {r['track_id']:r for r in a.read(arm / 'consensus' / s / 'automatic_tracks.json')['tracks']}
        byid = {r['observation_id']:r for r in rawobs}
        owner = {oid:t['track_id'] for t in rawtracks for oid in t['observation_ids']}
        keptids = {r['observation_id'] for r in a.jsonlines(arm/'hierarchy'/s/'automatic_observations.jsonl')}
        allpoints = {r['observation_id']:a.points(r['point_indices_path']) for r in rawobs}
        sp = np.load(job['paths']['processed'], mmap_mode='r')[:,9].astype(np.int64)
        labels, nums = np.unique(sp,return_counts=True)
        sizes = dict(zip(map(int,labels),map(int,nums)))
        visible, visible_counts, observations = {}, {}, {}
        for oid, raw in byid.items():
            fi = int(raw['frame_index'])
            if fi not in visible:
                visible[fi] = np.load(Path(raw['point_indices_path']).parent/'visibility.npy')
                ids, cc = np.unique(sp[visible[fi]], return_counts=True)
                visible_counts[fi] = dict(zip(map(int,ids),map(int,cc)))
            pp = allpoints[oid]
            ids, cc = np.unique(sp[pp], return_counts=True)
            observations[oid] = dict(frame_index=fi, visible_counts=visible_counts[fi],
                inside_counts=dict(zip(map(int,ids),map(int,cc))),
                inside_point_indices_by_superpoint={int(k):pp[sp[pp]==k] for k in ids})
        parents = a.read(ed/'parents.json')
        bs = a.read(ed/'blocks.json')
        evaluated = {b['block_id']:b for b in a.read(a.OUT/'evaluation'/s/'blocks.json')}
        links = a.read(ed/'block_observations.json')
        linkindex = defaultdict(dict)
        for o in links:
            assert o['observation_id'] not in linkindex[o['block_id']]
            linkindex[o['block_id']][o['observation_id']] = o
        scope = {r['candidate_id']:r for r in a.read(a.OUT/'evaluation/parent_scopes.json') if r['scene']==s}
        gt = np.loadtxt(job['GT'], dtype=np.int64)
        assert a.sha(job['GT']) == evaluation['GT_inputs'][job['GT']]
        authoritative = {b['block_id']:b for b in a.read(a.OLD/'predictions'/s/'blocks.json')}
        assert set(authoritative) == set(evaluated) == {b['block_id'] for b in bs}
        with np.load(ed/'pointsets.npz') as z, np.load(a.BLOCKS/'predictions'/s/'data.npz') as original:
            for parent in parents:
                cid = parent['candidate_id']
                pp = z[cid+'_parent']
                pset = set(map(int,pp))
                lineage = set(parent['source_track_ids'])
                initial = set()
                kept = set()
                for track in parent['original_track_records']:
                    tid = track['track_id']
                    ks, diagnostic, framerows, init = historical.refine_tracklet_superpoints(
                        track['observation_ids'], observations, sizes, .1, .3, 3, .5, 2, .3, .55)
                    assert init == set(parent['exact_D1_initial'][str(tid)])
                    assert ks == set(d1tracks[tid]['superpoint_ids'])
                    initial |= init
                    kept |= ks
                    counts['historical_D1_reconstructions'] += 1
                ownframes = defaultdict(list)
                for oid in parent['observation_ids']:
                    ownframes[int(byid[oid]['frame_index'])].append(allpoints[oid])
                maskframes = {fi:a.union(chunks) for fi,chunks in ownframes.items()}
                pbs = [b for b in bs if b['candidate_id']==cid]
                for b in pbs:
                    bid = b['block_id']
                    block = z[bid+'_block']
                    assert np.array_equal(block, original[bid+'_block'])
                    assert np.array_equal(z[bid+'_clip'], original[bid+'_clipped_proposal'])
                    assert b['prior_ownership_state'] == authoritative[bid]['state']
                    assert not pset & set(map(int,block))
                    blockmask = np.zeros(len(sp),bool)
                    blockmask[block] = True
                    expected_sets = {k:set() for k in ('all_raw','post_hierarchy','lineage_observed','other_tracks','kept_unassigned')}
                    expected_ids = set()
                    for oid, op in allpoints.items():
                        part = op[blockmask[op]]
                        if not len(part):
                            assert oid not in linkindex[bid]
                            continue
                        expected_ids.add(oid)
                        link = linkindex[bid][oid]
                        raw = byid[oid]
                        assert link['point_indices_path'] == raw['point_indices_path']
                        assert link['frame_id'] == raw['frame_id'] and link['frame_index'] == raw['frame_index']
                        assert link['raw_track_id'] == owner.get(oid)
                        assert link['block_hits'] == len(part)
                        assert link['parent_hits'] == sum(int(k) in pset for k in op)
                        fi = int(raw['frame_index'])
                        assert visible[fi][part].all()
                        partset = set(map(int,part))
                        expected_sets['all_raw'] |= partset
                        if oid in keptids:
                            expected_sets['post_hierarchy'] |= partset
                        if owner.get(oid) in lineage:
                            expected_sets['lineage_observed'] |= partset
                        elif oid in owner:
                            expected_sets['other_tracks'] |= partset
                        elif oid in keptids:
                            expected_sets['kept_unassigned'] |= partset
                        counts['block_observation_links'] += 1
                    assert expected_ids == set(linkindex[bid])
                    for name, wanted in expected_sets.items():
                        assert set(map(int,z[bid+'_'+name])) == wanted, (s,bid,name)
                    for name, ids in [('d1_initial',initial),('d1_kept',kept)]:
                        wanted = {int(k) for k in block if int(sp[k]) in ids}
                        assert wanted == set(map(int,z[bid+'_'+name]))
                    assert np.array_equal(z[bid+'_d1_kept'],z[bid+'_merge_input'])
                    assert np.array_equal(z[bid+'_d1_kept'],z[bid+'_merge_removed'])
                    assert not len(z[bid+'_final_T'])
                    ev = evaluated[bid]
                    target = scope[cid]['target'] if scope[cid]['stable'] else None
                    for name, expected in ev['stage_GT'].items():
                        calc = independent_census(z[bid+'_'+name],gt,target)
                        assert all(expected[k]==v for k,v in calc.items())
                        counts['GT_stage_censuses'] += 1
                    counts['blocks'] += 1
                    support = np.zeros(len(block),np.int32)
                    denom = np.zeros(len(block),np.int32)
                    for fi, mask in maskframes.items():
                        support += np.isin(block,mask)
                        denom += visible[fi][block]
                    assert (support<=denom).all()
                    histogram = []
                    for sup, vis in sorted(set(zip(map(int,support),map(int,denom)))):
                        q = block[(support==sup)&(denom==vis)]
                        histogram.append(dict(source_support_frames=sup, source_visible_frames=vis,
                                              **independent_census(q,gt,target)))
                    vr = dict(scene=s,candidate_id=cid,block_id=bid,parent_scope=scope[cid]['scope'],
                        target=target,opportunity=ev['opportunity_contribution'],histogram=histogram)
                    votes.append(vr)
                    if ev['opportunity_contribution']:
                        opportunities.append(dict(scene=s,candidate_id=cid,block_id=bid,
                            counts=b['stages'],block_GT=ev['block_GT'],removed=b['removed_superpoint_evidence'],
                            lineage_observations=[o for o in links if o['block_id']==bid and o['in_parent_lineage']]))
                    for detail in b['removed_superpoint_evidence']:
                        rem = block[sp[block]==detail['superpoint_id']]
                        failure_groups.append(dict(scene=s,candidate_id=cid,block_id=bid,
                            parent_scope=scope[cid]['scope'],opportunity=ev['opportunity_contribution'],
                            evidence=detail,GT=independent_census(rem,gt,target)))
                    if ev['stable'] and ev['accepted_NT_block'] and ev['block_GT']['other']:
                        accepted_negatives.append(dict(scene=s,block_id=bid,block_GT=ev['block_GT'],
                            counts=b['stages'],parent_lineage_ids=b['lineage_hit_observation_ids']))
                for mode in ('lineage_observed','d1_initial','merge_input','other_tracks','all_raw'):
                    parts = a.union(z[b['block_id']+'_'+mode] for b in pbs)
                    result = np.union1d(pp,parts)
                    assert np.array_equal(result,z[cid+'_'+mode+'_union'])
                    rr = next(r for r in all_union_rows if r['scene']==s and r['candidate_id']==cid and r['mode']==mode)
                    assert rr['added'] == len(parts)
                    cc = independent_census(parts,gt,scope[cid]['target'] if scope[cid]['stable'] else None)
                    if scope[cid]['stable']:
                        assert cc['target']==rr['added_target'] and cc['other']==rr['added_other']
                        assert cc['unknown']==rr['unknown_edits']
                        target = scope[cid]['target']
                        nh = np.count_nonzero(gt[result]==target)
                        iou = nh/(len(result)+np.count_nonzero(gt==target)-nh) if len(result)>=100 else 0
                        assert abs(iou-rr['repaired_iou'])<1e-12
                    counts['raw_restoration_unions'] += 1
                counts['parents'] += 1
    print('checking_original_input_hashes',len(inputhashes),flush=True)
    for p,h in inputhashes.items():
        assert a.sha(p)==h,p
    counts['original_input_hashes']=len(inputhashes)
    assert counts['parents']==207 and counts['blocks']==1308 and counts['block_observation_links']==19336
    unique = {}
    for mode in sorted({r['mode'] for r in all_union_rows}):
        selected=[r for r in all_union_rows if r['mode']==mode and r.get('pool_delta',-1)>=.02-1e-12]
        unique[mode]=dict(parents=len(selected),targets=len({(r['scene'],r['target']) for r in selected}),
                          scenes=len({r['scene'] for r in selected}))
    freq=defaultdict(lambda:Counter())
    for v in votes:
        if v['parent_scope']!='stable':
            continue
        for r in v['histogram']:
            for label in ('target','other','unknown'):
                freq[(r['source_support_frames'],r['source_visible_frames'])][label]+=r[label]
    collisions=[dict(source_support_frames=k[0],source_visible_frames=k[1],**v)
                for k,v in sorted(freq.items()) if v['target'] and v['other']]
    summary=dict(unique_gain02=unique, accepted_negative_blocks=accepted_negatives,
        opportunity_blocks=len(opportunities),source_vote_histogram_collisions=collisions,
        stable_target_points_in_colliding_votes=sum(v['target'] for v in collisions),
        stable_other_points_in_colliding_votes=sum(v['other'] for v in collisions),
        repeated_source_support_census={label:sum(r[label] for v in votes if v['parent_scope']=='stable'
             for r in v['histogram'] if r['source_support_frames']>=2) for label in ('target','other','unknown')},
        interpretation='Postseal descriptive census only. >=2 is a report of repeated frames, not a tested repair gate. No source-vote threshold was tuned and no new method was run.')
    a.seal(a.OUT/'supplementary_source_vote_census.json',dict(blocks=votes,summary=summary))
    a.seal(a.OUT/'opportunity_source_examples.json',opportunities)
    a.seal(a.OUT/'refinement_failure_census.json',failure_groups)
    a.seal(a.OUT/'verification.json',dict(complete=True,counts=dict(counts),
        source_evidence_seal_sha256=a.sha(a.OUT/'source_evidence_frozen.json'),
        evaluation_seal_sha256=a.sha(a.OUT/'evaluation_frozen.json'),
        historical_consensus_independent_of_vectorized_export=True,
        all_observation_presence_and_absence_checked=True,
        all_pointsets_and_GT_stage_counts_checked=True,unique_gain02=unique,
        code_sha256=a.sha(__file__)))
    print('verified',dict(counts),flush=True)


if __name__=='__main__':
    main()
