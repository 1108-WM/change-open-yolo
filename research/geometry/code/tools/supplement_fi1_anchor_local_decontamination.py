#!/usr/bin/env python3
"""Post-seal visibility/mask evidence reconstruction and sample-scope audit."""
from pathlib import Path
from collections import Counter,defaultdict
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from tools import run_fi1_anchor_local_decontamination as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha
from tools.verify_fi1_anchor_local_decontamination import overlap


def main():
    c=run.read(run.OUT/'contract.json');assert run.read(run.OUT/'independent_verified.json')['complete']
    sys.addaudithook(run.Guard(c,labels=True).hook)
    run.verify_files(run.read(run.OUT/'geometry_frozen.json')['outputs']);run.verify_files(c['frozen_SL_models'])
    tally=Counter();rows=[];group=defaultdict(Counter)
    for s in c['scenes']:
        j=c['jobs'][s];pd=run.OUT/'plan'/s;gd=run.OUT/'geometry'/s
        plans=run.read(pd/'candidates.json');byid={r['candidate_id']:r for r in plans}
        views=run.read(gd/'views.json');manifest=run.read(j['paths']['manifest'])
        fm={int(f['frame_id']):f for f in manifest['frames']};raw=np.load(j['paths']['processed'],mmap_mode='r',allow_pickle=False)
        assert sha(j['GT'])==c['GT_inputs'][j['GT']]
        gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True)
        valid=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[valid],sizes[valid]
        byframe=defaultdict(list)
        for v in views:byframe[v['frame']].append(v)
        with np.load(pd/'arrays.npz',allow_pickle=False) as arrays,np.load(gd/'evidence.npz',allow_pickle=False) as ev:
            for f,vs in byframe.items():
                entry=fm[f];sensor=np.asarray(Image.open(entry['depth']['path']),dtype=np.float64)/1000.
                width,height=Image.open(entry['color']['path']).size;dh,dw=sensor.shape
                pose=np.loadtxt(entry['pose']['path']);camera=np.column_stack((raw[:,:3],np.ones(len(raw))))@np.linalg.inv(pose).T
                with np.load(gd/'observations'/f'frame{f:06d}.npz',allow_pickle=False) as z:
                    pixel=z['pixels'];stored=z['reliable']
                # RGB resolution is at least depth resolution. Round-trip of integer
                # depth pixels through rounded RGB coordinates is exact here.
                assert width>=dw and height>=dh
                dp=np.rint(pixel*np.array([dw/width,dh/height])).astype(np.int64)
                first=byid[vs[0]['candidate_id']]
                q=next(q for q in first['views'] if q['frame']==f)
                visibility=np.load(Path(q['old_points_path']).parent/'visibility.npy',allow_pickle=False)
                known=(dp[:,0]>=0)&(dp[:,0]<dw)&(dp[:,1]>=0)&(dp[:,1]<dh)
                rgb_known=(pixel[:,0]>=0)&(pixel[:,0]<width)&(pixel[:,1]>=0)&(pixel[:,1]<height)
                depth=np.zeros(len(raw));ix=np.flatnonzero(known);depth[ix]=sensor[dp[ix,1],dp[ix,0]]
                expected=visibility&known&rgb_known&(depth>0)&(camera[:,2]>0)&(np.abs(depth-camera[:,2])<=c['parameters']['measured_depth_tolerance'])
                different=np.flatnonzero(expected!=stored)
                assert not len(different),(s,f,'sensor visibility mismatch',len(different))
                tally['sensor_visibility_frames']+=1;tally['sensor_visibility_points']+=len(raw)
                for v in vs:
                    cid=v['candidate_id'];vi=v['view_index'];parent=arrays[cid+'_parent'];anchor=arrays[cid+'_core']
                    visible_anchor=anchor[expected[anchor]]
                    assert np.isin(v['prompt_ids'],visible_anchor).all()
                    with np.load(gd/'observations'/f'{cid}_v{vi}.npz',allow_pickle=False) as masks:
                        for arm in run.ARMS:
                            mask=masks[arm];inside=np.zeros(len(raw),bool);excluded=np.zeros(len(raw),bool)
                            ix=np.flatnonzero(rgb_known)
                            inside[ix]=mask[pixel[ix,1],pixel[ix,0]]
                            margin=binary_dilation(mask,iterations=c['parameters']['pixel_exclusion_margin'])
                            excluded[ix]=~margin[pixel[ix,1],pixel[ix,0]]
                            assert np.array_equal(ev[f'{cid}_v{vi}_{arm}_inside'],inside[parent])
                            assert np.array_equal(ev[f'{cid}_v{vi}_{arm}_excluded'],excluded[parent])
                            assert np.array_equal(ev[f'{cid}_v{vi}_{arm}_reliable'],expected[parent])
                            assert np.array_equal(ev[f'{cid}_v{vi}_{arm}_region'],np.flatnonzero(expected&inside))
                            tally['mask_evidence_arm_views']+=1
                    tally['old_box_SAM_replays']+=1
            for p in plans:
                parent=arrays[p['candidate_id']+'_parent'];oi=overlap(parent,gt,ids,sizes)
                gid=int(ids[int(np.argmax(oi))]) if len(ids) and oi.max()>0 else None
                hit=int(np.sum(gt[parent]==gid)) if gid is not None else 0
                purity=hit/len(parent);targetsize=int(np.sum(gt==gid)) if gid is not None else 0
                row=dict(scene=s,candidate_id=p['candidate_id'],source=p['source'],anchored=p['accepted'],
                         best_valid_gt=gid,best_iou=float(oi.max(initial=0)),purity=purity,
                         target_recall=hit/max(1,targetsize),pollution_points=len(parent)-hit,
                         diagnostic_pollution_fraction_over05=bool(gid is not None and purity<.95),
                         diagnostic_pollution_fraction_over25=bool(gid is not None and purity<.75))
                rows.append(row)
                for name in ('all',p['source'],'anchored' if p['accepted'] else 'anchor_abstain'):
                    group[name]['parents']+=1;group[name]['no_valid_target']+=gid is None
                    group[name]['best_iou_gt075']+=row['best_iou']>.75
                    group[name]['pollution_fraction_gt05']+=row['diagnostic_pollution_fraction_over05']
                    group[name]['pollution_fraction_gt25']+=row['diagnostic_pollution_fraction_over25']
    result=dict(complete=True,checks=dict(tally),sample_groups={n:dict(v) for n,v in group.items()},
                sample_rows=rows,raw_GT_read=True,postseal_diagnosis_only=True,
                new_training=False,new_GPU_inference=False,new_geometry=False,new_instance_AP=False,
                interpretation='Pollution descriptors are post-GT diagnostic strata, not candidate selection or a new gate. Identity chosen by best valid-target IoU; unknown/ignored cannot be interpreted as clean.',
                inputs={str(run.OUT/n):sha(run.OUT/n) for n in ('contract.json','geometry_frozen.json','independent_verified.json')},
                script_sha256=sha(__file__))
    freeze_json(run.OUT/'supplemental_audit.json',result)
    lines=['# 观测重建与样本范围补充','',
           f"独立从原始传感器深度、pose、原缓存visibility重建{tally['sensor_visibility_frames']}帧／{tally['sensor_visibility_points']}点可靠可见性，逐位一致；从已保存两臂mask重建{tally['mask_evidence_arm_views']}个臂—视图的包含、带边距排除和可见区域点集，逐位一致。原实验{tally['old_box_SAM_replays']}次旧框SAM重放属于原执行证据，本补充没有新增GPU推理。",'',
           '|样本组|父候选|无有效GT交集|最佳IoU>.75|相对最佳GT污染>5%|污染>25%|',
           '|---|---:|---:|---:|---:|---:|']
    for n,v in group.items():lines.append('|'+n+'|'+'|'.join(str(v[k]) for k in
             ('parents','no_valid_target','best_iou_gt075','pollution_fraction_gt05','pollution_fraction_gt25'))+'|')
    lines+=['','污染分组仅是封存后诊断；无有效目标不能被说成干净候选，主GT也不能代表候选内所有身份。全部80候选仍为冻结非GT选择，没有按上述结果补选污染样本。',
            '', '本轮边界重提示未在既定观察/锚点/局部动作下形成净几何增量；有效锚与动作覆盖有限，不能据此认定所有污染分离或新提示分割没有空间。任何改变采样、核心、观测或动作都需新合同，不从本轮结果现场扫参数。',
            '', 'SL/生产权重再次核对不变；只复核现有封存输出并后置读取10场景GT，没有新训练、预测点集、自身语义或实例AP。']
    path=run.OUT/'supplemental_report.md';text='\n'.join(lines)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    freeze_json(run.OUT/'supplemental_verified.json',dict(complete=True,
                outputs={str(run.OUT/n):sha(run.OUT/n) for n in ('supplemental_audit.json','supplemental_report.md')},
                raw_GT_read_postseal=True,new_training=False,new_GPU_inference=False,new_instance_AP=False))
    print(dict(tally),flush=True)


if __name__=='__main__':main()
