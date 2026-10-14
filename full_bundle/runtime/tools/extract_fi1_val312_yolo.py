#!/usr/bin/env python3
"""Regenerate frozen val312 YOLO evidence with input hashes and resumable chunks."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_fi1_r0_r1_inputs import read, write, sha, digest


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify(files):
    for path, expected in files.items():
        require(sha(path) == expected, 'bound file changed: ' + str(path))


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    write(tmp, value)
    tmp.replace(path)


def paths_for_scene(root, scene, frequency):
    data = root / scene
    ids = list(range(0, len(list((data / 'poses').iterdir())), frequency))
    require(bool(ids), 'no sampled frames')
    return ids, [data / 'color' / f'{i}.jpg' for i in ids]


def validate_predictions(predictions, ids, class_count=198):
    import torch
    require(set(predictions) == {str(i) for i in ids}, 'missing/extra prediction frames')
    boxes = 0
    for row in predictions.values():
        b, c, s = (row[k] for k in ('bbox', 'labels', 'scores'))
        require(all(torch.is_tensor(v) and v.device.type == 'cpu' for v in (b, c, s)), 'CPU tensors required')
        require(b.ndim == 2 and b.shape[1] == 4 and c.ndim == s.ndim == 1 and len(b) == len(c) == len(s), 'invalid shapes')
        require(len(b) <= 100 and c.dtype in (torch.int64, torch.int32), 'invalid labels/budget')
        require(bool(torch.isfinite(b).all()) and bool(torch.isfinite(s).all()), 'nonfinite detection')
        require(bool(((c >= 0) & (c < class_count)).all()), 'out-of-vocabulary detection')
        require(bool((b[:, 2:] >= b[:, :2]).all()) and bool(((s > .08) & (s <= 1)).all()), 'invalid boxes/scores')
        boxes += len(b)
    return boxes


def code_files():
    paths = [Path(__file__).resolve(), ROOT/'utils/utils_2d.py', ROOT/'utils/__init__.py',
             ROOT/'tools/prepare_fi1_r0_r1_inputs.py']
    for folder in [ROOT/'models/YOLO-World/yolo_world', ROOT/'models/YOLO-World/third_party/mmyolo/configs']:
        paths.extend(p for p in folder.rglob('*.py') if 'work_dirs' not in p.parts)
    return {str(p): sha(p) for p in sorted(set(paths))}


def prepare(args):
    from PIL import Image
    prep, out = args.preparation.resolve(), args.output.resolve()
    frozen = read(prep/'yolo_frozen_configuration.json')
    inv = read(prep/'inventory/scene_inventory.json')
    sources = read(prep/'inventory/input_sources.json')
    root = Path(sources['sources']['val312']['prepared_root'])
    require(not out.exists(), 'refusing existing output directory')
    files = {str(prep/f): sha(prep/f) for f in ['yolo_frozen_configuration.json', 'inventory/scene_inventory.json', 'inventory/input_sources.json', 'inventory/val312.txt']}
    for name in ('config', 'model_config', 'weight'):
        key = 'config_path' if name == 'config' else name + '_file'
        files[frozen[key]] = frozen[name + '_sha256']
    files.update({str(p): sha(p) for p in (ROOT/'pretrained/clip-vit-base-patch32').iterdir() if p.is_file()})
    verify(files)
    out.mkdir(parents=True)
    jobs = []
    for row in inv:
        scene = row['scene_name']
        ids, paths = paths_for_scene(root, scene, 10)
        require(ids == row['yolo']['frame_ids'], 'sampled frame set differs from preparation')
        hashes, resolution = {}, None
        for image in paths:
            hashes[str(image)] = sha(image)
            with Image.open(image) as im:
                hw = [im.height, im.width]
            if resolution is None:
                resolution = hw
            require(hw == resolution, 'mixed scene image dimensions')
        with Image.open(root/scene/'depth'/f'{ids[0]}.png') as im:
            depth = [im.height, im.width]
        jobs.append({'scene_name': scene, 'frame_ids': ids, 'images': hashes,
                     'image_resolution': resolution, 'depth_resolution': depth})
        print('bound RGB', scene, len(ids), flush=True)
    require(len(jobs) == 312 and sum(len(j['frame_ids']) for j in jobs) == 54090, 'full312 frame coverage differs')
    plan = {'version': 'fi1_val312_fresh_yolo', 'preparation': str(prep), 'output': str(out),
            'prepared_root': str(root), 'records_root': sources['sources']['val312']['records_root'],
            'frozen': frozen, 'inputs': files, 'code': code_files(), 'scenes': jobs,
            'chunk_size': 32, 'new_training': False, 'Qwen': False, 'GT_content_read': False,
            'legacy_provenance': 'No verifiable generation receipt found; old caches remain unchanged. Fresh inference required.'}
    write(out/'plan.json', plan)
    print('Frozen fresh YOLO plan', out/'plan.json', flush=True)


def initialize_network(plan):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(ROOT/'models/YOLO-World'))
    import torch
    import yaml
    from utils.utils_2d import Network_2D
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(1)
    config = yaml.safe_load(Path(plan['frozen']['config_path']).read_text())
    require(config['network2d'] == plan['frozen']['network2d'], '2D config changed')
    network = Network_2D(config)
    runtime = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
               'packages': {n: importlib.metadata.version(n) for n in ['mmengine','mmcv','mmdet','mmyolo','transformers','numpy']}}
    return network, config, runtime


def bind_adapter(plan, scene, output_file):
    link = Path(plan['records_root'])/scene/'yoloworld_bboxes_2d'/f'{scene}.pt'
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        require(link.is_symlink() and link.resolve() == output_file.resolve(), 'refusing unrelated adapter path')
    else:
        link.symlink_to(output_file.resolve())


def run(args):
    import torch
    from types import SimpleNamespace
    plan = read(args.plan)
    out = Path(plan['output'])
    require(args.plan.resolve() == out/'plan.json', 'plan path differs')
    verify(plan['inputs']); verify(plan['code'])
    plan_hash = sha(args.plan)
    network = config = runtime = None
    summaries = []
    for job in plan['scenes'][:args.limit_scenes]:
        scene = job['scene_name']; target = out/scene
        receipt = target/'receipt.json'
        if receipt.exists():
            r = read(receipt)
            require(r['plan_sha256'] == plan_hash, 'resume plan differs')
            verify(job['images']); verify(r['outputs'])
            bind_adapter(plan, scene, target/f'{scene}.pt')
            summaries.append(r['summary'])
            continue
        verify(job['images'])
        if network is None:
            network, config, runtime = initialize_network(plan)
            if (out/'runtime.json').exists():
                require(read(out/'runtime.json') == runtime, 'runtime changed on resume')
            else:
                write(out/'runtime.json', runtime)
        target.mkdir(exist_ok=True)
        chunks = target/'chunks'; chunks.mkdir(exist_ok=True)
        started = time.monotonic()
        predictions = {}
        images = [str(Path(plan['prepared_root'])/scene/'color'/f'{i}.jpg') for i in job['frame_ids']]
        require(set(images) == set(job['images']), 'RGB manifest coverage differs')
        network.resolution = None
        for first in range(0, len(images), plan['chunk_size']):
            paths = images[first:first+plan['chunk_size']]
            ids = job['frame_ids'][first:first+plan['chunk_size']]
            chunk = chunks/f'{first:05d}.pt'; cr = chunk.with_suffix('.json')
            if cr.exists():
                old = read(cr)
                require(old['plan_sha256'] == plan_hash and old['images'] == {p: job['images'][p] for p in paths}, 'chunk identity differs')
                verify({str(chunk): old['sha256']})
                part = torch.load(chunk, map_location='cpu')
            else:
                part = network.get_bounding_boxes(paths)
                validate_predictions(part, ids)
                tmp = chunk.with_suffix('.tmp')
                torch.save(part, tmp); tmp.replace(chunk)
                atomic_json(cr, {'plan_sha256': plan_hash, 'images': {p:job['images'][p] for p in paths}, 'sha256':sha(chunk)})
            validate_predictions(part, ids)
            predictions.update(part)
            atomic_json(out/'progress.json', {'complete': False, 'completed_scenes':len(summaries),
                        'active_scene':scene, 'active_frames_complete':len(predictions), 'active_frames_total':len(images),
                        'new_training':False, 'new_AP':False})
        boxes = validate_predictions(predictions, job['frame_ids'])
        from utils import OpenYolo3D
        adapter = OpenYolo3D.__new__(OpenYolo3D)
        adapter.openyolo3d_config = config; adapter.datatype = 'mesh'
        adapter.world2cam = SimpleNamespace(image_resolution=job['image_resolution'], depth_resolution=job['depth_resolution'])
        metadata = adapter._build_2d_cache_metadata(scene, str(Path(plan['prepared_root'])/scene), None)
        path = target/f'{scene}.pt'; tmp = path.with_suffix('.tmp')
        torch.save({'metadata':metadata, 'predictions':predictions, 'provenance':{
            'fresh_inference':True, 'plan_sha256':plan_hash, 'RGB_sha256':job['images'],
            'runtime_sha256':sha(out/'runtime.json')}}, tmp)
        tmp.replace(path)
        verify(job['images']); verify(plan['code'])
        summary = {'scene_name':scene, 'frames':len(images), 'boxes':boxes, 'complete':True,
                   'seconds_this_attempt':time.monotonic()-started, 'fresh_inference':True}
        atomic_json(receipt, {'plan_sha256':plan_hash, 'outputs':{str(path):sha(path)}, 'summary':summary})
        bind_adapter(plan, scene, path)
        summaries.append(summary)
        print('fresh YOLO complete', scene, len(images), boxes, flush=True)
    complete = len(summaries) == len(plan['scenes'])
    atomic_json(out/'summary.json', {'complete':complete, 'scene_count':len(summaries),
        'frame_count':sum(r['frames'] for r in summaries), 'rows':summaries, 'plan_sha256':plan_hash,
        'new_visual_inference':True, 'new_training':False, 'GT_content_read':False, 'new_AP':False})
    atomic_json(out/'progress.json', {'complete':complete, 'completed_scenes':len(summaries)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--preparation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--limit-scenes', type=int, default=312)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'run from the project root')
    def deny_gt(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
            name = str(Path(os.fsdecode(values[0])).resolve())
            if any(s in name for s in ['ground_truth', '.labels.ply', '.aggregation.json', '.segs.json']):
                raise PermissionError('YOLO extraction must not open GT: ' + name)
    sys.addaudithook(deny_gt)
    if args.stage == 'prepare':
        prepare(args)
    else:
        require(0 < args.limit_scenes <= 312, 'invalid scene limit')
        run(args)
