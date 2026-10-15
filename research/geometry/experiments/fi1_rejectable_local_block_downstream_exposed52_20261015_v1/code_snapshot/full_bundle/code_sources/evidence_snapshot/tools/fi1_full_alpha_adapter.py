"""Reuse the complete DM-SMS Alpha recipe without argmax or SMS deletion."""
from pathlib import Path

import numpy as np
from PIL import Image

from tools.build_dm_sms1_alpha_view_manifest import SCALE_EXPANSIONS, select_top_visible_views, square_crop_box
from tools.build_official100_fi1_b2_multisemantic import sha256_file
from tools.fi1_unified_evidence_core import aggregate_alpha


def make_views(points, context, max_views=20):
    if max_views != 20:
        raise ValueError('full Alpha adapter fixes at most 20 views')
    if not len(points):
        return []
    visibility = context['visibility']
    if points[-1] >= visibility.shape[1]:
        raise ValueError('Alpha geometry exceeds projection domain')
    world = context['world']
    frames = [Path(p).stem for p in world.color_paths]
    counts = visibility[:, points].sum(axis=1)
    height, width = map(int, world.image_resolution)
    sy, sx = context['scaling']
    views = []
    for index in select_top_visible_views(counts, frames, max_views):
        visible = points[visibility[index, points]]
        coords = context['projections'][index, visible] / [sx, sy]
        if not np.isfinite(coords).all() or (coords < 0).any() or (coords[:,0] > width-1).any() or (coords[:,1] > height-1).any():
            raise ValueError('invalid visible Alpha projection')
        tight = [float(coords[:,0].min()), float(coords[:,1].min()), float(coords[:,0].max()), float(coords[:,1].max())]
        views.append({'frame_id': frames[index], 'frame_index': int(index),
            'visible_point_count': len(visible), 'visible_ratio': len(visible)/len(points),
            'sam_box_prompt_xyxy': tight, 'image_path': str(Path(world.color_paths[index]).resolve()),
            'image_sha256': sha256_file(world.color_paths[index]), 'image_shape': [height, width],
            'crop_scales': [{'scale_index': j, 'expansion': expansion,
                'crop_xyxy': square_crop_box(tight, expansion, width, height)[1]}
                for j, expansion in enumerate(SCALE_EXPANSIONS)]})
    return views


def export_alpha(views, state, predictor, artifact_root, *, encoder=None):
    """One SAM call per geometry/view, one encoding per valid view/scale.

    Invalid transformed alpha excludes the entire three-scale view, preserving
    the visibility-weighted feature aggregation recipe for remaining views.
    """
    if encoder is None:
        from tools.build_dm_sms1_alpha_embedding_ledger import _encode_alpha_batch
        encoder = _encode_alpha_batch
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=False)
    kept, ratios, results = [], [], []
    for rank, original in enumerate(views):
        row = {**original, 'available': False, 'missing_reason': None, 'scales': []}
        results.append(row)
        with Image.open(row['image_path']) as source:
            rgb = source.convert('RGB')
        if sha256_file(row['image_path']) != row['image_sha256'] or list(np.asarray(rgb).shape[:2]) != row['image_shape']:
            raise ValueError('Alpha RGB changed since view selection')
        predictor.set_image(np.ascontiguousarray(np.asarray(rgb)))
        masks, ious, _ = predictor.predict(box=np.asarray(row['sam_box_prompt_xyxy'], dtype=np.float32),
                                           multimask_output=True, return_logits=False)
        masks, ious = np.asarray(masks), np.asarray(ious)
        if masks.ndim != 3 or masks.shape[1:] != tuple(row['image_shape']) or ious.shape != (len(masks),):
            raise ValueError('SAM output dimensions differ')
        if not len(ious) or not np.isfinite(ious).all():
            row['missing_reason'] = 'invalid_sam_scores'
            continue
        selected = int(np.argmax(ious))
        mask = masks[selected].astype(bool)
        mask_path = root / f'view{rank}_sam.png'
        Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
        row.update(sam_selected_mask_index=selected, sam_predicted_iou=float(ious[selected]),
                   sam_mask_area=int(mask.sum()), sam_mask_path=mask_path.name,
                   sam_mask_sha256=sha256_file(mask_path))
        if not mask.any():
            row['missing_reason'] = 'empty_sam_mask'
            continue
        images, alphas = [], []
        for scale in row['crop_scales']:
            box = scale['crop_xyxy']
            x1,y1,x2,y2 = box
            crop = rgb.crop(tuple(box))
            alpha = Image.fromarray(mask[y1:y2,x1:x2].astype(np.uint8)*255)
            transformed = state['mask_transform'](alpha)
            effective = int((transformed > 0).sum().item())
            name = f"view{rank}_scale{scale['scale_index']}"
            crop.save(root / (name+'_rgb.png'))
            alpha.save(root / (name+'_alpha.png'))
            row['scales'].append({**scale, 'preprocessed_foreground_pixels': effective,
                'input_resolution': int(state['input_resolution']), 'rgb_path': name+'_rgb.png',
                'rgb_sha256': sha256_file(root/(name+'_rgb.png')), 'mask_path': name+'_alpha.png',
                'mask_sha256': sha256_file(root/(name+'_alpha.png'))})
            images.append(crop); alphas.append(alpha)
        if not all(s['preprocessed_foreground_pixels'] > 0 for s in row['scales']):
            row['missing_reason'] = 'empty_preprocessed_mask'
            continue
        features = np.asarray(encoder(state, images, alphas), dtype=np.float32)
        if features.shape != (3, state['feature_dim']):
            raise ValueError('Alpha encoder output dimensions differ')
        # Validate all scale and text features before persisting a usable view.
        aggregate_alpha(features[None], [row['visible_ratio']], state['text_features'])
        path = root / f'view{rank}_features.npy'
        np.save(path, features, allow_pickle=False)
        row.update(available=True, feature_path=path.name, feature_sha256=sha256_file(path))
        kept.append(features); ratios.append(row['visible_ratio'])
    if kept:
        summary = aggregate_alpha(np.stack(kept), ratios, state['text_features'])
    else:
        summary = {'available': False, 'cosine': None, 'view_scale_std': None,
                   'aggregate_feature': None, 'missing_reason': 'no_valid_three_scale_view'}
    return {**summary, 'views': results, 'selected_view_count': len(views), 'valid_view_count': len(kept),
            'recipe': '20 visible views; SAM ViT-B best predicted IoU; scales 0/.2/.4; L2Norm(sum visibility*feature)',
            'sms_applied': False, 'top1_applied': False}
