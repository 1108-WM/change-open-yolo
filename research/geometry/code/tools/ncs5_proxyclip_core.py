#!/usr/bin/env python3
"""Frozen local-only model and pooling primitives for NCS-5."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROXYCLIP_ROOT = PROJECT_ROOT / "_external/ProxyCLIP"
CLIP_CHECKPOINT = PROJECT_ROOT / "pretrained/checkpoints/ViT-B-16.pt"
DINO_CHECKPOINT = PROJECT_ROOT / "pretrained/checkpoints/dinov2_vits14_pretrain.pth"
EXPECTED_CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
EXPECTED_DINO_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
FRAME_HEIGHT = 336
FRAME_WIDTH = 448
TOKEN_HEIGHT = 24
TOKEN_WIDTH = 32
TOKEN_DIM = 512
BETA = 1.2
GAMMA = 3.0
CLIP_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
DINO_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_assets(
    clip_checkpoint: Path = CLIP_CHECKPOINT,
    dino_checkpoint: Path = DINO_CHECKPOINT,
) -> dict[str, str]:
    clip_checkpoint = Path(clip_checkpoint)
    dino_checkpoint = Path(dino_checkpoint)
    if not clip_checkpoint.is_file() or not dino_checkpoint.is_file():
        raise FileNotFoundError("NCS-5 checkpoint is missing")
    hashes = {
        "clip_sha256": sha256_file(clip_checkpoint),
        "dino_sha256": sha256_file(dino_checkpoint),
    }
    if hashes["clip_sha256"] != EXPECTED_CLIP_SHA256:
        raise ValueError("OpenAI CLIP ViT-B/16 checkpoint SHA-256 mismatch")
    if hashes["dino_sha256"] != EXPECTED_DINO_SHA256:
        raise ValueError("DINOv2 ViT-S/14 checkpoint SHA-256 mismatch")
    return hashes


def _import_proxyclip_open_clip():
    root = str(PROXYCLIP_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import open_clip  # type: ignore

    module_path = Path(open_clip.__file__).resolve()
    if PROXYCLIP_ROOT.resolve() not in module_path.parents:
        raise RuntimeError(f"wrong open_clip implementation imported: {module_path}")
    return open_clip


def load_proxyclip_model(
    checkpoint: Path = CLIP_CHECKPOINT,
    device: str | torch.device = "cpu",
):
    """Load the custom ProxyCLIP ViT-B/16 without any download path."""
    checkpoint = Path(checkpoint)
    if sha256_file(checkpoint) != EXPECTED_CLIP_SHA256:
        raise ValueError("OpenAI CLIP ViT-B/16 checkpoint SHA-256 mismatch")
    open_clip = _import_proxyclip_open_clip()
    model = open_clip.create_model(
        "ViT-B-16", pretrained=None, precision="fp32", device="cpu"
    )
    # The official OpenAI file is a TorchScript archive.  Explicit jit loading
    # avoids torch 1.12 dispatching through torch.load without map_location.
    state = torch.jit.load(str(checkpoint), map_location="cpu").state_dict()
    for key in ("input_resolution", "context_length", "vocab_size"):
        state.pop(key, None)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"strict CLIP load failed: {incompatible}")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_dinov2_model(
    checkpoint: Path = DINO_CHECKPOINT,
    device: str | torch.device = "cpu",
):
    """Load frozen DINOv2-S/14 at the registered 336x448 frame size."""
    checkpoint = Path(checkpoint)
    if sha256_file(checkpoint) != EXPECTED_DINO_SHA256:
        raise ValueError("DINOv2 ViT-S/14 checkpoint SHA-256 mismatch")
    try:
        import timm
        from timm.layers.pos_embed import resample_abs_pos_embed
    except ImportError as error:
        raise RuntimeError("timm with resample_abs_pos_embed is required") from error
    model = timm.create_model(
        "vit_small_patch14_dinov2",
        pretrained=False,
        num_classes=0,
        img_size=(FRAME_HEIGHT, FRAME_WIDTH),
    )
    state = torch.load(str(checkpoint), map_location="cpu")
    state["pos_embed"] = resample_abs_pos_embed(
        state["pos_embed"],
        new_size=(TOKEN_HEIGHT, TOKEN_WIDTH),
        num_prefix_tokens=1,
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or set(incompatible.unexpected_keys) != {"mask_token"}:
        raise ValueError(
            "DINOv2 checkpoint mismatch: "
            f"missing={incompatible.missing_keys[:5]} "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def preprocess_full_rgb(image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("RGB image must be uint8 HxWx3")
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            (FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.BICUBIC
        ),
        dtype=np.float32,
    ) / 255.0
    clip = ((resized - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)
    dino = ((resized - DINO_MEAN) / DINO_STD).transpose(2, 0, 1)
    return torch.from_numpy(clip.copy()), torch.from_numpy(dino.copy())


@torch.inference_mode()
def extract_dense_tokens(
    proxyclip_model,
    dinov2_model,
    image: np.ndarray,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    clip_tensor, dino_tensor = preprocess_full_rgb(image)
    clip_tensor = clip_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    dino_tensor = dino_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    external = dinov2_model.get_intermediate_layers(
        dino_tensor, n=1, reshape=True
    )[0]
    if tuple(external.shape) != (1, 384, TOKEN_HEIGHT, TOKEN_WIDTH):
        raise ValueError(f"unexpected DINOv2 feature shape: {tuple(external.shape)}")
    tokens = proxyclip_model.encode_image(
        clip_tensor, external_feats=external, beta=BETA, gamma=GAMMA
    )
    if tuple(tokens.shape) != (1, TOKEN_HEIGHT * TOKEN_WIDTH, TOKEN_DIM):
        raise ValueError(f"unexpected ProxyCLIP token shape: {tuple(tokens.shape)}")
    tokens = F.normalize(tokens.float(), dim=-1)
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("ProxyCLIP produced non-finite dense tokens")
    return tokens[0].reshape(TOKEN_HEIGHT, TOKEN_WIDTH, TOKEN_DIM).cpu()


def mask_token_coverage(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    resized = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
            (FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    ) > 0
    coverage = resized.reshape(TOKEN_HEIGHT, 14, TOKEN_WIDTH, 14).mean(axis=(1, 3))
    if coverage.shape != (TOKEN_HEIGHT, TOKEN_WIDTH):
        raise AssertionError("invalid token coverage shape")
    return coverage.astype(np.float32)


def pool_mask_tokens(tokens: torch.Tensor | np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    values = torch.as_tensor(tokens, dtype=torch.float32)
    if tuple(values.shape) != (TOKEN_HEIGHT, TOKEN_WIDTH, TOKEN_DIM):
        raise ValueError(f"unexpected dense token shape: {tuple(values.shape)}")
    coverage = torch.from_numpy(mask_token_coverage(mask))
    denominator = float(coverage.sum())
    if denominator <= 0:
        return None
    pooled = (values * coverage.unsqueeze(-1)).sum(dim=(0, 1)) / denominator
    pooled = F.normalize(pooled, dim=0)
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError("mask-token pooling produced non-finite values")
    return pooled.numpy().astype(np.float32)


def aggregate_visible_views(features: np.ndarray, visible_ratios: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    weights = np.asarray(visible_ratios, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != TOKEN_DIM or not len(features):
        raise ValueError("view features must be a non-empty Nx512 matrix")
    if weights.shape != (len(features),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("visible ratios must be finite positive weights")
    pooled = np.sum(features * weights[:, None], axis=0) / float(weights.sum())
    norm = float(np.linalg.norm(pooled))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("visibility aggregation produced invalid norm")
    return (pooled / norm).astype(np.float32)
