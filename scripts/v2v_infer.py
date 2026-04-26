#!/usr/bin/env python3
"""
PyTorch V2V inference (ControlNet-style) with a temporal latent KV cache.

This is a checkpoint-agnostic *inference harness*: to get photorealistic results,
swap in your trained ControlNet/video-to-video weights (or replace the model class).

Inputs
------
- G-buffer: Depth + Object Mask tensors saved as a Torch file (.pt).
  Supported layouts:
    1) Single-frame dict:
         {"depth": Float[H,W] or [1,H,W], "mask": Long[H,W] or [1,H,W]}
    2) Video dict:
         {"depth": Float[T,H,W], "mask": Long[T,H,W]}
- Text prompt: a string

Outputs
-------
- Per-frame RGB in PPM format (no extra deps) under --out-dir.

Run
---
python3 scripts/v2v_infer.py \
  --gbuffer-pt /abs/path/gbuffer.pt \
  --prompt "golden hour, cinematic, photoreal" \
  --out-dir /abs/path/out \
  --checkpoint /abs/path/model.pt
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class TemporalKVCache:
    """
    Minimal temporal cache passed between frames.

    - prev_latent: the previous frame's final latent (temporal consistency anchor)
    - kv: optional model-specific attention keys/values (if your model supports it)
    """

    prev_latent: torch.Tensor | None = None
    kv: Mapping[str, torch.Tensor] | None = None


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    frames: int | None = None  # None -> infer from gbuffer
    steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 0
    device: str = "cuda"
    dtype: str = "float16"
    latent_channels: int = 4
    latent_downsample: int = 8
    # Physics/conditioning scaling
    depth_scale: float = 1.0
    mask_scale: float = 1.0


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_gbuffer_pt(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (depth, mask).

    - depth: Float[T,H,W]
    - mask: Long[T,H,W]
    """

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("G-buffer .pt must be a dict with keys depth/mask")
    if "depth" not in payload or "mask" not in payload:
        raise ValueError("G-buffer .pt missing 'depth' or 'mask'")

    depth = payload["depth"]
    mask = payload["mask"]
    if not torch.is_tensor(depth) or not torch.is_tensor(mask):
        raise ValueError("depth/mask must be torch tensors")

    depth = depth.float()
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    elif depth.ndim == 3:
        pass
    else:
        raise ValueError("depth must have shape [H,W] or [T,H,W]")

    if mask.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
        mask = mask.long()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 3:
        pass
    else:
        raise ValueError("mask must have shape [H,W] or [T,H,W]")

    if depth.shape[0] != mask.shape[0] or depth.shape[1:] != mask.shape[1:]:
        raise ValueError(f"depth/mask shape mismatch: {tuple(depth.shape)} vs {tuple(mask.shape)}")

    return depth, mask.long()


def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    """
    Normalizes depth per-frame to [0,1] robustly (percentile-like clamp via min/max).
    depth: Float[T,H,W]
    returns: Float[T,1,H,W]
    """

    t, h, w = depth.shape
    d = depth.view(t, -1)
    dmin = d.min(dim=1).values.view(t, 1, 1, 1)
    dmax = d.max(dim=1).values.view(t, 1, 1, 1)
    out = (depth.view(t, 1, h, w) - dmin) / (dmax - dmin + 1e-6)
    return out.clamp(0.0, 1.0)


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Converts integer object index mask into a [0,1] float channel.
    mask: Long[T,H,W]
    returns: Float[T,1,H,W]
    """

    t, h, w = mask.shape
    m = mask.view(t, -1)
    mmax = m.max(dim=1).values.view(t, 1, 1, 1).float()
    out = mask.view(t, 1, h, w).float() / (mmax + 1.0)
    return out.clamp(0.0, 1.0)


def build_control(depth: torch.Tensor, mask: torch.Tensor, *, cfg: InferenceConfig) -> torch.Tensor:
    """
    Control tensor at full resolution: Float[T,2,H,W] (depth, mask).
    """

    d = normalize_depth(depth) * cfg.depth_scale
    m = normalize_mask(mask) * cfg.mask_scale
    return torch.cat([d, m], dim=1)


class SimpleTextEncoder(nn.Module):
    """
    Tiny local text encoder so the harness runs without extra deps.

    Replace with a real text encoder (e.g., T5/CLIP) for meaningful prompt control.
    """

    def __init__(self, embed_dim: int = 768, vocab_size: int = 4096) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.vocab_size = vocab_size

    def forward(self, prompt: str, *, device: torch.device) -> torch.Tensor:
        # Hash bytes into token ids (stable, simple).
        b = prompt.encode("utf-8", errors="ignore")
        if not b:
            ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            ids = torch.tensor([[(x % self.vocab_size) for x in b[:256]]], dtype=torch.long, device=device)
        emb = self.embed(ids)  # [1, L, D]
        pooled = emb.mean(dim=1)  # [1, D]
        return self.proj(pooled)  # [1, D]


class ControlNetStyleV2V(nn.Module):
    """
    Minimal ControlNet-style latent diffusion core.

    Forward signature is designed to support temporal KV cache:
      noise_pred, new_cache = model(latent, t, text_emb, control, cache)

    Swap this module with your trained 2026 V2V/ControlNet model if desired.
    """

    def __init__(self, *, latent_ch: int = 4, text_dim: int = 768, control_ch: int = 2) -> None:
        super().__init__()
        self.latent_ch = latent_ch
        self.text_dim = text_dim

        self.control_down = nn.Sequential(
            nn.Conv2d(control_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.temporal_in = nn.Conv2d(latent_ch, 64, 1)

        self.text_film = nn.Sequential(nn.Linear(text_dim, 128), nn.SiLU(), nn.Linear(128, 128))
        self.unet = nn.Sequential(
            nn.Conv2d(latent_ch, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, latent_ch, 3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        latent: torch.Tensor,  # [B,C,h,w]
        t: torch.Tensor,  # [B]
        text_emb: torch.Tensor,  # [B,D]
        control_fullres: torch.Tensor,  # [B,2,H,W]
        cache: TemporalKVCache,
    ) -> tuple[torch.Tensor, TemporalKVCache]:
        b, _, h, w = latent.shape
        control = F.interpolate(control_fullres, size=(h, w), mode="bilinear", align_corners=False)
        cfeat = self.control_down(control)

        if cache.prev_latent is None:
            tfeat = torch.zeros((b, 64, h, w), device=latent.device, dtype=latent.dtype)
        else:
            prev = cache.prev_latent
            if prev.shape[-2:] != (h, w):
                prev = F.interpolate(prev, size=(h, w), mode="bilinear", align_corners=False)
            tfeat = self.temporal_in(prev)

        film = self.text_film(text_emb).view(b, 128, 1, 1)
        scale = torch.tanh(film[:, :64])
        shift = film[:, 64:]
        x = latent
        h0 = self.unet[0](x)
        h0 = h0 + (cfeat * (1.0 + scale) + shift) + tfeat
        h0 = self.unet[1](h0)
        h0 = self.unet[2](h0)
        h0 = self.unet[3](h0)
        noise_pred = self.unet[4](h0)

        # Update KV cache with something tangible; in a real model, you'd pass attention KV tensors.
        new_cache = TemporalKVCache(prev_latent=cache.prev_latent, kv=cache.kv)
        return noise_pred, new_cache

    def decode(self, latent: torch.Tensor, *, out_hw: tuple[int, int]) -> torch.Tensor:
        x = self.decoder(latent)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


def ddim_step(x: torch.Tensor, eps: torch.Tensor, *, alpha: float, alpha_prev: float, eta: float = 0.0) -> torch.Tensor:
    """
    A small DDIM-like update with a simple scalar schedule.
    This is intentionally minimal; swap in your scheduler if desired.
    """

    # Predict x0 and step toward it.
    sqrt_alpha = alpha**0.5
    sqrt_one_minus = (1.0 - alpha) ** 0.5
    x0 = (x - sqrt_one_minus * eps) / (sqrt_alpha + 1e-8)

    sqrt_alpha_prev = alpha_prev**0.5
    sigma = eta * ((1.0 - alpha_prev) / (1.0 - alpha) * (1.0 - alpha / alpha_prev)) ** 0.5 if alpha_prev > 0 else 0.0
    dir_xt = ((1.0 - alpha_prev - sigma * sigma) ** 0.5) * eps
    if sigma > 0:
        noise = sigma * torch.randn_like(x)
    else:
        noise = torch.zeros_like(x)
    return sqrt_alpha_prev * x0 + dir_xt + noise


def simple_alpha_schedule(step: int, steps: int) -> float:
    # Cosine-ish schedule in (0,1]; stable enough for a harness.
    s = (step + 1) / steps
    return max(1e-4, float((1.0 - s) ** 2))


def _encode_prompt(text_encoder: SimpleTextEncoder, prompt: str, *, device: torch.device) -> torch.Tensor:
    return text_encoder(prompt, device=device)


@torch.no_grad()
def run_ddim(
    *,
    model: ControlNetStyleV2V,
    text_emb: torch.Tensor,
    control_fullres: torch.Tensor,
    cfg: InferenceConfig,
    cache: TemporalKVCache,
) -> tuple[torch.Tensor, TemporalKVCache]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    h_full, w_full = int(control_fullres.shape[-2]), int(control_fullres.shape[-1])
    h_lat = max(1, h_full // cfg.latent_downsample)
    w_lat = max(1, w_full // cfg.latent_downsample)

    g = torch.Generator(device=device)
    g.manual_seed(cfg.seed)
    x = torch.randn((1, cfg.latent_channels, h_lat, w_lat), device=device, dtype=dtype, generator=g)

    # Classifier-free guidance: unconditional embedding = zeros.
    uncond = torch.zeros_like(text_emb)
    for i in range(cfg.steps):
        t = torch.full((1,), float(cfg.steps - i), device=device, dtype=dtype)

        eps_u, cache_u = model(x, t, uncond, control_fullres, cache)
        eps_c, cache_c = model(x, t, text_emb, control_fullres, cache)
        eps = eps_u + cfg.guidance_scale * (eps_c - eps_u)

        alpha = simple_alpha_schedule(i, cfg.steps)
        alpha_prev = simple_alpha_schedule(i + 1, cfg.steps) if i + 1 < cfg.steps else 1e-4
        x = ddim_step(x, eps, alpha=alpha, alpha_prev=alpha_prev, eta=0.0)

        # Keep a simple cache behavior: allow the conditional path to update.
        cache = cache_c

    rgb = model.decode(x, out_hw=(h_full, w_full))
    # Update temporal cache for next frame.
    cache.prev_latent = x.detach()
    return rgb, cache


def save_ppm(path: str, rgb01: torch.Tensor) -> None:
    """
    Saves a single image as binary PPM (P6). Expects rgb in [0,1], shape [1,3,H,W] or [3,H,W].
    """

    if rgb01.ndim == 4:
        rgb01 = rgb01[0]
    if rgb01.shape[0] != 3:
        raise ValueError("Expected RGB with 3 channels")
    rgb = (rgb01.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).contiguous().cpu()
    h, w, _ = rgb.shape
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        try:
            f.write(rgb.numpy().tobytes())  # type: ignore[no-any-return]
        except Exception:
            # Fallback if NumPy is unavailable in the runtime environment.
            f.write(bytes(rgb.view(-1).tolist()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gbuffer-pt", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint", required=False, default=None, help="Optional .pt state_dict for the model stub.")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--latent-downsample", type=int, default=8)
    p.add_argument("--frames", type=int, default=None, help="Override number of frames to render.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = InferenceConfig(
        frames=args.frames,
        steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
        latent_downsample=int(args.latent_downsample),
    )

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    dtype = _dtype_from_name(cfg.dtype)

    depth, mask = load_gbuffer_pt(args.gbuffer_pt)
    control = build_control(depth, mask, cfg=cfg)  # [T,2,H,W]
    t_total = int(control.shape[0]) if cfg.frames is None else min(int(cfg.frames), int(control.shape[0]))

    model = ControlNetStyleV2V(latent_ch=cfg.latent_channels).to(device=device, dtype=dtype)
    text_encoder = SimpleTextEncoder().to(device=device, dtype=dtype)

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(sd, Mapping) and "state_dict" in sd:
            sd = sd["state_dict"]
        if not isinstance(sd, Mapping):
            raise ValueError("Checkpoint must be a state_dict dict or an object with key 'state_dict'")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print("Loaded checkpoint with non-strict match.")
            if missing:
                print("  missing:", missing)
            if unexpected:
                print("  unexpected:", unexpected)

    cache = TemporalKVCache(prev_latent=None, kv=None)
    os.makedirs(args.out_dir, exist_ok=True)

    text_emb = _encode_prompt(text_encoder, args.prompt, device=device)

    for t in range(t_total):
        control_t = control[t : t + 1].to(device=device, dtype=dtype)  # [1,2,H,W]
        rgb, cache = run_ddim(model=model, text_emb=text_emb, control_fullres=control_t, cfg=cfg, cache=cache)
        out_path = os.path.join(args.out_dir, f"frame_{t:04d}.ppm")
        save_ppm(out_path, rgb.float())


if __name__ == "__main__":
    main()
