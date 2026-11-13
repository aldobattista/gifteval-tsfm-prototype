"""8-layer Patch-Transformer backbone with origin-anchored RoPE and GroupAttention.

All normalization layers execute in float32 for numerical stability while the
Transformer blocks support mixed ``bfloat16`` execution on Apple Silicon MPS.
Temporal attention masks enforce strict intra-segment causality, blanked-key
exclusion, and an automatic self-diagonal safety guard on 100%-NaN windows;
GroupAttention layers zero-initialize ``out_proj.weight`` and mask by
``has_group_peer`` so univariate rows remain invariant to packing order.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

try:
    from jaxtyping import Bool, Float, Int
except ImportError:  # pragma: no cover

    class _JaxTypingFallback:
        def __getitem__(self, item: object) -> object:
            return Tensor

    Bool = Float = Int = _JaxTypingFallback()  # type: ignore[assignment]

from tsfm_proto.config import TSFMConfig
from tsfm_proto.heads import build_quantile_head


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rms).to(x.dtype) * self.weight.to(x.dtype)


class OriginAnchoredRoPE(nn.Module):
    """Rotary position embeddings indexed by signed patch offset from ``T_origin``."""

    def __init__(self, head_dim: int, max_pos: int = 128, base: float = 10000.0) -> None:
        super().__init__()
        self.max_pos = max_pos
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: Float[Tensor, "br h l hd"],
        k: Float[Tensor, "br h l hd"],
        rel_patch_idx: Int[Tensor, "br l"],
    ) -> tuple[Float[Tensor, "br h l hd"], Float[Tensor, "br h l hd"]]:
        pos = rel_patch_idx.clamp(-self.max_pos, self.max_pos).float()
        angles = torch.einsum("bl,d->bld", pos, self.inv_freq)
        cos = angles.cos().unsqueeze(1).to(q.dtype)
        sin = angles.sin().unsqueeze(1).to(q.dtype)
        return self._rotate_half(q, cos, sin), self._rotate_half(k, cos, sin)

    @staticmethod
    def _rotate_half(
        x: Float[Tensor, "br h l hd"],
        cos: Float[Tensor, "br 1 l hd2"],
        sin: Float[Tensor, "br 1 l hd2"],
    ) -> Float[Tensor, "br h l hd"]:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def build_temporal_mask(
    segment_id: Int[Tensor, "b r l"],
    rel_patch_idx: Int[Tensor, "b r l"],
    patch_obs_any: Bool[Tensor, "b r l"],
    exclude_blank_keys: bool = True,
) -> Bool[Tensor, "br 1 l l"]:
    """Build causal intra-segment temporal mask with blank-key exclusion and all-NaN fallback.

    Within each packed segment, patch ``q`` may attend to ``k`` only when ``k <= q``
    (via ``rel_patch_idx``). When ``exclude_blank_keys`` is enabled, keys whose
    32 timesteps contain zero observed values (missing spans or blanked horizon
    slots) are excluded from softmax keys. If an entire sequence is missing
    (zero observed keys), the self-diagonal fallback prevents ``0/0 = NaN`` in SDPA.
    """
    l_len = segment_id.shape[2]
    same_seg = (segment_id[:, :, :, None] == segment_id[:, :, None, :]) & (segment_id[:, :, :, None] >= 0)
    causal = rel_patch_idx[:, :, :, None] >= rel_patch_idx[:, :, None, :]
    valid_kv = patch_obs_any[:, :, None, :] if exclude_blank_keys else torch.ones_like(causal)
    base_mask = same_seg & causal & valid_kv

    eye_l = torch.eye(l_len, dtype=torch.bool, device=segment_id.device).view(1, 1, l_len, l_len)
    has_any_key = base_mask.any(dim=-1, keepdim=True)
    time_mask = base_mask | (~has_any_key & eye_l)
    return rearrange(time_mask, "b r q k -> (b r) 1 q k")


class PatchResStem(nn.Module):
    """Residual patch encoder mapping ``[B, R, L, P, C_in]`` to ``[B, R, L, D]``."""

    def __init__(self, c_in: int, d_model: int, patch_size: int, norm_eps: float, use_cov_emb: bool) -> None:
        super().__init__()
        in_dim = patch_size * c_in
        self.skip_proj = nn.Linear(in_dim, d_model, bias=False)
        self.fc1 = nn.Linear(in_dim, d_model, bias=True)
        self.fc2 = nn.Linear(d_model, d_model, bias=True)
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.cov_role_emb = nn.Parameter(torch.zeros(d_model)) if use_cov_emb else None

    def forward(
        self,
        patch_inputs: Float[Tensor, "b r l p c_in"],
        patch_is_cov: Bool[Tensor, "b r l"] | None = None,
    ) -> Float[Tensor, "b r l d"]:
        x = rearrange(patch_inputs, "b r l p c -> b r l (p c)")
        h = self.norm(self.skip_proj(x) + self.fc2(F.silu(self.fc1(x))))
        if self.cov_role_emb is not None and patch_is_cov is not None:
            h = h + patch_is_cov.unsqueeze(-1).to(h.dtype) * self.cov_role_emb.to(h.dtype)
        return h


class TemporalAttention(nn.Module):
    """Multi-head temporal self-attention with QK-RMSNorm and origin-anchored RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope_max_pos: int,
        rope_base: float,
        norm_eps: float,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = (d_model // n_heads) if head_dim is None else head_dim
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = OriginAnchoredRoPE(self.head_dim, max_pos=rope_max_pos, base=rope_base)

    def forward(
        self,
        h: Float[Tensor, "b r l d"],
        rel_patch_idx: Int[Tensor, "b r l"],
        time_mask: Bool[Tensor, "br 1 l l"],
    ) -> Float[Tensor, "b r l d"]:
        b, r, _, _ = h.shape
        x = rearrange(h, "b r l d -> (b r) l d")
        rel = rearrange(rel_patch_idx, "b r l -> (b r) l")
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "br l (three h hd) -> three br h l hd", three=3, h=self.n_heads).unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, rel)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=time_mask)
        out = self.out_proj(rearrange(attn_out, "br h l hd -> br l (h hd)"))
        return rearrange(out, "(b r) l d -> b r l d", b=b, r=r)


class GroupAttention(nn.Module):
    """Column-wise cross-variate & past-covariate attention with one-way firewall.

    Guarantees three invariants:
      1. Zero-initialized ``out_proj.weight`` preserves univariate backbone
         behavior identically at initialization while receiving full-rank
         gradients once downstream output layers wake (Step 2 in unit SGD when
         ``head.fc2`` is zero-initialized, or after the ``15%`` adapter warmup
         freeze in ``scripts/train.py``) when multivariate peers exist.
      2. Diagonal ``eye_r`` mask prevents ``0/0 = NaN`` on slack cells (``group_id < 0``).
      3. Multiplying the projected update by ``has_group_peer.to(dtype)`` ensures
         univariate rows and slack cells receive an exact ``0.0`` residual update
         regardless of what other sequences share the batch strip.
    """

    def __init__(self, d_model: int, n_heads: int, norm_eps: float, head_dim: int | None = None) -> None:
        super().__init__()
        self.n_heads = n_heads
        hd = (d_model // n_heads) if head_dim is None else head_dim
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(hd, eps=norm_eps)
        self.k_norm = RMSNorm(hd, eps=norm_eps)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        h: Float[Tensor, "b r l d"],
        patch_group_id: Int[Tensor, "b r l"],
        patch_is_cov: Bool[Tensor, "b r l"],
        patch_obs_any: Bool[Tensor, "b r l"] | None = None,
        rel_patch_idx: Int[Tensor, "b r l"] | None = None,
    ) -> Float[Tensor, "b r l d"]:
        b, _, l_len, _ = h.shape
        x = rearrange(h, "b r l d -> (b l) r d")
        gid = rearrange(patch_group_id, "b r l -> (b l) r")
        icov = rearrange(patch_is_cov, "b r l -> (b l) r")

        same_group = (gid[:, :, None] == gid[:, None, :]) & (gid[:, :, None] >= 0)
        # Historical covariates only inform themselves and targets; never read from targets.
        firewall = ~(icov[:, :, None] & (~icov[:, None, :]))
        if patch_obs_any is None:
            valid_peer_key = torch.ones((gid.size(0), 1, gid.size(1)), dtype=torch.bool, device=gid.device)
        else:
            obs = rearrange(patch_obs_any, "b r l -> (b l) r")
            if rel_patch_idx is not None:
                is_hor = rearrange(rel_patch_idx > 0, "b r l -> (b l) r")
                # Context columns require observed peer patches; horizon columns allow supervised co-targets only.
                key_ok = obs | (is_hor & (~icov))
            else:
                key_ok = (~icov) | obs
            valid_peer_key = key_ok[:, None, :]

        peer_mask = same_group & firewall & valid_peer_key
        eye_r = torch.eye(gid.size(1), dtype=torch.bool, device=gid.device).unsqueeze(0)
        has_group_peer = (peer_mask & ~eye_r).any(dim=-1, keepdim=True)
        group_attn_mask = (peer_mask | eye_r).unsqueeze(1)

        xn = self.norm(x)
        q, k, v = rearrange(self.qkv_proj(xn), "bl r (three h hd) -> three bl h r hd", three=3, h=self.n_heads).unbind(
            0
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=group_attn_mask)
        upd = self.out_proj(rearrange(attn_out, "bl h r hd -> bl r (h hd)"))
        x_out = x + upd * has_group_peer.to(x.dtype)
        return rearrange(x_out, "(b l) r d -> b r l d", b=b, l=l_len)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (``d_ff = 1024``)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with optional column-wise GroupAttention."""

    def __init__(self, cfg: TSFMConfig, has_group_attn: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = TemporalAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            rope_max_pos=cfg.rope_max_pos,
            rope_base=cfg.rope_base,
            norm_eps=cfg.norm_eps,
            head_dim=cfg.head_dim,
        )
        self.group_attn = (
            GroupAttention(d_model=cfg.d_model, n_heads=cfg.n_heads, norm_eps=cfg.norm_eps, head_dim=cfg.head_dim)
            if has_group_attn
            else None
        )
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(
        self,
        h: Float[Tensor, "b r l d"],
        rel_patch_idx: Int[Tensor, "b r l"],
        time_mask: Bool[Tensor, "br 1 l l"],
        patch_group_id: Int[Tensor, "b r l"],
        patch_is_cov: Bool[Tensor, "b r l"],
        patch_obs_any: Bool[Tensor, "b r l"] | None = None,
    ) -> Float[Tensor, "b r l d"]:
        h = h + self.attn(self.attn_norm(h), rel_patch_idx=rel_patch_idx, time_mask=time_mask)
        if self.group_attn is not None:
            h = self.group_attn(
                h,
                patch_group_id=patch_group_id,
                patch_is_cov=patch_is_cov,
                patch_obs_any=patch_obs_any,
                rel_patch_idx=rel_patch_idx,
            )
        h = h + self.ffn(self.ffn_norm(h))
        return h


class TSFMModel(nn.Module):
    """Unified 8-layer Patch-Transformer time-series forecasting model."""

    def __init__(self, cfg: TSFMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stem = PatchResStem(
            c_in=cfg.c_in,
            d_model=cfg.d_model,
            patch_size=cfg.patch_size,
            norm_eps=cfg.norm_eps,
            use_cov_emb=cfg.use_group_attn,
        )
        grp_layers = set(cfg.group_attn_layers) if cfg.use_group_attn else set()
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, has_group_attn=(idx in grp_layers)) for idx in range(cfg.n_layers)]
        )
        self.head = build_quantile_head(cfg)

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(
        cls,
        run_name_or_dir: str | Path,
        checkpoints_root: Path | None = None,
        hf_repo_id: str | None = None,
        filename: str = "model_soup.safetensors",
        device: str | torch.device = "cpu",
    ) -> tuple["TSFMModel", TSFMConfig]:
        """Load a trained checkpoint and config from local disk or HF Hub."""
        from tsfm_proto.hub import from_pretrained as _hub_from_pretrained

        return _hub_from_pretrained(
            run_name_or_dir=run_name_or_dir,
            checkpoints_root=checkpoints_root,
            hf_repo_id=hf_repo_id,
            filename=filename,
            device=device,
        )

    def forward(
        self,
        patch_inputs: Float[Tensor, "b r l p c_in"],
        rel_patch_idx: Int[Tensor, "b r l"],
        segment_id: Int[Tensor, "b r l"],
        patch_obs_any: Bool[Tensor, "b r l"],
        patch_group_id: Int[Tensor, "b r l"],
        patch_is_cov: Bool[Tensor, "b r l"],
    ) -> Float[Tensor, "b r t_max q"]:
        time_mask = build_temporal_mask(
            segment_id=segment_id,
            rel_patch_idx=rel_patch_idx,
            patch_obs_any=patch_obs_any,
            exclude_blank_keys=self.cfg.exclude_blank_keys,
        )
        h = self.stem(patch_inputs, patch_is_cov=patch_is_cov)
        for block in self.blocks:
            h = block(
                h,
                rel_patch_idx=rel_patch_idx,
                time_mask=time_mask,
                patch_group_id=patch_group_id,
                patch_is_cov=patch_is_cov,
                patch_obs_any=patch_obs_any,
            )
        return self.head(h)
