"""Monotone Q21 quantile decoding heads.

Two zero-overhead parameterizations sharing identical parameter counts:
  - ``LeftAnchoredSoftplusHead`` (baseline runs): anchors the bottom quantile
    ``q=0.01`` directly and accumulates positive ``softplus(u) + 1e-5`` gaps
    rightward. At zero initialization (``fc2 == 0``), this gives
    ``q_k = k * (ln(2) + 1e-5)`` (~4.43 Step-0 ``arcsinh`` CRPS, ~3.0% ``Cov80``).
  - ``MedianAnchoredProbitHead`` (calibrated runs): anchors the median
    ``q=0.50`` directly and accumulates left/right quantile spacings scaled by
    exact ``arcsinh(Phi^-1(q))`` Probit differences plus a ``1e-5`` monotonicity
    floor. At zero initialization (``fc2 == 0``), predictions match
    ``arcsinh(Phi^-1(q))`` up to the ``+/-10 * 1e-5`` floor
    (``max |sinh(pred) - Phi^-1(q)| = 2.54e-4``, analytical ``N(0,1)``
    ``Cov80 = 80.005%``, ~0.44 Step-0 CRPS in ``arcsinh`` space) while
    remaining strictly non-crossing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

try:
    from jaxtyping import Float
except ImportError:  # pragma: no cover

    class _JaxTypingFallback:
        def __getitem__(self, item: object) -> object:
            return Tensor

    Float = _JaxTypingFallback()  # type: ignore[assignment]

from tsfm_proto.config import TSFMConfig


class _HeadRMSNorm(nn.Module):
    """Local RMSNorm copy to keep heads self-contained without circular imports."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rms).to(x.dtype) * self.weight.to(x.dtype)


class _BaseQuantileHead(nn.Module):
    """Shared residual MLP projection from patch hidden states to raw logits ``[B, R, T, Q]``."""

    def __init__(self, d_model: int, patch_size: int, n_quantiles: int, norm_eps: float) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_quantiles = n_quantiles
        self.final_norm = _HeadRMSNorm(d_model, eps=norm_eps)
        self.fc1 = nn.Linear(d_model, d_model, bias=True)
        self.fc2 = nn.Linear(d_model, patch_size * n_quantiles, bias=True)
        # Zero final projection guarantees clean analytical Step-0 quantile calibration.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _raw_logits(self, h: Float[Tensor, "b r l d"]) -> Float[Tensor, "b r t q"]:
        x = self.final_norm(h)
        x = x + F.silu(self.fc1(x))
        raw = self.fc2(x).float()
        return rearrange(raw, "b r l (p q) -> b r (l p) q", p=self.patch_size, q=self.n_quantiles)


class LeftAnchoredSoftplusHead(_BaseQuantileHead):
    """Left-anchored cumulative softplus head (``q_0`` + cumulative positive gaps)."""

    def __init__(self, cfg: TSFMConfig) -> None:
        super().__init__(cfg.d_model, cfg.patch_size, cfg.n_quantiles, cfg.norm_eps)

    def forward(self, h: Float[Tensor, "b r l d"]) -> Float[Tensor, "b r t q"]:
        u = self._raw_logits(h)
        q0 = u[..., :1]
        gaps = F.softplus(u[..., 1:]) + 1e-5
        upper = q0 + torch.cumsum(gaps, dim=-1)
        return torch.cat((q0, upper), dim=-1)


class MedianAnchoredProbitHead(_BaseQuantileHead):
    """Median-anchored cumulative head scaled by ``arcsinh(Phi^-1(q))`` spacings."""

    def __init__(self, cfg: TSFMConfig) -> None:
        super().__init__(cfg.d_model, cfg.patch_size, cfg.n_quantiles, cfg.norm_eps)
        q_t = torch.tensor(cfg.quantiles, dtype=torch.float32)
        if 0.5 not in cfg.quantiles:
            raise ValueError("MedianAnchoredProbitHead requires 0.50 in cfg.quantiles")
        m_idx = cfg.quantiles.index(0.5)
        self.median_idx = m_idx
        # Equivariance with y_hat = mu + sigma * sinh(q_tilde) implies q_tilde ~ asinh(Phi^-1(q)).
        z_tilde = torch.asinh(math.sqrt(2.0) * torch.erfinv(2.0 * q_t - 1.0))
        dz_right = z_tilde[m_idx + 1 :] - z_tilde[m_idx:-1]
        dz_left = z_tilde[1 : m_idx + 1].flip(0) - z_tilde[:m_idx].flip(0)
        self.register_buffer("dz_right", dz_right, persistent=False)
        self.register_buffer("dz_left", dz_left, persistent=False)

    def forward(self, h: Float[Tensor, "b r l d"]) -> Float[Tensor, "b r t q"]:
        u = self._raw_logits(h)
        m = self.median_idx
        inv_ln2 = 1.0 / math.log(2.0)

        med = u[..., m : m + 1]
        right_gaps = self.dz_right * (F.softplus(u[..., m + 1 :]) * inv_ln2) + 1e-5
        left_gaps = self.dz_left * (F.softplus(u[..., :m].flip(-1)) * inv_ln2) + 1e-5

        right_q = med + torch.cumsum(right_gaps, dim=-1)
        left_q = (med - torch.cumsum(left_gaps, dim=-1)).flip(-1)
        return torch.cat((left_q, med, right_q), dim=-1)


def build_quantile_head(cfg: TSFMConfig) -> nn.Module:
    """Instantiate the configured monotone quantile head."""
    if cfg.head_type == "left_softplus":
        return LeftAnchoredSoftplusHead(cfg)
    if cfg.head_type == "median_probit":
        return MedianAnchoredProbitHead(cfg)
    raise ValueError(f"Unknown head_type: {cfg.head_type}")
