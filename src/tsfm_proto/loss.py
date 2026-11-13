"""Exact 2x pinball CRPS loss (flat token vs group-balanced damped) and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

try:
    from jaxtyping import Bool, Float, Int
except ImportError:  # pragma: no cover

    class _JaxTypingFallback:
        def __getitem__(self, item: object) -> object:
            return Tensor

    Bool = Float = Int = _JaxTypingFallback()  # type: ignore[assignment]

from tsfm_proto.config import TSFMConfig


@dataclass(frozen=True)
class LossMetrics:
    """Differentiable training loss plus detached calibration and safety diagnostics."""

    loss: Float[Tensor, ""]
    raw_crps: Float[Tensor, ""]
    cov80: Float[Tensor, ""]
    crossing_rate: Float[Tensor, ""]
    active_steps: int


def compute_pinball_crps_loss(
    pred_quantiles: Float[Tensor, "b r t q"],
    target_scaled: Float[Tensor, "b r t"],
    loss_mask: Bool[Tensor, "b r t"],
    series_id: Int[Tensor, "b r t"],
    group_target_count: Float[Tensor, "b r t"],
    cfg: TSFMConfig,
) -> LossMetrics:
    """Compute exact 2x average pinball loss over Q21 in float32.

    When ``cfg.loss_mode == "flat_token"``, every observed horizon step receives
    equal weight. When ``cfg.loss_mode == "per_series_damped"``, each active
    series horizon is averaged first, smoothly damped by
    ``tau * ln(1 + s / tau)`` (bounding spike outliers), and divided by
    ``group_target_count`` so multivariate groups do not outweigh univariate ones.
    """
    q_levels = torch.tensor(cfg.quantiles, dtype=torch.float32, device=pred_quantiles.device)
    pred = pred_quantiles.float()
    tgt = torch.where(loss_mask, target_scaled.float(), 0.0)

    diff = tgt.unsqueeze(-1) - pred
    pinball = torch.maximum(q_levels * diff, (q_levels - 1.0) * diff)
    step_crps = 2.0 * pinball.mean(dim=-1)  # Exact CRPS convention (matches GluonTS 2x pinball).

    mask_f = loss_mask.float()
    n_active = int(loss_mask.sum().item())
    denom_flat = mask_f.sum().clamp_min(1.0)
    raw_crps = (step_crps * mask_f).sum() / denom_flat

    if cfg.loss_mode == "flat_token" or n_active == 0:
        loss = raw_crps
    elif cfg.loss_mode == "per_series_damped":
        valid_step = loss_mask & (series_id >= 0)
        vals = step_crps[valid_step]
        v_cnts = group_target_count[valid_step].float().clamp_min(1.0)
        if vals.numel() == 0:
            loss = raw_crps
        else:
            raw_sids = series_id[valid_step]
            b_stride = int(raw_sids.max().item()) + 1
            b_idx = (
                torch.arange(pred.shape[0], device=pred.device, dtype=series_id.dtype)
                .view(-1, 1, 1)
                .expand_as(series_id)
            )
            sids = b_idx[valid_step] * b_stride + raw_sids
            n_series = int(sids.max().item()) + 1
            sum_per_s = torch.zeros(n_series, dtype=torch.float32, device=pred.device)
            cnt_per_s = torch.zeros(n_series, dtype=torch.float32, device=pred.device)
            grp_weight = torch.zeros(n_series, dtype=torch.float32, device=pred.device)

            sum_per_s.scatter_add_(0, sids, vals)
            cnt_per_s.scatter_add_(0, sids, torch.ones_like(vals))
            grp_weight.scatter_add_(0, sids, 1.0 / v_cnts)

            present = cnt_per_s > 0
            mean_s = sum_per_s[present] / cnt_per_s[present]
            tau = float(cfg.loss_damping_tau)
            damped_s = tau * torch.log1p(mean_s / tau)
            # grp_weight sums (1 / V_g) over cnt_per_s valid steps, so dividing recovers exact 1 / V_g.
            w_s = grp_weight[present] / cnt_per_s[present]
            loss = (damped_s * w_s).sum() / w_s.sum().clamp_min(1e-8)
    else:
        raise ValueError(f"Unknown loss_mode: {cfg.loss_mode}")

    with torch.no_grad():
        idx_10 = cfg.quantiles.index(0.10)
        idx_90 = cfg.quantiles.index(0.90)
        in_80 = (tgt >= pred[..., idx_10]) & (tgt <= pred[..., idx_90])
        cov80 = (in_80.float() * mask_f).sum() / denom_flat
        crossing = ((pred[..., 1:] < pred[..., :-1]).float().mean(dim=-1) * mask_f).sum() / denom_flat

    return LossMetrics(
        loss=loss,
        raw_crps=raw_crps.detach(),
        cov80=cov80.detach(),
        crossing_rate=crossing.detach(),
        active_steps=n_active,
    )
