"""Aligned strip packer and universal modulo-32 origin aligner.

Packs heterogeneous univariate, multivariate, and covariate-informed windows into
fixed ``[B=1, R=32, L=128]`` patch canvases (4,096 timesteps per row) so no
forward-pass FLOPs are wasted on padded sequence length while keeping forecast
origin ``T_origin`` locked at index 31 of patch ``rel_patch_idx = 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
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
from tsfm_proto.scaling import build_group_patch_features
from tsfm_proto.synthetic import SeriesItem


@dataclass
class PackedBatch:
    """Static-shape patch batch ready for ``TSFMModel`` and CRPS loss computation."""

    patch_inputs: Float[Tensor, "b r l p c_in"]
    target_scaled: Float[Tensor, "b r t_max"]
    loss_mask: Bool[Tensor, "b r t_max"]
    rel_patch_idx: Int[Tensor, "b r l"]
    segment_id: Int[Tensor, "b r l"]
    patch_obs_any: Bool[Tensor, "b r l"]
    patch_group_id: Int[Tensor, "b r l"]
    patch_is_cov: Bool[Tensor, "b r l"]
    series_id: Int[Tensor, "b r t_max"]
    group_target_count: Float[Tensor, "b r t_max"]


def trim_leading_missing(
    target: np.ndarray,
    past_cov: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Strip joint all-NaN leading columns while always keeping ``T >= 1`` step."""
    t_len = target.shape[1]
    if t_len <= 1:
        return target, past_cov, 0
    any_fin = np.isfinite(target).any(axis=0)
    if not any_fin.any():
        # Preserve the last step so completely NaN series keep T_ctx >= 1 safely.
        lead = t_len - 1
    else:
        lead = int(np.argmax(any_fin))
        if lead >= t_len:
            lead = t_len - 1
    tgt_out = target[:, lead:]
    cov_out = past_cov[:, lead:] if (past_cov is not None and past_cov.shape[1] == t_len) else past_cov
    return tgt_out, cov_out, lead


def compute_left_pad(t_ctx_real: int, patch_size: int = 32) -> tuple[int, int]:
    """Compute universal modulo-``patch_size`` left-padding and context patch count."""
    if t_ctx_real < 1:
        raise ValueError(f"Expected t_ctx_real >= 1, got {t_ctx_real}")
    left_pad = (-t_ctx_real) % patch_size
    n_ctx = (t_ctx_real + left_pad) // patch_size
    return left_pad, n_ctx


def apply_history_blackout(
    y_ctx_real: np.ndarray,
    rng: np.random.Generator,
    cov_ctx_real: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Blank a contiguous 20..64-step span (50% flush pre-origin, 50% interior)."""
    t_len = y_ctx_real.shape[1]
    if t_len < 36:
        return y_ctx_real, cov_ctx_real
    max_span = min(64, t_len - 16)
    span = int(rng.integers(20, max_span + 1))
    y_out = y_ctx_real.copy()
    c_out = cov_ctx_real.copy() if cov_ctx_real is not None else None
    if float(rng.random()) < 0.5:
        start = t_len - span
    else:
        max_start = max(1, t_len - span - 1)
        start = int(rng.integers(0, max_start + 1))
    y_out[:, start : start + span] = np.nan
    if c_out is not None and c_out.shape[1] == t_len:
        c_out[:, start : start + span] = np.nan
    return y_out, c_out


def _sample_window_and_horizon(
    rng: np.random.Generator,
    max_patches: int,
    max_hor_patches: int,
) -> tuple[int, int]:
    """Sample multi-scale patch budget N in [4, 128] and term-stratified horizon patches K."""
    min_budget = min(4, max(2, max_patches))
    if max_patches <= min_budget or float(rng.random()) < 0.5:
        n_budget = max(2, max_patches)
    else:
        hi_budget = max(min_budget + 1, min(64, max_patches))
        n_budget = int(np.exp(rng.uniform(math.log(float(min_budget)), math.log(float(hi_budget)))))
        n_budget = max(min_budget, min(max_patches, n_budget))

    max_h = max(1, min(max_hor_patches, n_budget - 1))
    u = float(rng.random())
    if u < 0.50 or max_h < 3:
        k_hor = int(rng.integers(1, min(2, max_h) + 1))
    elif u < 0.75 or max_h < 13:
        k_hor = int(rng.integers(min(3, max_h), min(12, max_h) + 1))
    else:
        k_hor = int(rng.integers(13, max_h + 1))
    k_hor = max(1, min(k_hor, max_hor_patches, n_budget - 1))
    return n_budget, k_hor


def pack_strip_batch(
    sampler_fn: Callable[[np.random.Generator], SeriesItem],
    rng: np.random.Generator,
    cfg: TSFMConfig,
) -> PackedBatch:
    """Pack real/synthetic draws into an aligned ``[1, R=32, L=128]`` training batch."""
    r_rows = cfg.strip_rows
    l_cols = cfg.max_patches
    p_size = cfg.patch_size
    c_in = cfg.c_in
    t_max = l_cols * p_size

    patch_inputs = np.zeros((r_rows, l_cols, p_size, c_in), dtype=np.float32)
    target_scaled = np.zeros((r_rows, t_max), dtype=np.float32)
    loss_mask = np.zeros((r_rows, t_max), dtype=bool)
    rel_patch_idx = np.zeros((r_rows, l_cols), dtype=np.int64)
    segment_id = np.full((r_rows, l_cols), -1, dtype=np.int64)
    patch_obs_any = np.zeros((r_rows, l_cols), dtype=bool)
    patch_group_id = np.full((r_rows, l_cols), -1, dtype=np.int64)
    patch_is_cov = np.zeros((r_rows, l_cols), dtype=bool)
    series_id = np.full((r_rows, t_max), -1, dtype=np.int64)
    group_target_count = np.ones((r_rows, t_max), dtype=np.float32)

    row_cursor = np.zeros(r_rows, dtype=np.int64)
    next_seg_id = 0
    next_grp_id = 0
    next_ser_id = 0
    attempts = 0
    max_attempts = r_rows * 16

    while attempts < max_attempts:
        attempts += 1
        min_cursor = int(row_cursor.min())
        if min_cursor > l_cols - 2:
            break

        item = sampler_fn(rng)
        tgt_full, cov_full, _ = trim_leading_missing(item.target, item.past_cov)
        t_avail = tgt_full.shape[1]
        if t_avail < 2:
            continue

        # Task-mixed compact subgroup packing preserves high strip occupancy and independent series throughput.
        n_tgt_raw = tgt_full.shape[0]
        n_cov_raw = cov_full.shape[0] if cov_full is not None else 0
        has_mv_channels = (n_tgt_raw > 1) or (n_cov_raw > 0)
        pack_as_group = cfg.use_group_attn and has_mv_channels and (
            item.dataset_name == "synth_coupled_mv" or float(rng.random()) < 0.35
        )
        if not pack_as_group:
            v_sel = int(rng.integers(0, n_tgt_raw))
            tgt_sel = tgt_full[v_sel : v_sel + 1]
            cov_sel = np.zeros((0, t_avail), dtype=np.float32)
        else:
            max_g = 6
            n_cov_keep = min(n_cov_raw, 2) if (cov_full is not None and n_cov_raw > 0) else 0
            n_tgt_keep = min(n_tgt_raw, max(1, max_g - n_cov_keep))
            tgt_idx = (
                np.sort(rng.choice(n_tgt_raw, size=n_tgt_keep, replace=False))
                if n_tgt_raw > n_tgt_keep
                else np.arange(n_tgt_raw)
            )
            tgt_sel = tgt_full[tgt_idx]
            if cov_full is not None and n_cov_keep > 0:
                cov_idx = (
                    np.sort(rng.choice(n_cov_raw, size=n_cov_keep, replace=False))
                    if n_cov_raw > n_cov_keep
                    else np.arange(n_cov_raw)
                )
                cov_sel = cov_full[cov_idx]
            else:
                cov_sel = np.zeros((0, t_avail), dtype=np.float32)

        v_tgt = tgt_sel.shape[0]
        v_cov = cov_sel.shape[0]
        v_group = v_tgt + v_cov

        # Select lowest-cursor rows and fall back to a 1-row target if aligning rows would open slack holes.
        order = np.argsort(row_cursor, kind="stable")
        chosen_rows = order[:v_group]
        if v_group > 1 and (
            int(row_cursor[chosen_rows].max()) - int(row_cursor[chosen_rows].min()) > 4
            or l_cols - int(row_cursor[chosen_rows].max()) < 4
        ):
            v_pick = int(rng.integers(0, tgt_sel.shape[0]))
            tgt_sel = tgt_sel[v_pick : v_pick + 1]
            cov_sel = np.zeros((0, t_avail), dtype=np.float32)
            v_tgt, v_cov, v_group = 1, 0, 1
            chosen_rows = order[:1]

        col_start = int(row_cursor[chosen_rows].max())
        remaining_cols = l_cols - col_start
        if remaining_cols < 2:
            if v_group > 1 and min_cursor <= l_cols - 2:
                tgt_sel = tgt_sel[:1]
                cov_sel = np.zeros((0, t_avail), dtype=np.float32)
                v_tgt, v_cov, v_group = 1, 0, 1
                chosen_rows = order[:1]
                col_start = int(row_cursor[chosen_rows[0]])
                remaining_cols = l_cols - col_start
            else:
                continue

        n_budget, k_hor = _sample_window_and_horizon(rng, l_cols, cfg.max_horizon_patches)
        n_budget = min(n_budget, remaining_cols)
        k_hor = min(k_hor, max(1, n_budget - 1))

        h_actual = min(k_hor * p_size, max(1, t_avail - 1))
        if t_avail >= 64 and t_avail - h_actual < 16:
            h_actual = max(1, t_avail // 3)
        k_hor_actual = math.ceil(h_actual / p_size)
        n_ctx_cap = max(1, n_budget - k_hor_actual)
        t_ctx_cap = min(t_avail - h_actual, n_ctx_cap * p_size)
        if t_ctx_cap < 1:
            continue

        t_ctx_draw = (
            t_ctx_cap
            if t_ctx_cap <= 32 or float(rng.random()) < 0.7
            else int(rng.integers(max(16, t_ctx_cap // 2), t_ctx_cap + 1))
        )
        total_slice = t_ctx_draw + h_actual
        max_start = t_avail - total_slice
        start_step = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0

        combined = np.concatenate([tgt_sel, cov_sel], axis=0) if v_cov > 0 else tgt_sel
        win = combined[:, start_step : start_step + total_slice]
        ctx_raw = win[:, :t_ctx_draw]
        hor_raw = win[:, t_ctx_draw:]

        # Re-trim leading all-NaN steps in case a random sub-window started inside a missing span.
        ctx_tgt, ctx_cov, _ = trim_leading_missing(ctx_raw[:v_tgt], ctx_raw[v_tgt:] if v_cov > 0 else None)
        if ctx_cov is not None and v_cov > 0:
            keep_cov = np.isfinite(ctx_cov).any(axis=1)
            if not bool(keep_cov.all()):
                ctx_cov = ctx_cov[keep_cov]
                v_cov = int(ctx_cov.shape[0])
                v_group = v_tgt + v_cov
                chosen_rows = chosen_rows[:v_group]
        ctx_raw = np.concatenate([ctx_tgt, ctx_cov], axis=0) if (ctx_cov is not None and v_cov > 0) else ctx_tgt
        t_ctx_real = ctx_raw.shape[1]

        # Ensure at least one finite horizon label exists before committing canvas slots.
        hor_tgt = hor_raw[:v_tgt]
        if not np.isfinite(hor_tgt).any():
            continue

        fallback_mu_sigma: tuple[np.ndarray, np.ndarray] | None = None
        if cfg.history_blackout_prob > 0.0 and float(rng.random()) < cfg.history_blackout_prob:
            # Compute moments strictly post-blackout (matching eval_metrics blackout stress tests),
            # falling back safely to (0, 1) inside compute_scale_stats if zero finite steps survive.
            ctx_raw, _ = apply_history_blackout(ctx_raw, rng)

        left_pad, n_ctx = compute_left_pad(t_ctx_real, p_size)
        n_seg = n_ctx + k_hor_actual
        if col_start + n_seg > l_cols:
            continue

        patched, obs_any, mu, sigma = build_group_patch_features(
            y_ctx_real=ctx_raw,
            left_pad=left_pad,
            n_ctx_patches=n_ctx,
            n_hor_patches=k_hor_actual,
            patch_size=p_size,
            scaling_mode=cfg.scaling_mode,
            use_local_scale_and_lags=cfg.use_local_scale_and_lags,
            eps_sigma=cfg.eps_sigma,
            fallback_mu_sigma=fallback_mu_sigma,
        )

        col_end = col_start + n_seg
        hor_step_start = (col_start + n_ctx) * p_size
        hor_step_end = hor_step_start + h_actual
        rel_idx_seg = np.arange(-(n_ctx - 1), k_hor_actual + 1, dtype=np.int64)

        for local_i, r_idx in enumerate(chosen_rows):
            is_cov = local_i >= v_tgt
            patch_inputs[r_idx, col_start:col_end] = patched[local_i]
            rel_patch_idx[r_idx, col_start:col_end] = rel_idx_seg
            segment_id[r_idx, col_start:col_end] = next_seg_id
            patch_obs_any[r_idx, col_start:col_end] = obs_any[local_i]
            patch_group_id[r_idx, col_start:col_end] = next_grp_id
            patch_is_cov[r_idx, col_start:col_end] = is_cov

            if not is_cov:
                fin_h = np.isfinite(hor_tgt[local_i])
                if fin_h.any():
                    raw_safe = np.where(fin_h, hor_tgt[local_i], 0.0).astype(np.float32)
                    normed = np.arcsinh(
                        (raw_safe.astype(np.float64) - float(mu[local_i])) / float(sigma[local_i])
                    ).astype(np.float32)
                    target_scaled[r_idx, hor_step_start:hor_step_end] = np.where(fin_h, normed, 0.0)
                    loss_mask[r_idx, hor_step_start:hor_step_end] = fin_h
                    series_id[r_idx, hor_step_start:hor_step_end] = next_ser_id
                    group_target_count[r_idx, hor_step_start:hor_step_end] = float(v_tgt)
                    next_ser_id += 1

            row_cursor[r_idx] = col_end
            next_seg_id += 1

        next_grp_id += 1
        if float((row_cursor >= l_cols - 1).mean()) >= 0.98:
            break

    return PackedBatch(
        patch_inputs=torch.from_numpy(patch_inputs).unsqueeze(0),
        target_scaled=torch.from_numpy(target_scaled).unsqueeze(0),
        loss_mask=torch.from_numpy(loss_mask).unsqueeze(0),
        rel_patch_idx=torch.from_numpy(rel_patch_idx).unsqueeze(0),
        segment_id=torch.from_numpy(segment_id).unsqueeze(0),
        patch_obs_any=torch.from_numpy(patch_obs_any).unsqueeze(0),
        patch_group_id=torch.from_numpy(patch_group_id).unsqueeze(0),
        patch_is_cov=torch.from_numpy(patch_is_cov).unsqueeze(0),
        series_id=torch.from_numpy(series_id).unsqueeze(0),
        group_target_count=torch.from_numpy(group_target_count).unsqueeze(0),
    )
