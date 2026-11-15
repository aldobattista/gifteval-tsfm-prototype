"""Scaling, local-window normalization, and observed-aware seasonal lag features.

All transformations are strictly causal with respect to the forecast origin
``T_origin = T_ctx_pad - 1``: global and local ``arcsinh`` moments are frozen at
``T_origin``, missingness is never leaked across future steps, and seasonal lags
search backward up to four cycles solely within observed context observations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from einops import rearrange

EPS_SIGMA: float = 0.01
SEASONAL_LAGS: tuple[int, ...] = (7, 12, 24, 48, 52, 96)
SINH_LATENT_CLAMP: float = 15.0


def _scale_equivariant_stats_1d(vals_64: np.ndarray, eps_zero: float = EPS_SIGMA) -> tuple[float, float]:
    """Compute unit-equivariant ``(mu, sigma)`` satisfying ``sigma(c * y) == c * sigma(y)`` for ``c > 0``.

    Preserves exact dynamic range on small SI-unit series (such as ``era5``/``cmip6`` specific
    humidity ``~1e-6 kg/kg`` or ``bitbrains`` fractional utilization ``~6e-4``) where an additive
    ``sqrt(var + 0.01**2)`` floor would inflate ``sigma`` by up to ``500,000x``, while safely
    guarding ``float32``-constant and identically-zero windows down to a ``1e-12`` numerical
    underflow threshold before falling back to ``eps_zero``.
    """
    if vals_64.size == 0:
        return 0.0, 1.0
    mu = float(vals_64.mean())
    s = float(vals_64.std())
    if s > max(1e-6 * abs(mu), 1e-12):
        return mu, s
    if abs(mu) > 1e-12:
        return mu, 1e-3 * abs(mu)
    return 0.0, float(eps_zero)


def compute_scale_stats(
    y_ctx_real: np.ndarray,
    mode: Literal["naive_zero_fill", "observed_frozen_origin"],
    eps_sigma: float = EPS_SIGMA,
    fallback_mu_sigma: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frozen-origin ``(mu, sigma)`` statistics per row ``(V, T_ctx_real)``.

    In ``naive_zero_fill`` mode (baseline iterations), missing entries are zeroed
    before computing unit-equivariant moments. In ``observed_frozen_origin`` mode, only
    finite context observations contribute; if a cropped evaluation window has zero
    finite points, pre-crop fallback moments (or ``(0, 1)``) are used.
    """
    n_rows = y_ctx_real.shape[0]
    mu = np.zeros(n_rows, dtype=np.float64)
    sigma = np.ones(n_rows, dtype=np.float64)
    y64 = np.asarray(y_ctx_real, dtype=np.float64)

    if mode == "naive_zero_fill":
        filled = np.where(np.isfinite(y64), y64, 0.0)
        for v in range(n_rows):
            mu[v], sigma[v] = _scale_equivariant_stats_1d(filled[v], eps_zero=eps_sigma)
        return mu.astype(np.float32), sigma.astype(np.float32)
    if mode != "observed_frozen_origin":
        raise ValueError(f"Unknown scaling_mode: {mode}")

    for v in range(n_rows):
        row = y64[v]
        obs = row[np.isfinite(row)]
        if obs.size > 0:
            mu[v], sigma[v] = _scale_equivariant_stats_1d(obs, eps_zero=eps_sigma)
        elif fallback_mu_sigma is not None:
            fb_mu = float(fallback_mu_sigma[0][v])
            fb_sig = float(fallback_mu_sigma[1][v])
            if np.isfinite(fb_mu) and np.isfinite(fb_sig) and fb_sig > 0.0:
                mu[v] = fb_mu
                sigma[v] = fb_sig
            else:
                mu[v] = 0.0
                sigma[v] = 1.0
        else:
            mu[v] = 0.0
            sigma[v] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def causal_forward_fill(y_ctx_real: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Forward-fill ``NaN`` entries causally along time, seeding leading gaps with ``mu``."""
    n_rows, t_len = y_ctx_real.shape
    out = np.empty((n_rows, t_len), dtype=np.float32)
    t_idx = np.arange(t_len, dtype=np.int64)
    for v in range(n_rows):
        row = y_ctx_real[v]
        fin = np.isfinite(row)
        if not fin.any():
            out[v, :] = float(mu[v])
            continue
        last_obs_idx = np.maximum.accumulate(np.where(fin, t_idx, -1))
        valid = last_obs_idx >= 0
        out[v, valid] = row[last_obs_idx[valid]]
        out[v, ~valid] = float(mu[v])
    return out


def invert_quantiles_np(
    q_scaled: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Exact inverse ``y_hat = mu + sigma * sinh(q_scaled)`` computed in float64."""
    clamped = np.clip(np.asarray(q_scaled, dtype=np.float64), -SINH_LATENT_CLAMP, SINH_LATENT_CLAMP)
    out64 = np.asarray(mu, dtype=np.float64) + np.asarray(sigma, dtype=np.float64) * np.sinh(clamped)
    return out64.astype(np.float32)


def build_group_patch_features(
    y_ctx_real: np.ndarray,
    left_pad: int,
    n_ctx_patches: int,
    n_hor_patches: int,
    patch_size: int = 32,
    scaling_mode: Literal["naive_zero_fill", "observed_frozen_origin"] = "observed_frozen_origin",
    use_local_scale_and_lags: bool = False,
    eps_sigma: float = EPS_SIGMA,
    fallback_mu_sigma: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build stem patch features ``[V, L_seg, P, C_in]``, ``patch_obs_any``, ``mu``, and ``sigma``.

    Base channels (``C_in = 4``):
      0. ``scaled_val`` (``arcsinh``-normalized context, zeroed on slack/blank patches/horizon)
      1. ``obs_mask`` (1.0 solely on actually observed context steps)
      2. ``intra_patch_ramp`` (fixed ``linspace(-1, 1, P)`` phase coordinate)
      3. ``rel_origin_pos`` (``clip((t - T_origin) / 1024, -2, 1)``, disambiguates Layer-0 1-patch horizon residual slots)

    Extended channels (``C_in = 17`` when ``use_local_scale_and_lags=True``):
      4. ``local_scaled_val`` (trailing-64 observed-point local ``arcsinh`` frame)
      5..10. ``lag_val[S]`` for ``S in (7, 12, 24, 48, 52, 96)`` (4-cycle observed fallback)
      11..16. ``lag_ok[S]`` binary reliability flags
    """
    n_rows, t_ctx_real = y_ctx_real.shape
    t_ctx_pad = n_ctx_patches * patch_size
    if left_pad + t_ctx_real != t_ctx_pad:
        raise ValueError(
            f"Alignment mismatch: left_pad({left_pad}) + T_ctx({t_ctx_real}) != N_ctx*P({t_ctx_pad})"
        )
    l_seg = n_ctx_patches + n_hor_patches
    t_tot = l_seg * patch_size
    t_origin = t_ctx_pad - 1
    c_in = 17 if use_local_scale_and_lags else 4

    mu, sigma = compute_scale_stats(
        y_ctx_real=y_ctx_real,
        mode=scaling_mode,
        eps_sigma=eps_sigma,
        fallback_mu_sigma=fallback_mu_sigma,
    )

    # Assemble context-length [V, T_ctx_pad] arrays for observation mask and raw values.
    y_ctx_pad = np.full((n_rows, t_ctx_pad), np.nan, dtype=np.float32)
    y_ctx_pad[:, left_pad:] = y_ctx_real.astype(np.float32, copy=False)
    is_left_pad = np.zeros(t_ctx_pad, dtype=bool)
    is_left_pad[:left_pad] = True
    obs_ctx = np.isfinite(y_ctx_pad) & (~is_left_pad[None, :])

    patch_obs_ctx = obs_ctx.reshape(n_rows, n_ctx_patches, patch_size).any(axis=-1)
    patch_obs_any = np.zeros((n_rows, l_seg), dtype=bool)
    patch_obs_any[:, :n_ctx_patches] = patch_obs_ctx

    feats = np.zeros((n_rows, t_tot, c_in), dtype=np.float32)

    mu64 = mu.astype(np.float64)[:, None]
    sig64 = sigma.astype(np.float64)[:, None]

    if scaling_mode == "naive_zero_fill":
        raw_filled = np.where(obs_ctx, y_ctx_pad.astype(np.float64), 0.0)
        scaled_ctx = np.arcsinh((raw_filled - mu64) / sig64).astype(np.float32)
        scaled_ctx[:, :left_pad] = 0.0
    else:
        ffill_real = causal_forward_fill(y_ctx_real, mu).astype(np.float64)
        scaled_real = np.arcsinh((ffill_real - mu64) / sig64).astype(np.float32)
        scaled_ctx = np.zeros((n_rows, t_ctx_pad), dtype=np.float32)
        scaled_ctx[:, left_pad:] = scaled_real
        # All-missing context patches carry zero values so masked keys hold no stale state.
        patch_keep_step = np.repeat(patch_obs_ctx, patch_size, axis=-1)
        scaled_ctx = np.where(patch_keep_step, scaled_ctx, 0.0)

    feats[:, :t_ctx_pad, 0] = scaled_ctx
    feats[:, :t_ctx_pad, 1] = obs_ctx.astype(np.float32)

    # Channel 2: intra-patch phase coordinate across all segments.
    ramp = np.linspace(-1.0, 1.0, patch_size, dtype=np.float32)
    feats[:, :, 2] = np.tile(ramp, l_seg)[None, :]
    if left_pad > 0:
        feats[:, :left_pad, 2] = 0.0

    # Channel 3: signed normalized distance from forecast origin T_origin.
    t_grid = np.arange(t_tot, dtype=np.float32)
    rel_pos = np.clip((t_grid - float(t_origin)) / 1024.0, -2.0, 1.0).astype(np.float32)
    rel_pos[:left_pad] = 0.0
    feats[:, :, 3] = rel_pos[None, :]

    if use_local_scale_and_lags:
        # Channel 4: trailing-64-observed-point local arcsinh scale (resists pre-origin blackouts).
        mu_loc = np.empty(n_rows, dtype=np.float64)
        sig_loc = np.empty(n_rows, dtype=np.float64)
        for v in range(n_rows):
            obs_vals = y_ctx_real[v][np.isfinite(y_ctx_real[v])]
            tail64 = obs_vals[-64:].astype(np.float64)
            if tail64.size >= 8:
                mu_loc[v], sig_loc[v] = _scale_equivariant_stats_1d(tail64, eps_zero=eps_sigma)
            else:
                mu_loc[v] = float(mu[v])
                sig_loc[v] = float(sigma[v])

        mu_loc64 = mu_loc[:, None]
        sig_loc64 = sig_loc[:, None]
        if scaling_mode == "naive_zero_fill":
            loc_filled = np.where(obs_ctx, y_ctx_pad.astype(np.float64), 0.0)
            local_ctx = np.arcsinh((loc_filled - mu_loc64) / sig_loc64).astype(np.float32)
            local_ctx[:, :left_pad] = 0.0
        else:
            local_real = np.arcsinh((ffill_real - mu_loc64) / sig_loc64).astype(np.float32)
            local_ctx = np.zeros((n_rows, t_ctx_pad), dtype=np.float32)
            local_ctx[:, left_pad:] = local_real
            patch_keep_step = np.repeat(patch_obs_ctx, patch_size, axis=-1)
            local_ctx = np.where(patch_keep_step, local_ctx, 0.0)
        feats[:, :t_ctx_pad, 4] = local_ctx

        # Channels 5..16: 6 observed-aware 4-cycle modular seasonal lags (+ binary reliability masks).
        t_int = np.arange(t_tot, dtype=np.int64)
        for lag_idx, period in enumerate(SEASONAL_LAGS):
            k_base = np.where(t_int <= t_origin, 1, (t_int - t_origin + period - 1) // period)
            lag_val = np.zeros((n_rows, t_tot), dtype=np.float32)
            lag_ok = np.zeros((n_rows, t_tot), dtype=np.float32)
            unresolved = np.ones((n_rows, t_tot), dtype=bool)
            unresolved[:, :left_pad] = False

            for cycle_m in range(4):
                tau = t_int - (k_base + cycle_m) * period
                in_bounds = (tau >= left_pad) & (tau < t_ctx_pad)
                tau_safe = np.clip(tau, 0, t_ctx_pad - 1)
                hit = unresolved & in_bounds[None, :] & obs_ctx[:, tau_safe]
                if not hit.any():
                    continue
                vals = np.arcsinh((y_ctx_pad[:, tau_safe].astype(np.float64) - mu64) / sig64).astype(
                    np.float32
                )
                lag_val[hit] = vals[hit]
                lag_ok[hit] = 1.0
                unresolved &= ~hit

            feats[:, :, 5 + lag_idx] = lag_val
            feats[:, :, 11 + lag_idx] = lag_ok

    patched = rearrange(feats, "v (l p) c -> v l p c", l=l_seg, p=patch_size)
    return patched, patch_obs_any, mu, sigma
