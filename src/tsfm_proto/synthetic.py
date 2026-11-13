"""Coupled synthetic generators (KernelSynth, TSMixup, coupled MV systems).

Active solely when ``synth_prob > 0`` (the final experimental iteration). Real
series used in ``TSMixup`` are drawn strictly from training-pool ``SeriesItem``s
after stripping ``[-H_freq:]`` validation holdout tails in ``SubsetDatasetStore``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SeriesItem:
    """Minimal canonical representation shared by real and synthetic time series."""

    dataset_name: str
    item_id: str
    freq: str
    target: np.ndarray  # [V_t, T], float32
    past_cov: np.ndarray  # [V_c, T], float32 (may have V_c == 0)


def _sample_kernel_synth_1d(rng: np.random.Generator, length: int) -> np.ndarray:
    """Generate a composite random-Fourier + periodic + piecewise-trend 1D series."""
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    signal = np.zeros(length, dtype=np.float64)

    # Random smooth GP approximation via cosine random Fourier features.
    n_basis = int(rng.integers(3, 8))
    freqs = rng.uniform(0.5, 18.0, size=n_basis)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_basis)
    weights = rng.normal(0.0, 1.0 / np.sqrt(n_basis), size=n_basis)
    for w, f, ph in zip(weights, freqs, phases):
        signal += w * np.cos(2.0 * np.pi * f * t + ph)

    # Periodic seasonal component anchored to realistic calendar periods.
    period = float(rng.choice([7, 12, 24, 48, 52, 96]))
    amp_s = float(rng.uniform(0.3, 1.8))
    phase_s = float(rng.uniform(0.0, 2.0 * np.pi))
    idx = np.arange(length, dtype=np.float64)
    signal += amp_s * np.sin(2.0 * np.pi * idx / period + phase_s)
    if rng.random() < 0.5:
        signal += 0.4 * amp_s * np.cos(4.0 * np.pi * idx / period + 0.5 * phase_s)

    # Optional linear drift + changepoint step so local vs global scaling both get exercised.
    signal += float(rng.uniform(-1.5, 1.5)) * t
    if rng.random() < 0.35 and length > 64:
        cp = int(rng.integers(length // 4, (3 * length) // 4))
        signal[cp:] += float(rng.uniform(-2.0, 2.0))

    noise_std = float(rng.uniform(0.03, 0.25))
    signal += rng.normal(0.0, noise_std, size=length)

    # Apply random affine scale/shift (and optional positive/integer projection).
    scale = float(np.exp(rng.uniform(-1.5, 4.0)))
    shift = float(rng.normal(0.0, 2.0)) * scale
    out = signal * scale + shift
    if rng.random() < 0.3:
        out = np.maximum(0.0, out - float(np.min(out)))
    return out.astype(np.float32)


def _sample_tsmixup_1d(
    rng: np.random.Generator,
    train_pool: Sequence[SeriesItem],
    length: int,
) -> np.ndarray:
    """Convex combination of 2-3 z-normalized real training windows from ``train_views``."""
    k = int(rng.integers(2, 4))
    weights = rng.dirichlet(np.ones(k, dtype=np.float64))
    mixed = np.zeros(length, dtype=np.float64)
    n_pool = len(train_pool)

    for w in weights:
        chosen = train_pool[int(rng.integers(0, n_pool))]
        v_idx = int(rng.integers(0, chosen.target.shape[0]))
        arr = np.asarray(chosen.target[v_idx], dtype=np.float64)
        fin_mask = np.isfinite(arr)
        if not fin_mask.any():
            part = _sample_kernel_synth_1d(rng, length).astype(np.float64)
        else:
            if arr.size >= length:
                start = int(rng.integers(0, arr.size - length + 1))
                part = arr[start : start + length].copy()
            else:
                reps = int(np.ceil(length / max(1, arr.size)))
                part = np.tile(arr, reps)[:length].copy()
            # Interpolate/fill any missing points before z-normalization.
            ok = np.isfinite(part)
            if not ok.any():
                part = np.zeros(length, dtype=np.float64)
            elif not ok.all():
                x_all = np.arange(length)
                part = np.interp(x_all, x_all[ok], part[ok])
            std = float(np.std(part))
            if std > 1e-6:
                part = (part - float(np.mean(part))) / std
            else:
                part = part - float(np.mean(part))
        mixed += float(w) * part

    scale = float(np.exp(rng.uniform(-1.0, 3.0)))
    return (mixed * scale + float(rng.normal(0.0, 1.0)) * scale).astype(np.float32)


def _sample_coupled_mv(rng: np.random.Generator, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a coupled V in {2..6} dynamical system split into targets and past covariates."""
    n_total = int(rng.integers(2, 7))
    n_cov = int(rng.integers(0, n_total))  # 0 .. n_total-1 historical covariates
    n_tgt = n_total - n_cov

    # Two shared latent factors: smooth trend/oscillation + damped AR(2) cycle.
    max_lag = min(32, max(1, (length // 3) - 1))
    buf_pad = 36
    z1 = _sample_kernel_synth_1d(rng, length + buf_pad).astype(np.float64)
    z1 = (z1 - z1.mean()) / (z1.std() + 1e-6)
    z2 = np.zeros(length + buf_pad, dtype=np.float64)
    eps = rng.normal(0.0, 0.35, size=length + buf_pad)
    for t in range(2, length + buf_pad):
        z2[t] = 1.15 * z2[t - 1] - 0.35 * z2[t - 2] + eps[t]
    z2 = (z2 - z2.mean()) / (z2.std() + 1e-6)

    lag_choices = np.asarray([lag for lag in (0, 2, 4, 8, 12, 24, 32) if lag <= max_lag], dtype=np.int64)
    if lag_choices.size == 0:
        lag_choices = np.asarray([0], dtype=np.int64)

    rows = []
    for _ in range(n_total):
        c1 = float(rng.normal(0.8, 0.5))
        c2 = float(rng.normal(0.4, 0.5))
        lag = int(rng.choice(lag_choices))
        offset = buf_pad - lag
        driver = c1 * z1[offset : offset + length] + c2 * z2[offset : offset + length]
        idio = rng.normal(0.0, float(rng.uniform(0.1, 0.45)), size=length)
        scale = float(np.exp(rng.uniform(-0.5, 2.5)))
        rows.append(((driver + idio) * scale + float(rng.normal(0.0, scale))).astype(np.float32))

    mat = np.stack(rows, axis=0)
    target = mat[:n_tgt]
    past_cov = mat[n_tgt:] if n_cov > 0 else np.zeros((0, length), dtype=np.float32)
    return target, past_cov


def sample_synthetic_series(
    rng: np.random.Generator,
    train_pool: Sequence[SeriesItem],
    length: int | None = None,
) -> SeriesItem:
    """Draw one synthetic series: 40% KernelSynth, 30% TSMixup, 30% coupled MV system."""
    if length is None:
        if float(rng.random()) < 0.40:
            length = int(rng.integers(36, 256))
        else:
            length = int(rng.integers(256, 4097))
    u = float(rng.random())
    if u < 0.40:
        arr = _sample_kernel_synth_1d(rng, length)[None, :]
        cov = np.zeros((0, length), dtype=np.float32)
        tag = "synth_kernelsynth"
    elif u < 0.70 and len(train_pool) > 0:
        arr = _sample_tsmixup_1d(rng, train_pool, length)[None, :]
        cov = np.zeros((0, length), dtype=np.float32)
        tag = "synth_tsmixup"
    else:
        arr, cov = _sample_coupled_mv(rng, length)
        tag = "synth_coupled_mv"

    return SeriesItem(
        dataset_name=tag,
        item_id=f"{tag}_{int(rng.integers(0, 10**9))}",
        freq="H",
        target=arr,
        past_cov=cov,
    )
