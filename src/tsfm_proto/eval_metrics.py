"""Batched zero-shot evaluation, exact GluonTS-parity scoring, and cluster bootstrap CIs.

Every evaluation window and multivariate group is assigned a distinct ``segment_id``
and ``patch_group_id`` so rolling windows never contaminate one another. Inference
dynamically expands the canvas row capacity ``R_eval = max(64, max_group_rows)`` to
keep high-dimensional covariate groups (such as 37-variate ``bizitobs_application``)
intact while shrinking sequence width ``L_batch <= 128`` to the batch maximum.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from tsfm_proto.config import DECILE_INDICES_IN_Q21, DECILE_LEVELS, Q21_LEVELS, TSFMConfig
from tsfm_proto.data import ValExample, _align_cov_width, normalize_pandas_freq
from tsfm_proto.model import TSFMModel
from tsfm_proto.packing import compute_left_pad, trim_leading_missing
from tsfm_proto.scaling import (
    build_group_patch_features,
    compute_scale_stats,
    invert_quantiles_np,
)


@dataclass
class EvalEntry:
    """Single inference item with optional ground-truth horizon for scoring."""

    target: np.ndarray  # [V_tgt, T_ctx]
    past_cov: np.ndarray | None  # [V_cov, T_ctx] or None
    prediction_length: int
    future_target: np.ndarray | None = None  # [V_tgt, H]
    season_length: int = 1


@dataclass
class PredictionOutput:
    """Raw, non-negativity-snapped, and full 21-quantile predictions per entry."""

    raw_deciles: list[np.ndarray]  # list of [9, H, V_tgt]
    clamp_deciles: list[np.ndarray]  # list of [9, H, V_tgt]
    raw_q21: list[np.ndarray]  # list of [21, H, V_tgt]


@dataclass
class _ChunkJob:
    entry_idx: int
    target_ctx: np.ndarray
    past_cov_ctx: np.ndarray | None
    v_tgt: int
    v_cov: int
    h_chunk: int
    k_hor: int
    n_ctx: int
    left_pad: int
    n_seg: int
    patched: np.ndarray  # [V_group, N_seg, P, C_in]
    obs_any: np.ndarray  # [V_group, N_seg]
    mu_tgt: np.ndarray  # [V_tgt]
    sigma_tgt: np.ndarray  # [V_tgt]


def _prepare_chunk_job(
    entry_idx: int,
    target_full: np.ndarray,
    past_cov_full: np.ndarray | None,
    h_chunk: int,
    cfg: TSFMConfig,
    pre_blackout_steps: int = 0,
) -> _ChunkJob:
    """Slice maximally recent context with pre-crop fallback stats and build patch features."""
    p_size = cfg.patch_size
    k_hor = math.ceil(h_chunk / p_size)
    max_ctx_patches = max(1, cfg.max_patches - k_hor)
    max_ctx_steps = max_ctx_patches * p_size

    tgt = np.asarray(target_full, dtype=np.float32)
    cov_raw = (
        np.asarray(past_cov_full, dtype=np.float32)
        if (past_cov_full is not None and cfg.use_group_attn)
        else None
    )
    cov = _align_cov_width(cov_raw, tgt.shape[1]) if (cov_raw is not None and cov_raw.shape[0] > 0) else None
    v_tgt = tgt.shape[0]
    v_cov = cov.shape[0] if (cov is not None and cov.shape[0] > 0) else 0

    if pre_blackout_steps > 0 and tgt.shape[1] >= pre_blackout_steps + 32:
        tgt = tgt.copy()
        tgt[:, -pre_blackout_steps:] = np.nan
        if cov is not None and cov.shape[1] == tgt.shape[1]:
            cov = cov.copy()
            cov[:, -pre_blackout_steps:] = np.nan

    combined_full = np.concatenate([tgt, cov], axis=0) if (cov is not None and v_cov > 0) else tgt
    # Freeze fallback moments BEFORE cropping to 128-K_hor patches so tail-missing crops recover scale.
    fallback_mu, fallback_sigma = compute_scale_stats(
        combined_full,
        mode=cfg.scaling_mode,
        eps_sigma=cfg.eps_sigma,
        fallback_mu_sigma=None,
    )

    tgt_trim, cov_trim, _ = trim_leading_missing(tgt, cov)
    if tgt_trim.shape[1] > max_ctx_steps:
        tgt_trim = tgt_trim[:, -max_ctx_steps:]
        if cov_trim is not None:
            cov_trim = cov_trim[:, -max_ctx_steps:]

    ctx_combined = (
        np.concatenate([tgt_trim, cov_trim], axis=0)
        if (cov_trim is not None and v_cov > 0)
        else tgt_trim
    )
    left_pad, n_ctx = compute_left_pad(ctx_combined.shape[1], p_size)
    n_seg = n_ctx + k_hor
    patched, obs_any, mu, sigma = build_group_patch_features(
        y_ctx_real=ctx_combined,
        left_pad=left_pad,
        n_ctx_patches=n_ctx,
        n_hor_patches=k_hor,
        patch_size=p_size,
        scaling_mode=cfg.scaling_mode,
        use_local_scale_and_lags=cfg.use_local_scale_and_lags,
        eps_sigma=cfg.eps_sigma,
        fallback_mu_sigma=(fallback_mu, fallback_sigma),
    )

    return _ChunkJob(
        entry_idx=entry_idx,
        target_ctx=tgt,
        past_cov_ctx=cov,
        v_tgt=v_tgt,
        v_cov=v_cov,
        h_chunk=h_chunk,
        k_hor=k_hor,
        n_ctx=n_ctx,
        left_pad=left_pad,
        n_seg=n_seg,
        patched=patched,
        obs_any=obs_any,
        mu_tgt=mu[:v_tgt],
        sigma_tgt=sigma[:v_tgt],
    )


def _run_chunk_batch(
    model: TSFMModel,
    cfg: TSFMConfig,
    jobs: list[_ChunkJob],
    device: torch.device,
    use_autocast: bool,
) -> list[np.ndarray]:
    """Execute one homogeneous-width forward batch using dynamic row packing and return ``[21, H, V_tgt]``."""
    p_size = cfg.patch_size
    c_in = 17 if cfg.use_local_scale_and_lags else 4

    # Pack jobs sequentially into mini-batches where each job keeps its rows contiguous and isolated.
    max_v = max(j.v_tgt + j.v_cov for j in jobs)
    r_eval = max(64, max_v)

    outputs: list[np.ndarray | None] = [None] * len(jobs)
    idx = 0
    while idx < len(jobs):
        batch_jobs: list[_ChunkJob] = []
        rows_used = 0
        while idx < len(jobs):
            vg = jobs[idx].v_tgt + jobs[idx].v_cov
            if batch_jobs and rows_used + vg > r_eval:
                break
            batch_jobs.append(jobs[idx])
            rows_used += vg
            idx += 1

        l_chunk = min(cfg.max_patches, max(j.n_seg for j in batch_jobs))
        patch_inputs = torch.zeros((1, r_eval, l_chunk, p_size, c_in), dtype=torch.float32)
        rel_patch_idx = torch.zeros((1, r_eval, l_chunk), dtype=torch.long)
        segment_id = torch.full((1, r_eval, l_chunk), -1, dtype=torch.long)
        patch_obs_any = torch.zeros((1, r_eval, l_chunk), dtype=torch.bool)
        patch_group_id = torch.full((1, r_eval, l_chunk), -1, dtype=torch.long)
        patch_is_cov = torch.zeros((1, r_eval, l_chunk), dtype=torch.bool)

        row_ptr = 0
        job_placements: list[tuple[_ChunkJob, int, int]] = []
        for g_id, job in enumerate(batch_jobs):
            v_g = job.v_tgt + job.v_cov
            n_seg = job.n_seg
            r0, r1 = row_ptr, row_ptr + v_g
            patch_inputs[0, r0:r1, :n_seg] = torch.from_numpy(job.patched)
            patch_obs_any[0, r0:r1, :n_seg] = torch.from_numpy(job.obs_any)

            rel_grid = torch.cat(
                [
                    torch.arange(-job.n_ctx + 1, 1, dtype=torch.long),
                    torch.arange(1, job.k_hor + 1, dtype=torch.long),
                ]
            )
            rel_patch_idx[0, r0:r1, :n_seg] = rel_grid.unsqueeze(0).expand(v_g, n_seg)
            patch_group_id[0, r0:r1, :n_seg] = g_id

            seg_grid = (g_id * max(r_eval, 1024)) + torch.arange(v_g, dtype=torch.long).unsqueeze(1)
            segment_id[0, r0:r1, :n_seg] = seg_grid.expand(v_g, n_seg)

            if job.v_cov > 0:
                patch_is_cov[0, r0 + job.v_tgt : r1, :n_seg] = True

            job_placements.append((job, r0, g_id))
            row_ptr = r1

        patch_inputs = patch_inputs.to(device)
        rel_patch_idx = rel_patch_idx.to(device)
        segment_id = segment_id.to(device)
        patch_obs_any = patch_obs_any.to(device)
        patch_group_id = patch_group_id.to(device)
        patch_is_cov = patch_is_cov.to(device)

        with torch.inference_mode():
            if use_autocast and device.type in {"mps", "cuda"}:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pred_q_all = model(
                        patch_inputs,
                        rel_patch_idx,
                        segment_id,
                        patch_obs_any,
                        patch_group_id,
                        patch_is_cov,
                    )
            else:
                pred_q_all = model(
                    patch_inputs,
                    rel_patch_idx,
                    segment_id,
                    patch_obs_any,
                    patch_group_id,
                    patch_is_cov,
                )
        pred_np = pred_q_all[0].float().cpu().numpy()  # [R_eval, T_chunk, 21]

        for job, r0, _ in job_placements:
            t_hor_start = job.n_ctx * p_size
            t_hor_end = t_hor_start + job.h_chunk
            q_scaled = pred_np[r0 : r0 + job.v_tgt, t_hor_start:t_hor_end, :]  # [V_tgt, H_chunk, 21]
            q_raw = invert_quantiles_np(
                q_scaled,
                job.mu_tgt[:, None, None],
                job.sigma_tgt[:, None, None],
            )
            if not np.isfinite(q_raw).all():
                raise FloatingPointError("Non-finite forecast generated during eval_metrics.")
            # Return [21, H_chunk, V_tgt]
            outputs[job.entry_idx] = np.transpose(q_raw, (2, 1, 0)).astype(np.float32, copy=False)

    return [out for out in outputs if out is not None]


def _visible_target_1d(row: np.ndarray, blackout_steps: int) -> np.ndarray:
    """Return only target observations visible after pre-origin blackout masking."""
    if blackout_steps > 0 and row.shape[0] >= blackout_steps + 32:
        return row[:-blackout_steps]
    return row


def predict_entries(
    model: TSFMModel,
    cfg: TSFMConfig,
    entries: list[EvalEntry],
    device: torch.device,
    batch_size: int = 32,
    apply_blackout_steps: int = 0,
    sym_avg: bool = False,
    use_autocast: bool = True,
) -> PredictionOutput:
    """Predict calibrated decile and 21-quantile forecasts with single-pass or multi-chunk rollout."""
    model.eval()
    max_single_h = min(cfg.max_horizon_patches, max(1, cfg.max_patches - 1)) * cfg.patch_size
    n_entries = len(entries)

    raw_q21_list: list[np.ndarray | None] = [None] * n_entries
    clamp_dec_list: list[np.ndarray | None] = [None] * n_entries
    raw_dec_list: list[np.ndarray | None] = [None] * n_entries
    med_idx_in_q21 = cfg.quantiles.index(0.5)
    dec_idx = np.asarray(DECILE_INDICES_IN_Q21, dtype=np.int64)

    def _predict_pass(pass_entries: list[EvalEntry]) -> list[np.ndarray]:
        pass_q21: list[np.ndarray] = [np.empty((0,), dtype=np.float32) for _ in pass_entries]
        for start in range(0, len(pass_entries), batch_size):
            batch = pass_entries[start : start + batch_size]
            cur_tgts = [np.asarray(ent.target, dtype=np.float32) for ent in batch]
            cur_covs = [
                (np.asarray(ent.past_cov, dtype=np.float32) if ent.past_cov is not None else None)
                for ent in batch
            ]
            done_h = [0] * len(batch)
            acc_chunks: list[list[np.ndarray]] = [[] for _ in batch]
            chunk_idx = 0

            while True:
                pending = [i for i, ent in enumerate(batch) if done_h[i] < ent.prediction_length]
                if not pending:
                    break
                jobs: list[_ChunkJob] = []
                b_blackout = apply_blackout_steps if chunk_idx == 0 else 0
                for local_i, b_i in enumerate(pending):
                    h_rem = batch[b_i].prediction_length - done_h[b_i]
                    h_chunk = min(h_rem, max_single_h)
                    jobs.append(
                        _prepare_chunk_job(
                            entry_idx=local_i,
                            target_full=cur_tgts[b_i],
                            past_cov_full=cur_covs[b_i],
                            h_chunk=h_chunk,
                            cfg=cfg,
                            pre_blackout_steps=b_blackout,
                        )
                    )
                preds_chunk = _run_chunk_batch(model, cfg, jobs, device, use_autocast=use_autocast)
                for local_i, b_i in enumerate(pending):
                    q_chunk = preds_chunk[local_i]  # [21, H_chunk, V_tgt]
                    acc_chunks[b_i].append(q_chunk)
                    done_h[b_i] += q_chunk.shape[1]
                    if done_h[b_i] < batch[b_i].prediction_length:
                        job_i = jobs[local_i]
                        med_c = np.transpose(q_chunk[med_idx_in_q21], (1, 0))  # [V_tgt, H_chunk]
                        cur_tgts[b_i] = np.concatenate([job_i.target_ctx.astype(np.float32), med_c], axis=1)
                        if job_i.past_cov_ctx is not None and job_i.past_cov_ctx.shape[0] > 0:
                            cov_pad = np.full(
                                (job_i.past_cov_ctx.shape[0], med_c.shape[1]),
                                np.nan,
                                dtype=np.float32,
                            )
                            cur_covs[b_i] = np.concatenate(
                                [job_i.past_cov_ctx.astype(np.float32), cov_pad], axis=1
                            )
                        else:
                            cur_covs[b_i] = None
                chunk_idx += 1

            for b_i, chunks in enumerate(acc_chunks):
                pass_q21[start + b_i] = (
                    chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=1)
                )
        return pass_q21

    pos_preds = _predict_pass(entries)
    if sym_avg:
        q_arr = np.asarray(cfg.quantiles, dtype=np.float64)
        if not np.allclose(q_arr + q_arr[::-1], 1.0, atol=1e-6):
            raise ValueError("sym_avg=True requires symmetric quantile grid q_k + q_{Q-1-k} == 1.0")
        signed_indices: list[int] = []
        signed_neg_entries: list[EvalEntry] = []
        for idx_e, ent in enumerate(entries):
            has_neg = False
            for v in range(ent.target.shape[0]):
                vis = _visible_target_1d(ent.target[v], apply_blackout_steps)
                fin = vis[np.isfinite(vis)]
                if fin.size > 0 and bool((fin < 0.0).any()):
                    has_neg = True
                    break
            if has_neg:
                signed_indices.append(idx_e)
                signed_neg_entries.append(
                    EvalEntry(
                        target=-ent.target,
                        past_cov=(-ent.past_cov if ent.past_cov is not None else None),
                        prediction_length=ent.prediction_length,
                        future_target=None,
                        season_length=ent.season_length,
                    )
                )
        neg_preds_signed = _predict_pass(signed_neg_entries) if signed_neg_entries else []
        neg_by_idx = dict(zip(signed_indices, neg_preds_signed))
        final_q21 = []
        for idx_e, (ent, p) in enumerate(zip(entries, pos_preds)):
            if idx_e not in neg_by_idx:
                final_q21.append(p)
                continue
            n = neg_by_idx[idx_e]
            avg = p.copy()
            for v in range(ent.target.shape[0]):
                vis = _visible_target_1d(ent.target[v], apply_blackout_steps)
                fin = vis[np.isfinite(vis)]
                # Restrict reflection averaging strictly to signed series; preserve raw quantiles on >= 0 series.
                if fin.size > 0 and bool((fin < 0.0).any()):
                    avg[:, :, v] = 0.5 * (p[:, :, v] - n[::-1, :, v])
            final_q21.append(avg)
    else:
        final_q21 = pos_preds

    for i, (ent, q21) in enumerate(zip(entries, final_q21)):
        raw_q21_list[i] = q21
        dec = q21[dec_idx]  # [9, H, V_tgt]
        raw_dec_list[i] = dec

        # Optional physical post-processing: non-negative clamp + exact < 2^23 integer snap.
        clamped = dec.copy()
        for v in range(ent.target.shape[0]):
            vis = _visible_target_1d(ent.target[v], apply_blackout_steps)
            fin = vis[np.isfinite(vis)]
            if fin.size == 0:
                continue
            if bool((fin >= 0.0).all()):
                clamped[:, :, v] = np.maximum(clamped[:, :, v], 0.0)
            if (
                fin.size >= 4
                and float(np.max(np.abs(fin))) < float(1 << 23)
                and bool(np.all(fin == np.floor(fin)))
                and bool(np.any(fin != 0.0))
            ):
                clamped[:, :, v] = np.rint(clamped[:, :, v])
        clamp_dec_list[i] = clamped

    return PredictionOutput(
        raw_deciles=[x for x in raw_dec_list if x is not None],
        clamp_deciles=[x for x in clamp_dec_list if x is not None],
        raw_q21=[x for x in raw_q21_list if x is not None],
    )


def seasonal_error_masked(past_target_2d: np.ndarray, seasonality: int) -> np.ma.MaskedArray:
    """Match ``gluonts.model.evaluation.seasonal_error`` over masked ``[V, T]`` arrays."""
    t_len = past_target_2d.shape[-1]
    seas = 1 if seasonality > t_len else max(1, int(seasonality))
    if t_len <= seas:
        return np.ma.masked_all((past_target_2d.shape[0], 1), dtype=np.float64)
    masked = np.ma.masked_invalid(np.asarray(past_target_2d, dtype=np.float64))
    y_t = masked[:, seas:]
    y_tm = masked[:, : t_len - seas]
    return np.abs(y_t - y_tm).mean(axis=-1, keepdims=True)


def score_task_fast(
    entries: list[EvalEntry],
    decile_preds: list[np.ndarray],
) -> dict[str, float]:
    """Compute exact GluonTS-parity ``MASE[0.5]`` and ``mean_weighted_sum_quantile_loss``."""
    mase_sum = 0.0
    mase_count = 0
    ql_sums = np.zeros(len(DECILE_LEVELS), dtype=np.float64)
    abs_label_sum = 0.0
    cov80_sum = 0.0
    cov80_count = 0

    q_arr = np.asarray(DECILE_LEVELS, dtype=np.float64)[:, None, None]  # [9, 1, 1]

    for ent, pred_dec in zip(entries, decile_preds):
        if ent.future_target is None:
            raise ValueError("score_task_fast requires EvalEntry.future_target")
        label = np.asarray(ent.future_target, dtype=np.float64)  # [V_tgt, H]
        pred_vh = np.transpose(np.asarray(pred_dec, dtype=np.float64), (0, 2, 1))  # [9, V_tgt, H]
        valid = np.isfinite(label)

        # Seasonal error per variate [V_tgt, 1].
        seas_err = seasonal_error_masked(ent.target, ent.season_length)
        seas_arr = np.where(np.ma.getmaskarray(seas_err), np.nan, np.asarray(seas_err, dtype=np.float64))
        abs_err = np.abs(pred_vh[4] - label)  # index 4 = 0.5 median
        with np.errstate(divide="ignore", invalid="ignore"):
            ase = abs_err / seas_arr  # [V_tgt, H]
        ase_finite = valid & np.isfinite(ase)
        mase_sum += float(np.sum(ase[ase_finite]))
        mase_count += int(np.sum(ase_finite))

        # Quantile loss over valid label cells: 2 * |(prediction - label) * ((prediction >= label) - q)|
        if valid.any():
            lbl_v = label[None, :, :]  # [1, V_tgt, H]
            diff = pred_vh - lbl_v
            ql = 2.0 * np.abs(diff * ((pred_vh >= lbl_v) - q_arr))
            ql_sums += np.sum(ql[:, valid], axis=-1)
            abs_label_sum += float(np.sum(np.abs(label[valid])))

            in_80 = (label >= pred_vh[0]) & (label <= pred_vh[8])  # [0.1, 0.9] -> 80% interval
            cov80_sum += float(np.sum(in_80[valid]))
            cov80_count += int(np.sum(valid))

    mase = mase_sum / max(mase_count, 1)
    if abs_label_sum > 0.0:
        mwql = float(np.mean(ql_sums / abs_label_sum))
    else:
        mwql = float("nan")
    cov80 = cov80_sum / max(cov80_count, 1)
    return {
        "MASE[0.5]": float(mase),
        "mean_weighted_sum_quantile_loss": float(mwql),
        "coverage_80": float(cov80),
        "n_valid_labels": float(cov80_count),
    }


def _fallback_seasonality(freq_str: str) -> int:
    norm_f = normalize_pandas_freq(freq_str)
    m = re.fullmatch(r"(\d*)([A-Za-z]+)(-[A-Za-z]+)?", norm_f)
    if not m:
        return 1
    mult = int(m.group(1)) if m.group(1) else 1
    base = m.group(2)
    base_season = {
        "s": 3600,
        "min": 1440,
        "h": 24,
        "B": 5,
        "D": 1,
        "W": 1,
        "ME": 12,
        "MS": 12,
        "BME": 12,
        "BMS": 12,
        "QE": 4,
        "QS": 4,
        "BQE": 4,
        "BQS": 4,
        "YE": 1,
        "YS": 1,
        "BYE": 1,
        "BYS": 1,
    }.get(base)
    if base_season is None or mult <= 0:
        return 1
    return base_season // mult if (base_season % mult == 0) else 1


def evaluate_val_split(
    model: TSFMModel,
    cfg: TSFMConfig,
    val_examples: list[ValExample],
    device: torch.device,
    batch_size: int = 32,
    use_autocast: bool = True,
) -> dict[str, float]:
    """Score the fixed 1,500-series internal validation holdout on MASE, MWQL, CRPS, and Cov80."""
    try:
        from gluonts.time_feature import get_seasonality
    except ImportError:  # pragma: no cover
        get_seasonality = None

    def _safe_season(freq_str: str) -> int:
        if get_seasonality is None:
            return _fallback_seasonality(freq_str)
        try:
            return int(get_seasonality(normalize_pandas_freq(freq_str)))
        except Exception:
            return _fallback_seasonality(freq_str)

    entries = [
        EvalEntry(
            target=ex.context_target,
            past_cov=(ex.context_past_cov if ex.context_past_cov.shape[0] > 0 else None),
            prediction_length=ex.label_target.shape[1],
            future_target=ex.label_target,
            season_length=_safe_season(ex.freq),
        )
        for ex in val_examples
    ]
    preds = predict_entries(
        model, cfg, entries, device=device, batch_size=batch_size, use_autocast=use_autocast
    )
    dec_metrics = score_task_fast(entries, preds.raw_deciles)

    # Group validation series by source dataset and compute Seasonal-Naive normalized geomean (GIFT-Eval style).
    by_ds_entries: dict[str, list[EvalEntry]] = defaultdict(list)
    by_ds_pred_dec: dict[str, list[np.ndarray]] = defaultdict(list)
    by_ds_sn_dec: dict[str, list[np.ndarray]] = defaultdict(list)
    for ex, ent, pred_dec in zip(val_examples, entries, preds.raw_deciles):
        t_len = ent.target.shape[1]
        eff_s = ent.season_length if ent.season_length <= t_len else 1
        idx = [t_len - eff_s + (h % eff_s) for h in range(ent.prediction_length)]
        sn_vh = ent.target[:, idx].copy()
        for v in range(sn_vh.shape[0]):
            fin = ent.target[v][np.isfinite(ent.target[v])]
            fill = float(fin[-1]) if fin.size > 0 else 0.0
            sn_vh[v] = np.where(np.isfinite(sn_vh[v]), sn_vh[v], fill)
        sn_dec = np.repeat(np.transpose(sn_vh, (1, 0))[None, :, :], len(DECILE_LEVELS), axis=0).astype(
            np.float32
        )
        by_ds_entries[ex.dataset_name].append(ent)
        by_ds_pred_dec[ex.dataset_name].append(pred_dec)
        by_ds_sn_dec[ex.dataset_name].append(sn_dec)

    nmase_vals: list[float] = []
    nmwql_vals: list[float] = []
    for ds_name, ds_ents in by_ds_entries.items():
        m_ds = score_task_fast(ds_ents, by_ds_pred_dec[ds_name])
        sn_ds = score_task_fast(ds_ents, by_ds_sn_dec[ds_name])
        mase_m, mase_sn = m_ds["MASE[0.5]"], sn_ds["MASE[0.5]"]
        mwql_m, mwql_sn = (
            m_ds["mean_weighted_sum_quantile_loss"],
            sn_ds["mean_weighted_sum_quantile_loss"],
        )
        if np.isfinite(mase_m) and np.isfinite(mase_sn) and mase_m > 0.0 and mase_sn > 0.0:
            nmase_vals.append(mase_m / mase_sn)
        if np.isfinite(mwql_m) and np.isfinite(mwql_sn) and mwql_m > 0.0 and mwql_sn > 0.0:
            nmwql_vals.append(mwql_m / mwql_sn)

    dec_metrics["val_nmase"] = float(np.exp(np.mean(np.log(nmase_vals)))) if nmase_vals else float("nan")
    dec_metrics["val_nmwql"] = float(np.exp(np.mean(np.log(nmwql_vals)))) if nmwql_vals else float("nan")

    # Also compute normalized scaled CRPS across the 21 quantiles for direct comparison with train loss.
    q21_t = torch.tensor(Q21_LEVELS, dtype=torch.float32).view(21, 1, 1)
    crps_vals: list[float] = []
    for ex, q21 in zip(val_examples, preds.raw_q21):
        # Always score val_crps in the fixed observed-only arcsinh frame so runs are directly comparable.
        mu, sigma = compute_scale_stats(
            ex.context_target, mode="observed_frozen_origin", eps_sigma=cfg.eps_sigma
        )
        mu64 = mu.astype(np.float64)[:, None]
        sig64 = sigma.astype(np.float64)[:, None]
        y_true = np.arcsinh((ex.label_target.astype(np.float64) - mu64) / sig64).astype(np.float32)
        q_scaled = np.arcsinh(
            (q21.astype(np.float64) - mu.astype(np.float64)[None, None, :])
            / sigma.astype(np.float64)[None, None, :]
        ).astype(np.float32)
        q_scaled = np.transpose(q_scaled, (0, 2, 1))  # [21, V_tgt, H]
        valid = np.isfinite(y_true)
        if not valid.any():
            continue
        err = torch.from_numpy(y_true[None, :, :] - q_scaled)
        pinball = torch.maximum(q21_t * err, (q21_t - 1.0) * err) * 2.0
        crps_vals.append(float(pinball[:, valid].mean().item()))

    dec_metrics["val_crps"] = float(np.mean(crps_vals)) if crps_vals else float("nan")
    return dec_metrics


def paired_bootstrap_geomean_ci(
    ratios: np.ndarray,
    family_ids: list[str] | np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute task-level and 28-family cluster-robust paired bootstrap 95% CIs on geometric-mean ratios."""
    arr = np.asarray(ratios, dtype=np.float64)
    fam = np.asarray(family_ids)
    valid = np.isfinite(arr) & (arr > 0.0)
    arr = arr[valid]
    fam = fam[valid]

    log_r = np.log(arr)
    point_gmr = float(np.exp(np.mean(log_r)))
    win_rate = float(np.mean(arr < 1.0))

    rng = np.random.default_rng(seed)
    n_tasks = arr.size

    # 1. Task-level paired bootstrap across the 97 tasks.
    idx_task = rng.integers(0, n_tasks, size=(n_boot, n_tasks))
    gmr_task = np.exp(np.mean(log_r[idx_task], axis=1))
    ci_lo, ci_hi = np.percentile(gmr_task, [2.5, 97.5])

    # 2. Cluster-robust paired bootstrap over dataset families (avoids horizon pseudo-replication).
    uniq_fam = np.unique(fam)
    fam_groups = [log_r[fam == f] for f in uniq_fam]
    n_fam = len(fam_groups)
    gmr_clust = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.integers(0, n_fam, size=n_fam)
        vals = np.concatenate([fam_groups[c] for c in chosen])
        gmr_clust[b] = np.exp(np.mean(vals))
    c_lo, c_hi = np.percentile(gmr_clust, [2.5, 97.5])

    return {
        "gmr": point_gmr,
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
        "cluster_ci95_lo": float(c_lo),
        "cluster_ci95_hi": float(c_hi),
        "win_rate": win_rate,
        "n_tasks": float(n_tasks),
        "n_families": float(n_fam),
    }
