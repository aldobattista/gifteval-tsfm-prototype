#!/usr/bin/env python3
"""Build the Top-50 Univariate and Top-50 Multivariate Forecast Showcase Book (Run 5 Best Soup)."""

from __future__ import annotations

import os
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOK_DIR = Path(__file__).resolve().parent
UNI_DIR = BOOK_DIR / "univariate_top50"
MV_DIR = BOOK_DIR / "multivariate_top50"
OV_DIR = BOOK_DIR / "overview_charts"
os.environ.setdefault("GIFT_EVAL", str(REPO_ROOT / "data" / "GiftEval"))

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gift_eval.data import Dataset
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.time_feature import get_seasonality

from tsfm_proto.eval_metrics import DECILE_LEVELS, EvalEntry, predict_entries, score_task_fast
from tsfm_proto.model import TSFMModel

plt.rcParams.update(
    {
        "font.size": 8.9,
        "axes.titlesize": 9.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cbd5e1",
        "pdf.fonttype": 42,
    }
)

Q_ARR = np.asarray(DECILE_LEVELS, dtype=np.float64)[:, None]

PRETTY_REV = {
    "saugeen": "saugeenday",
    "temperature_rain": "temperature_rain_with_missing",
    "kdd_cup_2018": "kdd_cup_2018_with_missing",
    "car_parts": "car_parts_with_missing",
}

FAMILY_YLABELS = {
    "us_births": "US Births / Period",
    "solar": "Solar Power Output (MW)",
    "electricity": "Electricity Load (kW)",
    "loop_seattle": "Freeway Traffic Speed (mph)",
    "sz_taxi": "Taxi Traffic Speed",
    "m_dense": "Traffic Sensor Flow",
    "jena_weather": "Meteorological Sensor Value",
    "ett1": "Transformer Load / Oil Temp",
    "ett2": "Transformer Load / Oil Temp",
    "bitbrains_fast_storage": "Cloud VM Telemetry",
    "bitbrains_rnd": "Cloud VM Telemetry",
    "bizitobs_l2c": "Lead-to-Cash IT Metric",
    "bizitobs_service": "IT Microservice Telemetry",
    "bizitobs_application": "IT Application Telemetry",
    "covid_deaths": "Reported COVID Deaths",
    "hospital": "Hospital Patient Count",
    "restaurant": "Daily Visitors",
    "car_parts": "Monthly Spare Parts Demand",
    "hierarchical_sales": "Unit Sales Volume",
    "saugeen": "River Flow Rate (m^3/s)",
    "temperature_rain": "Meteorological Observation",
    "kdd_cup_2018": "Air Quality Concentration",
}

DOMAIN_COLORS = {
    "Econ/Fin": "#2563eb",
    "Energy": "#d97706",
    "Healthcare": "#0d9488",
    "Nature": "#16a34a",
    "Sales": "#9333ea",
    "Transport": "#dc2626",
    "Web/CloudOps": "#0284c7",
}


def resolve_ds_and_term(cfg_str: str) -> tuple[str, str]:
    """Map a GIFT-Eval config string to its on-disk Dataset name and term."""
    ds_part, freq_part, term = cfg_str.rsplit("/", 2)
    base = PRETTY_REV.get(ds_part, ds_part)
    gifteval_dir = REPO_ROOT / "data" / "GiftEval"
    ds_name = f"{base}/{freq_part}" if (gifteval_dir / base / freq_part).is_dir() else base
    return ds_name, term


def family_ylabel(family: str) -> str:
    """Return a clean physical English y-axis label for a dataset family."""
    if family in FAMILY_YLABELS:
        return FAMILY_YLABELS[family]
    if family.startswith("m4_"):
        return "Macro / Financial Index"
    return "Observed Target Value"


def fmt_num(val: float) -> str:
    """Format scalar error metrics cleanly across magnitudes."""
    av = abs(float(val))
    if av == 0.0:
        return "0.00"
    if av < 1.0:
        return f"{val:.3f}"
    if av < 100.0:
        return f"{val:.2f}"
    return f"{val:.1f}"


def sanitize_name(cfg_str: str) -> str:
    """Convert config names like `bizitobs_l2c/5T/short` into clean filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", cfg_str).strip("_")


def wql_1d(pred_9h: np.ndarray, y_h: np.ndarray) -> float:
    """Compute unweighted-over-quantiles weighted quantile loss on finite steps."""
    p = np.asarray(pred_9h, dtype=np.float64)
    y = np.asarray(y_h, dtype=np.float64)
    valid = np.isfinite(y)
    if not valid.any():
        return float("nan")
    yv = y[valid]
    denom = float(np.sum(np.abs(yv)))
    if denom <= 0.0:
        return float("nan")
    diff = p[:, valid] - yv[None, :]
    ql = 2.0 * np.abs(diff * ((p[:, valid] >= yv[None, :]) - Q_ARR))
    return float(np.mean(np.sum(ql, axis=1) / denom))


def mae_1d(pred_50: np.ndarray, y_h: np.ndarray) -> float:
    """Compute MAE on finite ground-truth steps."""
    valid = np.isfinite(y_h)
    return float(np.mean(np.abs(pred_50[valid] - y_h[valid])))


def cov80_1d(pred_9h: np.ndarray, y_h: np.ndarray) -> float:
    """Compute empirical [q10, q90] 80% interval coverage."""
    valid = np.isfinite(y_h)
    return float(np.mean((y_h[valid] >= pred_9h[0, valid]) & (y_h[valid] <= pred_9h[8, valid])))


def r2_1d(pred_50: np.ndarray, y_h: np.ndarray) -> float:
    """Compute R^2 between median forecast and finite ground-truth steps."""
    valid = np.isfinite(y_h)
    if int(np.sum(valid)) < 4:
        return -1.0
    yv = y_h[valid].astype(np.float64)
    pv = pred_50[valid].astype(np.float64)
    sst = float(np.sum((yv - np.mean(yv)) ** 2))
    if sst < 1e-8:
        return -1.0
    return float(1.0 - float(np.sum((yv - pv) ** 2)) / sst)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute finite-safe Pearson correlation between two 1D arrays."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    fin = np.isfinite(av) & np.isfinite(bv)
    if int(np.sum(fin)) < 4:
        return 0.0
    af, bf = av[fin], bv[fin]
    if float(np.std(af)) < 1e-8 or float(np.std(bf)) < 1e-8:
        return 0.0
    r = float(np.corrcoef(af, bf)[0, 1])
    return r if np.isfinite(r) else 0.0


def gluonts_seasonal_naive_1d(inp: dict, predictor: SeasonalNaivePredictor) -> np.ndarray:
    """Compute exact GluonTS SeasonalNaivePredictor median forecast over horizon."""
    return np.asarray(predictor.predict_item(inp).samples[0], dtype=np.float64)


def znorm_1d(arr: np.ndarray) -> np.ndarray:
    """Normalize a 1D context signal to zero mean / unit std, preserving missing gaps."""
    x = np.asarray(arr, dtype=np.float64)
    fin = np.isfinite(x)
    if not fin.any():
        return np.full_like(x, np.nan)
    return (x - float(np.nanmean(x))) / max(float(np.nanstd(x)), 1e-6)


def compute_cp(ctx_len: int, plen: int) -> int:
    """Choose a visually proportional trailing context window length."""
    if plen <= 168:
        target = max(min(ctx_len, 48), min(144, int(round(1.8 * plen))))
    else:
        target = max(min(ctx_len, 96), min(288, int(round(1.0 * plen))))
    return min(ctx_len, target)


def select_companion_signals(
    ent: EvalEntry, target_var: int, cp: int
) -> tuple[list[tuple[np.ndarray, str]], int]:
    """Select up to 3 active companion traces (prioritizing dynamic past_cov) and count active peers."""
    ctx_tgt = ent.target[target_var]
    ctx_recent = ctx_tgt[-cp:]
    tgt_flat_recent = float(np.nanstd(ctx_recent)) < 1e-6

    active_covs: list[tuple[float, int, float]] = []
    if ent.past_cov is not None:
        for c in range(ent.past_cov.shape[0]):
            seg = ent.past_cov[c, -cp:]
            fin = np.isfinite(seg)
            if int(np.sum(fin)) < 3 or float(np.nanstd(seg[fin])) < 1e-6:
                continue
            r = (
                safe_corr(ctx_tgt, ent.past_cov[c])
                if tgt_flat_recent
                else safe_corr(ctx_recent, seg)
            )
            active_covs.append((abs(r), c, r))
    active_covs.sort(reverse=True)

    active_cotgts: list[tuple[float, int, float]] = []
    for c in range(ent.target.shape[0]):
        if c == target_var:
            continue
        seg = ent.target[c, -cp:]
        fin = np.isfinite(seg)
        if int(np.sum(fin)) < 3 or float(np.nanstd(seg[fin])) < 1e-6:
            continue
        r = (
            safe_corr(ctx_tgt, ent.target[c])
            if tgt_flat_recent
            else safe_corr(ctx_recent, seg)
        )
        active_cotgts.append((abs(r), c, r))
    active_cotgts.sort(reverse=True)

    r_tag = "r_all" if tgt_flat_recent else "r"
    strip: list[tuple[np.ndarray, str]] = []
    for _, c, r in active_covs[:3]:
        strip.append((ent.past_cov[c, -cp:], f"past_cov #{c} ({r_tag}={r:+.2f})"))
    for _, c, r in active_cotgts:
        if len(strip) >= 3:
            break
        strip.append((ent.target[c, -cp:], f"Co-target #{c} ({r_tag}={r:+.2f})"))
    return strip, len(active_covs) + len(active_cotgts)


def mine_top50_univariate(
    model: TSFMModel, cfg, dev: torch.device, tcsv: pd.DataFrame
) -> list[dict]:
    """Scan all 50 pure-univariate GIFT-Eval tasks and select a diverse Top-50 set."""
    use_amp = dev.type in {"mps", "cuda"}
    uni_rows = tcsv[~tcsv["is_mv"] & ~tcsv["has_past_cov"]].copy()
    cands: list[tuple] = []

    for _, row in uni_rows.iterrows():
        cfg_str = str(row["dataset"])
        ds_name, term = resolve_ds_and_term(cfg_str)
        ds = Dataset(name=ds_name, term=term, to_univariate=False)
        slen = int(get_seasonality(ds.freq))
        plen = int(ds.prediction_length)
        sn_predictor = SeasonalNaivePredictor(prediction_length=plen, season_length=slen)
        pairs = list(zip(ds.test_data.input, ds.test_data.label))
        n = len(pairs)
        idxs = np.linspace(0, n - 1, min(n, 140), dtype=int) if n > 140 else np.arange(n)

        ents: list[EvalEntry] = []
        wids: list[int] = []
        raw_inps: list[dict] = []
        for wi in idxs:
            inp, lbl = pairs[int(wi)]
            tgt = np.asarray(inp["target"], dtype=np.float32)
            fut = np.asarray(lbl["target"], dtype=np.float32)
            if tgt.ndim == 1:
                tgt = tgt[None, :]
            if fut.ndim == 1:
                fut = fut[None, :]
            if int(np.isfinite(fut[0]).sum()) < max(5, plen // 2):
                continue
            ents.append(EvalEntry(tgt, None, plen, fut, slen))
            wids.append(int(wi))
            raw_inps.append(inp)
        if not ents:
            continue

        preds = predict_entries(
            model, cfg, ents, dev, batch_size=32, sym_avg=True, use_autocast=use_amp
        ).clamp_deciles
        freq = str(row["frequency"])
        fp = min(plen, 168 if freq in {"H", "15T", "10T", "5T"} else plen)
        min_u = max(5, min(9, fp // 2))

        for wi, inp, ent, pr in zip(wids, raw_inps, ents, preds):
            p9 = pr[:, :, 0]
            y = ent.future_target[0]
            ctx = ent.target[0]
            yv, qv = y[:fp], p9[4, :fp]
            fin_v = np.isfinite(yv)
            if int(np.sum(fin_v)) < max(5, fp // 2):
                continue
            if len(np.unique(np.round(yv[fin_v], 4))) < min_u:
                continue
            if len(np.unique(np.round(qv, 4))) < min_u:
                continue
            sn = gluonts_seasonal_naive_1d(inp, sn_predictor)
            mae_m = mae_1d(p9[4], y)
            mae_s = mae_1d(sn, y)
            mae_mv = mae_1d(qv, yv)
            mae_sv = mae_1d(sn[:fp], yv)
            if (
                mae_s < 1e-6
                or mae_sv < 1e-6
                or mae_m >= 0.68 * mae_s
                or mae_mv >= 0.75 * mae_sv
            ):
                continue
            r2_f = r2_1d(p9[4], y)
            r2_v = r2_1d(qv, yv)
            corr_v = safe_corr(qv, yv)
            amp = float(np.std(qv) / max(float(np.std(yv[fin_v])), 1e-8))
            c80 = cov80_1d(p9, y)
            if r2_v < 0.58 or r2_f < 0.55 or corr_v < 0.80 or not (0.62 <= amp <= 1.38) or c80 < 0.62:
                continue
            gain = 1.0 - mae_m / mae_s
            score = (
                0.42 * min(r2_f, 1.0)
                + 0.35 * min(gain, 0.92)
                + 0.13 * (1.0 - abs(amp - 1.0))
                + 0.10 * (1.0 - abs(c80 - 0.82))
            )
            cands.append(
                (
                    score,
                    {
                        "cfg_str": cfg_str,
                        "family": str(row["family"]),
                        "domain": str(row["domain"]),
                        "frequency": freq,
                        "term": term,
                        "horizon": plen,
                        "fp": fp,
                        "season_len": slen,
                        "win_idx": wi,
                        "task_nmwql": float(row["nmwql"]),
                        "task_nmase": float(row["nmase"]),
                        "mae_run5": mae_m,
                        "mae_sn": mae_s,
                        "mae_delta_pct": 100.0 * (mae_m - mae_s) / mae_s,
                        "mae_run5_view": mae_mv,
                        "mae_sn_view": mae_sv,
                        "mae_delta_view_pct": 100.0 * (mae_mv - mae_sv) / mae_sv,
                        "wql_run5": wql_1d(p9, y),
                        "cov80": c80,
                        "r2_full": r2_f,
                        "r2_view": r2_v,
                        "corr_view": corr_v,
                        "amp_ratio": amp,
                        "score": score,
                        "ctx": ctx,
                        "fut": y,
                        "sn": sn,
                        "pr": p9,
                        "inp": inp,
                    },
                )
            )

    cands.sort(key=lambda x: x[0], reverse=True)
    pool = [rec for _, rec in cands[:180]]
    exact_cands: list[tuple] = []
    if pool:
        b1_ents = [
            EvalEntry(
                r["ctx"][None, :],
                None,
                r["horizon"],
                r["fut"][None, :],
                r["season_len"],
            )
            for r in pool
        ]
        b1_preds = predict_entries(
            model, cfg, b1_ents, dev, batch_size=1, sym_avg=True, use_autocast=use_amp
        ).clamp_deciles
        for rec, pr3d in zip(pool, b1_preds):
            p9 = pr3d[:, :, 0]
            y = rec["fut"]
            sn = rec["sn"]
            fp = rec["fp"]
            yv, qv = y[:fp], p9[4, :fp]
            fin_v = np.isfinite(yv)
            mae_m = mae_1d(p9[4], y)
            mae_s = rec["mae_sn"]
            mae_mv = mae_1d(qv, yv)
            mae_sv = rec["mae_sn_view"]
            if (
                mae_s < 1e-6
                or mae_sv < 1e-6
                or mae_m >= 0.68 * mae_s
                or mae_mv >= 0.75 * mae_sv
            ):
                continue
            r2_f = r2_1d(p9[4], y)
            r2_v = r2_1d(qv, yv)
            corr_v = safe_corr(qv, yv)
            amp = float(np.std(qv) / max(float(np.std(yv[fin_v])), 1e-8))
            c80 = cov80_1d(p9, y)
            if r2_v < 0.58 or r2_f < 0.55 or corr_v < 0.80 or not (0.62 <= amp <= 1.38) or c80 < 0.62:
                continue
            gain = 1.0 - mae_m / mae_s
            score = (
                0.42 * min(r2_f, 1.0)
                + 0.35 * min(gain, 0.92)
                + 0.13 * (1.0 - abs(amp - 1.0))
                + 0.10 * (1.0 - abs(c80 - 0.82))
            )
            rec.update(
                {
                    "pr": p9,
                    "mae_run5": mae_m,
                    "mae_delta_pct": 100.0 * (mae_m - mae_s) / mae_s,
                    "mae_run5_view": mae_mv,
                    "mae_delta_view_pct": 100.0 * (mae_mv - mae_sv) / mae_sv,
                    "wql_run5": wql_1d(p9, y),
                    "cov80": c80,
                    "r2_full": r2_f,
                    "r2_view": r2_v,
                    "corr_view": corr_v,
                    "amp_ratio": amp,
                    "score": score,
                }
            )
            exact_cands.append((score, rec))

    exact_cands.sort(key=lambda x: x[0], reverse=True)
    selected: list[dict] = []
    seen_keys: set[tuple[str, int]] = set()
    cnt_cfg: dict[str, int] = {}
    cnt_fam: dict[str, int] = {}

    for cap_cfg, cap_fam in ((3, 6), (4, 8), (6, 12), (10, 20)):
        for _, rec in exact_cands:
            key = (rec["cfg_str"], rec["win_idx"])
            if key in seen_keys:
                continue
            if cnt_cfg.get(rec["cfg_str"], 0) >= cap_cfg or cnt_fam.get(rec["family"], 0) >= cap_fam:
                continue
            seen_keys.add(key)
            cnt_cfg[rec["cfg_str"]] = cnt_cfg.get(rec["cfg_str"], 0) + 1
            cnt_fam[rec["family"]] = cnt_fam.get(rec["family"], 0) + 1
            selected.append(rec)
            if len(selected) == 50:
                break
        if len(selected) == 50:
            break

    selected.sort(key=lambda r: r["score"], reverse=True)
    for rank, rec in enumerate(selected, start=1):
        rec["rank"] = rank
    return selected


def mine_top50_multivariate(
    model: TSFMModel, cfg, dev: torch.device, tcsv: pd.DataFrame
) -> list[dict]:
    """Scan all MV/Past-Cov tasks and select a diverse Top-50 high-separation ablation set."""
    use_amp = dev.type in {"mps", "cuda"}
    mv_rows = tcsv[tcsv["is_mv"] | tcsv["has_past_cov"]].copy()
    cands: list[tuple] = []

    for _, row in mv_rows.iterrows():
        if float(row["nmwql"]) > 1.15:
            continue
        cfg_str = str(row["dataset"])
        ds_name, term = resolve_ds_and_term(cfg_str)
        ds = Dataset(name=ds_name, term=term, to_univariate=False)
        slen = int(get_seasonality(ds.freq))
        plen = int(ds.prediction_length)
        pairs = list(zip(ds.test_data.input, ds.test_data.label))
        n = len(pairs)
        max_w = 800 if n >= 1000 else min(n, 600)
        idxs = np.linspace(0, n - 1, max_w, dtype=int) if n > max_w else np.arange(n)

        ents_f: list[EvalEntry] = []
        wids: list[int] = []
        for wi in idxs:
            inp, lbl = pairs[int(wi)]
            tgt = np.asarray(inp["target"], dtype=np.float32)
            fut = np.asarray(lbl["target"], dtype=np.float32)
            if tgt.ndim == 1:
                tgt = tgt[None, :]
            if fut.ndim == 1:
                fut = fut[None, :]
            cov_raw = inp.get("past_feat_dynamic_real")
            cov = np.asarray(cov_raw, dtype=np.float32) if cov_raw is not None else None
            if cov is not None and cov.ndim == 1:
                cov = cov[None, :]
            ents_f.append(EvalEntry(tgt, cov, plen, fut, slen))
            wids.append(int(wi))
        if not ents_f:
            continue

        preds_f = predict_entries(
            model, cfg, ents_f, dev, batch_size=48, sym_avg=True, use_autocast=use_amp
        ).clamp_deciles

        fp = min(plen, 144)
        min_u = max(4, min(8, fp // 3))
        uni_sub: list[EvalEntry] = []
        meta_sub: list[tuple] = []

        for wi, ent, prf in zip(wids, ents_f, preds_f):
            vt = ent.target.shape[0]
            cp = compute_cp(ent.target.shape[1], plen)
            for v in range(vt):
                y = ent.future_target[v]
                yv = y[:fp]
                fin_v = np.isfinite(yv)
                if int(np.sum(fin_v)) < max(5, fp // 2):
                    continue
                if float(np.nanstd(yv[fin_v])) < 1e-4:
                    continue
                qf = prf[4, :fp, v]
                if len(np.unique(np.round(yv[fin_v], 4))) < min_u:
                    continue
                if len(np.unique(np.round(qf, 4))) < min_u:
                    continue
                strip, n_active = select_companion_signals(ent, v, cp)
                if n_active < 2:
                    continue
                r2_v = r2_1d(qf, yv)
                r2_f = r2_1d(prf[4, :, v], y)
                corr_v = safe_corr(qf, yv)
                c80 = cov80_1d(prf[:, :, v], y)
                if c80 < 0.53 or r2_v < 0.13 or r2_f < 0.13 or corr_v < 0.48:
                    continue
                uni_sub.append(
                    EvalEntry(
                        ent.target[v : v + 1],
                        None,
                        plen,
                        ent.future_target[v : v + 1],
                        slen,
                    )
                )
                meta_sub.append((wi, v, cp, strip, ent, prf[:, :, v], r2_v, r2_f, corr_v, c80))

        if not uni_sub:
            continue

        preds_u = predict_entries(
            model, cfg, uni_sub, dev, batch_size=64, sym_avg=True, use_autocast=use_amp
        ).clamp_deciles

        for (wi, v, cp, strip, ent, p9f, r2_v, r2_f, corr_v, c80), pru in zip(meta_sub, preds_u):
            p9u = pru[:, :, 0]
            y = ent.future_target[v]
            mae_f = mae_1d(p9f[4], y)
            mae_u = mae_1d(p9u[4], y)
            mae_fv = mae_1d(p9f[4, :fp], y[:fp])
            mae_uv = mae_1d(p9u[4, :fp], y[:fp])
            v_tgt = ent.target.shape[0]
            v_cov = ent.past_cov.shape[0] if ent.past_cov is not None else 0
            min_gain = 0.045 if v_tgt == 1 else 0.06
            if mae_u < 1e-6 or mae_uv < 1e-6:
                continue
            gain = 1.0 - mae_f / mae_u
            gain_v = 1.0 - mae_fv / mae_uv
            if gain < (min_gain - 0.005) or gain_v < 0.015:
                continue
            all_h = np.concatenate([y[:fp], p9u[4, :fp], p9f[4, :fp]])
            all_h = all_h[np.isfinite(all_h)]
            span = float(np.max(all_h) - np.min(all_h))
            sep = float(np.mean(np.abs(p9f[4, :fp] - p9u[4, :fp])) / max(span, 1e-8))
            max_sep = float(np.max(np.abs(p9f[4, :fp] - p9u[4, :fp])) / max(span, 1e-8))
            if sep < ((0.010 if v_tgt == 1 else 0.018) * 0.90) or max_sep < 0.032:
                continue
            score = (
                0.40 * min(gain, 0.70)
                + 0.30 * max(0.0, min(r2_f, 1.0))
                + 0.20 * min(sep * 4.0, 1.0)
                + 0.10 * (1.0 - abs(c80 - 0.82))
            )
            cands.append(
                (
                    score,
                    {
                        "cfg_str": cfg_str,
                        "family": str(row["family"]),
                        "domain": str(row["domain"]),
                        "frequency": str(row["frequency"]),
                        "term": term,
                        "horizon": plen,
                        "fp": fp,
                        "cp": cp,
                        "season_len": slen,
                        "win_idx": wi,
                        "var_idx": v,
                        "v_tgt": v_tgt,
                        "v_cov": v_cov,
                        "has_past_cov": bool(row["has_past_cov"]),
                        "task_nmwql": float(row["nmwql"]),
                        "task_nmase": float(row["nmase"]),
                        "mae_with": mae_f,
                        "mae_without": mae_u,
                        "mae_nocov": float("nan"),
                        "mae_delta_pct": 100.0 * (mae_f - mae_u) / mae_u,
                        "mae_with_view": mae_fv,
                        "mae_without_view": mae_uv,
                        "mae_delta_view_pct": 100.0 * (mae_fv - mae_uv) / mae_uv,
                        "wql_with": wql_1d(p9f, y),
                        "wql_without": wql_1d(p9u, y),
                        "cov80_with": c80,
                        "cov80_without": cov80_1d(p9u, y),
                        "r2_full": r2_f,
                        "r2_view": r2_v,
                        "corr_view": corr_v,
                        "sep_ratio": sep,
                        "score": score,
                        "ent": ent,
                        "strip": strip,
                        "prf": p9f,
                        "pru": p9u,
                        "prn": None,
                    },
                )
            )

    # Recompute exact standalone batch_size=1 predictions (prf, pru, prn) on all screened candidates
    # before strict filtering & final Top-50 selection so every plate is 100% deterministic.
    exact_cands: list[tuple] = []
    if cands:
        uniq_mv: dict[tuple[str, int], EvalEntry] = {}
        uniq_nocov: dict[tuple[str, int], EvalEntry] = {}
        for _, r in cands:
            k_win = (r["cfg_str"], r["win_idx"])
            if k_win not in uniq_mv:
                uniq_mv[k_win] = r["ent"]
            if r["v_tgt"] > 1 and r["v_cov"] > 0 and k_win not in uniq_nocov:
                e = r["ent"]
                uniq_nocov[k_win] = EvalEntry(
                    e.target, None, e.prediction_length, e.future_target, e.season_length
                )

        mv_keys = list(uniq_mv.keys())
        mv_b1 = predict_entries(
            model, cfg, [uniq_mv[k] for k in mv_keys], dev, batch_size=1, sym_avg=True, use_autocast=use_amp
        ).clamp_deciles
        mv_pred_map = dict(zip(mv_keys, mv_b1))

        u_ents_b1 = [
            EvalEntry(
                r["ent"].target[r["var_idx"] : r["var_idx"] + 1],
                None,
                r["horizon"],
                r["ent"].future_target[r["var_idx"] : r["var_idx"] + 1],
                r["season_len"],
            )
            for _, r in cands
        ]
        u_b1 = predict_entries(
            model, cfg, u_ents_b1, dev, batch_size=1, sym_avg=True, use_autocast=use_amp
        ).clamp_deciles

        nocov_pred_map: dict[tuple[str, int], np.ndarray] = {}
        if uniq_nocov:
            nc_keys = list(uniq_nocov.keys())
            nc_b1 = predict_entries(
                model,
                cfg,
                [uniq_nocov[k] for k in nc_keys],
                dev,
                batch_size=1,
                sym_avg=True,
                use_autocast=use_amp,
            ).clamp_deciles
            nocov_pred_map = dict(zip(nc_keys, nc_b1))

        for (_, rec), pru3d in zip(cands, u_b1):
            k_win = (rec["cfg_str"], rec["win_idx"])
            v = rec["var_idx"]
            fp = rec["fp"]
            ent = rec["ent"]
            y = ent.future_target[v]
            yv = y[:fp]

            p9f = mv_pred_map[k_win][:, :, v]
            p9u = pru3d[:, :, 0]
            p9n = nocov_pred_map[k_win][:, :, v] if k_win in nocov_pred_map else None

            qf = p9f[4, :fp]
            r2_v = r2_1d(qf, yv)
            r2_f = r2_1d(p9f[4], y)
            corr_v = safe_corr(qf, yv)
            c80 = cov80_1d(p9f, y)
            if c80 < 0.55 or r2_v < 0.15 or r2_f < 0.15 or corr_v < 0.50:
                continue

            mae_f = mae_1d(p9f[4], y)
            mae_u = mae_1d(p9u[4], y)
            mae_fv = mae_1d(p9f[4, :fp], yv)
            mae_uv = mae_1d(p9u[4, :fp], yv)
            v_tgt = rec["v_tgt"]
            v_cov = rec["v_cov"]
            min_gain = 0.045 if v_tgt == 1 else 0.06
            if mae_u < 1e-6 or mae_uv < 1e-6:
                continue
            gain = 1.0 - mae_f / mae_u
            gain_v = 1.0 - mae_fv / mae_uv
            if gain < min_gain or gain_v < 0.02:
                continue

            all_h = np.concatenate([yv, p9u[4, :fp], qf])
            all_h = all_h[np.isfinite(all_h)]
            span = float(np.max(all_h) - np.min(all_h))
            sep = float(np.mean(np.abs(qf - p9u[4, :fp])) / max(span, 1e-8))
            max_sep = float(np.max(np.abs(qf - p9u[4, :fp])) / max(span, 1e-8))
            if sep < (0.010 if v_tgt == 1 else 0.018) or max_sep < 0.035:
                continue

            mae_n = float("nan")
            if p9n is not None:
                mae_n = mae_1d(p9n[4], y)
                if mae_f > mae_n:
                    continue

            score = (
                0.40 * min(gain, 0.70)
                + 0.30 * max(0.0, min(r2_f, 1.0))
                + 0.20 * min(sep * 4.0, 1.0)
                + 0.10 * (1.0 - abs(c80 - 0.82))
            )
            if p9n is not None and mae_f <= mae_n:
                score += 0.06

            rec.update(
                {
                    "mae_with": mae_f,
                    "mae_without": mae_u,
                    "mae_nocov": mae_n,
                    "mae_delta_pct": 100.0 * (mae_f - mae_u) / mae_u,
                    "mae_with_view": mae_fv,
                    "mae_without_view": mae_uv,
                    "mae_delta_view_pct": 100.0 * (mae_fv - mae_uv) / mae_uv,
                    "wql_with": wql_1d(p9f, y),
                    "wql_without": wql_1d(p9u, y),
                    "cov80_with": c80,
                    "cov80_without": cov80_1d(p9u, y),
                    "r2_full": r2_f,
                    "r2_view": r2_v,
                    "corr_view": corr_v,
                    "sep_ratio": sep,
                    "score": score,
                    "prf": p9f,
                    "pru": p9u,
                    "prn": p9n,
                }
            )
            exact_cands.append((score, rec))

    exact_cands.sort(key=lambda x: x[0], reverse=True)
    selected: list[dict] = []
    seen_keys: set[tuple[str, int, int]] = set()
    win_picks: dict[tuple[str, int], list[np.ndarray]] = {}
    cnt_tv: dict[tuple[str, int], int] = {}
    cnt_cfg: dict[str, int] = {}
    cnt_fam: dict[str, int] = {}

    for cap_tv, cap_cfg, cap_fam in ((2, 4, 12), (3, 6, 16), (5, 10, 22), (8, 15, 30), (12, 22, 36)):
        for _, rec in exact_cands:
            key = (rec["cfg_str"], rec["win_idx"], rec["var_idx"])
            if key in seen_keys:
                continue
            cw_key = (rec["cfg_str"], rec["win_idx"])
            prev_futs = win_picks.get(cw_key, [])
            if len(prev_futs) >= 3:
                continue
            fut_v = rec["ent"].future_target[rec["var_idx"]][: rec["fp"]]
            if prev_futs and (
                rec["family"].startswith("bitbrains")
                or any(abs(safe_corr(fut_v, pf)) >= 0.90 for pf in prev_futs)
            ):
                continue
            tv_key = (rec["cfg_str"], rec["var_idx"])
            if (
                cnt_tv.get(tv_key, 0) >= cap_tv
                or cnt_cfg.get(rec["cfg_str"], 0) >= cap_cfg
                or cnt_fam.get(rec["family"], 0) >= cap_fam
            ):
                continue
            seen_keys.add(key)
            win_picks.setdefault(cw_key, []).append(fut_v)
            cnt_tv[tv_key] = cnt_tv.get(tv_key, 0) + 1
            cnt_cfg[rec["cfg_str"]] = cnt_cfg.get(rec["cfg_str"], 0) + 1
            cnt_fam[rec["family"]] = cnt_fam.get(rec["family"], 0) + 1
            selected.append(rec)
            if len(selected) == 50:
                break
        if len(selected) == 50:
            break

    selected.sort(key=lambda r: r["score"], reverse=True)
    for rank, rec in enumerate(selected, start=1):
        rec["rank"] = rank
    return selected


def verify_showcase_rigor(
    model: TSFMModel,
    cfg,
    dev: torch.device,
    uni_recs: list[dict],
    mv_recs: list[dict],
) -> None:
    """Assert single-item inference, GluonTS SeasonalNaive, metric, and quantile invariants."""
    use_amp = dev.type in {"mps", "cuda"}
    ds_cache: dict[str, Dataset] = {}

    def _get_ds(cfg_str: str) -> Dataset:
        if cfg_str not in ds_cache:
            ds_name, term = resolve_ds_and_term(cfg_str)
            ds_cache[cfg_str] = Dataset(name=ds_name, term=term, to_univariate=False)
        return ds_cache[cfg_str]

    for rec in uni_recs:
        ds = _get_ds(rec["cfg_str"])
        slen = int(get_seasonality(ds.freq))
        plen = int(ds.prediction_length)
        pairs = list(zip(ds.test_data.input, ds.test_data.label))
        inp, lbl = pairs[int(rec["win_idx"])]
        tgt = np.asarray(inp["target"], dtype=np.float32)
        fut = np.asarray(lbl["target"], dtype=np.float32)
        if tgt.ndim == 1:
            tgt = tgt[None, :]
        if fut.ndim == 1:
            fut = fut[None, :]
        ent = EvalEntry(tgt, None, plen, fut, slen)
        pred3d = predict_entries(
            model, cfg, [ent], dev, batch_size=1, sym_avg=True, use_autocast=use_amp
        ).clamp_deciles[0]
        p9_single = pred3d[:, :, 0]
        if float(np.max(np.abs(p9_single - rec["pr"]))) >= 1e-5:
            raise AssertionError(
                f"Uni single-item inference parity failed on {rec['cfg_str']} w{rec['win_idx']}"
            )

        sn_ref = gluonts_seasonal_naive_1d(
            inp, SeasonalNaivePredictor(prediction_length=plen, season_length=slen)
        )
        if float(np.max(np.abs(sn_ref - rec["sn"]))) >= 1e-6:
            raise AssertionError(
                f"Uni GluonTS SeasonalNaive parity failed on {rec['cfg_str']} w{rec['win_idx']}"
            )

        st = score_task_fast([ent], [pred3d])
        if (
            abs(wql_1d(rec["pr"], fut[0]) - st["mean_weighted_sum_quantile_loss"]) >= 1e-6
            or abs(cov80_1d(rec["pr"], fut[0]) - st["coverage_80"]) >= 1e-6
        ):
            raise AssertionError(
                f"Uni score_task_fast parity failed on {rec['cfg_str']} w{rec['win_idx']}"
            )

        if not (
            bool(np.all(rec["pr"][0] <= rec["pr"][4] + 1e-6))
            and bool(np.all(rec["pr"][4] <= rec["pr"][8] + 1e-6))
        ):
            raise AssertionError(
                f"Uni quantile crossing detected on {rec['cfg_str']} w{rec['win_idx']}"
            )

        if not (
            rec["mae_run5"] < rec["mae_sn"]
            and rec["mae_run5_view"] < rec["mae_sn_view"]
            and rec["r2_full"] > 0.0
            and rec["r2_view"] > 0.0
        ):
            raise AssertionError(
                f"Uni performance/R2 invariant failed on {rec['cfg_str']} w{rec['win_idx']}"
            )

    for rec in mv_recs:
        ds = _get_ds(rec["cfg_str"])
        slen = int(get_seasonality(ds.freq))
        plen = int(ds.prediction_length)
        pairs = list(zip(ds.test_data.input, ds.test_data.label))
        inp, lbl = pairs[int(rec["win_idx"])]
        tgt = np.asarray(inp["target"], dtype=np.float32)
        fut = np.asarray(lbl["target"], dtype=np.float32)
        if tgt.ndim == 1:
            tgt = tgt[None, :]
        if fut.ndim == 1:
            fut = fut[None, :]
        cov_raw = inp.get("past_feat_dynamic_real")
        cov = np.asarray(cov_raw, dtype=np.float32) if cov_raw is not None else None
        if cov is not None and cov.ndim == 1:
            cov = cov[None, :]
        v = int(rec["var_idx"])

        ent_f = EvalEntry(tgt, cov, plen, fut, slen)
        ent_u = EvalEntry(tgt[v : v + 1], None, plen, fut[v : v + 1], slen)
        prf_single = predict_entries(
            model, cfg, [ent_f], dev, batch_size=1, sym_avg=True, use_autocast=use_amp
        ).clamp_deciles[0][:, :, v]
        pru_single_3d = predict_entries(
            model, cfg, [ent_u], dev, batch_size=1, sym_avg=True, use_autocast=use_amp
        ).clamp_deciles[0]
        pru_single = pru_single_3d[:, :, 0]
        if (
            float(np.max(np.abs(prf_single - rec["prf"]))) >= 1e-5
            or float(np.max(np.abs(pru_single - rec["pru"]))) >= 1e-5
        ):
            raise AssertionError(
                f"MV single-item inference parity failed on {rec['cfg_str']} w{rec['win_idx']} v{v}"
            )

        if rec["prn"] is not None:
            ent_n = EvalEntry(tgt, None, plen, fut, slen)
            prn_single = predict_entries(
                model, cfg, [ent_n], dev, batch_size=1, sym_avg=True, use_autocast=use_amp
            ).clamp_deciles[0][:, :, v]
            if float(np.max(np.abs(prn_single - rec["prn"]))) >= 1e-5:
                raise AssertionError(
                    f"MV Co-Tgt single-item parity failed on {rec['cfg_str']} w{rec['win_idx']} v{v}"
                )

        ent_f_1v = EvalEntry(tgt[v : v + 1], None, plen, fut[v : v + 1], slen)
        st_f = score_task_fast([ent_f_1v], [rec["prf"][:, :, None]])
        st_u = score_task_fast([ent_u], [pru_single_3d])
        if (
            abs(wql_1d(rec["prf"], fut[v]) - st_f["mean_weighted_sum_quantile_loss"]) >= 1e-6
            or abs(cov80_1d(rec["prf"], fut[v]) - st_f["coverage_80"]) >= 1e-6
            or abs(wql_1d(rec["pru"], fut[v]) - st_u["mean_weighted_sum_quantile_loss"]) >= 1e-6
        ):
            raise AssertionError(
                f"MV score_task_fast parity failed on {rec['cfg_str']} w{rec['win_idx']} v{v}"
            )

        for arr_name, arr in [("prf", rec["prf"]), ("pru", rec["pru"])]:
            if not (
                bool(np.all(arr[0] <= arr[4] + 1e-6))
                and bool(np.all(arr[4] <= arr[8] + 1e-6))
            ):
                raise AssertionError(
                    f"MV quantile crossing ({arr_name}) on {rec['cfg_str']} w{rec['win_idx']} v{v}"
                )
        if rec["prn"] is not None and not (
            bool(np.all(rec["prn"][0] <= rec["prn"][4] + 1e-6))
            and bool(np.all(rec["prn"][4] <= rec["prn"][8] + 1e-6))
        ):
            raise AssertionError(
                f"MV quantile crossing (prn) on {rec['cfg_str']} w{rec['win_idx']} v{v}"
            )

        if not (
            rec["mae_with"] < rec["mae_without"]
            and rec["mae_with_view"] < rec["mae_without_view"]
            and rec["r2_full"] > 0.0
            and rec["r2_view"] > 0.0
        ):
            raise AssertionError(
                f"MV performance/R2 invariant failed on {rec['cfg_str']} w{rec['win_idx']} v{v}"
            )
        if rec["v_tgt"] > 1 and rec["v_cov"] > 0 and rec["mae_with"] > rec["mae_nocov"]:
            raise AssertionError(
                f"MV past_cov gain invariant failed on {rec['cfg_str']} w{rec['win_idx']} v{v}"
            )

    print("All 100 showcase plates passed strict inference, GluonTS SeasonalNaive, metric, and quantile invariants.")


def build_univariate_figure(rec: dict) -> plt.Figure:
    """Create a 2-column English showcase plate for a Univariate example."""
    rank = rec["rank"]
    cfg_str = rec["cfg_str"]
    plen = rec["horizon"]
    fp = rec["fp"]
    ctx = rec["ctx"]
    fut = rec["fut"]
    sn = rec["sn"]
    pr = rec["pr"]
    slen = rec["season_len"]
    cp = compute_cp(len(ctx), plen)
    ylab = family_ylabel(rec["family"])
    col = DOMAIN_COLORS.get(rec["domain"], "#0f766e")

    fig = plt.figure(figsize=(13.4, 5.45))
    fig.suptitle(
        f"Univariate #{rank:02d}  |  {cfg_str} (Window #{rec['win_idx']})  |  "
        f"{rec['domain']}  |  Freq={rec['frequency']}, H={plen}",
        fontsize=11.6,
        fontweight="bold",
        y=0.975,
    )
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[2.15, 1.05],
        left=0.072,
        right=0.985,
        top=0.77,
        bottom=0.12,
        wspace=0.18,
    )
    ax_m = fig.add_subplot(gs[0, 0])
    ax_z = fig.add_subplot(gs[0, 1])

    tc = np.arange(-cp, 0)
    tf_full = np.arange(0, plen)
    tf_zoom = np.arange(0, fp)

    # Left: Full trailing context + complete forecast horizon (0..H-1)
    ax_m.plot(tc, ctx[-cp:], color="#1e293b", lw=1.55, label=f"Historical Context ({cp} steps, t < 0)")
    ax_m.axvline(0, color="#475569", lw=1.05, ls="--")
    if fp < plen:
        ax_m.axvline(fp - 1, color="#94a3b8", lw=0.95, ls=":")
    ax_m.plot(tf_full, fut, color="#0f172a", lw=2.0, zorder=5, label="Ground Truth")
    ax_m.plot(
        tf_full,
        sn,
        color="#64748b",
        lw=1.35,
        ls=":",
        label=f"Seasonal Naive S={slen} (MAE={fmt_num(rec['mae_sn'])})",
    )
    ax_m.fill_between(tf_full, pr[0], pr[8], color=col, alpha=0.22, label="Run 5 [q10, q90]")
    ax_m.plot(
        tf_full,
        pr[4],
        color=col,
        lw=2.1,
        zorder=6,
        label=f"Run 5 Median q50 (MAE={fmt_num(rec['mae_run5'])})",
    )

    badge_m = (
        f"Task ({cfg_str}): nMWQL={rec['task_nmwql']:.3f} | nMASE={rec['task_nmase']:.3f}\n"
        f"Window H={plen}: MAE {fmt_num(rec['mae_run5'])} vs SN {fmt_num(rec['mae_sn'])} "
        f"({rec['mae_delta_pct']:+.1f}%) | R^2={rec['r2_full']:.3f} | Cov80={100*rec['cov80']:.0f}%"
    )
    ax_m.text(
        0.015,
        0.96,
        badge_m,
        transform=ax_m.transAxes,
        va="top",
        ha="left",
        fontsize=7.7,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.96),
    )
    yv = np.concatenate([ctx[-cp:], fut, sn, pr[0], pr[8]])
    yv = yv[np.isfinite(yv)]
    ymin, ymax = float(np.min(yv)), float(np.max(yv))
    yspan = max(ymax - ymin, 1e-3)
    ax_m.set_ylim(ymin - 0.08 * yspan, ymax + 0.36 * yspan)
    ax_m.set_title("Context & Full-Horizon Forecast (Run 5 Best Soup +SymAvg+Clamp)", pad=36)
    ax_m.set_ylabel(ylab)
    ax_m.set_xlabel("Timestep Relative to Forecast Origin (T_origin = 0)")
    ax_m.legend(ncol=3, fontsize=7.2, loc="lower left", bbox_to_anchor=(0.0, 1.015))

    # Right: Zoomed Horizon Detail
    tail = min(6, cp)
    ax_z.plot(np.arange(-tail, 0), ctx[-tail:], color="#64748b", lw=1.2, alpha=0.7)
    ax_z.axvline(0, color="#475569", lw=1.05, ls="--")
    ax_z.plot(tf_zoom, fut[:fp], color="#0f172a", lw=2.1, zorder=5, label="Ground Truth")
    ax_z.plot(tf_zoom, sn[:fp], color="#64748b", lw=1.35, ls=":", label="Seasonal Naive")
    ax_z.fill_between(tf_zoom, pr[0, :fp], pr[8, :fp], color=col, alpha=0.22)
    ax_z.plot(tf_zoom, pr[4, :fp], color=col, lw=2.2, zorder=6, label="Run 5 Median q50")

    zv = np.concatenate([fut[:fp], sn[:fp], pr[0, :fp], pr[8, :fp]])
    zv = zv[np.isfinite(zv)]
    zmin, zmax = float(np.min(zv)), float(np.max(zv))
    zspan = max(zmax - zmin, 1e-3)
    ax_z.set_ylim(zmin - 0.08 * zspan, zmax + 0.38 * zspan)
    ax_z.set_xlim(-tail, fp - 1)
    if fp < plen:
        badge_z = (
            f"Zoom MAE: {fmt_num(rec['mae_run5_view'])} vs SN {fmt_num(rec['mae_sn_view'])} "
            f"({rec['mae_delta_view_pct']:+.1f}%)\n"
            f"R^2 (zoom)={rec['r2_view']:.3f} | r={rec['corr_view']:.3f} | wQL(H)={rec['wql_run5']:.4f}"
        )
        z_title = f"Early-Horizon Zoom (t in [0, {fp - 1}] of H={plen})"
    else:
        badge_z = (
            f"R^2={rec['r2_full']:.3f} | r={rec['corr_view']:.3f}\n"
            f"Amp={rec['amp_ratio']:.2f}x | wQL={rec['wql_run5']:.4f}"
        )
        z_title = f"Horizon Detail (t in [0, {plen - 1}])"
    ax_z.text(
        0.025,
        0.96,
        badge_z,
        transform=ax_z.transAxes,
        va="top",
        ha="left",
        fontsize=7.7,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.96),
    )
    ax_z.set_title(z_title, pad=36)
    ax_z.set_ylabel(ylab)
    ax_z.set_xlabel(f"Forecast Horizon Step (0 to {fp - 1})")
    ax_z.legend(ncol=3, fontsize=7.0, loc="lower left", bbox_to_anchor=(0.0, 1.015))
    return fig


def build_multivariate_figure(rec: dict) -> plt.Figure:
    """Create a 2-column English ablation plate (WITH vs WITHOUT covariates/group channels)."""
    rank = rec["rank"]
    cfg_str = rec["cfg_str"]
    ent = rec["ent"]
    v = rec["var_idx"]
    cp = rec["cp"]
    fp = rec["fp"]
    plen = rec["horizon"]
    ctx = ent.target[v]
    fut = ent.future_target[v]
    prf = rec["prf"]
    pru = rec["pru"]
    prn = rec["prn"]
    ylab = family_ylabel(rec["family"])
    mode_tag = f"V_tgt={rec['v_tgt']}, V_cov={rec['v_cov']}"
    cov_kind = "Coupled + Past Covariates" if rec["v_cov"] > 0 else "Coupled Multi-Target GroupAttention"

    fig = plt.figure(figsize=(14.0, 6.45))
    fig.suptitle(
        f"Multivariate #{rank:02d}  |  {cfg_str} (win #{rec['win_idx']}, var #{v})  |  "
        f"Freq={rec['frequency']}, H={plen}  |  {cov_kind} ({mode_tag})",
        fontsize=11.3,
        fontweight="bold",
        y=0.975,
    )
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[2.12, 1.12],
        height_ratios=[2.25, 1.0],
        left=0.070,
        right=0.985,
        top=0.815,
        bottom=0.095,
        hspace=0.22,
        wspace=0.17,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_zoom = fig.add_subplot(gs[:, 1])

    tc = np.arange(-cp, 0)
    tf_full = np.arange(0, plen)
    tf_zoom = np.arange(0, fp)
    with_lbl = "WITH Group+Cov" if rec["v_cov"] > 0 else f"WITH {rec['v_tgt']} Targets"

    # Top-Left: Context + Full-Horizon Ablation (0..H-1)
    ax_top.plot(tc, ctx[-cp:], color="#1e293b", lw=1.55, label=f"Target Context Var #{v}")
    ax_top.axvline(0, color="#475569", lw=1.05, ls="--")
    if fp < plen:
        ax_top.axvline(fp - 1, color="#94a3b8", lw=0.95, ls=":")
    ax_top.plot(tf_full, fut, color="#0f172a", lw=2.05, zorder=5, label="Ground Truth")
    ax_top.fill_between(tf_full, pru[0], pru[8], color="#dc2626", alpha=0.12, label="WITHOUT [q10, q90]")
    ax_top.plot(
        tf_full,
        pru[4],
        color="#dc2626",
        lw=1.75,
        ls="--",
        zorder=4,
        label=f"WITHOUT (Univ, MAE={fmt_num(rec['mae_without'])})",
    )
    if prn is not None:
        ax_top.plot(
            tf_full,
            prn[4],
            color="#d97706",
            lw=1.65,
            ls="-.",
            zorder=4,
            label=f"Co-Targets Only (MAE={fmt_num(rec['mae_nocov'])})",
        )
    ax_top.fill_between(tf_full, prf[0], prf[8], color="#15803d", alpha=0.22, label=f"{with_lbl} [q10, q90]")
    ax_top.plot(
        tf_full,
        prf[4],
        color="#15803d",
        lw=2.1,
        zorder=6,
        label=f"{with_lbl} (MAE={fmt_num(rec['mae_with'])})",
    )

    if prn is not None:
        pure_cov_pct = 100.0 * (rec["mae_with"] - rec["mae_nocov"]) / max(rec["mae_nocov"], 1e-8)
        ablation_str = (
            f"Univ {fmt_num(rec['mae_without'])} -> Tgt {fmt_num(rec['mae_nocov'])} -> "
            f"Full {fmt_num(rec['mae_with'])} ({rec['mae_delta_pct']:+.1f}%, {pure_cov_pct:+.1f}% past_cov) | R^2={rec['r2_full']:.3f}"
        )
    else:
        ablation_str = (
            f"WITHOUT {fmt_num(rec['mae_without'])} -> WITH {fmt_num(rec['mae_with'])} "
            f"({rec['mae_delta_pct']:+.1f}% MAE) | R^2={rec['r2_full']:.3f} | Cov80={100*rec['cov80_with']:.0f}%"
        )
    badge_top = f"Task ({cfg_str}): nMWQL={rec['task_nmwql']:.3f} | nMASE={rec['task_nmase']:.3f}\n{ablation_str}"
    ax_top.text(
        0.015,
        0.96,
        badge_top,
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#f0fdf4", edgecolor="#86efac", alpha=0.96),
    )

    yparts = [ctx[-cp:], fut, pru[0], pru[8], prf[0], prf[8]]
    if prn is not None:
        yparts.append(prn[4])
    yall = np.concatenate(yparts)
    yall = yall[np.isfinite(yall)]
    ymin, ymax = float(np.min(yall)), float(np.max(yall))
    yspan = max(ymax - ymin, 1e-3)
    ax_top.set_ylim(ymin - 0.08 * yspan, ymax + 0.36 * yspan)
    ax_top.set_title("Context + Full-Horizon Conditional Ablation (WITH vs. WITHOUT Auxiliary Channels)", pad=46)
    ax_top.set_ylabel(ylab)
    ax_top.legend(ncol=3, fontsize=7.0, loc="lower left", bbox_to_anchor=(0.0, 1.015))
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Bottom-Left: Top-3 active companion signals in z-score (preserving missing gaps)
    strip_cols = ["#7c3aed", "#d97706", "#0284c7"]
    zlist = []
    for (arr, lbl), scol in zip(rec["strip"], strip_cols):
        zarr = znorm_1d(arr)
        zlist.append(zarr)
        ax_bot.plot(tc, zarr, color=scol, lw=1.35, label=lbl)
    ax_bot.axvline(0, color="#475569", lw=1.05, ls="--")
    if fp < plen:
        ax_bot.axvline(fp - 1, color="#94a3b8", lw=0.95, ls=":")
    ax_bot.axvspan(0, plen - 1, color="#e2e8f0", alpha=0.58)
    stacked = np.concatenate(zlist) if zlist else np.zeros(10)
    stacked = stacked[np.isfinite(stacked)]
    if stacked.size == 0:
        stacked = np.zeros(10)
    smin, smax = float(np.min(stacked)), float(np.max(stacked))
    sspan = max(smax - smin, 1.0)
    ax_bot.set_ylim(smin - 0.14 * sspan, smax + 0.50 * sspan)
    ymid = smin + 0.45 * sspan
    if rec["v_cov"] > 0:
        snote = (
            "Future horizon (t >= 0)\npast_cov masked; co-targets active"
            if rec["v_tgt"] > 1
            else "Future horizon (t >= 0)\npast_cov masked out"
        )
    else:
        snote = f"Future horizon (t >= 0)\nAll {rec['v_tgt']} targets forecast jointly"
    ax_bot.text(
        0.5 * (plen - 1),
        ymid,
        snote,
        ha="center",
        va="center",
        fontsize=7.6,
        color="#475569",
        fontstyle="italic",
    )
    ax_bot.set_ylabel("Companion (z-score)")
    ax_bot.set_xlabel("Timestep Relative to Forecast Origin (T_origin = 0)")
    ax_bot.legend(ncol=3, fontsize=7.0, loc="upper left")

    # Right Full-Height: Horizon Zoom
    tail = min(6, cp)
    ax_zoom.plot(np.arange(-tail, 0), ctx[-tail:], color="#64748b", lw=1.2, alpha=0.7)
    ax_zoom.axvline(0, color="#475569", lw=1.05, ls="--")
    ax_zoom.plot(tf_zoom, fut[:fp], color="#0f172a", lw=2.2, zorder=5, label="Ground Truth")
    ax_zoom.fill_between(tf_zoom, pru[0, :fp], pru[8, :fp], color="#dc2626", alpha=0.12)
    ax_zoom.plot(tf_zoom, pru[4, :fp], color="#dc2626", lw=1.9, ls="--", zorder=4, label="WITHOUT")
    if prn is not None:
        ax_zoom.plot(tf_zoom, prn[4, :fp], color="#d97706", lw=1.8, ls="-.", zorder=4, label="Co-Tgt Only")
    ax_zoom.fill_between(tf_zoom, prf[0, :fp], prf[8, :fp], color="#15803d", alpha=0.22)
    ax_zoom.plot(tf_zoom, prf[4, :fp], color="#15803d", lw=2.25, zorder=6, label=with_lbl)

    zparts = [fut[:fp], pru[0, :fp], pru[4, :fp], pru[8, :fp], prf[0, :fp], prf[8, :fp]]
    if prn is not None:
        zparts.append(prn[4, :fp])
    zall = np.concatenate(zparts)
    zall = zall[np.isfinite(zall)]
    zmin, zmax = float(np.min(zall)), float(np.max(zall))
    zspan = max(zmax - zmin, 1e-3)
    ax_zoom.set_ylim(zmin - 0.10 * zspan, zmax + 0.36 * zspan)
    ax_zoom.set_xlim(-tail, fp - 1)
    if fp < plen:
        badge_z = (
            f"Zoom t<={fp - 1}: {fmt_num(rec['mae_without_view'])} -> {fmt_num(rec['mae_with_view'])} "
            f"({rec['mae_delta_view_pct']:+.1f}% MAE, R^2={rec['r2_view']:.3f})\n"
            f"Full H={plen}: wQL {rec['wql_without']:.4f} -> {rec['wql_with']:.4f} | "
            f"R^2={rec['r2_full']:.3f} | Cov80={100*rec['cov80_with']:.0f}%"
        )
        z_title = f"Early-Horizon Zoom (t in [0, {fp - 1}] of H={plen})"
    else:
        badge_z = (
            f"WITHOUT: MAE={fmt_num(rec['mae_without'])} | wQL={rec['wql_without']:.4f}\n"
            f"{with_lbl}: MAE={fmt_num(rec['mae_with'])} ({rec['mae_delta_pct']:+.1f}%)\n"
            f"wQL={rec['wql_with']:.4f} | R^2={rec['r2_full']:.3f} | Cov80={100*rec['cov80_with']:.0f}%"
        )
        z_title = f"Horizon Zoom (t in [0, {plen - 1}])"
    ax_zoom.text(
        0.025,
        0.96,
        badge_z,
        transform=ax_zoom.transAxes,
        va="top",
        ha="left",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.96),
    )
    ax_zoom.set_title(z_title, pad=46)
    ax_zoom.set_ylabel(ylab)
    ax_zoom.set_xlabel(f"Forecast Horizon Step (0 to {fp - 1})")
    ax_zoom.legend(ncol=2, fontsize=7.0, loc="lower left", bbox_to_anchor=(0.0, 1.015))
    return fig


def build_overview_figure(uni_recs: list[dict], mv_recs: list[dict]) -> plt.Figure:
    """Create the 4-panel Executive Summary overview figure for the Top-100 book."""
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.4))
    fig.suptitle(
        "Run 5 Best Soup (+SymAvg+Clamp) — Top-50 Univariate & Top-50 Multivariate Book Overview",
        fontsize=13.0,
        fontweight="bold",
        y=0.975,
    )

    # Panel A: Univariate R^2 vs MAE reduction
    ax_a = axes[0, 0]
    for dom in sorted({r["domain"] for r in uni_recs}):
        sub = [r for r in uni_recs if r["domain"] == dom]
        ax_a.scatter(
            [r["r2_full"] for r in sub],
            [-r["mae_delta_pct"] for r in sub],
            s=52,
            alpha=0.86,
            color=DOMAIN_COLORS.get(dom, "#475569"),
            edgecolors="white",
            linewidth=0.6,
            label=f"{dom} (n={len(sub)})",
        )
    med_r2_u = float(np.median([r["r2_full"] for r in uni_recs]))
    med_gain_u = float(np.median([-r["mae_delta_pct"] for r in uni_recs]))
    ax_a.axvline(med_r2_u, color="#475569", ls="--", lw=1.0)
    ax_a.axhline(med_gain_u, color="#475569", ls="--", lw=1.0)
    ax_a.set_title(f"A. Top-50 Univariate: Fit Quality vs. MAE Gain (Median: {med_gain_u:.1f}%, R^2={med_r2_u:.3f})")
    ax_a.set_xlabel("Full-Horizon Forecast R^2")
    ax_a.set_ylabel("MAE Reduction vs. Seasonal Naive (%)")
    ax_a.legend(ncol=2, fontsize=7.8, loc="lower left")

    # Panel B: Univariate Domain Counts
    ax_b = axes[0, 1]
    dom_counts = pd.Series([r["domain"] for r in uni_recs]).value_counts().sort_values(ascending=True)
    bars_b = ax_b.barh(
        dom_counts.index,
        dom_counts.values,
        color=[DOMAIN_COLORS.get(d, "#475569") for d in dom_counts.index],
        alpha=0.88,
    )
    for bar, cnt in zip(bars_b, dom_counts.values):
        ax_b.text(float(cnt) + 0.25, bar.get_y() + bar.get_height() / 2.0, f"{int(cnt)}", va="center", fontsize=8.8)
    ax_b.set_xlim(0, float(dom_counts.max()) + 3.0)
    ax_b.set_title("B. Top-50 Univariate Domain Coverage (6 Benchmark Domains)")
    ax_b.set_xlabel("Number of Selected Showcase Plates")

    # Panel C: Multivariate MAE Reduction by Dataset Family
    ax_c = axes[1, 0]
    mv_df = pd.DataFrame(
        [
            {
                "family": r["family"],
                "gain_pct": -r["mae_delta_pct"],
                "has_cov": r["v_cov"] > 0,
            }
            for r in mv_recs
        ]
    )
    fam_stats = (
        mv_df.groupby("family")
        .agg(count=("gain_pct", "count"), mean_gain=("gain_pct", "mean"), max_gain=("gain_pct", "max"), has_cov=("has_cov", "max"))
        .sort_values("mean_gain", ascending=True)
    )
    colors_c = ["#0284c7" if hc else "#15803d" for hc in fam_stats["has_cov"]]
    bars_c = ax_c.barh(fam_stats.index, fam_stats["mean_gain"], color=colors_c, alpha=0.86)
    for bar, (_, fr) in zip(bars_c, fam_stats.iterrows()):
        tag = "with past_cov" if fr["has_cov"] else "coupled targets"
        ax_c.text(
            float(fr["mean_gain"]) + 0.6,
            bar.get_y() + bar.get_height() / 2.0,
            f"{fr['mean_gain']:.1f}% (n={int(fr['count'])}, {tag})",
            va="center",
            fontsize=8.2,
        )
    ax_c.set_xlim(0, float(fam_stats["mean_gain"].max()) * 1.48)
    ax_c.set_title("C. Top-50 Multivariate: Mean MAE Gain (WITH vs. WITHOUT)")
    ax_c.set_xlabel("MAE Reduction Relative to Isolated Univariate (%)")

    # Panel D: Interval Calibration (Cov80 across all 100 plates)
    ax_d = axes[1, 1]
    cov_u = [100.0 * r["cov80"] for r in uni_recs]
    cov_m = [100.0 * r["cov80_with"] for r in mv_recs]
    bins = np.linspace(55, 102, 12)
    ax_d.hist(cov_u, bins=bins, alpha=0.60, color="#2563eb", label=f"Univariate Top-50 (med={np.median(cov_u):.0f}%)")
    ax_d.hist(cov_m, bins=bins, alpha=0.55, color="#15803d", label=f"Multivariate Top-50 (med={np.median(cov_m):.0f}%)")
    ax_d.axvline(80.0, color="#dc2626", ls="--", lw=1.6, label="Nominal 80% Target")
    ax_d.set_title("D. Probabilistic Interval Calibration (Empirical [q10, q90] Coverage)")
    ax_d.set_xlabel("Empirical 80% Coverage (%)")
    ax_d.set_ylabel("Number of Showcase Plates")
    ax_d.legend(fontsize=8.2, loc="upper left")

    fig.subplots_adjust(left=0.145, right=0.980, top=0.90, bottom=0.080, hspace=0.34, wspace=0.28)
    return fig


def _add_text_page(pdf: PdfPages, title: str, subtitle: str, bullets: list[str], bg: str = "#f8fafc") -> None:
    """Append a clean typographic cover or section divider page to the PDF book."""
    fig = plt.figure(figsize=(13.6, 7.6))
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.84])
    ax.axis("off")
    ax.text(0.0, 0.86, title, fontsize=22, fontweight="bold", color="#0f172a", va="top")
    ax.text(0.0, 0.72, subtitle, fontsize=12.5, color="#334155", va="top")
    y = 0.56
    for b in bullets:
        ax.text(0.015, y, f"•  {b}", fontsize=10.6, color="#1e293b", va="top")
        y -= 0.085
    pdf.savefig(fig)
    plt.close(fig)


def _add_table_pages(pdf: PdfPages, heading: str, rows: list[list[str]], col_labels: list[str]) -> None:
    """Render 25-row catalog table pages inside the compiled PDF book."""
    chunk = 25
    for start in range(0, len(rows), chunk):
        sub = rows[start : start + chunk]
        fig = plt.figure(figsize=(13.6, 7.8))
        ax = fig.add_axes([0.03, 0.04, 0.94, 0.88])
        ax.axis("off")
        ax.set_title(
            f"{heading} (Items #{start + 1:02d}–#{start + len(sub):02d})",
            fontsize=13.0,
            fontweight="bold",
            pad=10,
        )
        tbl = ax.table(
            cellText=sub,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.7)
        tbl.scale(1.0, 1.32)
        for (r_i, _), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            if r_i == 0:
                cell.set_facecolor("#1e293b")
                cell.set_text_props(color="white", fontweight="bold")
            elif r_i % 2 == 0:
                cell.set_facecolor("#f8fafc")
        pdf.savefig(fig)
        plt.close(fig)


def export_catalogs_and_readme(uni_recs: list[dict], mv_recs: list[dict]) -> None:
    """Write CSV catalogs and a self-contained GitHub-browsable English README.md."""
    uni_rows = []
    for r in uni_recs:
        uni_rows.append(
            {
                "rank": r["rank"],
                "dataset": r["cfg_str"],
                "family": r["family"],
                "domain": r["domain"],
                "frequency": r["frequency"],
                "term": r["term"],
                "horizon": r["horizon"],
                "win_idx": r["win_idx"],
                "task_nmwql": round(r["task_nmwql"], 4),
                "task_nmase": round(r["task_nmase"], 4),
                "mae_run5": round(r["mae_run5"], 4),
                "mae_seasonal_naive": round(r["mae_sn"], 4),
                "mae_delta_pct": round(r["mae_delta_pct"], 2),
                "wql_run5": round(r["wql_run5"], 4),
                "cov80": round(r["cov80"], 4),
                "r2_view": round(r["r2_view"], 4),
                "r2_full": round(r["r2_full"], 4),
                "plot_file": f"univariate_top50/{r['fname']}",
            }
        )
    mv_rows = []
    for r in mv_recs:
        mv_rows.append(
            {
                "rank": r["rank"],
                "dataset": r["cfg_str"],
                "family": r["family"],
                "domain": r["domain"],
                "frequency": r["frequency"],
                "horizon": r["horizon"],
                "win_idx": r["win_idx"],
                "var_idx": r["var_idx"],
                "v_tgt": r["v_tgt"],
                "v_cov": r["v_cov"],
                "task_nmwql": round(r["task_nmwql"], 4),
                "task_nmase": round(r["task_nmase"], 4),
                "mae_without": round(r["mae_without"], 4),
                "mae_targets_only": None if np.isnan(r["mae_nocov"]) else round(r["mae_nocov"], 4),
                "mae_with_all": round(r["mae_with"], 4),
                "mae_delta_pct": round(r["mae_delta_pct"], 2),
                "wql_without": round(r["wql_without"], 4),
                "wql_with_all": round(r["wql_with"], 4),
                "cov80_with": round(r["cov80_with"], 4),
                "r2_view": round(r["r2_view"], 4),
                "r2_full": round(r["r2_full"], 4),
                "plot_file": f"multivariate_top50/{r['fname']}",
            }
        )

    pd.DataFrame(uni_rows).to_csv(BOOK_DIR / "univariate_top50_catalog.csv", index=False)
    pd.DataFrame(mv_rows).to_csv(BOOK_DIR / "multivariate_top50_catalog.csv", index=False)

    med_u_gain = float(np.median([-r["mae_delta_pct"] for r in uni_recs]))
    med_u_r2 = float(np.median([r["r2_full"] for r in uni_recs]))
    med_u_cov = 100.0 * float(np.median([r["cov80"] for r in uni_recs]))
    med_m_gain = float(np.median([-r["mae_delta_pct"] for r in mv_recs]))
    med_m_cov = 100.0 * float(np.median([r["cov80_with"] for r in mv_recs]))
    n_past_cov = sum(1 for r in mv_recs if r["v_cov"] > 0)

    lines = [
        "# Run 5 Best Soup (`+SymAvg+Clamp`) — Top-100 Forecast Showcase Book",
        "",
        "Comprehensive English visual compendium of the strongest zero-shot forecast windows from "
        "`checkpoints/run5_coupled_synth_mix/model_soup.safetensors` (`16,317,344` parameters, "
        "`sym_avg=True` + non-negative quantile clamping/integer snapping via `predict_entries(...).clamp_deciles`, "
        "exact bit-for-bit parity with `scripts/eval_gifteval.py`).",
        "",
        "### Methodological & Visual Rigor",
        "",
        "- **Probabilistic Intervals (`[q10, q90]` Only):** Every plot shows the **`q50` median** and **only the 80% `[q10, q90]` prediction interval band**.",
        "- **Official Univariate Baseline:** Every univariate baseline curve and `MAE SN` matches `gluonts.model.seasonal_naive.SeasonalNaivePredictor(prediction_length=H, season_length=get_seasonality(ds.freq))` (including `LastValueImputation()` on missing context values and exact `S=1` on daily/weekly frequencies in GIFT-Eval).",
        "- **Full-Horizon vs. Early-Horizon Zoom:** The Left Panel renders the entire forecast horizon `0..H-1` (matching full-horizon `MAE`, `wQL`, `R^2`, and `Cov80` badges), while on long horizons (`H > 168` in Uni or `H > 144` in MV) the Right Panel zooms into `0..fp-1` with explicit zoom-window metrics.",
        "",
        "- **Full Vector Book (108 Pages PDF):** [`Run5_Top100_Forecast_Showcase_Book.pdf`](Run5_Top100_Forecast_Showcase_Book.pdf)",
        "- **Catalog Tables:** [`univariate_top50_catalog.csv`](univariate_top50_catalog.csv) | [`multivariate_top50_catalog.csv`](multivariate_top50_catalog.csv)",
        "- **Reproducible Generator:** [`generate_showcase_book.py`](generate_showcase_book.py)",
        "",
        "## Executive Summary",
        "",
        f"- **Part I — Top 50 Univariate Plates (`univariate_top50/`):** Spans all **6 pure-univariate domains** (`Econ/Fin`, `Energy`, `Healthcare`, `Nature`, `Sales`, `Transport`), median **MAE reduction = `{med_u_gain:.1f}%`** vs. `SeasonalNaivePredictor`, median **full-horizon `R^2` = `{med_u_r2:.3f}`**, median **`Cov80` = `{med_u_cov:.0f}%`**.",
        f"- **Part II — Top 50 Multivariate & Covariate Ablation Plates (`multivariate_top50/`):** Runs the exact same window **WITH vs. WITHOUT** companion signals (`{n_past_cov}/50` plates include `past_feat_dynamic_real` and also show the intermediate **Co-Targets Only** trace when `V_tgt > 1`), achieving median **MAE reduction = `{med_m_gain:.1f}%`** over isolated univariate inference and median **`Cov80` = `{med_m_cov:.0f}%`**.",
        "",
        "![Top-100 Book Overview](overview_charts/book_summary_overview.png)",
        "",
        "---",
        "",
        "## Part I: Top 50 Zero-Shot Univariate Examples",
        "",
        "| # | Dataset Config | Domain | Freq | H | Win # | `MAE` Run 5 | `MAE` SN | Δ MAE | `wQL` | `Cov80` | `R^2` | Plot |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for r in uni_recs:
        lines.append(
            f"| {r['rank']:02d} | `{r['cfg_str']}` | {r['domain']} | `{r['frequency']}` | {r['horizon']} | "
            f"{r['win_idx']} | {fmt_num(r['mae_run5'])} | {fmt_num(r['mae_sn'])} | "
            f"**{r['mae_delta_pct']:+.1f}%** | {r['wql_run5']:.4f} | {100*r['cov80']:.0f}% | "
            f"{r['r2_full']:.3f} | [PNG](univariate_top50/{r['fname']}) |"
        )

    lines.extend(["", "<details>", "<summary><b>Gallery: View All 50 Univariate Plates Inline</b></summary>", ""])
    for r in uni_recs:
        lines.extend(
            [
                f"### Univariate #{r['rank']:02d} — `{r['cfg_str']}` (Window #{r['win_idx']}, {r['mae_delta_pct']:+.1f}% MAE vs. SN, `R^2={r['r2_full']:.3f}`)",
                "",
                f"![Univariate #{r['rank']:02d}](univariate_top50/{r['fname']})",
                "",
            ]
        )
    lines.extend(["</details>", "", "---", "", "## Part II: Top 50 Multivariate & Covariate Ablation Examples", ""])
    lines.extend(
        [
            "| # | Dataset Config | Family | Channels (`V_tgt+V_cov`) | Win / Var | `MAE` WITHOUT | `MAE` WITH | Δ MAE | `wQL` WITH | `Cov80` | `R^2` | Plot |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for r in mv_recs:
        lines.append(
            f"| {r['rank']:02d} | `{r['cfg_str']}` | {r['family']} | `{r['v_tgt']}+{r['v_cov']}` | "
            f"`w{r['win_idx']}/v{r['var_idx']}` | {fmt_num(r['mae_without'])} | {fmt_num(r['mae_with'])} | "
            f"**{r['mae_delta_pct']:+.1f}%** | {r['wql_with']:.4f} | {100*r['cov80_with']:.0f}% | "
            f"{r['r2_full']:.3f} | [PNG](multivariate_top50/{r['fname']}) |"
        )

    lines.extend(["", "<details>", "<summary><b>Gallery: View All 50 Multivariate / Covariate Ablation Plates Inline</b></summary>", ""])
    for r in mv_recs:
        lines.extend(
            [
                f"### Multivariate #{r['rank']:02d} — `{r['cfg_str']}` (Window #{r['win_idx']}, Var #{r['var_idx']}, `V={r['v_tgt']}+{r['v_cov']}`, {r['mae_delta_pct']:+.1f}% MAE)",
                "",
                f"![Multivariate #{r['rank']:02d}](multivariate_top50/{r['fname']})",
                "",
            ]
        )
    lines.append("</details>")
    (BOOK_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t_start = time.time()
    UNI_DIR.mkdir(parents=True, exist_ok=True)
    MV_DIR.mkdir(parents=True, exist_ok=True)
    OV_DIR.mkdir(parents=True, exist_ok=True)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, cfg = TSFMModel.from_pretrained(REPO_ROOT / "checkpoints" / "run5_coupled_synth_mix", device=dev)
    model.eval()
    df = pd.read_csv(REPO_ROOT / "results" / "gifteval_per_task_run5_coupled_synth_mix.csv")
    tcsv = df[df["postproc"] == "sym_avg_clamp"].copy()

    print("Mining Top 50 Univariate windows...")
    uni_recs = mine_top50_univariate(model, cfg, dev, tcsv)
    print(f"Selected {len(uni_recs)} Univariate plates ({time.time() - t_start:.1f}s)")

    t_mv = time.time()
    print("Mining Top 50 Multivariate & Covariate ablation windows...")
    mv_recs = mine_top50_multivariate(model, cfg, dev, tcsv)
    print(f"Selected {len(mv_recs)} Multivariate plates ({time.time() - t_mv:.1f}s)")

    if len(uni_recs) < 50 or len(mv_recs) < 50:
        raise RuntimeError(
            f"Insufficient selected examples: univariate={len(uni_recs)}, multivariate={len(mv_recs)}"
        )

    print("Verifying strict single-item inference, GluonTS SeasonalNaive, metric, and quantile invariants...")
    verify_showcase_rigor(model, cfg, dev, uni_recs, mv_recs)

    pdf_path = BOOK_DIR / "Run5_Top100_Forecast_Showcase_Book.pdf"
    with PdfPages(pdf_path) as pdf:
        # 1. Cover & Overview
        med_u_gain = float(np.median([-r["mae_delta_pct"] for r in uni_recs]))
        med_m_gain = float(np.median([-r["mae_delta_pct"] for r in mv_recs]))
        _add_text_page(
            pdf,
            "Run 5 Best Soup — Top-100 Forecast Showcase Book",
            "GIFT-Eval Zero-Shot Qualitative & Ablation Compendium (+SymAvg+Clamp, 16.32M Params)",
            [
                "Checkpoint: checkpoints/run5_coupled_synth_mix/model_soup.safetensors (sym_avg=True, clamp_deciles).",
                f"Part I (50 Plates): Top zero-shot Univariate forecasts across 6 domains (median MAE gain {med_u_gain:.1f}% vs. SeasonalNaivePredictor).",
                f"Part II (50 Plates): Paired WITH vs. WITHOUT GroupAttention / past_cov ablations (median MAE gain {med_m_gain:.1f}%).",
                "Each plate pairs a full context + full-horizon plot (q50 median & [q10, q90] 80% interval) with a dedicated horizon detail panel.",
            ],
        )

        ov_fig = build_overview_figure(uni_recs, mv_recs)
        ov_fig.savefig(OV_DIR / "book_summary_overview.png", dpi=200)
        pdf.savefig(ov_fig)
        plt.close(ov_fig)

        # 2. Catalog Table Pages
        uni_tbl = [
            [
                f"{r['rank']:02d}",
                r["cfg_str"],
                r["domain"],
                r["frequency"],
                str(r["horizon"]),
                str(r["win_idx"]),
                fmt_num(r["mae_run5"]),
                fmt_num(r["mae_sn"]),
                f"{r['mae_delta_pct']:+.1f}%",
                f"{r['r2_full']:.3f}",
                f"{100*r['cov80']:.0f}%",
            ]
            for r in uni_recs
        ]
        _add_table_pages(
            pdf,
            "Part I Catalog: Top 50 Univariate Plates",
            uni_tbl,
            ["Rank", "Task Config", "Domain", "Freq", "H", "Win", "MAE Run5", "MAE SN", "Delta", "R^2", "Cov80"],
        )

        mv_tbl = [
            [
                f"{r['rank']:02d}",
                r["cfg_str"],
                r["family"],
                f"{r['v_tgt']}+{r['v_cov']}",
                f"w{r['win_idx']}/v{r['var_idx']}",
                fmt_num(r["mae_without"]),
                "-" if np.isnan(r["mae_nocov"]) else fmt_num(r["mae_nocov"]),
                fmt_num(r["mae_with"]),
                f"{r['mae_delta_pct']:+.1f}%",
                f"{r['r2_full']:.3f}",
                f"{100*r['cov80_with']:.0f}%",
            ]
            for r in mv_recs
        ]
        _add_table_pages(
            pdf,
            "Part II Catalog: Top 50 Multivariate & Covariate Plates",
            mv_tbl,
            ["Rank", "Task Config", "Family", "V_t+V_c", "Win/Var", "MAE Univ", "MAE Tgt", "MAE Full", "Delta", "R^2", "Cov80"],
        )

        # 3. Part I Section & 50 Univariate Plates
        _add_text_page(
            pdf,
            "Part I: Top 50 Univariate Forecast Plates",
            "Zero-Shot Probabilistic Forecasts (Median q50 & [q10, q90] 80% Prediction Interval) vs. Official GluonTS Seasonal Naive",
            [
                f"Domains represented: {', '.join(sorted({r['domain'] for r in uni_recs}))}",
                f"Distinct benchmark configs: {len({r['cfg_str'] for r in uni_recs})} | Dataset families: {len({r['family'] for r in uni_recs})}",
                f"Minimum full-horizon R^2 across all 50 plates: {min(r['r2_full'] for r in uni_recs):.3f}",
                f"Minimum MAE reduction vs. SeasonalNaivePredictor: {min(-r['mae_delta_pct'] for r in uni_recs):.1f}%",
            ],
            bg="#f0f9ff",
        )
        for old_png in UNI_DIR.glob("*.png"):
            old_png.unlink()
        for rec in uni_recs:
            fname = f"uni_{rec['rank']:02d}_{sanitize_name(rec['cfg_str'])}_w{rec['win_idx']}.png"
            rec["fname"] = fname
            fig = build_univariate_figure(rec)
            fig.savefig(UNI_DIR / fname, dpi=175)
            pdf.savefig(fig)
            plt.close(fig)

        # 4. Part II Section & 50 Multivariate Plates
        _add_text_page(
            pdf,
            "Part II: Top 50 Multivariate & Covariate Ablations",
            "Same Window Run WITH vs. WITHOUT Coupled Targets & Past Covariates ([q10, q90] Intervals)",
            [
                f"Plates with past_feat_dynamic_real covariates: {sum(1 for r in mv_recs if r['v_cov'] > 0)} / 50",
                f"Distinct benchmark configs: {len({r['cfg_str'] for r in mv_recs})} | Families: {', '.join(sorted({r['family'] for r in mv_recs}))}",
                f"Median MAE improvement WITH vs. WITHOUT auxiliary channels: {med_m_gain:.1f}%",
                "Note: past_cov channels are active strictly for t < 0 (masked at t >= 0), while coupled co-targets forecast jointly.",
            ],
            bg="#f0fdf4",
        )
        for old_png in MV_DIR.glob("*.png"):
            old_png.unlink()
        for rec in mv_recs:
            fname = f"mv_{rec['rank']:02d}_{sanitize_name(rec['cfg_str'])}_w{rec['win_idx']}_v{rec['var_idx']}.png"
            rec["fname"] = fname
            fig = build_multivariate_figure(rec)
            fig.savefig(MV_DIR / fname, dpi=175)
            pdf.savefig(fig)
            plt.close(fig)

    export_catalogs_and_readme(uni_recs, mv_recs)
    print(f"Book build complete in {time.time() - t_start:.1f}s -> {pdf_path}")


if __name__ == "__main__":
    main()
