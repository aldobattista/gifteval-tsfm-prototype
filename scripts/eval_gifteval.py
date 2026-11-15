"""Evaluate trained TSFM prototype soup checkpoints on all 97 official GIFT-Eval configs.

Supports `--smoke` (3 critical edge-case tasks with mandatory < 1e-6 GluonTS parity
assertion), full 97-task evaluation with Raw Neural vs +Clamp/Snap vs +SymAvg
decompositions, and 20/64-step pre-origin blackout stress testing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tsfm_proto.config import DECILE_LEVELS, TSFMConfig
from tsfm_proto.data import normalize_pandas_freq
from tsfm_proto.eval_metrics import (
    EvalEntry,
    predict_entries,
    score_task_fast,
)
from tsfm_proto.model import TSFMModel

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_gifteval_dir() -> Path:
    env_dir = os.environ.get("GIFT_EVAL")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_PROJECT_ROOT / "data" / "GiftEval"


def _default_official_dir() -> Path | None:
    env_dir = os.environ.get("GIFT_EVAL_OFFICIAL_RESULTS")
    if env_dir:
        cand_env = Path(env_dir)
        if cand_env.exists():
            return cand_env
    candidate = DEFAULT_PROJECT_ROOT / "data" / "gift_eval_official_results"
    return candidate if candidate.exists() else None


DEFAULT_GIFTEVAL_DIR = _default_gifteval_dir()

PRETTY_REV = {
    "saugeen": "saugeenday",
    "temperature_rain": "temperature_rain_with_missing",
    "kdd_cup_2018": "kdd_cup_2018_with_missing",
    "car_parts": "car_parts_with_missing",
}

SMOKE_CONFIGS = (
    "m4_yearly/A/short",
    "electricity/W/short",
    "bizitobs_application/10S/short",
)


@dataclass(frozen=True)
class TaskSpec:
    config_str: str
    family: str
    ds_name: str
    term: str
    domain: str
    frequency: str
    sn_mase: float
    sn_mwql: float


def _resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_task_specs(results_dir: Path, gifteval_dir: Path, smoke: bool) -> list[TaskSpec]:
    sn_df = pd.read_csv(results_dir / "seasonal_naive_official_reference.csv")
    props = json.loads((results_dir / "gift_eval_dataset_properties.json").read_text())

    specs: list[TaskSpec] = []
    for _, row in sn_df.iterrows():
        cfg_str = str(row["dataset"])
        if smoke and cfg_str not in SMOKE_CONFIGS:
            continue
        ds_part, freq_part, term = cfg_str.rsplit("/", 2)
        base = PRETTY_REV.get(ds_part, ds_part)
        name = f"{base}/{freq_part}" if (gifteval_dir / base / freq_part).is_dir() else base
        prop = props[ds_part]
        specs.append(
            TaskSpec(
                config_str=cfg_str,
                family=ds_part,
                ds_name=name,
                term=term,
                domain=str(prop["domain"]),
                frequency=freq_part,
                sn_mase=float(row["eval_metrics/MASE[0.5]"]),
                sn_mwql=float(row["eval_metrics/mean_weighted_sum_quantile_loss"]),
            )
        )
    return specs


def _load_dataset_entries(ds) -> tuple[list[EvalEntry], list[dict], int]:
    """Extract normalized 2D ``[V, T]`` context/covariate entries plus GluonTS inputs."""
    try:
        from gluonts.time_feature import get_seasonality
    except ImportError as exc:
        raise ImportError(
            "`gluonts` is required for live GIFT-Eval dataset loading. Install via `pip install -e \".[eval]\"`."
        ) from exc

    season_len = int(get_seasonality(normalize_pandas_freq(ds.freq)))
    pred_len = int(ds.prediction_length)
    entries: list[EvalEntry] = []
    raw_inputs: list[dict] = []

    for inp, lbl in zip(ds.test_data.input, ds.test_data.label):
        tgt = np.asarray(inp["target"], dtype=np.float32)
        if tgt.ndim == 1:
            tgt = tgt[None, :]
        fut = np.asarray(lbl["target"], dtype=np.float32)
        if fut.ndim == 1:
            fut = fut[None, :]
        cov_raw = inp.get("past_feat_dynamic_real")
        if cov_raw is not None:
            cov = np.asarray(cov_raw, dtype=np.float32)
            if cov.ndim == 1:
                cov = cov[None, :]
        else:
            cov = None
        entries.append(
            EvalEntry(
                target=tgt,
                past_cov=cov,
                prediction_length=pred_len,
                future_target=fut,
                season_length=season_len,
            )
        )
        raw_inputs.append(inp)
    return entries, raw_inputs, season_len


def _verify_gluonts_parity(
    ds,
    raw_inputs: list[dict],
    decile_preds: list[np.ndarray],
    fast_metrics: dict[str, float],
    season_len: int,
    cfg_str: str,
) -> None:
    """Assert vectorized ``score_task_fast`` matches official GluonTS to < 1e-6."""
    try:
        from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
        from gluonts.model import evaluate_forecasts
        from gluonts.model.forecast import QuantileForecast
    except ImportError as exc:
        raise ImportError(
            "`gluonts` is required for `--smoke` GluonTS parity verification. Install via `pip install -e \".[eval]\"`."
        ) from exc

    keys = [str(q) for q in DECILE_LEVELS]
    forecasts: list[QuantileForecast] = []
    for inp, pred_dec in zip(raw_inputs, decile_preds):
        arr = pred_dec[:, :, 0] if inp["target"].ndim == 1 else pred_dec
        forecasts.append(
            QuantileForecast(
                forecast_arrays=arr,
                forecast_keys=keys,
                start_date=inp["start"] + inp["target"].shape[-1],
                item_id=inp.get("item_id"),
            )
        )
    ref_df = evaluate_forecasts(
        forecasts,
        test_data=ds.test_data,
        metrics=[MASE(), MeanWeightedSumQuantileLoss(DECILE_LEVELS)],
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_len,
    )
    ref_mase = float(ref_df["MASE[0.5]"].iloc[0])
    ref_mwql = float(ref_df["mean_weighted_sum_quantile_loss"].iloc[0])
    diff_mase = abs(fast_metrics["MASE[0.5]"] - ref_mase)
    diff_mwql = abs(fast_metrics["mean_weighted_sum_quantile_loss"] - ref_mwql)
    if diff_mase >= 1e-6 or diff_mwql >= 1e-6:
        raise AssertionError(
            f"GluonTS parity failure on {cfg_str}: "
            f"MASE diff={diff_mase:.3e}, MWQL diff={diff_mwql:.3e}"
        )


def _geo_mean(vals: pd.Series) -> float:
    arr = vals.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    return float(np.exp(np.mean(np.log(arr)))) if arr.size > 0 else float("nan")


def _compute_slices(df: pd.DataFrame, run_name: str, postproc: str) -> list[dict[str, object]]:
    sub = df[df["postproc"] == postproc]
    if sub.empty:
        return []
    freq_base = (
        sub["frequency"]
        .astype(str)
        .str.split("-")
        .str[0]
        .str.replace(r"^\d+", "", regex=True)
        .str.upper()
    )
    is_low_res = freq_base.isin(["A", "AS", "AE", "Y", "YS", "YE", "Q", "QS", "QE", "M", "MS", "ME", "W"])
    slices: dict[str, pd.Series] = {
        "Overall (97)": pd.Series(True, index=sub.index),
        "Term: short": sub["term"] == "short",
        "Term: medium": sub["term"] == "medium",
        "Term: long": sub["term"] == "long",
        "Horizon <= 512": sub["horizon"] <= 512,
        "Horizon > 512": sub["horizon"] > 512,
        "Univariate": ~sub["is_mv"],
        "Multivariate": sub["is_mv"],
        "Past Covariates": sub["has_past_cov"],
        "Missing-Flagged Datasets": sub["is_missing_ds"],
        "Low-Res Freq (A/Q/M/W)": is_low_res,
        "High-Res Freq (D/H/Min/S)": ~is_low_res,
    }
    for dom in sorted(sub["domain"].unique()):
        slices[f"Domain: {dom}"] = sub["domain"] == dom

    rows: list[dict[str, object]] = []
    for sname, mask in slices.items():
        part = sub[mask]
        if part.empty:
            continue
        rows.append(
            {
                "run_name": run_name,
                "postproc": postproc,
                "slice": sname,
                "n_tasks": int(len(part)),
                "geo_nmwql": round(_geo_mean(part["nmwql"]), 6),
                "geo_nmase": round(_geo_mean(part["nmase"]), 6),
                "mean_cov80": round(float(part["cov80"].mean()), 6),
            }
        )
    return rows


def _build_seasonal_naive_deciles(entries: list[EvalEntry]) -> list[np.ndarray]:
    """Build deterministic Seasonal-Naive decile forecasts matching official horizon padding."""
    sn_decs: list[np.ndarray] = []
    for ent in entries:
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
        sn_decs.append(sn_dec)
    return sn_decs


def _export_leaderboard_comparison(
    results_dir: Path,
    official_dir: Path | None,
    per_run_raw_dfs: dict[str, pd.DataFrame],
    per_run_sym_dfs: dict[str, pd.DataFrame],
) -> None:
    """Compare all 6 prototype runs against official 97-task GIFT-Eval baselines on exact SN parity."""
    if official_dir is None or not official_dir.exists():
        return

    sn_path = results_dir / "seasonal_naive_official_reference.csv"
    if not sn_path.exists():
        return
    sn_df = pd.read_csv(sn_path)[
        ["dataset", "eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss"]
    ].rename(
        columns={
            "eval_metrics/MASE[0.5]": "sn_mase",
            "eval_metrics/mean_weighted_sum_quantile_loss": "sn_mwql",
        }
    )

    curated_models: list[tuple[str, str]] = [
        ("Naive", "Statistical"),
        ("Auto_Theta", "Statistical"),
        ("Seasonal_Naive", "Statistical"),
        ("Auto_Arima", "Statistical"),
        ("TimesFM", "Pretrained TSFM (200M)"),
        ("Chronos_small", "Pretrained TSFM (46M)"),
        ("Chronos_base", "Pretrained TSFM (200M)"),
        ("Moirai_small", "Pretrained TSFM (14M)"),
        ("Chronos_large", "Pretrained TSFM (710M)"),
        ("iTransformer", "Task-Trained Deep"),
        ("Moirai_base", "Pretrained TSFM (91M)"),
        ("TFT", "Task-Trained Deep"),
        ("Moirai_large", "Pretrained TSFM (311M)"),
        ("PatchTST", "Task-Trained Deep"),
        ("chronos_bolt_small", "Pretrained TSFM (48M)"),
        ("chronos_bolt_base", "Pretrained TSFM (205M)"),
        ("sundial_base_128m", "Pretrained TSFM (128M)"),
        ("timesfm_2_0_500m", "Pretrained TSFM (500M)"),
        ("TabPFN-TS", "Pretrained TSFM"),
        ("Chronos-2-Synth", "Pretrained TSFM (120M, Synth)"),
        ("t0-alpha", "Pretrained TSFM (TFC)"),
        ("TimesFM-2.5", "Pretrained TSFM (200M)"),
        ("TiRex", "Pretrained TSFM (35M)"),
        ("Chronos-2", "Pretrained TSFM (120M)"),
        ("Timer-S1", "Pretrained TSFM"),
        ("t0-beta", "Pretrained TSFM (TFC)"),
        ("t0-gamma", "Pretrained TSFM (TFC)"),
    ]

    # Compute ranks across all complete 97-task submissions in official_dir plus our prototype runs.
    all_official_scores: dict[str, float] = {}
    for p in sorted(official_dir.glob("*/all_results.csv")):
        odf = pd.read_csv(p)
        if len(odf) != 97:
            continue
        merged = odf.merge(sn_df, on="dataset")
        if len(merged) != 97:
            continue
        all_official_scores[p.parent.name] = _geo_mean(
            merged["eval_metrics/mean_weighted_sum_quantile_loss"] / merged["sn_mwql"]
        )

    def _summarize_task_frame(label: str, category: str, df_97: pd.DataFrame) -> dict[str, object]:
        excl_10s = ~df_97["dataset"].astype(str).str.contains("/10S/")
        raw_geo_nmwql = _geo_mean(df_97["nmwql"])
        return {
            "model": label,
            "category": category,
            "geo_nmwql": round(raw_geo_nmwql, 6),
            "geo_nmase": round(_geo_mean(df_97["nmase"]), 6),
            "geo_nmwql_excl_10s": round(_geo_mean(df_97.loc[excl_10s, "nmwql"]), 6),
            "wins_vs_sn_nmwql": int((df_97["nmwql"] < 1.0).sum()),
            "wins_vs_sn_nmase": int((df_97["nmase"] < 1.0).sum()),
            "n_tasks": int(len(df_97)),
            "_raw_geo_nmwql": raw_geo_nmwql,
        }

    rows: list[dict[str, object]] = []
    for model_name, cat in curated_models:
        csv_p = official_dir / model_name / "all_results.csv"
        if not csv_p.exists():
            continue
        odf = pd.read_csv(csv_p)
        if len(odf) != 97:
            continue
        m = odf.merge(sn_df, on="dataset")
        m["nmwql"] = m["eval_metrics/mean_weighted_sum_quantile_loss"] / m["sn_mwql"]
        m["nmase"] = m["eval_metrics/MASE[0.5]"] / m["sn_mase"]
        rows.append(_summarize_task_frame(model_name, cat, m))

    for run_name, raw_df in per_run_raw_dfs.items():
        rows.append(_summarize_task_frame(f"{run_name} (Raw)", "Our Prototype (14.8M-16.3M)", raw_df))
        if run_name in per_run_sym_dfs:
            rows.append(
                _summarize_task_frame(
                    f"{run_name} (+SymAvg+Clamp)",
                    "Our Prototype (14.8M-16.3M, Postproc)",
                    per_run_sym_dfs[run_name],
                )
            )

    out_df = pd.DataFrame(rows).sort_values("_raw_geo_nmwql", ascending=True).reset_index(drop=True)
    ref_vals = sorted(all_official_scores.values())
    out_df["official_rank_of_141"] = [
        int(1 + sum(1 for v in ref_vals if v < float(g) - 1e-12))
        for g in out_df["_raw_geo_nmwql"]
    ]
    out_df = out_df.drop(columns=["_raw_geo_nmwql"])
    out_df.insert(0, "table_rank", np.arange(1, len(out_df) + 1))
    out_df.to_csv(results_dir / "gifteval_leaderboard_comparison.csv", index=False)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Evaluate TSFM soups on GIFT-Eval.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "run0_v0_baseline",
            "run1_data_balanced",
            "run2_missingness_suffix_cpm",
            "run3_probit_localscale_lags",
            "run4_multivariate_group_attn",
            "run5_coupled_synth_mix",
        ],
    )
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_PROJECT_ROOT / "checkpoints")
    parser.add_argument("--gifteval-dir", type=Path, default=DEFAULT_GIFTEVAL_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_PROJECT_ROOT / "results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--blackout-max-windows", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-blackout", action="store_true")
    parser.add_argument(
        "--recompute-summaries-only",
        action="store_true",
        help="Recompute summary, slice, and leaderboard CSVs from saved per-task CSVs.",
    )
    parser.add_argument(
        "--official-leaderboard-dir",
        type=Path,
        default=_default_official_dir(),
        help="Optional directory containing GIFT-Eval `<model>/all_results.csv` leaderboard files.",
    )
    parser.add_argument(
        "--sym-avg-runs",
        nargs="*",
        default=["run4_multivariate_group_attn", "run5_coupled_synth_mix"],
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=os.environ.get("TSFM_HF_REPO_ID") or None,
        help="Optional Hugging Face Hub repo ID fallback when local safetensors checkpoints are absent.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.recompute_summaries_only:
        summary_rows: list[dict[str, object]] = []
        slice_rows: list[dict[str, object]] = []
        per_run_raw_dfs: dict[str, pd.DataFrame] = {}
        per_run_sym_dfs: dict[str, pd.DataFrame] = {}
        existing_summary_path = args.results_dir / "gifteval_ablation_summary.csv"
        param_map: dict[str, int] = {}
        if existing_summary_path.exists():
            old_sum = pd.read_csv(existing_summary_path)
            for _, srow in old_sum.iterrows():
                param_map[str(srow["run_name"])] = int(srow["n_params"])

        for ckpt_arg in args.checkpoints:
            run_name = Path(ckpt_arg).name
            per_task_path = args.results_dir / f"gifteval_per_task_{run_name}.csv"
            if not per_task_path.exists():
                raise FileNotFoundError(f"Missing per-task CSV for --recompute-summaries-only: {per_task_path}")
            run_df = pd.read_csv(per_task_path)
            if "frequency" not in run_df.columns:
                run_df["frequency"] = run_df["dataset"].astype(str).apply(lambda s: s.rsplit("/", 2)[1])

            n_params = param_map.get(run_name, 0)
            if n_params == 0:
                cfg_path = (Path(ckpt_arg) if Path(ckpt_arg).exists() else args.ckpt_root / run_name) / "config.json"
                if cfg_path.exists():
                    n_params = TSFMModel(TSFMConfig.from_json(cfg_path)).count_parameters()
                else:
                    yaml_path = DEFAULT_PROJECT_ROOT / "configs" / f"{run_name}.yaml"
                    if yaml_path.exists():
                        n_params = TSFMModel(TSFMConfig.from_yaml(yaml_path)).count_parameters()

            per_run_raw_dfs[run_name] = run_df[run_df["postproc"] == "raw"].reset_index(drop=True)
            sym_sub = run_df[run_df["postproc"] == "sym_avg_clamp"]
            if not sym_sub.empty:
                per_run_sym_dfs[run_name] = sym_sub.reset_index(drop=True)

            for pp_name in run_df["postproc"].unique():
                sub_pp = run_df[run_df["postproc"] == pp_name]
                summary_rows.append(
                    {
                        "run_name": run_name,
                        "postproc": pp_name,
                        "n_params": n_params,
                        "n_tasks": int(len(sub_pp)),
                        "geo_nmwql": round(_geo_mean(sub_pp["nmwql"]), 6),
                        "geo_nmase": round(_geo_mean(sub_pp["nmase"]), 6),
                        "mean_cov80": round(float(sub_pp["cov80"].mean()), 6),
                    }
                )
                slice_rows.extend(_compute_slices(run_df, run_name, pp_name))

        updated_runs = {str(r["run_name"]) for r in summary_rows}
        for out_path, new_rows in (
            (args.results_dir / "gifteval_ablation_summary.csv", summary_rows),
            (args.results_dir / "gifteval_slices_summary.csv", slice_rows),
        ):
            new_df = pd.DataFrame(new_rows)
            if out_path.exists():
                old_df = pd.read_csv(out_path)
                if not updated_runs.issuperset(set(old_df["run_name"].astype(str))):
                    kept = old_df[~old_df["run_name"].astype(str).isin(updated_runs)]
                    new_df = pd.concat([kept, new_df], ignore_index=True)
            new_df.to_csv(out_path, index=False)

        for other_csv in sorted(args.results_dir.glob("gifteval_per_task_*.csv")):
            other_run = other_csv.stem[len("gifteval_per_task_") :]
            if other_run in per_run_raw_dfs:
                continue
            other_df = pd.read_csv(other_csv)
            per_run_raw_dfs[other_run] = other_df[other_df["postproc"] == "raw"].reset_index(drop=True)
            other_sym = other_df[other_df["postproc"] == "sym_avg_clamp"]
            if not other_sym.empty:
                per_run_sym_dfs[other_run] = other_sym.reset_index(drop=True)

        _export_leaderboard_comparison(args.results_dir, args.official_leaderboard_dir, per_run_raw_dfs, per_run_sym_dfs)
        if args.official_leaderboard_dir is not None and args.official_leaderboard_dir.exists():
            print("Recomputed summary, slice, and leaderboard CSVs.")
        else:
            print("Recomputed summary and slice CSVs (official leaderboard dir not present; kept existing leaderboard CSV).")
        return

    if not args.gifteval_dir.exists():
        raise FileNotFoundError(
            f"GIFT-Eval directory not found at {args.gifteval_dir}. "
            "Download via `huggingface-cli download Salesforce/gift-eval --repo-type dataset --local-dir data/GiftEval`, "
            "set `GIFT_EVAL=/path/to/GiftEval`, or pass `--recompute-summaries-only` for offline CSV recomputation."
        )

    os.environ["GIFT_EVAL"] = str(args.gifteval_dir)
    env_ge_src = os.environ.get("GIFT_EVAL_SRC")
    if env_ge_src and Path(env_ge_src).exists() and env_ge_src not in sys.path:
        sys.path.insert(0, env_ge_src)
    try:
        from gift_eval.data import Dataset
    except ImportError as exc:
        raise ImportError(
            "`gift_eval` is required for checkpoint evaluation. Install via "
            "`pip install git+https://github.com/SalesforceAIResearch/gift-eval.git` "
            "(or pass `GIFT_EVAL_SRC=/path/to/gift-eval/src`), or run with "
            "`--recompute-summaries-only` to rebuild summary CSVs offline."
        ) from exc

    device = _resolve_device(args.device)
    task_specs = _load_task_specs(args.results_dir, args.gifteval_dir, smoke=args.smoke)

    models_bundle: list[tuple[str, TSFMConfig, TSFMModel]] = []
    for ckpt_arg in args.checkpoints:
        model, cfg = TSFMModel.from_pretrained(
            ckpt_arg,
            checkpoints_root=args.ckpt_root,
            hf_repo_id=args.hf_repo_id,
            device=device,
        )
        models_bundle.append((cfg.run_name, cfg, model))

    print(
        f"Loaded {len(models_bundle)} soup checkpoints onto {device}; "
        f"evaluating {len(task_specs)} GIFT-Eval tasks in a single dataset pass..."
    )

    task_records_by_run: dict[str, list[dict[str, object]]] = {rname: [] for rname, _, _ in models_bundle}
    blackout_accum: dict[tuple[str, int], dict[str, list[float]]] = {}
    for rname, _, _ in models_bundle:
        for b_steps in (0, 20, 64):
            blackout_accum[(rname, b_steps)] = {"nmwql": [], "nmase": []}
    n_blackout_tasks = 0

    for t_idx, spec in enumerate(task_specs, start=1):
        ds = Dataset(name=spec.ds_name, term=spec.term, to_univariate=False)
        entries, raw_inputs, season_len = _load_dataset_entries(ds)
        is_mv = bool(ds.target_dim > 1)
        has_cov = bool(ds.past_feat_dynamic_real_dim > 0)
        is_miss = "_with_missing" in spec.ds_name
        med_ctx = float(np.median([e.target.shape[1] for e in entries]))
        do_blackout = (
            (not args.smoke)
            and (not args.skip_blackout)
            and (med_ctx >= 96.0)
            and (spec.frequency not in {"A", "Y", "Q"})
        )

        sub_idx: np.ndarray | None = None
        sub_entries: list[EvalEntry] = []
        sn_sub_mase = spec.sn_mase
        sn_sub_mwql = spec.sn_mwql
        if do_blackout:
            n_blackout_tasks += 1
            n_sub = min(len(entries), max(1, int(args.blackout_max_windows)))
            sub_idx = np.linspace(0, len(entries) - 1, n_sub, dtype=np.int64)
            sub_entries = [entries[int(i)] for i in sub_idx]
            sn_sub = score_task_fast(sub_entries, _build_seasonal_naive_deciles(sub_entries))
            if np.isfinite(sn_sub["MASE[0.5]"]) and sn_sub["MASE[0.5]"] > 0.0:
                sn_sub_mase = sn_sub["MASE[0.5]"]
            if (
                np.isfinite(sn_sub["mean_weighted_sum_quantile_loss"])
                and sn_sub["mean_weighted_sum_quantile_loss"] > 0.0
            ):
                sn_sub_mwql = sn_sub["mean_weighted_sum_quantile_loss"]

        for run_name, cfg, model in models_bundle:
            preds = predict_entries(
                model,
                cfg,
                entries,
                device=device,
                batch_size=args.batch_size,
                apply_blackout_steps=0,
                sym_avg=False,
                use_autocast=(device.type in {"mps", "cuda"}),
            )

            raw_m = score_task_fast(entries, preds.raw_deciles)
            clamp_m = score_task_fast(entries, preds.clamp_deciles)
            if args.smoke:
                _verify_gluonts_parity(
                    ds, raw_inputs, preds.raw_deciles, raw_m, season_len, spec.config_str
                )
                print(
                    f"  [{run_name} smoke parity OK <1e-6] {spec.config_str} | "
                    f"MASE={raw_m['MASE[0.5]']:.4f} MWQL={raw_m['mean_weighted_sum_quantile_loss']:.4f}"
                )

            for pp_name, m_dict in (("raw", raw_m), ("clamp_snap", clamp_m)):
                task_records_by_run[run_name].append(
                    {
                        "run_name": run_name,
                        "postproc": pp_name,
                        "dataset": spec.config_str,
                        "family": spec.family,
                        "domain": spec.domain,
                        "frequency": spec.frequency,
                        "term": spec.term,
                        "horizon": int(ds.prediction_length),
                        "is_mv": is_mv,
                        "has_past_cov": has_cov,
                        "is_missing_ds": is_miss,
                        "n_items": len(entries),
                        "mase": m_dict["MASE[0.5]"],
                        "mwql": m_dict["mean_weighted_sum_quantile_loss"],
                        "nmase": m_dict["MASE[0.5]"] / spec.sn_mase,
                        "nmwql": m_dict["mean_weighted_sum_quantile_loss"] / spec.sn_mwql,
                        "cov80": m_dict["coverage_80"],
                    }
                )

            if (not args.smoke) and (run_name in args.sym_avg_runs):
                sym_preds = predict_entries(
                    model,
                    cfg,
                    entries,
                    device=device,
                    batch_size=args.batch_size,
                    apply_blackout_steps=0,
                    sym_avg=True,
                    use_autocast=(device.type in {"mps", "cuda"}),
                )
                sym_m = score_task_fast(entries, sym_preds.clamp_deciles)
                task_records_by_run[run_name].append(
                    {
                        "run_name": run_name,
                        "postproc": "sym_avg_clamp",
                        "dataset": spec.config_str,
                        "family": spec.family,
                        "domain": spec.domain,
                        "frequency": spec.frequency,
                        "term": spec.term,
                        "horizon": int(ds.prediction_length),
                        "is_mv": is_mv,
                        "has_past_cov": has_cov,
                        "is_missing_ds": is_miss,
                        "n_items": len(entries),
                        "mase": sym_m["MASE[0.5]"],
                        "mwql": sym_m["mean_weighted_sum_quantile_loss"],
                        "nmase": sym_m["MASE[0.5]"] / spec.sn_mase,
                        "nmwql": sym_m["mean_weighted_sum_quantile_loss"] / spec.sn_mwql,
                        "cov80": sym_m["coverage_80"],
                    }
                )

            if do_blackout and sub_idx is not None:
                sub_raw0 = [preds.raw_deciles[int(i)] for i in sub_idx]
                bm0 = score_task_fast(sub_entries, sub_raw0)
                blackout_accum[(run_name, 0)]["nmwql"].append(
                    bm0["mean_weighted_sum_quantile_loss"] / sn_sub_mwql
                )
                blackout_accum[(run_name, 0)]["nmase"].append(bm0["MASE[0.5]"] / sn_sub_mase)
                for b_steps in (20, 64):
                    b_preds = predict_entries(
                        model,
                        cfg,
                        sub_entries,
                        device=device,
                        batch_size=args.batch_size,
                        apply_blackout_steps=b_steps,
                        sym_avg=False,
                        use_autocast=(device.type in {"mps", "cuda"}),
                    )
                    bm = score_task_fast(sub_entries, b_preds.raw_deciles)
                    blackout_accum[(run_name, b_steps)]["nmwql"].append(
                        bm["mean_weighted_sum_quantile_loss"] / sn_sub_mwql
                    )
                    blackout_accum[(run_name, b_steps)]["nmase"].append(bm["MASE[0.5]"] / sn_sub_mase)

        if (not args.smoke) and (t_idx % 10 == 0 or t_idx == len(task_specs)):
            print(f"  [{t_idx}/{len(task_specs)}] {spec.config_str} ({len(entries)} windows) complete")

    if args.smoke:
        print(
            f"Smoke parity verified (< 1e-6 vs GluonTS) across {len(models_bundle)} checkpoints x "
            f"{len(task_specs)} tasks (skipping full results/ CSV export)."
        )
        return

    summary_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    blackout_rows: list[dict[str, object]] = []
    per_run_raw_dfs: dict[str, pd.DataFrame] = {}
    per_run_sym_dfs: dict[str, pd.DataFrame] = {}

    for run_name, _, model in models_bundle:
        run_df = pd.DataFrame(task_records_by_run[run_name])
        per_task_path = args.results_dir / f"gifteval_per_task_{run_name}.csv"
        run_df.to_csv(per_task_path, index=False)

        raw_sub = run_df[run_df["postproc"] == "raw"].reset_index(drop=True)
        per_run_raw_dfs[run_name] = raw_sub
        sym_sub = run_df[run_df["postproc"] == "sym_avg_clamp"]
        if not sym_sub.empty:
            per_run_sym_dfs[run_name] = sym_sub.reset_index(drop=True)

        for pp_name in run_df["postproc"].unique():
            sub_pp = run_df[run_df["postproc"] == pp_name]
            summary_rows.append(
                {
                    "run_name": run_name,
                    "postproc": pp_name,
                    "n_params": model.count_parameters(),
                    "n_tasks": int(len(sub_pp)),
                    "geo_nmwql": round(_geo_mean(sub_pp["nmwql"]), 6),
                    "geo_nmase": round(_geo_mean(sub_pp["nmase"]), 6),
                    "mean_cov80": round(float(sub_pp["cov80"].mean()), 6),
                }
            )
            slice_rows.extend(_compute_slices(run_df, run_name, pp_name))

        if (not args.smoke) and (not args.skip_blackout) and n_blackout_tasks > 0:
            for b_steps in (0, 20, 64):
                acc = blackout_accum[(run_name, b_steps)]
                blackout_rows.append(
                    {
                        "run_name": run_name,
                        "blackout_steps": b_steps,
                        "n_tasks": n_blackout_tasks,
                        "geo_nmwql": round(_geo_mean(pd.Series(acc["nmwql"])), 6),
                        "geo_nmase": round(_geo_mean(pd.Series(acc["nmase"])), 6),
                    }
                )

    pd.DataFrame(summary_rows).to_csv(args.results_dir / "gifteval_ablation_summary.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(args.results_dir / "gifteval_slices_summary.csv", index=False)
    if blackout_rows:
        pd.DataFrame(blackout_rows).to_csv(args.results_dir / "gifteval_blackout_stress.csv", index=False)

    _export_leaderboard_comparison(
        args.results_dir, args.official_leaderboard_dir, per_run_raw_dfs, per_run_sym_dfs
    )

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
